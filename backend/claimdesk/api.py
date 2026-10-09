"""HTTP API. Run with: uvicorn claimdesk.api:app --reload"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .app_state import build_components
from .config import Settings
from .extraction import MAX_UPLOAD_BYTES, DocumentError, read_document
from .guardrails import mask_pii
from .models import (
    AnswerOut,
    AskRequest,
    ClaimAssessment,
    ClaimText,
    QueuedClaim,
    ReviewDecision,
    ReviewStatus,
)
from .ratelimit import RateLimiter, RequestIdFilter, new_request_id, request_id
from .review_queue import QueueError
from .store import list_samples, read_sample

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s [%(request_id)s]: %(message)s",
)
for handler in logging.getLogger().handlers:
    handler.addFilter(RequestIdFilter())
log = logging.getLogger("claimdesk.api")

# Routes that can call the model, and so cost money per request.
RATE_LIMITED = frozenset({"/api/ask", "/api/claims", "/api/claims/assess", "/api/claims/upload"})

# backend/claimdesk/api.py -> repository root -> frontend/dist
DEFAULT_STATIC_DIR = Path(__file__).resolve().parents[2] / "frontend" / "dist"


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    components = build_components(settings)

    app = FastAPI(
        title="ClaimDesk API",
        version=__version__,
        summary="Benefits Q&A with citations, a tool-calling agent and claim intake checks.",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.claimdesk_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "Retry-After"],
    )

    limiter = RateLimiter(settings.claimdesk_rate_limit_per_minute)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        token = request_id.set(new_request_id(request.headers.get("x-request-id")))
        try:
            wait = _limited(request)
            if wait:
                response = JSONResponse(
                    {"detail": "Too many requests. Try again shortly."},
                    status_code=429,
                    headers={"Retry-After": str(math.ceil(wait))},
                )
            else:
                response = await call_next(request)
            response.headers["X-Request-ID"] = request_id.get()
            elapsed = (time.perf_counter() - started) * 1000
            log.info(
                "%s %s %s %.0fms", request.method, request.url.path, response.status_code, elapsed
            )
            return response
        finally:
            request_id.reset(token)

    def _limited(request: Request) -> float:
        if request.method != "POST" or request.url.path not in RATE_LIMITED:
            return 0.0
        return limiter.check(request.client.host if request.client else "unknown")

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "llm_provider": components.provider,
            "retrieval": components.retriever.mode,
            "sections": len(components.retriever.chunks),
            "as_of": settings.claimdesk_as_of.isoformat(),
            "queue": components.queue.counts(),
        }

    @app.post("/api/ask", response_model=AnswerOut)
    def ask(body: AskRequest) -> AnswerOut:
        log.info("question: %s", mask_pii(body.question))
        return components.agent.answer(body.question, body.member_id, body.history)

    @app.post("/api/claims/assess", response_model=ClaimAssessment)
    def assess(body: ClaimText) -> ClaimAssessment:
        return components.assessor.assess(body.text)

    @app.post("/api/claims/upload", response_model=ClaimAssessment)
    async def upload(file: Annotated[UploadFile, File()]) -> ClaimAssessment:
        data = await file.read(MAX_UPLOAD_BYTES + 1)
        try:
            text = read_document(data, file.filename or "")
        except DocumentError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return components.assessor.assess(text)

    # ----- examiner queue --------------------------------------------------------------------

    @app.post("/api/claims", response_model=QueuedClaim, status_code=201)
    def submit(body: ClaimText) -> QueuedClaim:
        """Assesses a claim and puts it in the queue for a claims examiner."""
        return components.queue.submit(body.text, components.assessor.assess(body.text))

    @app.get("/api/claims", response_model=list[QueuedClaim])
    def queue(
        status: ReviewStatus | None = None, limit: Annotated[int, Query(ge=1, le=200)] = 50
    ) -> list[QueuedClaim]:
        return components.queue.list(status, limit)

    @app.get("/api/claims/{claim_id}", response_model=QueuedClaim)
    def queued_claim(claim_id: int) -> QueuedClaim:
        claim = components.queue.get(claim_id)
        if claim is None:
            raise HTTPException(status_code=404, detail="Claim not found.")
        return claim

    @app.post("/api/claims/{claim_id}/review", response_model=QueuedClaim)
    def review(claim_id: int, body: ReviewDecision) -> QueuedClaim:
        try:
            claim = components.queue.review(claim_id, body)
        except QueueError as error:
            code = 404 if "not found" in str(error) else 409
            raise HTTPException(status_code=code, detail=str(error)) from error
        log.info("claim %d %s by %s", claim_id, body.decision, mask_pii(body.reviewer))
        return claim

    @app.get("/api/samples")
    def samples() -> list[dict[str, str]]:
        return list_samples()

    @app.get("/api/samples/{name}")
    def sample(name: str) -> dict[str, str]:
        text = read_sample(name)
        if text is None:
            raise HTTPException(status_code=404, detail="Sample not found.")
        return {"name": name, "text": text}

    static_dir = settings.claimdesk_static_dir or DEFAULT_STATIC_DIR
    if static_dir.is_dir():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="web")
    return app


app = create_app()
