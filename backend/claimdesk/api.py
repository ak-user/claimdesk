"""HTTP API. Run with: uvicorn claimdesk.api:app --reload"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from . import __version__
from .app_state import build_components
from .config import Settings
from .extraction import MAX_UPLOAD_BYTES, DocumentError, read_document
from .guardrails import mask_pii
from .models import AnswerOut, AskRequest, ClaimAssessment, ClaimText
from .store import list_samples, read_sample

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("claimdesk.api")

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
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next):  # type: ignore[no-untyped-def]
        started = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - started) * 1000
        log.info("%s %s %s %.0fms", request.method, request.url.path, response.status_code, elapsed)
        return response

    @app.get("/api/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "llm_provider": components.provider,
            "retrieval": components.retriever.mode,
            "sections": len(components.retriever.chunks),
            "as_of": settings.claimdesk_as_of.isoformat(),
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
