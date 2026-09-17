"""API contracts shared by the agent, the claim assessor and the HTTP layer."""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from pydantic import BaseModel, Field


class Citation(BaseModel):
    section: str
    title: str
    snippet: str


class ToolStep(BaseModel):
    tool: str
    arguments: dict[str, Any]
    summary: str
    ms: float


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=4000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    member_id: str | None = Field(default=None, max_length=20)
    # Earlier turns of the conversation, oldest first, so follow-ups like "what about $90?" work.
    history: list[Turn] = Field(default=[], max_length=20)


class AnswerOut(BaseModel):
    answer: str
    citations: list[Citation]
    trace: list[ToolStep]
    provider: str
    refused: bool = False
    flags: list[str] = []


class ClaimText(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)


class ClaimFields(BaseModel):
    member_id: str | None = None
    patient_name: str | None = None
    service_type: str | None = None
    service_date: date | None = None
    provider_name: str | None = None
    provider_registration: str | None = None
    amount: float | None = None


class Issue(BaseModel):
    field: str | None
    severity: Literal["error", "warning", "info"]
    kind: Literal["not_payable", "needs_correction", "note"]
    message: str
    section: str | None = None


class Estimate(BaseModel):
    benefit: str
    benefit_display: str
    coinsurance_percent: int
    covered_share: float
    estimate: float
    remaining_before: float | None
    remaining_after: float | None
    capped_by_maximum: bool


class ClaimAssessment(BaseModel):
    status: Literal["ready_for_review", "needs_correction", "not_payable"]
    summary: str
    fields: ClaimFields
    extractor: Literal["rules", "llm"]
    benefit: str | None
    benefit_display: str | None
    issues: list[Issue]
    estimate: Estimate | None
    citations: list[Citation]
    flags: list[str]
    requires_human_review: bool = True
