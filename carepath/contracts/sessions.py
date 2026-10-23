"""Session, consent, workflow-run and event contracts."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field

from carepath.contracts.common import ContractModel, WorkflowStatus

DEMO_CONSENT_VERSION = "demo-consent-v1"
DEMO_CONSENT_PURPOSE = "demo_care_navigation"

DEMO_CONSENT_TEXT = (
    "This is a research prototype using synthetic data only. It is not a medical device, "
    "does not diagnose or prescribe, and must not be used for real health decisions. "
    "Do not enter real personal or health information."
)


class Modality(StrEnum):
    TEXT = "text"


class SessionStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


class ConsentDecision(StrEnum):
    ACCEPTED = "accepted"
    DECLINED = "declined"
    WITHDRAWN = "withdrawn"


class DemoConsentInput(ContractModel):
    purpose: Literal["demo_care_navigation"] = DEMO_CONSENT_PURPOSE
    version: Literal["demo-consent-v1"] = DEMO_CONSENT_VERSION
    accepted: bool


class CreateSessionRequest(ContractModel):
    actor_handle: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,39}$", examples=["demo-asha"])
    modality: Modality = Modality.TEXT
    consent: DemoConsentInput


class DemoActorView(ContractModel):
    id: uuid.UUID
    handle: str
    display_name: str
    is_synthetic: bool


class ConsentRecordView(ContractModel):
    id: uuid.UUID
    purpose: str
    consent_version: str
    decision: ConsentDecision
    decided_at: datetime
    revoked_at: datetime | None
    trace_id: str


class SessionSummary(ContractModel):
    id: uuid.UUID
    actor: DemoActorView
    status: SessionStatus
    modality: Modality
    is_demo: bool
    workflow_version: str
    created_at: datetime
    updated_at: datetime


class SessionDetail(SessionSummary):
    consents: list[ConsentRecordView]
    event_count: int
    run_count: int


class SessionCreated(ContractModel):
    session: SessionDetail
    run_id: uuid.UUID
    trace_id: str
    replayed: bool = Field(
        description="True when an Idempotency-Key matched an existing session and no new record was created."
    )


class WorkflowEventView(ContractModel):
    id: uuid.UUID
    aggregate_type: str
    aggregate_id: str
    sequence: int
    event_type: str
    schema_version: str
    actor: str
    session_id: uuid.UUID | None
    run_id: uuid.UUID | None
    trace_id: str
    occurred_at: datetime
    payload: dict[str, Any]


class ModelRunView(ContractModel):
    id: uuid.UUID
    provider: str
    model_id: str
    capability: str
    prompt_version: str | None
    status: str
    attempts: int
    latency_ms: float | None
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    error_code: str | None
    created_at: datetime


class WorkflowRunView(ContractModel):
    id: uuid.UUID
    session_id: uuid.UUID | None
    workflow_name: str
    workflow_version: str
    status: WorkflowStatus
    trace_id: str
    started_at: datetime
    finished_at: datetime | None
    error_code: str | None
    error_summary: str | None


class WorkflowRunDetail(WorkflowRunView):
    events: list[WorkflowEventView]
    model_runs: list[ModelRunView]


class DemoActorList(ContractModel):
    actors: list[DemoActorView]
