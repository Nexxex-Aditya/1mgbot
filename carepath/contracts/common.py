"""Shared contract primitives used across API, services and UI."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

CONTRACT_VERSION = "1.0"


def utcnow() -> datetime:
    return datetime.now(UTC)


class ContractModel(BaseModel):
    """Base for all API/runtime contracts: strict field names, immutable by default."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ErrorCode(StrEnum):
    VALIDATION_ERROR = "validation_error"
    NOT_FOUND = "not_found"
    UNAUTHORIZED = "unauthorized"
    CONSENT_REQUIRED = "consent_required"
    PAYLOAD_TOO_LARGE = "payload_too_large"
    RATE_LIMITED = "rate_limited"
    PERSISTENCE_UNAVAILABLE = "persistence_unavailable"
    MODEL_NOT_CONFIGURED = "model_not_configured"
    MODEL_TIMEOUT = "model_timeout"
    MODEL_RATE_LIMITED = "model_rate_limited"
    MODEL_PROVIDER_ERROR = "model_provider_error"
    MODEL_OUTPUT_INVALID = "model_output_invalid"
    INTERNAL_ERROR = "internal_error"


class ErrorBody(ContractModel):
    code: ErrorCode
    message: str
    trace_id: str | None = None
    details: list[dict[str, Any]] | None = None


class ErrorResponse(ContractModel):
    error: ErrorBody


class WorkflowStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkflowResult(ContractModel):
    """Envelope returned by orchestrated workflows to the UI."""

    status: WorkflowStatus
    trace_id: str
    run_id: str | None = None
    payload: dict[str, Any] | None = None
    warnings: list[str] = Field(default_factory=list)
    required_next_action: str | None = None
    elapsed_ms: float | None = None
    error_code: ErrorCode | None = None
