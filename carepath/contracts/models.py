"""Model gateway request/response contracts (provider-agnostic)."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import Field

from carepath.contracts.common import ContractModel, ErrorCode


class ChatMessage(ContractModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(max_length=200_000)


class ModelRequest(ContractModel):
    messages: list[ChatMessage] = Field(min_length=1)
    model: str | None = None
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=16_384)
    json_mode: bool = False


class TokenUsage(ContractModel):
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ProviderResponse(ContractModel):
    """Raw-but-normalized single provider attempt result."""

    content: str
    model: str
    finish_reason: str | None = None
    usage: TokenUsage | None = None


class ModelResponse(ContractModel):
    content: str
    provider: str
    model: str
    finish_reason: str | None
    usage: TokenUsage | None
    attempts: int
    latency_ms: float


class GatewayState(StrEnum):
    NOT_CONFIGURED = "not_configured"
    CONFIGURED_UNVERIFIED = "configured_unverified"
    HEALTHY = "healthy"
    FAILED = "failed"


class ModelCheckStatus(StrEnum):
    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    FAILED = "failed"


class ModelCheckResult(ContractModel):
    """Outcome of an explicit, user-triggered connectivity + structured-output probe."""

    status: ModelCheckStatus
    provider: str
    model: str
    checked_at: datetime
    trace_id: str
    run_id: str | None = None
    latency_ms: float | None = None
    attempts: int | None = None
    structured_output_valid: bool | None = None
    error_code: ErrorCode | None = None
    message: str


class ConnectivityProbe(ContractModel):
    """Typed output the model must return for the connectivity probe."""

    status: Literal["ok"]
    echo: str = Field(max_length=64)
