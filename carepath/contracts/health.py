"""Health, readiness and redacted configuration report contracts."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from carepath.contracts.common import ContractModel


class ComponentState(StrEnum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"  # configured but failing
    NOT_CONFIGURED = "not_configured"  # credentials/endpoint absent
    DISABLED = "disabled"  # feature flag off or not yet implemented
    CONFIGURED_UNVERIFIED = "configured_unverified"  # credentials present, not yet probed


class ComponentHealth(ContractModel):
    name: str
    state: ComponentState
    required: bool
    detail: str
    checked_at: datetime
    latency_ms: float | None = None


class LivenessResponse(ContractModel):
    status: Literal["alive"] = "alive"
    service: str = "carepath-api"
    version: str


class ReadinessResponse(ContractModel):
    status: Literal["ready", "degraded", "not_ready"]
    demo_mode: bool
    components: list[ComponentHealth]
    checked_at: datetime


class SecretStatus(StrEnum):
    CONFIGURED = "configured"
    MISSING = "missing"


class ConfigReport(ContractModel):
    """Configuration diagnostics. Never contains secret values."""

    app_env: str
    demo_mode: bool
    log_level: str
    auth_required: bool
    database_backend: str
    database_target: str
    model_base_url: str
    llm_model: str
    judge_model: str
    llm_timeout_seconds: float
    llm_max_retries: int
    max_concurrency: int
    embedding_provider: str | None
    embedding_model: str | None
    qdrant_mode: str
    qdrant_collection: str
    neo4j_enabled: bool
    searxng_configured: bool
    langsmith_tracing: bool
    voice_enabled: bool
    cors_origins: list[str]
    research_limits: dict[str, int]
    secrets: dict[str, SecretStatus]
