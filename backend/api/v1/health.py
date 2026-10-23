"""Health and readiness endpoints: GET /api/v1/health/live, /api/v1/health/ready, /api/v1/config."""

from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Request

from backend.dependencies import GatewayDep, SettingsDep
from carepath.contracts.health import (
    ComponentHealth,
    ComponentState,
    ConfigReport,
    LivenessResponse,
    ReadinessResponse,
    SecretStatus,
)
from carepath.models.gateway import GatewayState
from carepath.persistence.database import probe_database

router = APIRouter(prefix="/api/v1", tags=["health"])

APP_VERSION = "0.1.0"


@router.get("/health/live", response_model=LivenessResponse)
async def liveness() -> LivenessResponse:
    return LivenessResponse(version=APP_VERSION)


@router.get("/health/ready", response_model=ReadinessResponse)
async def readiness(request: Request, settings: SettingsDep, gateway: GatewayDep) -> ReadinessResponse:
    now = datetime.now(UTC)
    components: list[ComponentHealth] = []

    # --- database ---
    engine = request.app.state.engine
    db_probe = await probe_database(engine)
    if db_probe.reachable and db_probe.at_head:
        db_state = ComponentState.HEALTHY
        db_detail = f"revision={db_probe.current_revision}"
    elif db_probe.reachable:
        db_state = ComponentState.DEGRADED
        db_detail = f"current={db_probe.current_revision} head={db_probe.head_revision} — migration pending"
    else:
        db_state = ComponentState.UNAVAILABLE
        db_detail = db_probe.error or "unreachable"
    components.append(
        ComponentHealth(
            name="postgresql",
            state=db_state,
            required=True,
            detail=db_detail,
            checked_at=now,
            latency_ms=db_probe.latency_ms,
        )
    )

    # --- model gateway ---
    gw_state_enum = gateway.state()
    gw_map = {
        GatewayState.HEALTHY: ComponentState.HEALTHY,
        GatewayState.CONFIGURED_UNVERIFIED: ComponentState.CONFIGURED_UNVERIFIED,
        GatewayState.FAILED: ComponentState.UNAVAILABLE,
        GatewayState.NOT_CONFIGURED: ComponentState.NOT_CONFIGURED,
    }
    components.append(
        ComponentHealth(
            name="model_gateway",
            state=gw_map[gw_state_enum],
            required=False,
            detail=f"provider={gateway.provider_label} model={gateway.default_model}",
            checked_at=now,
        )
    )

    # --- optional services (disabled/not_configured until their phase) ---
    for svc_name, enabled in [
        ("qdrant", settings.qdrant_mode != "disabled"),
        ("neo4j", settings.neo4j_enabled),
        ("searxng", settings.searxng_base_url is not None),
        ("langsmith", settings.langsmith_tracing),
        ("voice", settings.voice_enabled),
    ]:
        components.append(
            ComponentHealth(
                name=svc_name,
                state=ComponentState.DISABLED if not enabled else ComponentState.CONFIGURED_UNVERIFIED,
                required=False,
                detail="disabled" if not enabled else "configured, not probed",
                checked_at=now,
            )
        )

    overall_ok = all(
        c.state in {ComponentState.HEALTHY, ComponentState.DEGRADED} for c in components if c.required
    )
    any_degraded = any(c.state == ComponentState.DEGRADED for c in components if c.required)

    return ReadinessResponse(
        status="ready" if overall_ok and not any_degraded else "degraded" if overall_ok else "not_ready",
        demo_mode=settings.demo_mode,
        components=components,
        checked_at=now,
    )


@router.get("/config", response_model=ConfigReport)
async def config_report(settings: SettingsDep) -> ConfigReport:
    def _secret_status(val: object) -> SecretStatus:
        if val is None:
            return SecretStatus.MISSING
        secret_val = getattr(val, "get_secret_value", lambda: None)()
        return SecretStatus.CONFIGURED if secret_val else SecretStatus.MISSING

    return ConfigReport(
        app_env=settings.app_env,
        demo_mode=settings.demo_mode,
        log_level=settings.log_level,
        auth_required=settings.auth_required,
        database_backend=settings.database_backend,
        database_target=settings.redacted_database_target(),
        model_base_url=settings.nvidia_base_url,
        llm_model=settings.llm_model,
        judge_model=settings.judge_model,
        llm_timeout_seconds=settings.llm_timeout_seconds,
        llm_max_retries=settings.llm_max_retries,
        max_concurrency=settings.max_concurrency,
        embedding_provider=settings.embedding_provider,
        embedding_model=settings.embedding_model,
        qdrant_mode=settings.qdrant_mode,
        qdrant_collection=settings.qdrant_collection,
        neo4j_enabled=settings.neo4j_enabled,
        searxng_configured=settings.searxng_base_url is not None,
        langsmith_tracing=settings.langsmith_tracing,
        voice_enabled=settings.voice_enabled,
        cors_origins=settings.cors_origins,
        research_limits={
            "max_workers": settings.research_max_workers,
            "max_steps": settings.research_max_steps,
            "max_sources": settings.research_max_sources,
            "job_timeout_seconds": settings.job_timeout_seconds,
        },
        secrets={
            "nvidia_api_key": _secret_status(settings.nvidia_api_key),
            "gemini_api_key": _secret_status(settings.gemini_api_key),
            "qdrant_api_key": _secret_status(settings.qdrant_api_key),
            "neo4j_password": _secret_status(settings.neo4j_password),
            "langsmith_api_key": _secret_status(settings.langsmith_api_key),
            "logfire_token": _secret_status(settings.logfire_token),
            "api_bearer_token": _secret_status(settings.api_bearer_token),
            "livekit_api_key": _secret_status(settings.livekit_api_key),
            "livekit_api_secret": _secret_status(settings.livekit_api_secret),
        },
    )
