"""Model connectivity check endpoint — user-triggered, not on every startup."""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime

from fastapi import APIRouter

from backend.dependencies import GatewayDep, RequireAuth
from carepath.contracts.common import ErrorCode
from carepath.contracts.models import (
    ChatMessage,
    ConnectivityProbe,
    ModelCheckResult,
    ModelCheckStatus,
    ModelRequest,
)
from carepath.models.errors import ModelGatewayError, ModelNotConfiguredError
from carepath.models.gateway import extract_json_object
from carepath.observability.context import current_trace_id, new_id

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["models"], dependencies=[RequireAuth])


@router.post("/models/check", response_model=ModelCheckResult)
async def check_model(gateway: GatewayDep) -> ModelCheckResult:
    """Explicit connectivity + structured-output probe (POST so it's never cached by proxies)."""
    from carepath.prompts import CONNECTIVITY_PROBE

    trace_id = current_trace_id()
    nonce = new_id()[:12]

    if not gateway.configured:
        result = ModelCheckResult(
            status=ModelCheckStatus.NOT_CONFIGURED,
            provider=gateway.provider_label,
            model=gateway.default_model,
            checked_at=datetime.now(UTC),
            trace_id=trace_id,
            message="NVIDIA_API_KEY is not configured. Set it in .env and restart.",
        )
        gateway.last_check = result
        return result

    started = time.perf_counter()
    try:
        request = ModelRequest(
            messages=[
                ChatMessage(role="system", content=CONNECTIVITY_PROBE.system),
                ChatMessage(role="user", content=CONNECTIVITY_PROBE.render_user(nonce=nonce)),
            ],
            temperature=0.0,
            max_tokens=64,
            json_mode=True,
        )
        response = await gateway.complete(request)
    except ModelGatewayError as exc:
        result = ModelCheckResult(
            status=ModelCheckStatus.FAILED,
            provider=gateway.provider_label,
            model=gateway.default_model,
            checked_at=datetime.now(UTC),
            trace_id=trace_id,
            latency_ms=round((time.perf_counter() - started) * 1000, 2),
            error_code=exc.code,
            message=str(exc),
        )
        gateway.last_check = result
        return result

    latency_ms = round((time.perf_counter() - started) * 1000, 2)

    # Validate the structured output
    valid_output = False
    try:
        probe = ConnectivityProbe.model_validate_json(extract_json_object(response.content))
        valid_output = probe.status == "ok" and probe.echo == nonce
    except Exception:
        pass

    result = ModelCheckResult(
        status=ModelCheckStatus.OK if valid_output else ModelCheckStatus.FAILED,
        provider=gateway.provider_label,
        model=response.model,
        checked_at=datetime.now(UTC),
        trace_id=trace_id,
        run_id=None,
        latency_ms=latency_ms,
        attempts=response.attempts,
        structured_output_valid=valid_output,
        error_code=None if valid_output else ErrorCode.MODEL_OUTPUT_INVALID,
        message=(
            f"Connected successfully. Structured output validated. Latency {latency_ms:.0f}ms."
            if valid_output
            else f"Connected but structured output validation failed. Response: {response.content[:200]}"
        ),
    )
    gateway.last_check = result
    return result
