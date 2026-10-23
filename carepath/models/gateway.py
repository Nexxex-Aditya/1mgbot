"""Central model gateway: concurrency limits, bounded retries and schema validation.

All model output is untrusted. ``complete_structured`` parses and validates it against a
Pydantic schema and allows at most ``repair_attempts`` repair round-trips before failing
visibly with :class:`ModelOutputValidationError`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from carepath.config.settings import Settings
from carepath.contracts.models import (
    ChatMessage,
    GatewayState,
    ModelCheckResult,
    ModelRequest,
    ModelResponse,
)
from carepath.models.errors import (
    ModelGatewayError,
    ModelNotConfiguredError,
    ModelOutputValidationError,
)
from carepath.models.providers import ModelProvider, OpenAICompatibleChatProvider

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)
SleepFn = Callable[[float], Awaitable[None]]

_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


@dataclass(frozen=True)
class StructuredResult[T: BaseModel]:
    value: T
    response: ModelResponse
    repair_attempts_used: int


def extract_json_object(text: str) -> str:
    """Extract a JSON object from model text, tolerating code fences and surrounding prose."""
    stripped = text.strip()
    fenced = _FENCE.match(stripped)
    if fenced:
        stripped = fenced.group(1).strip()
    if stripped.startswith("{"):
        return stripped
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        return stripped[start : end + 1]
    return stripped


def _summarize_validation_error(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        parts = [
            f"{'.'.join(str(p) for p in err['loc']) or '<root>'}: {err['type']}"
            for err in exc.errors()[:8]
        ]
        return "; ".join(parts)
    return type(exc).__name__


class ModelGateway:
    def __init__(
        self,
        *,
        provider: ModelProvider | None,
        default_model: str,
        max_retries: int,
        max_concurrency: int,
        provider_label: str = "nvidia",
        sleep: SleepFn = asyncio.sleep,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
    ) -> None:
        self._provider = provider
        self._default_model = default_model
        self._max_retries = max_retries
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._sleep = sleep
        self._base_backoff = base_backoff_seconds
        self._max_backoff = max_backoff_seconds
        self.provider_label = provider.name if provider else provider_label
        self.last_check: ModelCheckResult | None = None

    # ---- status ----
    @property
    def configured(self) -> bool:
        return self._provider is not None

    @property
    def default_model(self) -> str:
        return self._default_model

    def state(self) -> GatewayState:
        if not self.configured:
            return GatewayState.NOT_CONFIGURED
        if self.last_check is None:
            return GatewayState.CONFIGURED_UNVERIFIED
        return GatewayState.HEALTHY if self.last_check.status == "ok" else GatewayState.FAILED

    async def aclose(self) -> None:
        if self._provider is not None:
            await self._provider.aclose()

    # ---- calls ----
    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return retry_after
        exp = min(self._max_backoff, self._base_backoff * (2 ** (attempt - 1)))
        return exp * (0.5 + random.random() / 2)  # noqa: S311 - jitter, not crypto

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._provider is None:
            raise ModelNotConfiguredError(
                "Model provider is not configured (NVIDIA_API_KEY missing)."
            )
        model = request.model or self._default_model
        started = time.perf_counter()
        attempt = 0
        async with self._semaphore:
            while True:
                attempt += 1
                try:
                    result = await self._provider.complete(request, model=model)
                except ModelGatewayError as exc:
                    if not exc.retryable or attempt > self._max_retries:
                        exc.attempts = attempt
                        logger.warning(
                            "model call failed",
                            extra={"error_code": exc.code.value, "attempts": attempt, "model": model},
                        )
                        raise
                    delay = self._backoff(attempt, exc.retry_after)
                    logger.info(
                        "model call retry",
                        extra={"error_code": exc.code.value, "attempt": attempt, "delay_s": round(delay, 2)},
                    )
                    await self._sleep(delay)
                    continue
                latency_ms = (time.perf_counter() - started) * 1000
                return ModelResponse(
                    content=result.content,
                    provider=self.provider_label,
                    model=result.model,
                    finish_reason=result.finish_reason,
                    usage=result.usage,
                    attempts=attempt,
                    latency_ms=round(latency_ms, 2),
                )

    async def complete_structured(
        self,
        request: ModelRequest,
        schema: type[T],
        *,
        repair_attempts: int = 1,
    ) -> StructuredResult[T]:
        messages = list(request.messages)
        last_error = "no attempt"
        total_attempts = 0
        total_latency = 0.0
        for repair_round in range(repair_attempts + 1):
            current = request.model_copy(update={"messages": messages, "json_mode": True})
            response = await self.complete(current)
            total_attempts += response.attempts
            total_latency += response.latency_ms
            try:
                value = schema.model_validate_json(extract_json_object(response.content))
            except (ValidationError, ValueError) as exc:
                last_error = _summarize_validation_error(exc)
                logger.info(
                    "structured output invalid",
                    extra={"schema": schema.__name__, "repair_round": repair_round, "errors": last_error},
                )
                messages = [
                    *messages,
                    ChatMessage(role="assistant", content=response.content[:4000]),
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous reply did not validate against the required JSON schema "
                            f"({last_error}). Reply with only a corrected JSON object matching: "
                            f"{json.dumps(schema.model_json_schema())}"
                        ),
                    ),
                ]
                continue
            merged = response.model_copy(
                update={"attempts": total_attempts, "latency_ms": round(total_latency, 2)}
            )
            return StructuredResult(value=value, response=merged, repair_attempts_used=repair_round)
        err = ModelOutputValidationError(
            f"Model output failed schema validation for {schema.__name__}: {last_error}"
        )
        err.attempts = total_attempts
        raise err


def build_model_gateway(settings: Settings) -> ModelGateway:
    """Build the production gateway. Without a key the gateway reports ``not_configured``.

    The deterministic fake provider is intentionally *not* reachable from here; it is only
    injected by tests (see ``tests/fakes.py``).
    """
    provider: ModelProvider | None = None
    if settings.model_configured and settings.nvidia_api_key is not None:
        provider = OpenAICompatibleChatProvider(
            name="nvidia",
            base_url=settings.nvidia_base_url,
            api_key=settings.nvidia_api_key,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    return ModelGateway(
        provider=provider,
        default_model=settings.llm_model,
        max_retries=settings.llm_max_retries,
        max_concurrency=settings.max_concurrency,
    )
