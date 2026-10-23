"""Model providers behind the gateway.

``OpenAICompatibleChatProvider`` talks to NVIDIA's hosted OpenAI-compatible endpoint via
``httpx`` directly (see ADR-0004). It performs exactly one attempt per call; retry policy
lives in :class:`carepath.models.gateway.ModelGateway`.
"""

from __future__ import annotations

from typing import Any, Protocol

import httpx
from pydantic import SecretStr

from carepath.contracts.models import ModelRequest, ProviderResponse, TokenUsage
from carepath.models.errors import (
    ModelProviderError,
    ModelRateLimitedError,
    ModelTimeoutError,
    ModelTransientError,
)

MAX_RETRY_AFTER_SECONDS = 30.0


class ModelProvider(Protocol):
    name: str

    async def complete(self, request: ModelRequest, *, model: str) -> ProviderResponse: ...

    async def aclose(self) -> None: ...


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, min(seconds, MAX_RETRY_AFTER_SECONDS))


class OpenAICompatibleChatProvider:
    """Single-attempt client for ``POST {base_url}/chat/completions``."""

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: SecretStr,
        timeout_seconds: float,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.name = name
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            timeout=httpx.Timeout(timeout_seconds, connect=min(10.0, timeout_seconds)),
            headers={
                "Authorization": f"Bearer {api_key.get_secret_value()}",
                "Accept": "application/json",
            },
            transport=transport,
            follow_redirects=False,
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def complete(self, request: ModelRequest, *, model: str) -> ProviderResponse:
        body: dict[str, Any] = {
            "model": model,
            "messages": [m.model_dump() for m in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        if request.json_mode:
            body["response_format"] = {"type": "json_object"}
        try:
            response = await self._client.post("chat/completions", json=body)
        except httpx.TimeoutException as exc:
            raise ModelTimeoutError("Model request timed out.") from exc
        except httpx.TransportError as exc:
            raise ModelTransientError(f"Model transport error: {type(exc).__name__}.") from exc

        status = response.status_code
        if status == 429:
            raise ModelRateLimitedError(
                "Model provider rate limited the request.",
                retry_after=_parse_retry_after(response.headers.get("Retry-After")),
            )
        if status in {408, 504}:
            raise ModelTimeoutError(f"Model provider timeout (HTTP {status}).")
        if status >= 500:
            raise ModelTransientError(f"Model provider server error (HTTP {status}).")
        if status >= 400:
            # Do not echo the provider body: it may contain request content.
            raise ModelProviderError(f"Model provider rejected the request (HTTP {status}).")

        try:
            data = response.json()
            choice = data["choices"][0]
            content = choice["message"].get("content") or ""
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
            raise ModelProviderError("Model provider returned an unexpected response envelope.") from exc

        usage_raw = data.get("usage") if isinstance(data, dict) else None
        usage = None
        if isinstance(usage_raw, dict):
            usage = TokenUsage(
                prompt_tokens=usage_raw.get("prompt_tokens"),
                completion_tokens=usage_raw.get("completion_tokens"),
                total_tokens=usage_raw.get("total_tokens"),
            )
        return ProviderResponse(
            content=content,
            model=str(data.get("model") or model),
            finish_reason=choice.get("finish_reason"),
            usage=usage,
        )
