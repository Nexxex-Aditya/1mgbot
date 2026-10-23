"""Deterministic fake model provider for tests. Visibly labelled FAKE — never in production."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from carepath.contracts.models import ModelRequest, ProviderResponse, TokenUsage


@dataclass
class FakeModelProvider:
    """Returns deterministic canned responses. Each call records the request for assertions.

    Never used outside tests. The label ``fake-deterministic`` is visible in traces.
    """

    name: str = "fake-deterministic"
    calls: list[tuple[ModelRequest, str]] = field(default_factory=list)
    canned_responses: list[str] = field(default_factory=list)
    _call_index: int = field(default=0, init=False)

    async def complete(self, request: ModelRequest, *, model: str) -> ProviderResponse:
        if self._call_index < len(self.canned_responses):
            content = self.canned_responses[self._call_index]
        else:
            # Default: echo a JSON object matching ConnectivityProbe
            content = json.dumps({"status": "ok", "echo": "test"})
        self.calls.append((request, model))
        self._call_index += 1
        return ProviderResponse(
            content=content,
            model=model,
            finish_reason="stop",
            usage=TokenUsage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        )

    async def aclose(self) -> None:
        pass
