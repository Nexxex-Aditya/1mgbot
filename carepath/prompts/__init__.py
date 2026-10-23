"""Versioned prompt templates.

Prompts live here as named, versioned templates (not inline in route handlers) so that
model runs can record ``prompt_version`` and evaluations can compare versions.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PromptTemplate:
    name: str
    version: str
    system: str
    user: str

    @property
    def ref(self) -> str:
        return f"{self.name}@{self.version}"

    def render_user(self, **values: str) -> str:
        return self.user.format(**values)


CONNECTIVITY_PROBE = PromptTemplate(
    name="connectivity_probe",
    version="v1",
    system=(
        "You are a connectivity probe for a software health check. "
        "Reply with a single JSON object and nothing else."
    ),
    user=(
        'Return exactly this JSON object with the same values: {{"status": "ok", "echo": "{nonce}"}}'
    ),
)
