"""Best-effort redaction of credentials from strings before they reach logs or responses.

This is a defence-in-depth layer: the primary control is to never log secrets or raw
health content in the first place.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

REDACTED = "[REDACTED]"

_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Authorization headers / bearer tokens
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]{8,}"),
    # Provider key formats (NVIDIA nvapi-, OpenAI-style sk-, Google AIza, LangSmith lsv2_)
    re.compile(r"nvapi-[A-Za-z0-9_-]{8,}"),
    re.compile(r"\bsk-[A-Za-z0-9_-]{12,}"),
    re.compile(r"\bAIza[0-9A-Za-z_-]{20,}"),
    re.compile(r"\blsv2_[A-Za-z0-9_]{12,}"),
    # key=value / "key": "value" pairs for secret-looking names
    re.compile(
        r"(?i)((?:api[_-]?key|secret|password|passwd|token|authorization)[\"']?\s*[:=]\s*[\"']?)"
        r"[^\s\"',;&]{4,}"
    ),
    # Credentials embedded in URLs: scheme://user:password@host
    re.compile(r"(?i)([a-z][a-z0-9+.-]*://[^\s:/@]+:)[^\s@/]+(@)"),
)

_SENSITIVE_KEYS = {
    "authorization",
    "api_key",
    "apikey",
    "password",
    "secret",
    "token",
    "access_token",
    "refresh_token",
    "cookie",
    "set-cookie",
    "x-api-key",
}


def redact_text(value: str) -> str:
    redacted = value
    for pattern in _PATTERNS:
        if pattern.groups >= 2:
            redacted = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}{m.group(2)}", redacted)
        elif pattern.groups == 1:
            redacted = pattern.sub(lambda m: f"{m.group(1)}{REDACTED}", redacted)
        else:
            redacted = pattern.sub(REDACTED, redacted)
    return redacted


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively redact values under sensitive keys and secret-looking strings."""
    result: dict[str, Any] = {}
    for key, value in data.items():
        if key.lower() in _SENSITIVE_KEYS:
            result[key] = REDACTED
        elif isinstance(value, Mapping):
            result[key] = redact_mapping(value)
        elif isinstance(value, str):
            result[key] = redact_text(value)
        elif isinstance(value, list):
            result[key] = [
                redact_mapping(v) if isinstance(v, Mapping) else redact_text(v) if isinstance(v, str) else v
                for v in value
            ]
        else:
            result[key] = value
    return result
