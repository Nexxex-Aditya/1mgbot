"""Request/run/trace correlation IDs carried via context variables."""

from __future__ import annotations

import re
import uuid
from contextvars import ContextVar

_request_id: ContextVar[str | None] = ContextVar("carepath_request_id", default=None)
_trace_id: ContextVar[str | None] = ContextVar("carepath_trace_id", default=None)
_run_id: ContextVar[str | None] = ContextVar("carepath_run_id", default=None)

# Accept only conservative ID formats from clients (hex / uuid-ish, bounded length) so that
# arbitrary attacker-controlled strings never reach logs or persisted records.
_SAFE_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def new_id() -> str:
    return uuid.uuid4().hex


def sanitize_incoming_id(value: str | None) -> str | None:
    if value and _SAFE_ID.fullmatch(value):
        return value
    return None


def set_request_context(*, request_id: str, trace_id: str) -> None:
    _request_id.set(request_id)
    _trace_id.set(trace_id)


def set_run_id(run_id: str | None) -> None:
    _run_id.set(run_id)


def current_request_id() -> str | None:
    return _request_id.get()


def current_trace_id() -> str:
    """Return the active trace ID, creating one for non-request contexts (scripts, jobs)."""
    trace_id = _trace_id.get()
    if trace_id is None:
        trace_id = new_id()
        _trace_id.set(trace_id)
    return trace_id


def current_run_id() -> str | None:
    return _run_id.get()
