"""Structured JSON logging with correlation IDs and secret redaction."""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime

from carepath.observability.context import current_request_id, current_run_id
from carepath.observability.context import _trace_id as _trace_var
from carepath.observability.redaction import redact_text

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {"message", "asctime"}


class CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = current_request_id()
        record.trace_id = _trace_var.get()
        record.run_id = current_run_id()
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": redact_text(record.getMessage()),
        }
        for key in ("request_id", "trace_id", "run_id"):
            value = getattr(record, key, None)
            if value:
                payload[key] = value
        for key, value in record.__dict__.items():
            if key in _RESERVED or key in payload or key.startswith("_"):
                continue
            if key in {"request_id", "trace_id", "run_id"}:
                continue
            payload[key] = redact_text(value) if isinstance(value, str) else value
        if record.exc_info:
            payload["exc_type"] = record.exc_info[0].__name__ if record.exc_info[0] else None
            payload["exc"] = redact_text(self.formatException(record.exc_info))
        return json.dumps(payload, default=str)


_configured = False


def configure_logging(level: str = "INFO") -> None:
    """Idempotently configure root logging to emit redacted JSON lines on stderr."""
    global _configured
    root = logging.getLogger()
    root.setLevel(level)
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(CorrelationFilter())
    root.handlers = [handler]
    # Keep noisy client libraries from logging request URLs/headers at INFO.
    for noisy in ("httpx", "httpcore", "sqlalchemy.engine", "watchfiles"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    _configured = True
