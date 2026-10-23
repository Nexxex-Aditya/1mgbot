"""Request-scoped middleware: correlation IDs, size limits, sanitized error responses."""

from __future__ import annotations

import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from carepath.contracts.common import ErrorBody, ErrorCode, ErrorResponse
from carepath.observability.context import new_id, sanitize_incoming_id, set_request_context

logger = logging.getLogger(__name__)


class CorrelationMiddleware(BaseHTTPMiddleware):
    """Attaches request_id + trace_id context vars and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = sanitize_incoming_id(request.headers.get("X-Request-ID")) or new_id()
        trace_id = sanitize_incoming_id(request.headers.get("X-Trace-ID")) or new_id()
        set_request_context(request_id=request_id, trace_id=trace_id)

        started = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 2)

        response.headers["X-Request-ID"] = request_id
        response.headers["X-Trace-ID"] = trace_id

        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "elapsed_ms": elapsed_ms,
            },
        )
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject bodies larger than ``max_bytes`` before the router processes them."""

    def __init__(self, app: object, *, max_bytes: int) -> None:
        super().__init__(app)  # type: ignore[arg-type]
        self._max_bytes = max_bytes

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self._max_bytes:
            return _error_response(
                413,
                ErrorCode.PAYLOAD_TOO_LARGE,
                f"Request body exceeds the {self._max_bytes} byte limit.",
            )
        return await call_next(request)


def _error_response(status_code: int, code: ErrorCode, message: str) -> JSONResponse:
    body = ErrorResponse(error=ErrorBody(code=code, message=message))
    return JSONResponse(status_code=status_code, content=body.model_dump(mode="json"))
