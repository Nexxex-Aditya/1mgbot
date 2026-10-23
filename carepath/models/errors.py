"""Typed model-gateway errors with stable error codes."""

from __future__ import annotations

from carepath.contracts.common import ErrorCode


class ModelGatewayError(Exception):
    code: ErrorCode = ErrorCode.MODEL_PROVIDER_ERROR
    retryable: bool = False

    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.attempts: int | None = None


class ModelNotConfiguredError(ModelGatewayError):
    code = ErrorCode.MODEL_NOT_CONFIGURED


class ModelTimeoutError(ModelGatewayError):
    code = ErrorCode.MODEL_TIMEOUT
    retryable = True


class ModelRateLimitedError(ModelGatewayError):
    code = ErrorCode.MODEL_RATE_LIMITED
    retryable = True


class ModelTransientError(ModelGatewayError):
    """5xx / connection failures that may succeed on retry."""

    code = ErrorCode.MODEL_PROVIDER_ERROR
    retryable = True


class ModelProviderError(ModelGatewayError):
    """Non-retryable provider failure (4xx other than 429, malformed envelope...)."""

    code = ErrorCode.MODEL_PROVIDER_ERROR


class ModelOutputValidationError(ModelGatewayError):
    code = ErrorCode.MODEL_OUTPUT_INVALID
