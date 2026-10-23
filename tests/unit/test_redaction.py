"""Unit tests for the observability redaction module."""

from __future__ import annotations

import pytest

from carepath.observability.redaction import redact_mapping, redact_text


class TestRedaction:
    def test_redacts_bearer_token(self) -> None:
        text = "Authorization: Bearer sk-abc123def456"
        result = redact_text(text)
        assert "sk-abc123" not in result
        assert "REDACTED" in result

    def test_redacts_nvidia_key(self) -> None:
        text = "key=nvapi-abcdef1234567890long"
        result = redact_text(text)
        assert "nvapi-" not in result or "REDACTED" in result

    def test_passthrough_safe_strings(self) -> None:
        text = "This is a normal log message with no secrets"
        result = redact_text(text)
        assert result == text

    def test_redacts_in_dict(self) -> None:
        data = {"authorization": "Bearer sk-secret123456"}
        result = redact_mapping(data)
        assert isinstance(result, dict)
        assert "sk-secret" not in str(result)

    def test_redacts_url_credentials(self) -> None:
        text = "postgresql://user:my_password@localhost:5432/db"
        result = redact_text(text)
        assert "my_password" not in result
