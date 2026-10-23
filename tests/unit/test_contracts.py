"""Unit tests for Pydantic contracts — serialization and validation."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from carepath.contracts.sessions import (
    CreateSessionRequest,
    DemoActorView,
    DemoConsentInput,
    SessionDetail,
    SessionSummary,
)


class TestContracts:
    def test_create_session_consent_required_field(self) -> None:
        """The consent object must be provided explicitly — it has no default."""
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                actor_handle="demo-asha",
                modality="text",
            )

    def test_create_session_consent_declined(self) -> None:
        req = CreateSessionRequest(
            actor_handle="demo-asha",
            modality="text",
            consent=DemoConsentInput(accepted=False),
        )
        assert req.consent.accepted is False

    def test_create_session_valid(self) -> None:
        req = CreateSessionRequest(
            actor_handle="demo-asha",
            modality="text",
            consent=DemoConsentInput(accepted=True),
        )
        assert req.consent.accepted is True
        assert req.modality == "text"
        assert req.consent.purpose == "demo_care_navigation"
        assert req.consent.version == "demo-consent-v1"

    def test_session_detail_serialization(self) -> None:
        actor = DemoActorView(
            id=uuid.uuid4(),
            handle="demo-asha",
            display_name="Asha",
            is_synthetic=True,
        )
        detail = SessionDetail(
            id=uuid.uuid4(),
            actor=actor,
            status="active",
            modality="text",
            is_demo=True,
            workflow_version="1.0",
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            consents=[],
            event_count=0,
            run_count=0,
        )
        data = detail.model_dump(mode="json")
        assert isinstance(data["id"], str)
        assert data["status"] == "active"
        assert data["actor"]["handle"] == "demo-asha"

    def test_demo_actor_view(self) -> None:
        actor = DemoActorView(
            id=uuid.uuid4(),
            handle="demo-test",
            display_name="Test Actor",
            is_synthetic=True,
        )
        assert actor.is_synthetic is True
        assert actor.handle == "demo-test"

    def test_actor_handle_pattern(self) -> None:
        """Handles must match ^[a-z0-9][a-z0-9-]{2,39}$."""
        with pytest.raises(ValidationError):
            CreateSessionRequest(
                actor_handle="A",  # too short & uppercase
                modality="text",
                consent=DemoConsentInput(accepted=True),
            )
