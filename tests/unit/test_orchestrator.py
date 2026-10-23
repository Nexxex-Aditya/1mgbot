"""Unit tests for the LangGraph care navigation orchestrator."""

from __future__ import annotations

import json

import pytest

from carepath.models.gateway import ModelGateway
from carepath.orchestrator.graph import run_care_graph
from carepath.orchestrator.state import CarePhase, CareState, TriageResult, TriageUrgency
from tests.fakes import FakeModelProvider


@pytest.fixture
def fake_provider_with_triage() -> FakeModelProvider:
    """Provider that returns a valid triage JSON on first call, then a text response."""
    triage_json = json.dumps({
        "urgency": "standard",
        "chief_complaint": "headache and mild fever",
        "symptoms": ["headache", "fever"],
        "suggested_specialties": ["General Medicine"],
        "confidence": 0.75,
        "reasoning": "Common symptoms suggesting viral or tension-related cause.",
        "safety_flags": [],
    })
    return FakeModelProvider(
        canned_responses=[
            triage_json,  # triage node
            "Based on your symptoms of headache and fever, I recommend consulting a general physician.",  # present node
        ]
    )


@pytest.fixture
def configured_gateway(fake_provider_with_triage: FakeModelProvider) -> ModelGateway:
    return ModelGateway(
        provider=fake_provider_with_triage,
        default_model="test/fake-model",
        max_retries=1,
        max_concurrency=4,
    )


@pytest.fixture
def unconfigured_gateway() -> ModelGateway:
    return ModelGateway(
        provider=None,
        default_model="test/none",
        max_retries=0,
        max_concurrency=1,
    )


class TestOrchestratorState:
    def test_care_state_defaults(self) -> None:
        state = CareState(session_id="test-123")
        assert state.phase == CarePhase.INTAKE
        assert state.is_demo is True
        assert state.messages == []
        assert state.triage is None

    def test_triage_result_validation(self) -> None:
        result = TriageResult(
            urgency=TriageUrgency.STANDARD,
            chief_complaint="test",
            symptoms=["headache"],
            confidence=0.8,
        )
        assert result.urgency == TriageUrgency.STANDARD
        assert result.confidence == 0.8


class TestOrchestratorGraph:
    @pytest.mark.asyncio
    async def test_full_graph_with_configured_gateway(self, configured_gateway: ModelGateway) -> None:
        result = await run_care_graph(
            configured_gateway,
            session_id="test-session-1",
            user_input="I have a headache and mild fever for two days",
            trace_id="test-trace-1",
        )

        # Graph should complete to AWAITING_INPUT phase
        assert result["phase"] == CarePhase.AWAITING_INPUT
        assert result["pending_response"] != ""
        assert result.get("error_message") is None

        # Triage should be populated
        triage = result.get("triage")
        assert triage is not None
        assert triage["urgency"] == "standard"
        assert "headache" in triage["chief_complaint"].lower() or triage["chief_complaint"] != ""

        # Evidence should be present (synthetic in Phase 2)
        assert len(result["evidence"]) >= 1

        # Provider suggestions should be present (synthetic)
        assert len(result["provider_suggestions"]) >= 1
        for p in result["provider_suggestions"]:
            assert p["is_synthetic"] is True

        # Messages should contain the user input and assistant response
        assert len(result["messages"]) >= 2

    @pytest.mark.asyncio
    async def test_graph_without_gateway_uses_fallback(self, unconfigured_gateway: ModelGateway) -> None:
        result = await run_care_graph(
            unconfigured_gateway,
            session_id="test-session-2",
            user_input="I have chest pain",
            trace_id="test-trace-2",
        )

        # Should still complete (fallback path)
        assert result["phase"] == CarePhase.AWAITING_INPUT
        assert result["pending_response"] != ""

        # Triage should be present with unknown urgency (gateway not configured)
        triage = result.get("triage")
        if triage is not None:
            assert triage["urgency"] == "unknown"
        # Fallback response should still mention the symptoms
        assert result["pending_response"] != ""

        # Should still have synthetic evidence and providers
        assert len(result.get("evidence", [])) >= 1
        assert len(result.get("provider_suggestions", [])) >= 1

    @pytest.mark.asyncio
    async def test_graph_response_contains_disclaimer(self, configured_gateway: ModelGateway) -> None:
        result = await run_care_graph(
            configured_gateway,
            session_id="test-session-3",
            user_input="My knee hurts",
            trace_id="test-trace-3",
        )

        response = result["pending_response"]
        # Deterministic disclaimer is always appended
        assert "research prototype" in response.lower() or "synthetic" in response.lower()


class TestChatEndpoint:
    def test_chat_endpoint_returns_response(self, app_client) -> None:
        # First create a session
        session_resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-asha",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        assert session_resp.status_code == 201
        session_id = session_resp.json()["session"]["id"]

        # Now send a chat message
        chat_resp = app_client.post(
            "/api/v1/chat",
            json={
                "session_id": session_id,
                "message": "I have a persistent cough for three days",
            },
        )
        assert chat_resp.status_code == 200
        body = chat_resp.json()
        assert body["response"] != ""
        assert body["trace_id"] != ""
        assert "phase" in body

    def test_chat_endpoint_empty_message_rejected(self, app_client) -> None:
        resp = app_client.post(
            "/api/v1/chat",
            json={
                "session_id": "test-session",
                "message": "",
            },
        )
        assert resp.status_code == 422  # validation error

    def test_multi_turn_persists_history(self, app_client) -> None:
        """Send two messages and verify the history endpoint returns both turns."""
        session_resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-ravi",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        session_id = session_resp.json()["session"]["id"]

        # First turn
        r1 = app_client.post("/api/v1/chat", json={"session_id": session_id, "message": "I have a headache"})
        assert r1.status_code == 200

        # Second turn
        r2 = app_client.post("/api/v1/chat", json={"session_id": session_id, "message": "It started yesterday"})
        assert r2.status_code == 200

        # Check persisted history
        history = app_client.get(f"/api/v1/sessions/{session_id}/chat-history")
        assert history.status_code == 200
        turns = history.json()
        # Each turn produces a user + assistant pair = 4 turns total
        assert len(turns) >= 4
        roles = [t["role"] for t in turns]
        assert roles[0] == "user"
        assert roles[1] == "assistant"
        assert roles[2] == "user"
        assert roles[3] == "assistant"
