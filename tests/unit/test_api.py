"""Unit tests for health, session, and model endpoints."""

from __future__ import annotations

import uuid

import pytest
from fastapi.testclient import TestClient


class TestLiveness:
    def test_liveness_returns_200(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health/live")
        assert resp.status_code == 200
        body = resp.json()
        assert body["status"] == "alive"
        assert "version" in body

    def test_liveness_has_correlation_id(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health/live")
        assert "x-request-id" in resp.headers
        assert "x-trace-id" in resp.headers


class TestReadiness:
    def test_readiness_returns_components(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        body = resp.json()
        assert "components" in body
        names = {c["name"] for c in body["components"]}
        assert "model_gateway" in names

    def test_readiness_reports_demo_mode(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/health/ready")
        body = resp.json()
        assert body["demo_mode"] is True


class TestConfigReport:
    def test_config_never_exposes_secrets(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/config")
        assert resp.status_code == 200
        body = resp.json()
        # secrets field should only contain 'configured' or 'missing', never actual values
        for key, value in body.get("secrets", {}).items():
            assert value in ("configured", "missing"), f"Secret {key} has unexpected value: {value}"

    def test_config_has_research_limits(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/config")
        body = resp.json()
        assert "research_limits" in body
        assert "max_workers" in body["research_limits"]


class TestSessionEndpoints:
    def test_list_actors(self, app_client: TestClient) -> None:
        resp = app_client.get("/api/v1/actors")
        assert resp.status_code == 200
        body = resp.json()
        assert "actors" in body
        handles = {a["handle"] for a in body["actors"]}
        assert "demo-asha" in handles

    def test_create_session_with_consent(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-asha",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        assert resp.status_code == 201
        body = resp.json()
        assert body["session"]["status"] == "active"
        assert body["replayed"] is False

    def test_create_session_requires_consent(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-asha",
                "modality": "text",
                "consent": {"accepted": False},
            },
        )
        assert resp.status_code == 422
        assert "consent" in resp.json()["detail"].lower()

    def test_create_session_unknown_actor(self, app_client: TestClient) -> None:
        resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "nonexistent-actor",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        assert resp.status_code == 404

    def test_idempotency_replay(self, app_client: TestClient) -> None:
        key = "test-idempotent-key-1234"
        payload = {
            "actor_handle": "demo-ravi",
            "modality": "text",
            "consent": {"accepted": True},
        }
        r1 = app_client.post("/api/v1/sessions", json=payload, headers={"Idempotency-Key": key})
        r2 = app_client.post("/api/v1/sessions", json=payload, headers={"Idempotency-Key": key})
        assert r1.status_code == 201
        assert r2.status_code == 201
        assert r1.json()["session"]["id"] == r2.json()["session"]["id"]
        assert r2.json()["replayed"] is True

    def test_get_session_by_id(self, app_client: TestClient) -> None:
        create_resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-asha",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        session_id = create_resp.json()["session"]["id"]
        resp = app_client.get(f"/api/v1/sessions/{session_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == session_id

    def test_get_session_not_found(self, app_client: TestClient) -> None:
        resp = app_client.get(f"/api/v1/sessions/{uuid.uuid4()}")
        assert resp.status_code == 404

    def test_get_session_events(self, app_client: TestClient) -> None:
        create_resp = app_client.post(
            "/api/v1/sessions",
            json={
                "actor_handle": "demo-asha",
                "modality": "text",
                "consent": {"accepted": True},
            },
        )
        session_id = create_resp.json()["session"]["id"]
        resp = app_client.get(f"/api/v1/sessions/{session_id}/events")
        assert resp.status_code == 200
        events = resp.json()
        # Session creation should produce at least one audit event
        assert len(events) >= 1
        assert any(e["event_type"] == "session.created" for e in events)
