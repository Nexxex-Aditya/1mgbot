"""Typed state schema for the care navigation orchestrator graph.

State flows through graph nodes and is persisted per-session. Each field has a
clear owner (the node that writes it) and consumers (nodes that read it).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from carepath.contracts.common import utcnow


class CarePhase(StrEnum):
    """Which phase of the care navigation workflow we are in."""

    INTAKE = "intake"
    TRIAGE = "triage"
    EVIDENCE_SEARCH = "evidence_search"
    PROVIDER_MATCH = "provider_match"
    PRESENT = "present"
    AWAITING_INPUT = "awaiting_input"
    CLOSED = "closed"
    ERROR = "error"


class TriageUrgency(StrEnum):
    EMERGENCY = "emergency"
    URGENT = "urgent"
    STANDARD = "standard"
    SELF_CARE = "self_care"
    UNKNOWN = "unknown"


class ChatTurn(BaseModel):
    """A single chat message in the conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: datetime = Field(default_factory=utcnow)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TriageResult(BaseModel):
    """Output of the triage node."""

    urgency: TriageUrgency = TriageUrgency.UNKNOWN
    chief_complaint: str = ""
    symptoms: list[str] = Field(default_factory=list)
    suggested_specialties: list[str] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasoning: str = ""
    safety_flags: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    """A single piece of evidence found during search."""

    title: str
    snippet: str
    source_url: str | None = None
    source_type: str = "synthetic"
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class ProviderSuggestion(BaseModel):
    """A suggested provider match (synthetic in demo mode)."""

    provider_name: str
    specialty: str
    match_reason: str
    is_synthetic: bool = True
    availability_note: str = "⚠️ Synthetic availability — not real"


class CareState(BaseModel):
    """Complete state of a care navigation workflow run.

    This is the TypedDict-equivalent state that flows through the LangGraph graph.
    Using Pydantic for validation and serialization.
    """

    # --- identifiers ---
    session_id: str
    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    trace_id: str = ""

    # --- conversation ---
    messages: list[ChatTurn] = Field(default_factory=list)
    current_input: str = ""

    # --- workflow phase ---
    phase: CarePhase = CarePhase.INTAKE
    turn_count: int = 0

    # --- triage ---
    triage: TriageResult | None = None

    # --- evidence ---
    evidence: list[EvidenceItem] = Field(default_factory=list)
    evidence_summary: str = ""

    # --- provider matching ---
    provider_suggestions: list[ProviderSuggestion] = Field(default_factory=list)

    # --- response ---
    pending_response: str = ""

    # --- error tracking ---
    error_message: str | None = None
    node_errors: dict[str, str] = Field(default_factory=dict)

    # --- metadata ---
    is_demo: bool = True
    created_at: datetime = Field(default_factory=utcnow)
    updated_at: datetime = Field(default_factory=utcnow)
