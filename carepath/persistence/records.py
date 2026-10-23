"""SQLAlchemy ORM records (canonical store). Distinct from API contracts by design."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Text,
    UniqueConstraint,
    Uuid,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from carepath.contracts.common import utcnow
from carepath.persistence.types import JSONType, UTCDateTime

NAMING_CONVENTION = {
    "ix": "ix_%(column_0_label)s",
    "uq": "uq_%(table_name)s_%(column_0_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)


class DemoActorRecord(Base):
    """Synthetic demo identity. Holds no phone numbers, ABHA IDs or real names."""

    __tablename__ = "demo_actors"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    handle: Mapped[str] = mapped_column(String(40), unique=True, nullable=False)
    display_name: Mapped[str] = mapped_column(String(80), nullable=False)
    is_synthetic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class CareSessionRecord(Base):
    __tablename__ = "care_sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("demo_actors.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="active")
    modality: Mapped[str] = mapped_column(String(20), nullable=False, default="text")
    is_demo: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    workflow_version: Mapped[str] = mapped_column(String(40), nullable=False)
    idempotency_key: Mapped[str | None] = mapped_column(String(80), unique=True, nullable=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        UTCDateTime(), nullable=False, default=utcnow, onupdate=utcnow
    )

    actor: Mapped[DemoActorRecord] = relationship(lazy="joined")


class ChatTurnRecord(Base):
    """Per-session chat message. Ordered by sequence within a session."""

    __tablename__ = "chat_turns"
    __table_args__ = (
        UniqueConstraint("session_id", "sequence", name="uq_chat_turns_session_seq"),
        Index("ix_chat_turns_session_created", "session_id", "created_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("care_sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String(20), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    metadata_json: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)


class ConsentRecord(Base):
    __tablename__ = "consent_records"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    actor_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("demo_actors.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    purpose: Mapped[str] = mapped_column(String(60), nullable=False)
    consent_version: Mapped[str] = mapped_column(String(40), nullable=False)
    decision: Mapped[str] = mapped_column(String(20), nullable=False)
    decided_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    revoked_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False)


class WorkflowRunRecord(Base):
    __tablename__ = "workflow_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    workflow_name: Mapped[str] = mapped_column(String(80), nullable=False)
    workflow_version: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(UTCDateTime(), nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    error_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_ref: Mapped[str | None] = mapped_column(String(200), nullable=True)


class WorkflowEventRecord(Base):
    """Append-only event log. Payloads hold redacted metadata/references, not raw health text."""

    __tablename__ = "workflow_events"
    __table_args__ = (
        UniqueConstraint(
            "aggregate_type", "aggregate_id", "sequence", name="uq_workflow_events_aggregate_seq"
        ),
        Index("ix_workflow_events_session_occurred", "session_id", "occurred_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    aggregate_type: Mapped[str] = mapped_column(String(40), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(64), nullable=False)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    event_type: Mapped[str] = mapped_column(String(80), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(10), nullable=False, default="1")
    actor: Mapped[str] = mapped_column(String(80), nullable=False)
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("care_sessions.id", ondelete="RESTRICT"), nullable=True
    )
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    occurred_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
    payload: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False, default=dict)


class ModelRunRecord(Base):
    __tablename__ = "model_runs"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid, ForeignKey("workflow_runs.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    provider: Mapped[str] = mapped_column(String(40), nullable=False)
    model_id: Mapped[str] = mapped_column(String(160), nullable=False)
    capability: Mapped[str] = mapped_column(String(40), nullable=False)
    prompt_version: Mapped[str | None] = mapped_column(String(80), nullable=True)
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    total_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(60), nullable=True)
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(UTCDateTime(), nullable=False, default=utcnow)
