"""Session lifecycle service: synthetic session creation with demo consent and audit events."""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from carepath.contracts.common import WorkflowStatus, utcnow
from carepath.contracts.sessions import (
    ConsentDecision,
    ConsentRecordView,
    CreateSessionRequest,
    DemoActorView,
    SessionDetail,
    SessionStatus,
    SessionSummary,
    WorkflowEventView,
)
from carepath.persistence.events import append_event
from carepath.persistence.records import (
    CareSessionRecord,
    ConsentRecord,
    DemoActorRecord,
    WorkflowEventRecord,
    WorkflowRunRecord,
)

SESSION_WORKFLOW_NAME = "session.create"
SESSION_WORKFLOW_VERSION = "1"
CARE_WORKFLOW_VERSION = "carepath-intake-v0"
COMPONENT = "session_service"


class ConsentRequiredError(Exception):
    pass


class UnknownActorError(Exception):
    pass


class IdempotencyConflictError(Exception):
    pass


class SessionNotFoundError(Exception):
    pass


@dataclass(frozen=True)
class SessionCreation:
    detail: SessionDetail
    run_id: uuid.UUID
    trace_id: str
    replayed: bool


def actor_view(actor: DemoActorRecord) -> DemoActorView:
    return DemoActorView(
        id=actor.id,
        handle=actor.handle,
        display_name=actor.display_name,
        is_synthetic=actor.is_synthetic,
    )


def _summary_fields(record: CareSessionRecord) -> dict[str, object]:
    return {
        "id": record.id,
        "actor": actor_view(record.actor),
        "status": SessionStatus(record.status),
        "modality": record.modality,
        "is_demo": record.is_demo,
        "workflow_version": record.workflow_version,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
    }


def session_summary(record: CareSessionRecord) -> SessionSummary:
    return SessionSummary.model_validate(_summary_fields(record))


async def list_actors(db: AsyncSession) -> list[DemoActorView]:
    rows = await db.scalars(select(DemoActorRecord).order_by(DemoActorRecord.handle))
    return [actor_view(a) for a in rows]


async def list_sessions(db: AsyncSession, *, limit: int = 20) -> list[SessionSummary]:
    rows = await db.scalars(
        select(CareSessionRecord).order_by(CareSessionRecord.created_at.desc()).limit(limit)
    )
    return [session_summary(r) for r in rows.unique()]


async def get_session_detail(db: AsyncSession, session_id: uuid.UUID) -> SessionDetail:
    record = await db.get(CareSessionRecord, session_id)
    if record is None:
        raise SessionNotFoundError(str(session_id))
    consents = await db.scalars(
        select(ConsentRecord)
        .where(ConsentRecord.session_id == session_id)
        .order_by(ConsentRecord.decided_at)
    )
    event_count = await db.scalar(
        select(func.count()).select_from(WorkflowEventRecord).where(
            WorkflowEventRecord.session_id == session_id
        )
    )
    run_count = await db.scalar(
        select(func.count()).select_from(WorkflowRunRecord).where(
            WorkflowRunRecord.session_id == session_id
        )
    )
    return SessionDetail.model_validate(
        {
            **_summary_fields(record),
            "consents": [
                ConsentRecordView(
                    id=c.id,
                    purpose=c.purpose,
                    consent_version=c.consent_version,
                    decision=ConsentDecision(c.decision),
                    decided_at=c.decided_at,
                    revoked_at=c.revoked_at,
                    trace_id=c.trace_id,
                )
                for c in consents
            ],
            "event_count": event_count or 0,
            "run_count": run_count or 0,
        }
    )


def event_view(e: WorkflowEventRecord) -> WorkflowEventView:
    return WorkflowEventView(
        id=e.id,
        aggregate_type=e.aggregate_type,
        aggregate_id=e.aggregate_id,
        sequence=e.sequence,
        event_type=e.event_type,
        schema_version=e.schema_version,
        actor=e.actor,
        session_id=e.session_id,
        run_id=e.run_id,
        trace_id=e.trace_id,
        occurred_at=e.occurred_at,
        payload=e.payload,
    )


async def list_session_events(db: AsyncSession, session_id: uuid.UUID) -> list[WorkflowEventView]:
    if await db.get(CareSessionRecord, session_id) is None:
        raise SessionNotFoundError(str(session_id))
    rows = await db.scalars(
        select(WorkflowEventRecord)
        .where(WorkflowEventRecord.session_id == session_id)
        .order_by(WorkflowEventRecord.occurred_at, WorkflowEventRecord.sequence)
    )
    return [event_view(e) for e in rows]


async def _replay(
    db: AsyncSession, idempotency_key: str, request: CreateSessionRequest
) -> SessionCreation | None:
    existing = await db.scalar(
        select(CareSessionRecord).where(CareSessionRecord.idempotency_key == idempotency_key)
    )
    if existing is None:
        return None
    if existing.actor.handle != request.actor_handle or existing.modality != request.modality.value:
        raise IdempotencyConflictError("Idempotency-Key was already used with a different request.")
    run = await db.scalar(
        select(WorkflowRunRecord)
        .where(
            WorkflowRunRecord.session_id == existing.id,
            WorkflowRunRecord.workflow_name == SESSION_WORKFLOW_NAME,
        )
        .order_by(WorkflowRunRecord.started_at)
        .limit(1)
    )
    if run is None:  # pragma: no cover - invariant: created in same transaction
        raise RuntimeError("Session exists without its creation run.")
    return SessionCreation(
        detail=await get_session_detail(db, existing.id),
        run_id=run.id,
        trace_id=run.trace_id,
        replayed=True,
    )


async def create_session(
    db: AsyncSession,
    request: CreateSessionRequest,
    *,
    trace_id: str,
    idempotency_key: str | None = None,
) -> SessionCreation:
    """Create a synthetic care session, its demo consent record and audit events atomically."""
    if not request.consent.accepted:
        raise ConsentRequiredError("Demo consent must be accepted to start a synthetic session.")

    if idempotency_key:
        replay = await _replay(db, idempotency_key, request)
        if replay is not None:
            return replay

    actor = await db.scalar(
        select(DemoActorRecord).where(DemoActorRecord.handle == request.actor_handle)
    )
    if actor is None:
        raise UnknownActorError(request.actor_handle)

    now = utcnow()
    run = WorkflowRunRecord(
        workflow_name=SESSION_WORKFLOW_NAME,
        workflow_version=SESSION_WORKFLOW_VERSION,
        status=WorkflowStatus.RUNNING.value,
        trace_id=trace_id,
        started_at=now,
    )
    session = CareSessionRecord(
        actor_id=actor.id,
        status=SessionStatus.ACTIVE.value,
        modality=request.modality.value,
        is_demo=True,
        workflow_version=CARE_WORKFLOW_VERSION,
        idempotency_key=idempotency_key,
        created_at=now,
        updated_at=now,
    )
    db.add(session)
    try:
        await db.flush()
    except IntegrityError:
        # Concurrent request with the same Idempotency-Key won the race.
        await db.rollback()
        if idempotency_key:
            replay = await _replay(db, idempotency_key, request)
            if replay is not None:
                return replay
        raise
    run.session_id = session.id
    db.add(run)
    consent = ConsentRecord(
        actor_id=actor.id,
        session_id=session.id,
        purpose=request.consent.purpose,
        consent_version=request.consent.version,
        decision=ConsentDecision.ACCEPTED.value,
        decided_at=now,
        trace_id=trace_id,
    )
    db.add(consent)
    await db.flush()

    await append_event(
        db,
        aggregate_type="care_session",
        aggregate_id=session.id,
        event_type="session.created",
        actor=COMPONENT,
        trace_id=trace_id,
        session_id=session.id,
        run_id=run.id,
        payload={
            "actor_handle": actor.handle,
            "modality": session.modality,
            "is_demo": True,
            "workflow_version": session.workflow_version,
        },
    )
    await append_event(
        db,
        aggregate_type="care_session",
        aggregate_id=session.id,
        event_type="consent.recorded",
        actor=COMPONENT,
        trace_id=trace_id,
        session_id=session.id,
        run_id=run.id,
        payload={
            "consent_id": str(consent.id),
            "purpose": consent.purpose,
            "version": consent.consent_version,
            "decision": consent.decision,
        },
    )
    run.status = WorkflowStatus.SUCCEEDED.value
    run.finished_at = utcnow()
    run.output_ref = f"care_session:{session.id}"
    await db.commit()

    return SessionCreation(
        detail=await get_session_detail(db, session.id),
        run_id=run.id,
        trace_id=trace_id,
        replayed=False,
    )
