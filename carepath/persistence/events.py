"""Append-only workflow event log.

Events are appended inside the caller's transaction so state changes and their audit
events commit (or roll back) atomically. Per-aggregate ordering is enforced by a unique
``(aggregate_type, aggregate_id, sequence)`` constraint; concurrent writers retry on conflict
inside a savepoint.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from carepath.contracts.common import utcnow
from carepath.observability.redaction import redact_mapping
from carepath.persistence.records import WorkflowEventRecord

EVENT_SCHEMA_VERSION = "1"
_MAX_SEQUENCE_RETRIES = 5


async def append_event(
    db: AsyncSession,
    *,
    aggregate_type: str,
    aggregate_id: str | uuid.UUID,
    event_type: str,
    actor: str,
    trace_id: str,
    session_id: uuid.UUID | None = None,
    run_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
) -> WorkflowEventRecord:
    agg_id = str(aggregate_id)
    safe_payload = redact_mapping(payload or {})
    for _ in range(_MAX_SEQUENCE_RETRIES):
        current_max = await db.scalar(
            select(func.max(WorkflowEventRecord.sequence)).where(
                WorkflowEventRecord.aggregate_type == aggregate_type,
                WorkflowEventRecord.aggregate_id == agg_id,
            )
        )
        event = WorkflowEventRecord(
            aggregate_type=aggregate_type,
            aggregate_id=agg_id,
            sequence=(current_max or 0) + 1,
            event_type=event_type,
            schema_version=EVENT_SCHEMA_VERSION,
            actor=actor,
            session_id=session_id,
            run_id=run_id,
            trace_id=trace_id,
            occurred_at=utcnow(),
            payload=safe_payload,
        )
        try:
            async with db.begin_nested():
                db.add(event)
            return event
        except IntegrityError:
            continue
    raise RuntimeError(f"Could not allocate event sequence for {aggregate_type}:{agg_id}")
