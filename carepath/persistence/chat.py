"""Chat turn persistence: save and load conversation history per session."""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from carepath.contracts.common import utcnow
from carepath.persistence.records import ChatTurnRecord


async def append_chat_turn(
    db: AsyncSession,
    *,
    session_id: uuid.UUID,
    role: str,
    content: str,
    trace_id: str,
    metadata: dict[str, Any] | None = None,
) -> ChatTurnRecord:
    """Append a chat turn to the session. Auto-assigns the next sequence number."""
    current_max = await db.scalar(
        select(func.max(ChatTurnRecord.sequence)).where(
            ChatTurnRecord.session_id == session_id
        )
    )
    record = ChatTurnRecord(
        session_id=session_id,
        sequence=(current_max or 0) + 1,
        role=role,
        content=content,
        metadata_json=metadata or {},
        trace_id=trace_id,
        created_at=utcnow(),
    )
    db.add(record)
    await db.flush()
    return record


async def load_chat_history(
    db: AsyncSession,
    session_id: uuid.UUID,
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    """Load chat turns for a session, ordered by sequence."""
    rows = await db.scalars(
        select(ChatTurnRecord)
        .where(ChatTurnRecord.session_id == session_id)
        .order_by(ChatTurnRecord.sequence)
        .limit(limit)
    )
    return [
        {
            "role": r.role,
            "content": r.content,
            "timestamp": r.created_at.isoformat() if r.created_at else None,
            "metadata": r.metadata_json or {},
        }
        for r in rows
    ]


async def chat_turn_count(db: AsyncSession, session_id: uuid.UUID) -> int:
    """Return the number of chat turns in a session."""
    result = await db.scalar(
        select(func.count()).select_from(ChatTurnRecord).where(
            ChatTurnRecord.session_id == session_id
        )
    )
    return result or 0
