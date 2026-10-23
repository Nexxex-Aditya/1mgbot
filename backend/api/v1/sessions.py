"""Session lifecycle endpoints: POST/GET sessions, GET events."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Header, HTTPException, status

from backend.dependencies import DbSession, RequireAuth, SettingsDep
from carepath.contracts.sessions import (
    CreateSessionRequest,
    DemoActorList,
    SessionCreated,
    SessionDetail,
    WorkflowEventView,
)
from carepath.observability.context import current_trace_id, sanitize_incoming_id
from carepath.sessions.service import (
    ConsentRequiredError,
    IdempotencyConflictError,
    SessionNotFoundError,
    UnknownActorError,
    create_session,
    get_session_detail,
    list_actors,
    list_session_events,
    list_sessions,
)

router = APIRouter(prefix="/api/v1", tags=["sessions"], dependencies=[RequireAuth])


@router.get("/actors", response_model=DemoActorList)
async def get_actors(db: DbSession) -> DemoActorList:
    return DemoActorList(actors=await list_actors(db))


@router.get("/sessions")
async def get_sessions(db: DbSession, settings: SettingsDep) -> list[dict]:
    sessions = await list_sessions(db)
    return [s.model_dump(mode="json") for s in sessions]


@router.post("/sessions", response_model=SessionCreated, status_code=status.HTTP_201_CREATED)
async def post_session(
    request: CreateSessionRequest,
    db: DbSession,
    idempotency_key: str | None = Header(None, alias="Idempotency-Key"),
) -> SessionCreated:
    safe_key = sanitize_incoming_id(idempotency_key)
    trace_id = current_trace_id()
    try:
        result = await create_session(db, request, trace_id=trace_id, idempotency_key=safe_key)
    except ConsentRequiredError:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Demo consent must be accepted to start a synthetic session.",
        )
    except UnknownActorError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown demo actor: {exc}. Seed the database first.",
        )
    except IdempotencyConflictError:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Idempotency-Key was already used with a different request.",
        )
    return SessionCreated(
        session=result.detail,
        run_id=result.run_id,
        trace_id=result.trace_id,
        replayed=result.replayed,
    )


@router.get("/sessions/{session_id}", response_model=SessionDetail)
async def get_session(session_id: uuid.UUID, db: DbSession) -> SessionDetail:
    try:
        return await get_session_detail(db, session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")


@router.get("/sessions/{session_id}/events", response_model=list[WorkflowEventView])
async def get_events(session_id: uuid.UUID, db: DbSession) -> list[WorkflowEventView]:
    try:
        return await list_session_events(db, session_id)
    except SessionNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found.")
