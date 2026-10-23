"""Chat endpoint — connects the Streamlit UI to the LangGraph orchestrator.

Persists each user/assistant turn to the database and loads history on resume.
"""

from __future__ import annotations

import uuid
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.dependencies import DbSession, GatewayDep, RequireAuth, ResearchServiceDep
from carepath.observability.context import current_trace_id
from carepath.orchestrator.graph import run_care_graph
from carepath.persistence.chat import append_chat_turn, load_chat_history

router = APIRouter(prefix="/api/v1", tags=["chat"], dependencies=[RequireAuth])


class ChatRequest(BaseModel):
    session_id: str
    message: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    response: str
    triage: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    provider_suggestions: list[dict[str, Any]] = Field(default_factory=list)
    phase: str = ""
    trace_id: str = ""
    turn_count: int = 0
    error: str | None = None


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    gateway: GatewayDep,
    db: DbSession,
    research: ResearchServiceDep,
) -> ChatResponse:
    """Process a user message through the care navigation orchestrator."""
    trace_id = current_trace_id()

    try:
        session_uuid = uuid.UUID(request.session_id)
    except ValueError:
        raise HTTPException(status_code=422, detail="Invalid session_id format")

    # Load existing chat history from DB
    existing_messages = await load_chat_history(db, session_uuid)

    # Persist the user's message
    await append_chat_turn(
        db,
        session_id=session_uuid,
        role="user",
        content=request.message,
        trace_id=trace_id,
    )

    # Run the orchestrator
    result = await run_care_graph(
        gateway,
        session_id=request.session_id,
        user_input=request.message,
        trace_id=trace_id,
        existing_messages=existing_messages,
        research_service=research,
    )

    response_text = result.get("pending_response", "")

    # Persist the assistant's response
    if response_text:
        await append_chat_turn(
            db,
            session_id=session_uuid,
            role="assistant",
            content=response_text,
            trace_id=trace_id,
            metadata={
                "phase": result.get("phase", ""),
                "has_triage": result.get("triage") is not None,
                "evidence_count": len(result.get("evidence", [])),
                "provider_count": len(result.get("provider_suggestions", [])),
            },
        )

    await db.commit()

    return ChatResponse(
        response=response_text,
        triage=result.get("triage"),
        evidence=result.get("evidence", []),
        provider_suggestions=result.get("provider_suggestions", []),
        phase=result.get("phase", ""),
        trace_id=trace_id,
        turn_count=result.get("turn_count", 0),
        error=result.get("error_message"),
    )


@router.get("/sessions/{session_id}/chat-history")
async def get_chat_history(session_id: uuid.UUID, db: DbSession) -> list[dict[str, Any]]:
    """Load persisted chat history for a session."""
    return await load_chat_history(db, session_id)
