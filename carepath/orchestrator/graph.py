"""LangGraph care navigation graph — the primary orchestrator.

Single graph owns state transitions. Deterministic code controls routing (not LLM).
Phase 2 implements: intake → triage → evidence_search → provider_match → present.
"""

from __future__ import annotations

import logging
import operator
from functools import partial
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from carepath.models.gateway import ModelGateway
from carepath.orchestrator.nodes import (
    evidence_search_node,
    intake_node,
    present_node,
    provider_match_node,
    triage_node,
)
from carepath.orchestrator.state import CarePhase
from carepath.research import ResearchService

logger = logging.getLogger(__name__)


def _last_value(old: Any, new: Any) -> Any:
    """Reducer: always take the latest value (LangGraph default for non-list fields)."""
    return new


class GraphState(TypedDict, total=False):
    """Typed state schema for the LangGraph graph.

    Each field uses _last_value reducer so partial updates merge correctly
    and ALL keys survive across nodes.
    """

    session_id: Annotated[str, _last_value]
    trace_id: Annotated[str, _last_value]
    current_input: Annotated[str, _last_value]
    messages: Annotated[list[dict[str, Any]], _last_value]
    phase: Annotated[str, _last_value]
    turn_count: Annotated[int, _last_value]
    triage: Annotated[dict[str, Any] | None, _last_value]
    evidence: Annotated[list[dict[str, Any]], _last_value]
    evidence_summary: Annotated[str, _last_value]
    provider_suggestions: Annotated[list[dict[str, Any]], _last_value]
    pending_response: Annotated[str, _last_value]
    error_message: Annotated[str | None, _last_value]
    node_errors: Annotated[dict[str, str], _last_value]
    is_demo: Annotated[bool, _last_value]
    updated_at: Annotated[str, _last_value]


def _route_after_present(state: GraphState) -> str:
    """Deterministic routing: always go to END after presenting."""
    return END


def build_care_graph(
    gateway: ModelGateway,
    research_service: ResearchService | None = None,
) -> StateGraph:
    """Build the care navigation graph with the given model gateway and research service.

    The graph structure is:
        intake → triage → evidence_search → provider_match → present → END
    """
    graph = StateGraph(GraphState)

    # Bind the gateway and research service to nodes via partial application
    graph.add_node("intake", partial(intake_node, gateway=gateway))
    graph.add_node("triage", partial(triage_node, gateway=gateway))
    graph.add_node(
        "evidence_search",
        partial(evidence_search_node, gateway=gateway, research_service=research_service),
    )
    graph.add_node("provider_match", partial(provider_match_node, gateway=gateway))
    graph.add_node("present", partial(present_node, gateway=gateway))

    # Linear flow for Phase 2-4
    graph.set_entry_point("intake")
    graph.add_edge("intake", "triage")
    graph.add_edge("triage", "evidence_search")
    graph.add_edge("evidence_search", "provider_match")
    graph.add_edge("provider_match", "present")
    graph.add_edge("present", END)

    return graph


def compile_care_graph(
    gateway: ModelGateway,
    research_service: ResearchService | None = None,
):
    """Build and compile the graph, returning a runnable."""
    graph = build_care_graph(gateway, research_service=research_service)
    return graph.compile()


async def run_care_graph(
    gateway: ModelGateway,
    *,
    session_id: str,
    user_input: str,
    trace_id: str,
    existing_messages: list[dict[str, Any]] | None = None,
    research_service: ResearchService | None = None,
) -> dict[str, Any]:
    """Execute the care navigation graph for a single user turn.

    Returns the final state dict containing messages, triage, evidence,
    provider suggestions, and the pending response.
    """
    compiled = compile_care_graph(gateway, research_service=research_service)

    initial_state: dict[str, Any] = {
        "session_id": session_id,
        "trace_id": trace_id,
        "current_input": user_input,
        "messages": existing_messages or [],
        "phase": CarePhase.INTAKE,
        "turn_count": 0,
        "triage": None,
        "evidence": [],
        "evidence_summary": "",
        "provider_suggestions": [],
        "pending_response": "",
        "error_message": None,
        "node_errors": {},
        "is_demo": True,
        "updated_at": "",
    }

    try:
        result = await compiled.ainvoke(initial_state)
        logger.info(
            "care graph completed",
            extra={
                "session_id": session_id,
                "phase": result.get("phase", "unknown"),
                "turn_count": result.get("turn_count", 0),
            },
        )
        return dict(result)
    except Exception as exc:
        logger.error("care graph failed", extra={"session_id": session_id, "error": str(exc)})
        return {
            **initial_state,
            "phase": CarePhase.ERROR,
            "error_message": f"Orchestrator error: {type(exc).__name__}: {str(exc)[:200]}",
            "pending_response": (
                "I'm sorry, I encountered an error processing your request. "
                "Please try again or consult a healthcare provider directly.\n\n"
                "⚠️ *Research prototype — synthetic data only.*"
            ),
        }
