"""Graph node implementations for the care navigation orchestrator.

Each node:
- Receives the full CareState
- Returns a partial state update dict
- Uses the model gateway for LLM calls (deterministic code owns safety gates)
- Never diagnoses, prescribes, or makes unsupervised clinical decisions
"""

from __future__ import annotations

import json
import logging
from typing import Any

from carepath.contracts.common import utcnow
from carepath.contracts.models import ChatMessage, ModelRequest
from carepath.models.gateway import ModelGateway
from carepath.orchestrator.state import (
    CarePhase,
    CareState,
    ChatTurn,
    EvidenceItem,
    ProviderSuggestion,
    TriageResult,
    TriageUrgency,
)
from carepath.research import ResearchService, SearchQuery, get_research_service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates (inline for Phase 2; will move to carepath/prompts/ later)
# ---------------------------------------------------------------------------

TRIAGE_SYSTEM = """\
You are a symptom triage assistant in a research prototype using synthetic data only.
You do NOT diagnose. You do NOT prescribe. You classify the described symptoms by urgency
and suggest medical specialties that might be relevant.

Respond with ONLY a JSON object matching this schema:
{
  "urgency": "emergency|urgent|standard|self_care|unknown",
  "chief_complaint": "brief summary",
  "symptoms": ["symptom1", "symptom2"],
  "suggested_specialties": ["specialty1"],
  "confidence": 0.0 to 1.0,
  "reasoning": "brief clinical reasoning",
  "safety_flags": ["any safety concerns"]
}
"""

RESPONSE_SYSTEM = """\
You are a care navigation assistant in a research prototype. You help users understand
their symptoms and find appropriate care. You use ONLY the evidence and triage information
provided to you. You NEVER diagnose, prescribe medication, or make clinical decisions.

Always:
- State this is a research prototype with synthetic data
- Cite evidence sources when available
- Recommend consulting a real healthcare provider
- Flag any safety concerns prominently

Format your response in clear, empathetic language.
"""


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

async def intake_node(state: dict[str, Any], gateway: ModelGateway) -> dict[str, Any]:
    """Process the user's initial message and prepare for triage."""
    current_input = state.get("current_input", "")
    messages = list(state.get("messages", []))

    messages.append(
        ChatTurn(role="user", content=current_input).model_dump()
        if isinstance(state.get("messages", [{}])[0] if state.get("messages") else {}, dict)
        else ChatTurn(role="user", content=current_input).model_dump()
    )

    return {
        "messages": messages,
        "phase": CarePhase.TRIAGE,
        "turn_count": state.get("turn_count", 0) + 1,
        "updated_at": utcnow().isoformat(),
    }


async def triage_node(state: dict[str, Any], gateway: ModelGateway) -> dict[str, Any]:
    """Use the model gateway to classify symptoms by urgency and suggest specialties.

    The model output is UNTRUSTED — it is validated against the TriageResult schema.
    Deterministic code enforces safety flags for emergency urgency.
    """
    current_input = state.get("current_input", "")

    if not gateway.configured:
        logger.warning("triage: model gateway not configured, using fallback")
        return {
            "triage": TriageResult(
                urgency=TriageUrgency.UNKNOWN,
                chief_complaint=current_input[:200],
                reasoning="Model gateway not configured — unable to perform AI triage.",
            ).model_dump(),
            "phase": CarePhase.EVIDENCE_SEARCH,
            "node_errors": {**state.get("node_errors", {}), "triage": "gateway_not_configured"},
        }

    try:
        request = ModelRequest(
            messages=[
                ChatMessage(role="system", content=TRIAGE_SYSTEM),
                ChatMessage(role="user", content=f"Patient describes: {current_input}"),
            ],
            temperature=0.0,
            max_tokens=512,
            json_mode=True,
        )
        result = await gateway.complete_structured(request, TriageResult, repair_attempts=1)
        triage = result.value

        # Deterministic safety gate: emergency urgency triggers safety flags
        if triage.urgency == TriageUrgency.EMERGENCY and not triage.safety_flags:
            triage = triage.model_copy(
                update={"safety_flags": ["EMERGENCY: Seek immediate medical attention. Call emergency services."]}
            )

        logger.info(
            "triage complete",
            extra={"urgency": triage.urgency, "confidence": triage.confidence, "specialties": triage.suggested_specialties},
        )
        return {
            "triage": triage.model_dump(),
            "phase": CarePhase.EVIDENCE_SEARCH,
        }
    except Exception as exc:
        logger.error("triage failed", extra={"error": str(exc)})
        return {
            "triage": TriageResult(
                urgency=TriageUrgency.UNKNOWN,
                chief_complaint=current_input[:200],
                reasoning=f"Triage model call failed: {type(exc).__name__}",
            ).model_dump(),
            "phase": CarePhase.EVIDENCE_SEARCH,
            "node_errors": {**state.get("node_errors", {}), "triage": str(exc)[:200]},
        }


async def evidence_search_node(
    state: dict[str, Any],
    gateway: ModelGateway,
    research_service: ResearchService | None = None,
) -> dict[str, Any]:
    """Search for relevant evidence using the configured ResearchService (Qdrant, SearxNG, Synthetic)."""
    triage_data = state.get("triage", {}) or {}
    chief_complaint = triage_data.get("chief_complaint") or state.get("current_input", "general health inquiry")
    specialties = triage_data.get("suggested_specialties", [])
    symptoms = triage_data.get("symptoms", [])

    service = research_service or get_research_service()

    try:
        query = SearchQuery(
            text=chief_complaint,
            specialties=specialties,
            symptoms=symptoms,
            max_results=5,
        )
        results = await service.search(query)
    except Exception as exc:
        logger.warning("evidence search failed, using minimal synthetic fallback", extra={"error": str(exc)})
        results = []

    if results:
        evidence = [
            EvidenceItem(
                title=r.title,
                snippet=r.snippet,
                source_url=r.source_url,
                source_type=r.source_type,
                relevance_score=r.relevance_score,
            ).model_dump()
            for r in results
        ]
    else:
        # Fallback if no results returned
        evidence = [
            EvidenceItem(
                title=f"General guidance for: {chief_complaint}",
                snippet=(
                    "⚠️ SYNTHETIC EVIDENCE — This is a research prototype. "
                    "In the full system, this would be replaced by curated medical literature. "
                    "Always consult a qualified healthcare provider."
                ),
                source_type="synthetic",
                relevance_score=0.5,
            ).model_dump()
        ]

    source_types = ", ".join(sorted(set(e.get("source_type", "unknown") for e in evidence)))
    summary = f"Found {len(evidence)} evidence items for: {chief_complaint} (sources: {source_types})"

    return {
        "evidence": evidence,
        "evidence_summary": summary,
        "phase": CarePhase.PROVIDER_MATCH,
    }


async def provider_match_node(state: dict[str, Any], gateway: ModelGateway) -> dict[str, Any]:
    """Match providers based on triage. Phase 2 uses synthetic providers (clearly labelled)."""
    triage_data = state.get("triage", {})
    specialties = triage_data.get("suggested_specialties", ["General Medicine"])

    # Phase 2: synthetic providers (clearly labelled)
    suggestions = []
    synthetic_providers = [
        ("Dr. Sharma", "General Medicine", "Broad symptom evaluation"),
        ("Dr. Patel", "Internal Medicine", "Detailed diagnostic workup"),
        ("Dr. Gupta", "Specialist", "Targeted specialty care"),
    ]

    for name, specialty, reason in synthetic_providers[:len(specialties) + 1]:
        matched_specialty = specialties[0] if specialties else specialty
        suggestions.append(
            ProviderSuggestion(
                provider_name=f"{name} (SYNTHETIC)",
                specialty=matched_specialty,
                match_reason=reason,
                is_synthetic=True,
                availability_note="⚠️ Synthetic availability — not real booking",
            ).model_dump()
        )

    return {
        "provider_suggestions": suggestions,
        "phase": CarePhase.PRESENT,
    }


async def present_node(state: dict[str, Any], gateway: ModelGateway) -> dict[str, Any]:
    """Generate the final response using triage, evidence and provider data.

    The model synthesizes a user-friendly response. Deterministic code adds disclaimers.
    """
    triage_data = state.get("triage", {})
    evidence = state.get("evidence", [])
    providers = state.get("provider_suggestions", [])
    current_input = state.get("current_input", "")

    context_block = _build_context(triage_data, evidence, providers)

    if not gateway.configured:
        response_text = _build_fallback_response(triage_data, evidence, providers)
    else:
        try:
            request = ModelRequest(
                messages=[
                    ChatMessage(role="system", content=RESPONSE_SYSTEM),
                    ChatMessage(
                        role="user",
                        content=(
                            f"Patient said: {current_input}\n\n"
                            f"Available context:\n{context_block}\n\n"
                            "Generate a helpful, empathetic care navigation response. "
                            "Do NOT diagnose or prescribe. Cite sources. Recommend seeing a real provider."
                        ),
                    ),
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            response = await gateway.complete(request)
            response_text = response.content
        except Exception as exc:
            logger.error("present node model call failed", extra={"error": str(exc)})
            response_text = _build_fallback_response(triage_data, evidence, providers)

    # Deterministic safety: always append disclaimer
    disclaimer = (
        "\n\n---\n"
        "⚠️ *Research prototype — synthetic data only. This is not medical advice. "
        "Always consult a qualified healthcare provider for health concerns.*"
    )

    # Add safety flags prominently
    safety_flags = triage_data.get("safety_flags", [])
    if safety_flags:
        safety_block = "\n".join(f"🚨 **{flag}**" for flag in safety_flags)
        response_text = f"{safety_block}\n\n{response_text}"

    full_response = response_text + disclaimer

    messages = list(state.get("messages", []))
    messages.append(ChatTurn(role="assistant", content=full_response).model_dump())

    return {
        "messages": messages,
        "pending_response": full_response,
        "phase": CarePhase.AWAITING_INPUT,
        "updated_at": utcnow().isoformat(),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_context(triage: dict, evidence: list, providers: list) -> str:
    parts = []
    if triage:
        parts.append(f"**Triage:** {triage.get('urgency', 'unknown')} urgency — {triage.get('chief_complaint', 'N/A')}")
        if triage.get("suggested_specialties"):
            parts.append(f"**Specialties:** {', '.join(triage['suggested_specialties'])}")
        if triage.get("reasoning"):
            parts.append(f"**Reasoning:** {triage['reasoning']}")
    if evidence:
        parts.append(f"**Evidence:** {len(evidence)} items found")
        for ev in evidence[:3]:
            parts.append(f"  - {ev.get('title', 'Untitled')}: {ev.get('snippet', '')[:100]}")
    if providers:
        parts.append(f"**Providers:** {len(providers)} suggestions")
        for p in providers[:3]:
            parts.append(f"  - {p.get('provider_name', '?')} ({p.get('specialty', '?')})")
    return "\n".join(parts) if parts else "No context available."


def _build_fallback_response(triage: dict, evidence: list, providers: list) -> str:
    """Deterministic response when the model gateway is unavailable."""
    parts = ["Thank you for sharing your symptoms. Here's what I found:\n"]

    urgency = triage.get("urgency", "unknown")
    complaint = triage.get("chief_complaint", "your described symptoms")
    parts.append(f"**Assessment:** Based on your description of *{complaint}*, "
                 f"the preliminary triage suggests **{urgency}** urgency.\n")

    specialties = triage.get("suggested_specialties", [])
    if specialties:
        parts.append(f"**Suggested specialties:** {', '.join(specialties)}\n")

    if evidence:
        parts.append("**Evidence found:**")
        for ev in evidence:
            parts.append(f"- {ev.get('title', 'N/A')} *(source: {ev.get('source_type', 'unknown')})*")
        parts.append("")

    if providers:
        parts.append("**Provider suggestions** *(synthetic — not real bookings)*:")
        for p in providers:
            parts.append(f"- **{p.get('provider_name', '?')}** — {p.get('specialty', '?')} — {p.get('match_reason', '')}")
        parts.append("")

    parts.append("**Recommended next step:** Please consult a qualified healthcare provider for proper evaluation.")

    return "\n".join(parts)
