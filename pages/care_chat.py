"""Care Chat page — the primary user-facing interaction surface.

Phase 1: Session creation with consent, chat message display, placeholder for
LangGraph-driven conversation (Phase 2+).
"""

from __future__ import annotations

import streamlit as st
from ui.api_client import get, post

st.header("💬 Care Chat")

# ---------------------------------------------------------------------------
# Disclaimers
# ---------------------------------------------------------------------------
st.caption(
    "⚠️ **Research prototype — synthetic data only.** "
    "This is not a medical device. It does not diagnose, prescribe, or replace professional care. "
    "Do not enter real personal or health information."
)

# ---------------------------------------------------------------------------
# Session creation (if none active)
# ---------------------------------------------------------------------------
if st.session_state.active_session_id is None:
    st.subheader("Start a new care session")

    # Fetch demo actors
    actors_resp = None
    try:
        actors_resp = get("/api/v1/actors")
    except Exception as exc:
        st.error(f"Cannot reach backend: {exc}")

    if actors_resp and actors_resp.status_code == 200:
        actors = actors_resp.json().get("actors", [])
        if not actors:
            st.warning("No demo actors found. Run the seed script first.")
            st.stop()

        actor_options = {a["display_name"]: a["handle"] for a in actors}
        selected_name = st.selectbox(
            "Select demo patient (synthetic)",
            options=list(actor_options.keys()),
        )

        st.info(
            "**Demo consent:** This is a research prototype using synthetic data only. "
            "It is not a medical device, does not diagnose or prescribe, and must not be "
            "used for real health decisions. Do not enter real personal or health information."
        )
        consent = st.checkbox("I understand and accept the demo terms", value=False)

        if st.button("Start Session", type="primary", disabled=not consent):
            handle = actor_options[selected_name]
            resp = post(
                "/api/v1/sessions",
                json={
                    "actor_handle": handle,
                    "modality": "text",
                    "consent": {"accepted": True},
                },
            )
            if resp.status_code == 201:
                body = resp.json()
                st.session_state.active_session_id = body["session"]["id"]
                st.session_state.chat_messages = []
                st.session_state.session_detail = body["session"]
                st.rerun()
            else:
                st.error(f"Session creation failed: {resp.text}")
    elif actors_resp:
        st.error(f"Backend error: {actors_resp.status_code} — {actors_resp.text}")

# ---------------------------------------------------------------------------
# Active session: chat interface
# ---------------------------------------------------------------------------
else:
    session_id = st.session_state.active_session_id
    detail = st.session_state.get("session_detail", {})

    col1, col2 = st.columns([3, 1])
    with col1:
        actor_name = detail.get("actor", {}).get("display_name", "Unknown")
        st.markdown(f"**Session:** `{session_id[:8]}…` · **Patient:** {actor_name}")
    with col2:
        if st.button("End Session"):
            st.session_state.active_session_id = None
            st.session_state.chat_messages = []
            st.session_state.pop("session_detail", None)
            st.rerun()

    st.divider()

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input — sends to LangGraph orchestrator via /api/v1/chat
    if prompt := st.chat_input("Describe your symptoms or ask a question…"):
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Processing through care navigation workflow…"):
                try:
                    resp = post(
                        "/api/v1/chat",
                        json={
                            "session_id": session_id,
                            "message": prompt,
                        },
                        timeout=60.0,
                    )
                    if resp.status_code == 200:
                        body = resp.json()
                        response_text = body.get("response", "No response generated.")
                        st.markdown(response_text)
                        st.session_state.chat_messages.append({"role": "assistant", "content": response_text})

                        # Show triage/evidence/provider details in expandable sections
                        triage = body.get("triage")
                        if triage:
                            with st.expander("🏥 Triage Details"):
                                st.markdown(f"**Urgency:** {triage.get('urgency', 'unknown')}")
                                st.markdown(f"**Chief complaint:** {triage.get('chief_complaint', 'N/A')}")
                                if triage.get("suggested_specialties"):
                                    st.markdown(f"**Specialties:** {', '.join(triage['suggested_specialties'])}")
                                if triage.get("safety_flags"):
                                    for flag in triage["safety_flags"]:
                                        st.warning(flag)

                        evidence = body.get("evidence", [])
                        if evidence:
                            with st.expander(f"📚 Evidence ({len(evidence)} items)"):
                                for ev in evidence:
                                    st.markdown(f"- **{ev.get('title', 'N/A')}** ({ev.get('source_type', '?')})")
                                    st.caption(ev.get("snippet", "")[:200])

                        providers = body.get("provider_suggestions", [])
                        if providers:
                            with st.expander(f"👨‍⚕️ Provider Suggestions ({len(providers)})"):
                                for p in providers:
                                    st.markdown(f"- **{p.get('provider_name', '?')}** — {p.get('specialty', '?')}")
                                    st.caption(p.get("availability_note", ""))

                        if body.get("error"):
                            st.warning(f"Workflow note: {body['error']}")
                    else:
                        error_msg = f"Chat request failed ({resp.status_code}): {resp.text[:300]}"
                        st.error(error_msg)
                        st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
                except Exception as exc:
                    error_msg = f"Cannot reach backend: {exc}"
                    st.error(error_msg)
                    st.session_state.chat_messages.append({"role": "assistant", "content": f"⚠️ {error_msg}"})
