"""Session Browser page — list sessions and inspect their events."""

from __future__ import annotations

import streamlit as st
from ui.api_client import get

st.header("📋 Session Browser")

# ---------------------------------------------------------------------------
# List sessions
# ---------------------------------------------------------------------------
try:
    resp = get("/api/v1/sessions")
except Exception as exc:
    st.error(f"Cannot reach backend: {exc}")
    st.stop()

if resp.status_code != 200:
    st.error(f"Failed to list sessions: {resp.status_code}")
    st.stop()

sessions = resp.json()
if not sessions:
    st.info("No sessions yet. Start one from the Care Chat page.")
    st.stop()

# Display as a table
st.subheader(f"{len(sessions)} session(s)")
for session in sessions:
    actor = session.get("actor", {})
    with st.expander(
        f"🗂 {actor.get('display_name', '?')} — {session['status']} — "
        f"`{session['id'][:8]}…` — {session.get('created_at', '?')[:19]}"
    ):
        col1, col2, col3 = st.columns(3)
        col1.metric("Status", session["status"])
        col2.metric("Modality", session.get("modality", "?"))
        col3.metric("Demo", "Yes" if session.get("is_demo") else "No")

        # Fetch events for this session
        if st.button(f"Load events", key=f"events-{session['id']}"):
            events_resp = get(f"/api/v1/sessions/{session['id']}/events")
            if events_resp.status_code == 200:
                events = events_resp.json()
                if events:
                    for ev in events:
                        st.markdown(
                            f"**{ev['event_type']}** (seq {ev['sequence']}) — "
                            f"`{ev['occurred_at'][:19]}` — actor: `{ev['actor']}`"
                        )
                        if ev.get("payload"):
                            st.json(ev["payload"])
                else:
                    st.caption("No events recorded.")
            else:
                st.error(f"Failed to load events: {events_resp.status_code}")

        # Session detail
        if st.button(f"Full detail", key=f"detail-{session['id']}"):
            detail_resp = get(f"/api/v1/sessions/{session['id']}")
            if detail_resp.status_code == 200:
                detail = detail_resp.json()
                st.json(detail)
            else:
                st.error(f"Detail failed: {detail_resp.status_code}")
