"""CarePath AI — Streamlit application entry point.

Run with: streamlit run app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Page config (must be the very first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="CarePath AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

PAGES = {
    "💬 Care Chat": "pages.care_chat",
    "🔍 System Health": "pages.system_health",
    "📋 Sessions": "pages.session_browser",
}


def _init_session_state() -> None:
    """Initialise shared session state on first run."""
    if "backend_url" not in st.session_state:
        import os
        st.session_state.backend_url = os.environ.get("BACKEND_URL", "http://localhost:8000")
    if "active_session_id" not in st.session_state:
        st.session_state.active_session_id = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []


_init_session_state()

# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------
page = st.navigation(
    [
        st.Page("pages/care_chat.py", title="Care Chat", icon="💬", default=True),
        st.Page("pages/system_health.py", title="System Health", icon="🔍"),
        st.Page("pages/session_browser.py", title="Sessions", icon="📋"),
    ]
)
page.run()
