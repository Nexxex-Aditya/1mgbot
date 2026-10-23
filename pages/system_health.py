"""System Health page — live backend health check and configuration diagnostics."""

from __future__ import annotations

import streamlit as st
from ui.api_client import get, post

st.header("🔍 System Health")

# ---------------------------------------------------------------------------
# Liveness
# ---------------------------------------------------------------------------
st.subheader("Liveness")
try:
    live = get("/api/v1/health/live")
    if live.status_code == 200:
        body = live.json()
        st.success(f"API is alive — v{body.get('version', '?')}")
    else:
        st.error(f"Liveness check failed: {live.status_code}")
except Exception as exc:
    st.error(f"Cannot reach backend at `{st.session_state.backend_url}`: {exc}")
    st.info("Start the backend with: `uv run uvicorn backend.main:app --reload`")
    st.stop()

# ---------------------------------------------------------------------------
# Readiness (component health)
# ---------------------------------------------------------------------------
st.subheader("Readiness")
try:
    ready = get("/api/v1/health/ready")
    if ready.status_code == 200:
        body = ready.json()
        overall = body.get("status", "unknown")
        demo = body.get("demo_mode", False)

        status_colors = {"ready": "🟢", "degraded": "🟡", "not_ready": "🔴"}
        st.markdown(f"**Overall:** {status_colors.get(overall, '⚪')} {overall}  ·  **Demo mode:** {'Yes' if demo else 'No'}")

        components = body.get("components", [])
        if components:
            cols = st.columns(min(len(components), 4))
            state_icons = {
                "healthy": "🟢",
                "degraded": "🟡",
                "unavailable": "🔴",
                "not_configured": "⚪",
                "disabled": "⬜",
                "configured_unverified": "🔵",
            }
            for i, comp in enumerate(components):
                with cols[i % len(cols)]:
                    icon = state_icons.get(comp["state"], "❓")
                    st.metric(
                        label=comp["name"],
                        value=f"{icon} {comp['state']}",
                        help=comp.get("detail", ""),
                    )
    else:
        st.warning(f"Readiness probe returned {ready.status_code}")
except Exception as exc:
    st.error(f"Readiness check failed: {exc}")

# ---------------------------------------------------------------------------
# Model connectivity probe (user-triggered)
# ---------------------------------------------------------------------------
st.subheader("Model Gateway")
if st.button("🔌 Run Connectivity Probe"):
    with st.spinner("Probing model gateway…"):
        try:
            resp = post("/api/v1/models/check")
            if resp.status_code == 200:
                result = resp.json()
                status = result.get("status", "unknown")
                if status == "ok":
                    st.success(
                        f"✅ Connected to **{result['model']}** via {result['provider']}  ·  "
                        f"{result.get('latency_ms', '?')}ms  ·  "
                        f"Structured output: {'✓' if result.get('structured_output_valid') else '✗'}"
                    )
                elif status == "not_configured":
                    st.warning(f"⚠️ {result.get('message', 'Model gateway not configured')}")
                else:
                    st.error(f"❌ {result.get('message', 'Probe failed')}")
            else:
                st.error(f"Probe returned {resp.status_code}: {resp.text[:300]}")
        except Exception as exc:
            st.error(f"Probe error: {exc}")

# ---------------------------------------------------------------------------
# Configuration report
# ---------------------------------------------------------------------------
st.subheader("Configuration")
try:
    config = get("/api/v1/config")
    if config.status_code == 200:
        body = config.json()
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Application**")
            st.json({
                "environment": body.get("app_env"),
                "demo_mode": body.get("demo_mode"),
                "log_level": body.get("log_level"),
                "auth_required": body.get("auth_required"),
                "database": body.get("database_target"),
                "cors_origins": body.get("cors_origins"),
            })

        with col2:
            st.markdown("**Model Gateway**")
            st.json({
                "base_url": body.get("model_base_url"),
                "llm_model": body.get("llm_model"),
                "judge_model": body.get("judge_model"),
                "timeout_s": body.get("llm_timeout_seconds"),
                "max_retries": body.get("llm_max_retries"),
                "concurrency": body.get("max_concurrency"),
            })

        st.markdown("**Secrets** (configured / missing)")
        secrets = body.get("secrets", {})
        secret_cols = st.columns(3)
        for i, (key, val) in enumerate(secrets.items()):
            with secret_cols[i % 3]:
                icon = "✅" if val == "configured" else "❌"
                st.markdown(f"{icon} `{key}`")

        with st.expander("Research limits"):
            st.json(body.get("research_limits", {}))

        with st.expander("Optional services"):
            st.json({
                "qdrant": body.get("qdrant_mode"),
                "neo4j": body.get("neo4j_enabled"),
                "searxng": body.get("searxng_configured"),
                "langsmith": body.get("langsmith_tracing"),
                "voice": body.get("voice_enabled"),
            })
except Exception as exc:
    st.error(f"Config report failed: {exc}")
