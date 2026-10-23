"""Thin HTTP client for Streamlit → FastAPI backend calls."""

from __future__ import annotations

import httpx
import streamlit as st


def _base_url() -> str:
    return st.session_state.get("backend_url", "http://localhost:8000")


def _headers() -> dict[str, str]:
    import os
    token = os.environ.get("API_BEARER_TOKEN", "")
    h: dict[str, str] = {"Accept": "application/json"}
    if token:
        h["Authorization"] = f"Bearer {token}"
    return h


def get(path: str, *, timeout: float = 10.0) -> httpx.Response:
    return httpx.get(f"{_base_url()}{path}", headers=_headers(), timeout=timeout)


def post(path: str, *, json: dict | None = None, timeout: float = 30.0, extra_headers: dict[str, str] | None = None) -> httpx.Response:
    headers = _headers()
    if extra_headers:
        headers.update(extra_headers)
    return httpx.post(f"{_base_url()}{path}", headers=headers, json=json, timeout=timeout)
