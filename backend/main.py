"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.v1.chat import router as chat_router
from backend.api.v1.health import router as health_router
from backend.api.v1.models import router as models_router
from backend.api.v1.sessions import router as sessions_router
from backend.middleware import CorrelationMiddleware, RequestSizeLimitMiddleware
from carepath.config.settings import Settings, get_settings
from carepath.models.gateway import ModelGateway, build_model_gateway
from carepath.observability.logging import configure_logging
from carepath.persistence.database import create_engine, create_session_factory
from carepath.research import get_research_service

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared resources at startup, dispose at shutdown."""
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)

    engine = create_engine(
        settings.database_url.get_secret_value(),
        connect_timeout=settings.db_connect_timeout_seconds,
    )
    app.state.engine = engine
    app.state.db_session_factory = create_session_factory(engine)

    gateway = build_model_gateway(settings)
    app.state.gateway = gateway
    app.state.research_service = get_research_service(settings)

    gw_label = "configured" if gateway.configured else "NOT configured (NVIDIA_API_KEY missing)"
    logger.info(
        "startup",
        extra={
            "demo_mode": settings.demo_mode,
            "database": settings.redacted_database_target(),
            "model_gateway": gw_label,
        },
    )

    yield

    await gateway.aclose()
    await engine.dispose()
    logger.info("shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    if settings is None:
        settings = get_settings()

    app = FastAPI(
        title="CarePath AI API",
        description=(
            "Evidence-aware care navigation research prototype. "
            "Synthetic data only — not a medical device."
        ),
        version="0.1.0",
        lifespan=lifespan,
        docs_url="/api/docs" if settings.app_env == "development" else None,
        redoc_url="/api/redoc" if settings.app_env == "development" else None,
    )

    app.state.settings = settings

    # --- middleware (outermost first) ---
    app.add_middleware(RequestSizeLimitMiddleware, max_bytes=settings.max_request_bytes)
    app.add_middleware(CorrelationMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID", "X-Trace-ID"],
    )

    # --- routers ---
    app.include_router(health_router)
    app.include_router(chat_router)
    app.include_router(models_router)
    app.include_router(sessions_router)

    return app


# Module-level app for `uvicorn backend.main:app`
app = create_app()
