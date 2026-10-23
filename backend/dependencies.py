"""FastAPI dependency injection: shared settings, DB sessions, gateway and auth."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from carepath.config.settings import Settings
from carepath.models.gateway import ModelGateway
from carepath.research import ResearchService, get_research_service


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _gateway(request: Request) -> ModelGateway:
    return request.app.state.gateway


def _research_service(request: Request) -> ResearchService:
    if hasattr(request.app.state, "research_service"):
        return request.app.state.research_service
    return get_research_service(getattr(request.app.state, "settings", None))


SettingsDep = Annotated[Settings, Depends(_settings)]
GatewayDep = Annotated[ModelGateway, Depends(_gateway)]
ResearchServiceDep = Annotated[ResearchService, Depends(_research_service)]


async def _db_session(request: Request) -> AsyncIterator[AsyncSession]:
    factory = request.app.state.db_session_factory
    async with factory() as session:
        yield session


DbSession = Annotated[AsyncSession, Depends(_db_session)]


async def _require_auth(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
) -> None:
    settings: Settings = request.app.state.settings
    if not settings.auth_required:
        return
    expected = settings.api_bearer_token
    if expected is None:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
    token = authorization[7:]
    if token != expected.get_secret_value():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token.")


RequireAuth = Depends(_require_auth)
