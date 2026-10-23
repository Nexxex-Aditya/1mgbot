"""Async engine/session management and database health probes."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path

from alembic.config import Config
from alembic.script import ScriptDirectory
from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class PersistenceUnavailableError(RuntimeError):
    """Raised when an operation requires the canonical store and it is unreachable."""


def create_engine(database_url: str, *, connect_timeout: float = 5.0) -> AsyncEngine:
    kwargs: dict[str, object] = {"pool_pre_ping": True}
    if database_url.startswith("postgresql+asyncpg"):
        kwargs["connect_args"] = {"timeout": connect_timeout, "command_timeout": 30}
        kwargs["pool_size"] = 5
        kwargs["max_overflow"] = 5
    elif database_url.startswith("sqlite"):
        kwargs["connect_args"] = {"timeout": connect_timeout}
    return create_async_engine(database_url, **kwargs)


def create_session_factory(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(engine, expire_on_commit=False, autoflush=False)


async def session_scope(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    async with factory() as session:
        yield session


def alembic_config(database_url: str | None = None) -> Config:
    cfg = Config(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(PROJECT_ROOT / "migrations"))
    if database_url:
        cfg.attributes["database_url"] = database_url
    return cfg


def alembic_head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


@dataclass(frozen=True)
class DatabaseProbe:
    reachable: bool
    latency_ms: float | None
    current_revision: str | None
    head_revision: str | None
    error: str | None = None

    @property
    def at_head(self) -> bool:
        return self.reachable and self.current_revision == self.head_revision


async def probe_database(engine: AsyncEngine, *, timeout: float = 3.0) -> DatabaseProbe:
    head = alembic_head_revision()
    started = time.perf_counter()
    try:
        async with asyncio.timeout(timeout):
            async with engine.connect() as conn:
                await conn.execute(text("SELECT 1"))
                try:
                    result = await conn.execute(text("SELECT version_num FROM alembic_version"))
                    current = result.scalar_one_or_none()
                except SQLAlchemyError:
                    current = None
    except (TimeoutError, SQLAlchemyError, OSError) as exc:
        return DatabaseProbe(False, None, None, head, error=type(exc).__name__)
    return DatabaseProbe(True, round((time.perf_counter() - started) * 1000, 2), current, head)
