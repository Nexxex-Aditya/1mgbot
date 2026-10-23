"""Pytest fixtures: in-memory SQLite DB, FastAPI test client, and fake model gateway."""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator, Iterator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker

from carepath.config.settings import Settings
from carepath.models.gateway import ModelGateway
from carepath.persistence.records import Base, DemoActorRecord
from tests.fakes import FakeModelProvider

# Force test settings: demo mode, SQLite, no secrets.
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("DEMO_MODE", "true")
os.environ.setdefault("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    return Settings(
        app_env="test",
        demo_mode=True,
        database_url="sqlite+aiosqlite:///:memory:",  # type: ignore[arg-type]
        log_level="WARNING",
    )


@pytest_asyncio.fixture
async def engine():
    eng = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield eng
    await eng.dispose()


@pytest_asyncio.fixture
async def db_session(engine) -> AsyncIterator[AsyncSession]:
    factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
    async with factory() as session:
        yield session


@pytest_asyncio.fixture
async def seeded_db(db_session: AsyncSession) -> AsyncSession:
    """DB session with demo actors seeded."""
    for handle, name in [
        ("demo-asha", "Asha (Synthetic Demo Patient)"),
        ("demo-ravi", "Ravi (Synthetic Demo Patient)"),
    ]:
        db_session.add(
            DemoActorRecord(handle=handle, display_name=name, is_synthetic=True)
        )
    await db_session.commit()
    return db_session


@pytest.fixture
def fake_provider() -> FakeModelProvider:
    return FakeModelProvider()


@pytest.fixture
def fake_gateway(fake_provider: FakeModelProvider) -> ModelGateway:
    return ModelGateway(
        provider=fake_provider,
        default_model="test/fake-model",
        max_retries=1,
        max_concurrency=4,
    )


@pytest.fixture
def app_client(test_settings: Settings, fake_gateway: ModelGateway) -> Iterator[TestClient]:
    """Synchronous TestClient with SQLite backend and fake model gateway."""
    from backend.main import create_app

    app = create_app(settings=test_settings)

    # Override lifespan resources
    import sqlalchemy
    from sqlalchemy.ext.asyncio import create_async_engine as _create

    engine = _create("sqlite+aiosqlite:///:memory:")

    from contextlib import asynccontextmanager
    from collections.abc import AsyncIterator as _AI

    @asynccontextmanager
    async def test_lifespan(app_instance) -> _AI[None]:  # type: ignore[override]
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        # Seed demo actors
        factory = async_sessionmaker(engine, expire_on_commit=False, autoflush=False)
        async with factory() as session:
            for handle, name in [
                ("demo-asha", "Asha (Synthetic Demo Patient)"),
                ("demo-ravi", "Ravi (Synthetic Demo Patient)"),
                ("demo-priya", "Priya (Synthetic Demo Patient)"),
            ]:
                session.add(DemoActorRecord(handle=handle, display_name=name, is_synthetic=True))
            await session.commit()
        app_instance.state.engine = engine
        app_instance.state.db_session_factory = factory
        app_instance.state.gateway = fake_gateway
        app_instance.state.settings = test_settings
        yield
        await engine.dispose()

    app.router.lifespan_context = test_lifespan
    with TestClient(app, raise_server_exceptions=False) as client:
        yield client
