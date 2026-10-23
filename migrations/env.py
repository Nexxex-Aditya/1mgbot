"""Alembic migrations env.py — works with both sync PostgreSQL and sync SQLite."""

from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

# Import all models so Alembic sees the tables.
from carepath.persistence.records import Base  # noqa: F401

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _get_url() -> str:
    """Resolve database URL: explicit attribute > env var > settings (.env)."""
    if hasattr(config, "attributes") and config.attributes.get("database_url"):
        url = config.attributes["database_url"]
    else:
        url = os.environ.get("DATABASE_URL", "")
    if not url:
        try:
            from carepath.config.settings import get_settings
            url = get_settings().database_url.get_secret_value()
        except Exception:
            pass
    if not url:
        raise RuntimeError(
            "DATABASE_URL is not set. Provide it as an environment variable, in .env, or via config attributes."
        )
    # Alembic runs synchronous migrations; convert async drivers.
    if "+asyncpg" in url:
        url = url.replace("+asyncpg", "")
    if "+aiosqlite" in url:
        url = url.replace("+aiosqlite", "")
    return url


def run_migrations_offline() -> None:
    context.configure(
        url=_get_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        {"sqlalchemy.url": _get_url()},
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
