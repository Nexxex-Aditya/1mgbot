"""Seed synthetic demo actors. Idempotent: skips already-existing handles."""

from __future__ import annotations

import asyncio
import os
import sys

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

# Allow running as `python -m scripts.seed` from repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from carepath.persistence.database import create_engine, create_session_factory
from carepath.persistence.records import DemoActorRecord

DEMO_ACTORS = [
    {"handle": "demo-asha", "display_name": "Asha (Synthetic Demo Patient)", "is_synthetic": True},
    {"handle": "demo-ravi", "display_name": "Ravi (Synthetic Demo Patient)", "is_synthetic": True},
    {"handle": "demo-priya", "display_name": "Priya (Synthetic Demo Patient)", "is_synthetic": True},
]


async def seed(database_url: str) -> None:
    engine = create_engine(database_url)
    factory = create_session_factory(engine)
    async with factory() as session:
        for actor_data in DEMO_ACTORS:
            existing = await session.scalar(
                select(DemoActorRecord).where(DemoActorRecord.handle == actor_data["handle"])
            )
            if existing:
                print(f"  skip {actor_data['handle']} (already exists)")
                continue
            session.add(DemoActorRecord(**actor_data))
            print(f"  created {actor_data['handle']}")
        await session.commit()
    await engine.dispose()
    print("Seed complete.")


def main() -> None:
    url = os.environ.get("DATABASE_URL", "")
    if not url:
        print("ERROR: DATABASE_URL environment variable is required.", file=sys.stderr)
        sys.exit(1)
    asyncio.run(seed(url))


if __name__ == "__main__":
    main()
