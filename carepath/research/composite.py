"""Composite research service — combines multiple retrieval backends with fallback.

Queries real evidence providers (Qdrant, SearxNG) when configured, and falls
back to synthetic guidance when external services are unavailable or return no results.
This ensures text/demo mode remains fully functional and honest.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence

from carepath.config.settings import Settings, get_settings
from carepath.research.embeddings import get_embedding_service
from carepath.research.protocol import EvidenceResult, ResearchService, SearchQuery
from carepath.research.qdrant import QdrantResearchService
from carepath.research.searxng import SearxNGResearchService
from carepath.research.synthetic import SyntheticResearchService

logger = logging.getLogger(__name__)


class CompositeResearchService:
    """Dispatches search to multiple backends, merges and deduplicates results."""

    def __init__(
        self,
        backends: Sequence[ResearchService],
        fallback_service: ResearchService | None = None,
    ) -> None:
        self._backends = list(backends)
        self._fallback = fallback_service or SyntheticResearchService()

    @property
    def name(self) -> str:
        return "composite"

    @property
    def available(self) -> bool:
        return any(b.available for b in self._backends) or self._fallback.available

    @property
    def backends(self) -> list[ResearchService]:
        return list(self._backends)

    async def search(self, query: SearchQuery) -> list[EvidenceResult]:
        active_backends = [b for b in self._backends if b.available]

        results: list[EvidenceResult] = []

        if active_backends:
            tasks = [backend.search(query) for backend in active_backends]
            gathered = await asyncio.gather(*tasks, return_exceptions=True)

            for outcome in gathered:
                if isinstance(outcome, Exception):
                    logger.warning("Research backend search failed", extra={"error": str(outcome)})
                    continue
                results.extend(outcome)

        # If real backends yielded results, deduplicate and return
        if results:
            deduped = self._deduplicate(results)
            deduped.sort(key=lambda r: r.relevance_score, reverse=True)
            return deduped[: query.max_results]

        # Otherwise fallback to synthetic (demo mode / offline guarantee)
        if self._fallback and self._fallback.available:
            return await self._fallback.search(query)

        return []

    def _deduplicate(self, items: list[EvidenceResult]) -> list[EvidenceResult]:
        seen_keys: set[str] = set()
        unique: list[EvidenceResult] = []

        for item in items:
            key = (item.title.strip().lower(), item.source_url or "")
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique.append(item)

        return unique


def get_research_service(settings: Settings | None = None) -> CompositeResearchService:
    """Build the research service stack based on application settings."""
    cfg = settings or get_settings()
    backends: list[ResearchService] = []

    # 1. Qdrant vector retrieval (if configured and not disabled)
    if cfg.qdrant_mode != "disabled" and cfg.qdrant_url:
        embedding_svc = get_embedding_service(cfg)
        qdrant_api_key = cfg.qdrant_api_key.get_secret_value() if cfg.qdrant_api_key else None
        backends.append(
            QdrantResearchService(
                url=cfg.qdrant_url,
                api_key=qdrant_api_key,
                collection_name=cfg.qdrant_collection,
                embedding_service=embedding_svc,
            )
        )

    # 2. SearxNG web search (if configured)
    if cfg.searxng_base_url:
        backends.append(SearxNGResearchService(base_url=cfg.searxng_base_url))

    return CompositeResearchService(
        backends=backends,
        fallback_service=SyntheticResearchService(),
    )
