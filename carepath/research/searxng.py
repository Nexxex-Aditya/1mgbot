"""SearxNG research service — web search via a self-hosted SearxNG instance.

Optional: gracefully unavailable when SEARXNG_BASE_URL is not configured.
"""

from __future__ import annotations

import logging

import httpx

from carepath.research.protocol import EvidenceResult, SearchQuery

logger = logging.getLogger(__name__)


class SearxNGResearchService:
    """Search via a SearxNG instance for web evidence."""

    def __init__(self, base_url: str | None = None, timeout: float = 10.0) -> None:
        self._base_url = (base_url or "").rstrip("/")
        self._timeout = timeout
        self._available = bool(self._base_url)

    @property
    def name(self) -> str:
        return "searxng"

    @property
    def available(self) -> bool:
        return self._available and bool(self._base_url)

    async def search(self, query: SearchQuery) -> list[EvidenceResult]:
        if not self.available or not self._base_url:
            return []

        search_text = query.text
        if query.specialties:
            search_text += f" {' '.join(query.specialties[:2])}"

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(
                    f"{self._base_url}/search",
                    params={
                        "q": search_text,
                        "format": "json",
                        "categories": "general,science",
                        "language": "en",
                        "pageno": 1,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning("SearxNG search failed", extra={"error": str(exc)})
            self._available = False
            return []

        self._available = True
        results: list[EvidenceResult] = []
        for item in data.get("results", [])[: query.max_results]:
            results.append(
                EvidenceResult(
                    title=item.get("title", "Untitled"),
                    snippet=item.get("content", "")[:500],
                    source_url=item.get("url"),
                    source_type="web_search",
                    relevance_score=min(1.0, float(item.get("score") or 0.5)),
                    metadata={"engine": item.get("engine", "unknown")},
                )
            )

        return results
