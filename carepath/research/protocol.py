"""Research service protocol and contracts.

Defines the interface that all evidence search backends must implement.
This follows the master prompt directive: "Prefer a ResearchService interface first."
"""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, Field


class EvidenceResult(BaseModel):
    """A single piece of evidence returned by a research backend."""

    title: str
    snippet: str
    source_url: str | None = None
    source_type: str = "unknown"
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SearchQuery(BaseModel):
    """Structured search query for evidence retrieval."""

    text: str
    specialties: list[str] = Field(default_factory=list)
    symptoms: list[str] = Field(default_factory=list)
    max_results: int = Field(default=5, ge=1, le=20)


class ResearchService(Protocol):
    """Protocol for evidence search backends.

    Implementations:
    - SyntheticResearchService (always available, demo mode)
    - SearxNGResearchService (web search, optional)
    - QdrantResearchService (semantic retrieval, optional)
    - CompositeResearchService (combines multiple backends)
    """

    async def search(self, query: SearchQuery) -> list[EvidenceResult]:
        """Execute an evidence search."""
        ...

    @property
    def name(self) -> str:
        """Human-readable name for this backend."""
        ...

    @property
    def available(self) -> bool:
        """Whether this backend is currently available."""
        ...
