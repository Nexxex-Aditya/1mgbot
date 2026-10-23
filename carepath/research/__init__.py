"""CarePath research service — evidence search and retrieval."""

from carepath.research.composite import CompositeResearchService, get_research_service
from carepath.research.embeddings import (
    DeterministicEmbeddingService,
    EmbeddingService,
    NVIDIAEmbeddingService,
    get_embedding_service,
)
from carepath.research.protocol import EvidenceResult, ResearchService, SearchQuery
from carepath.research.qdrant import QdrantResearchService
from carepath.research.searxng import SearxNGResearchService
from carepath.research.synthetic import SyntheticResearchService

__all__ = [
    "CompositeResearchService",
    "DeterministicEmbeddingService",
    "EmbeddingService",
    "EvidenceResult",
    "NVIDIAEmbeddingService",
    "QdrantResearchService",
    "ResearchService",
    "SearchQuery",
    "SearxNGResearchService",
    "SyntheticResearchService",
    "get_embedding_service",
    "get_research_service",
]
