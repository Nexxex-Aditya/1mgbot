"""Qdrant research service — semantic vector retrieval over indexed medical evidence.

Gracefully degrades when Qdrant is disabled or unavailable.
Supports both local/remote Qdrant servers and in-memory instances for testing.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from qdrant_client import AsyncQdrantClient, models

from carepath.research.embeddings import DeterministicEmbeddingService, EmbeddingService
from carepath.research.protocol import EvidenceResult, SearchQuery

logger = logging.getLogger(__name__)


class QdrantResearchService:
    """Semantic evidence retrieval backend using Qdrant vector database."""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str = "carepath_evidence",
        embedding_service: EmbeddingService | None = None,
        client: AsyncQdrantClient | None = None,
        in_memory: bool = False,
        timeout: float = 5.0,
    ) -> None:
        self._collection_name = collection_name
        self._embedding_service = embedding_service or DeterministicEmbeddingService()
        self._timeout = timeout
        self._collection_initialized = False

        if client is not None:
            self._client = client
            self._available = True
        elif in_memory:
            self._client = AsyncQdrantClient(":memory:")
            self._available = True
        elif url:
            self._client = AsyncQdrantClient(
                url=url,
                api_key=api_key,
                timeout=timeout,
                check_compatibility=False,
            )
            self._available = True
        else:
            self._client = None
            self._available = False

    @property
    def name(self) -> str:
        return "qdrant"

    @property
    def available(self) -> bool:
        return self._available and self._client is not None

    async def ensure_collection(self) -> bool:
        """Create the collection if it does not already exist."""
        if not self.available or self._client is None:
            return False

        try:
            exists = await self._client.collection_exists(self._collection_name)
            if not exists:
                await self._client.create_collection(
                    collection_name=self._collection_name,
                    vectors_config=models.VectorParams(
                        size=self._embedding_service.dimension,
                        distance=models.Distance.COSINE,
                    ),
                )
            self._collection_initialized = True
            return True
        except Exception as exc:
            logger.warning("Failed to initialize Qdrant collection", extra={"error": str(exc)})
            self._available = False
            return False

    async def index_documents(self, documents: list[dict[str, Any]]) -> int:
        """Index a batch of documents/passages into the collection.

        Each doc should have:
        - text / snippet (str)
        - title (str, optional)
        - source_url (str, optional)
        - source_type (str, optional, defaults to 'vector_store')
        - metadata (dict, optional)
        """
        if not documents or not self.available or self._client is None:
            return 0

        if not self._collection_initialized:
            ok = await self.ensure_collection()
            if not ok:
                return 0

        texts = [doc.get("snippet") or doc.get("text", "") for doc in documents]
        vectors = await self._embedding_service.embed_documents(texts)

        points: list[models.PointStruct] = []
        for i, (doc, vec) in enumerate(zip(documents, vectors, strict=True)):
            raw_id = doc.get("id")
            if isinstance(raw_id, int):
                point_id: str | int = raw_id
            elif isinstance(raw_id, str):
                try:
                    point_id = str(uuid.UUID(raw_id))
                except ValueError:
                    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, raw_id))
            else:
                point_id = str(uuid.uuid4())

            payload = {
                "title": doc.get("title", f"Document {i + 1}"),
                "snippet": doc.get("snippet") or doc.get("text", ""),
                "source_url": doc.get("source_url"),
                "source_type": doc.get("source_type", "vector_store"),
                "metadata": doc.get("metadata", {}),
            }
            points.append(models.PointStruct(id=point_id, vector=vec, payload=payload))

        try:
            await self._client.upsert(
                collection_name=self._collection_name,
                points=points,
            )
            return len(points)
        except Exception as exc:
            logger.warning("Qdrant upsert failed", extra={"error": str(exc)})
            return 0

    async def search(self, query: SearchQuery) -> list[EvidenceResult]:
        if not self.available or self._client is None:
            return []

        try:
            # Check if collection exists before querying
            exists = await self._client.collection_exists(self._collection_name)
            if not exists:
                return []

            query_vec = await self._embedding_service.embed_query(query.text)
            response = await self._client.query_points(
                collection_name=self._collection_name,
                query=query_vec,
                limit=query.max_results,
            )

            results: list[EvidenceResult] = []
            for hit in response.points:
                payload = hit.payload or {}
                # Cosine distance in Qdrant produces scores in [-1.0, 1.0], normalize to [0.0, 1.0]
                raw_score = (hit.score + 1.0) / 2.0 if hit.score is not None else 0.5
                normalized_score = max(0.0, min(1.0, raw_score))
                results.append(
                    EvidenceResult(
                        title=payload.get("title", "Clinical Evidence"),
                        snippet=payload.get("snippet", ""),
                        source_url=payload.get("source_url"),
                        source_type=payload.get("source_type", "vector_store"),
                        relevance_score=normalized_score,
                        metadata=payload.get("metadata", {}),
                    )
                )

            return results
        except Exception as exc:
            logger.warning(
                "Qdrant search failed, gracefully degrading", extra={"error": str(exc)}
            )
            self._available = False
            return []
