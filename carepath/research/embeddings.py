"""Embedding generation services for evidence indexing and retrieval.

Provides:
- EmbeddingService: Protocol defining query and document embedding.
- DeterministicEmbeddingService: Fast, reproducible, offline embedding for tests and demo mode.
- NVIDIAEmbeddingService: Live embedding generation via NVIDIA API (when configured).
- get_embedding_service: Factory returning the appropriate service based on settings.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from typing import Protocol

import httpx

from carepath.config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

DEFAULT_EMBEDDING_DIM = 256
DEFAULT_NVIDIA_EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"


class EmbeddingService(Protocol):
    """Protocol for vector embedding generation."""

    @property
    def dimension(self) -> int:
        """Vector dimensionality."""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query text."""
        ...

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of document texts."""
        ...


class DeterministicEmbeddingService:
    """Offline, deterministic embedding generator using feature hashing and L2 normalization.

    Produces consistent unit-length vectors with non-zero similarity for overlapping terms,
    allowing full vector index validation without external API keys or heavy dependencies.
    """

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIM) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def _embed_text(self, text: str) -> list[float]:
        if not text.strip():
            return [0.0] * self._dim

        vec = [0.0] * self._dim
        words = re.findall(r"\w+", text.lower())

        if not words:
            return [0.0] * self._dim

        for word in words:
            # Deterministic hash to bucket and sign
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 1.0 if ((h >> 8) & 1) == 1 else -1.0
            vec[idx] += sign

        # Add character bi-grams for subword matching
        for i in range(len(text) - 1):
            bigram = text[i : i + 2].lower()
            h = int(hashlib.sha256(bigram.encode("utf-8")).hexdigest(), 16)
            idx = h % self._dim
            sign = 0.5 if ((h >> 4) & 1) == 1 else -0.5
            vec[idx] += sign

        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0.0:
            vec = [v / norm for v in vec]
        return vec

    async def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(t) for t in texts]


class NVIDIAEmbeddingService:
    """Live embedding generation via NVIDIA API endpoints."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        model: str = DEFAULT_NVIDIA_EMBEDDING_MODEL,
        dimension: int = 1024,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._dimension = dimension
        self._timeout = timeout

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed_query(self, text: str) -> list[float]:
        results = await self.embed_documents([text])
        return results[0] if results else [0.0] * self._dimension

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/embeddings",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "input": texts,
                        "model": self._model,
                        "input_type": "passage",
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                embeddings = [item["embedding"] for item in data.get("data", [])]
                if embeddings and len(embeddings[0]) != self._dimension:
                    self._dimension = len(embeddings[0])
                return embeddings
        except Exception as exc:
            logger.warning(
                "NVIDIA embedding call failed, falling back to deterministic",
                extra={"error": str(exc)},
            )
            fallback = DeterministicEmbeddingService(dimension=self._dimension)
            return await fallback.embed_documents(texts)


def get_embedding_service(settings: Settings | None = None) -> EmbeddingService:
    """Return configured embedding service (NVIDIA if configured, else Deterministic)."""
    cfg = settings or get_settings()

    if cfg.nvidia_api_key and cfg.embedding_provider == "nvidia":
        return NVIDIAEmbeddingService(
            api_key=cfg.nvidia_api_key.get_secret_value(),
            base_url=cfg.nvidia_base_url,
            model=cfg.embedding_model or DEFAULT_NVIDIA_EMBEDDING_MODEL,
        )

    return DeterministicEmbeddingService()
