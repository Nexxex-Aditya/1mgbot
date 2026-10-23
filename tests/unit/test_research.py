"""Unit tests for Phase 4: Evidence Research Services (SearxNG, Qdrant, Embeddings, Composite)."""

from __future__ import annotations

import math
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from carepath.models.gateway import ModelGateway
from carepath.orchestrator.graph import run_care_graph
from carepath.orchestrator.state import CarePhase
from carepath.research.composite import CompositeResearchService
from carepath.research.embeddings import DeterministicEmbeddingService
from carepath.research.protocol import EvidenceResult, SearchQuery
from carepath.research.qdrant import QdrantResearchService
from carepath.research.searxng import SearxNGResearchService
from carepath.research.synthetic import SyntheticResearchService
from tests.fakes import FakeModelProvider


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


class TestEmbeddingService:
    @pytest.mark.asyncio
    async def test_deterministic_embedding_dimension_and_norm(self) -> None:
        svc = DeterministicEmbeddingService(dimension=128)
        assert svc.dimension == 128

        vec = await svc.embed_query("Patient has acute chest pain and shortness of breath")
        assert len(vec) == 128
        norm = math.sqrt(sum(v * v for v in vec))
        assert abs(norm - 1.0) < 1e-4

    @pytest.mark.asyncio
    async def test_deterministic_embedding_repeatability(self) -> None:
        svc = DeterministicEmbeddingService(dimension=64)
        vec1 = await svc.embed_query("fever and chills")
        vec2 = await svc.embed_query("fever and chills")
        assert vec1 == vec2

    @pytest.mark.asyncio
    async def test_deterministic_embedding_batch(self) -> None:
        svc = DeterministicEmbeddingService(dimension=64)
        texts = ["headache", "migraine symptoms", "unrelated text"]
        vecs = await svc.embed_documents(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 64

    @pytest.mark.asyncio
    async def test_deterministic_embedding_semantic_similarity(self) -> None:
        svc = DeterministicEmbeddingService(dimension=256)
        query = await svc.embed_query("severe pediatric fever and cough")
        doc_relevant = await svc.embed_query("Management of pediatric fever and acute cough")
        doc_irrelevant = await svc.embed_query("Fracture of the femur in elderly patients")

        sim_rel = _cosine_similarity(query, doc_relevant)
        sim_irrel = _cosine_similarity(query, doc_irrelevant)
        assert sim_rel > sim_irrel


class TestSyntheticResearchService:
    @pytest.mark.asyncio
    async def test_synthetic_service_returns_labeled_evidence(self) -> None:
        svc = SyntheticResearchService()
        assert svc.name == "synthetic"
        assert svc.available is True

        query = SearchQuery(
            text="headache and stiff neck",
            specialties=["Neurology"],
            symptoms=["headache", "stiff neck"],
            max_results=3,
        )
        results = await svc.search(query)
        assert len(results) <= 3
        assert len(results) >= 1
        for item in results:
            assert item.source_type == "synthetic"
            assert "synthetic" in item.snippet.lower() or "synthetic" in item.title.lower()


class TestSearxNGResearchService:
    def test_unconfigured_searxng_is_unavailable(self) -> None:
        svc = SearxNGResearchService(base_url=None)
        assert svc.available is False
        assert svc.name == "searxng"

    @pytest.mark.asyncio
    async def test_unconfigured_searxng_returns_empty(self) -> None:
        svc = SearxNGResearchService(base_url="")
        results = await svc.search(SearchQuery(text="flu symptoms"))
        assert results == []

    @pytest.mark.asyncio
    async def test_searxng_success_response(self) -> None:
        svc = SearxNGResearchService(base_url="http://mock-searxng:8080")

        mock_payload = {
            "results": [
                {
                    "title": "Clinical Guideline for Asthma",
                    "content": "Asthma presentation involves wheezing and dyspnea.",
                    "url": "https://example.org/guidelines/asthma",
                    "score": 0.85,
                    "engine": "google",
                },
                {
                    "title": "Pediatric Asthma Review",
                    "content": "Early management guidelines.",
                    "url": "https://example.org/pediatric-asthma",
                    "score": 0.75,
                    "engine": "pubmed",
                },
            ]
        }

        mock_resp = httpx.Response(status_code=200, json=mock_payload, request=httpx.Request("GET", "http://test"))
        with patch("httpx.AsyncClient.get", new=AsyncMock(return_value=mock_resp)):
            results = await svc.search(SearchQuery(text="asthma wheezing", max_results=2))

        assert len(results) == 2
        assert results[0].title == "Clinical Guideline for Asthma"
        assert results[0].source_type == "web_search"
        assert results[0].source_url == "https://example.org/guidelines/asthma"
        assert results[0].metadata.get("engine") == "google"

    @pytest.mark.asyncio
    async def test_searxng_network_error_degrades_gracefully(self) -> None:
        svc = SearxNGResearchService(base_url="http://nonexistent-searxng.internal")
        with patch("httpx.AsyncClient.get", side_effect=httpx.ConnectError("Connection refused")):
            results = await svc.search(SearchQuery(text="test query"))

        assert results == []
        assert svc.available is False


class TestQdrantResearchService:
    @pytest.mark.asyncio
    async def test_in_memory_indexing_and_search(self) -> None:
        embedding_svc = DeterministicEmbeddingService(dimension=64)
        svc = QdrantResearchService(
            in_memory=True,
            collection_name="test_evidence",
            embedding_service=embedding_svc,
        )
        assert svc.available is True
        assert svc.name == "qdrant"

        # Index synthetic clinical documents
        docs = [
            {
                "id": "doc-cardio-1",
                "title": "Angina Pectoris Assessment Guideline",
                "snippet": (
                    "Substernal pressure exacerbated by exertion, radiating to jaw or left arm."
                ),
                "source_url": "https://evidence.carepath.internal/cardio/angina",
                "source_type": "vector_store",
                "metadata": {"specialty": "Cardiology", "version": "1.0"},
            },
            {
                "id": "doc-derm-1",
                "title": "Atopic Dermatitis Management",
                "snippet": "Pruritic erythematous lesions occurring on flexural surfaces.",
                "source_url": "https://evidence.carepath.internal/derm/eczema",
                "source_type": "vector_store",
                "metadata": {"specialty": "Dermatology", "version": "1.0"},
            },
        ]

        indexed_count = await svc.index_documents(docs)
        assert indexed_count == 2

        # Search for cardiac presentation
        query = SearchQuery(
            text="substernal chest pressure and pain radiating to arm",
            max_results=2,
        )
        results = await svc.search(query)
        assert len(results) >= 1
        top_match = results[0]
        assert top_match.source_type == "vector_store"
        assert "Angina" in top_match.title
        assert top_match.relevance_score > 0.0

    @pytest.mark.asyncio
    async def test_unconfigured_qdrant_is_unavailable(self) -> None:
        svc = QdrantResearchService(url=None)
        assert svc.available is False
        results = await svc.search(SearchQuery(text="test query"))
        assert results == []


class TestCompositeResearchService:
    @pytest.mark.asyncio
    async def test_composite_prioritizes_real_backends(self) -> None:
        fake_result = EvidenceResult(
            title="Real Evidence Document",
            snippet="Actual medical passage.",
            source_url="https://evidence.internal/doc",
            source_type="vector_store",
            relevance_score=0.9,
        )

        mock_backend = AsyncMock()
        mock_backend.available = True
        mock_backend.search = AsyncMock(return_value=[fake_result])

        composite = CompositeResearchService(
            backends=[mock_backend],
            fallback_service=SyntheticResearchService(),
        )

        results = await composite.search(SearchQuery(text="test query", max_results=5))
        assert len(results) == 1
        assert results[0].title == "Real Evidence Document"
        assert results[0].source_type == "vector_store"

    @pytest.mark.asyncio
    async def test_composite_falls_back_to_synthetic_when_empty(self) -> None:
        empty_backend = AsyncMock()
        empty_backend.available = True
        empty_backend.search = AsyncMock(return_value=[])

        composite = CompositeResearchService(
            backends=[empty_backend],
            fallback_service=SyntheticResearchService(),
        )

        results = await composite.search(SearchQuery(text="headache and nausea", max_results=3))
        assert len(results) >= 1
        assert results[0].source_type == "synthetic"

    @pytest.mark.asyncio
    async def test_composite_deduplication(self) -> None:
        dupe1 = EvidenceResult(
            title="Common Cold Care",
            snippet="Rest and hydration.",
            source_url="https://example.org/cold",
            source_type="web_search",
            relevance_score=0.7,
        )
        dupe2 = EvidenceResult(
            title="Common Cold Care",
            snippet="Rest and hydration.",
            source_url="https://example.org/cold",
            source_type="vector_store",
            relevance_score=0.8,
        )

        backend = AsyncMock()
        backend.available = True
        backend.search = AsyncMock(return_value=[dupe1, dupe2])

        composite = CompositeResearchService(backends=[backend])
        results = await composite.search(SearchQuery(text="cold symptoms"))
        assert len(results) == 1


class TestOrchestratorWithResearchService:
    @pytest.mark.asyncio
    async def test_orchestrator_receives_vector_evidence(self) -> None:
        embedding_svc = DeterministicEmbeddingService(dimension=64)
        qdrant_svc = QdrantResearchService(
            in_memory=True,
            collection_name="test_orchestrator_evidence",
            embedding_service=embedding_svc,
        )

        await qdrant_svc.index_documents([
            {
                "id": "evidence-migraine-1",
                "title": "Migraine Treatment Overview",
                "snippet": "First-line therapies include hydration and dark resting environment.",
                "source_url": "https://clinical.internal/neurology/migraine",
                "source_type": "vector_store",
            }
        ])

        composite = CompositeResearchService(
            backends=[qdrant_svc],
            fallback_service=SyntheticResearchService(),
        )

        triage_json = (
            '{"urgency": "standard", "chief_complaint": "throbbing headache", '
            '"symptoms": ["headache"], "suggested_specialties": ["Neurology"], '
            '"confidence": 0.8, "reasoning": "Classic migraine presentation.", '
            '"safety_flags": []}'
        )
        provider = FakeModelProvider(canned_responses=[
            triage_json,
            "Based on symptoms and guidelines, you may be experiencing a migraine.",
        ])
        gateway = ModelGateway(
            provider=provider,
            default_model="test/fake",
            max_retries=1,
            max_concurrency=2,
        )

        result = await run_care_graph(
            gateway,
            session_id="test-research-session",
            user_input="I have a throbbing unilateral headache with light sensitivity",
            trace_id="test-trace-research",
            research_service=composite,
        )

        assert result["phase"] == CarePhase.AWAITING_INPUT
        assert len(result["evidence"]) >= 1
        # Vector evidence should be present
        source_types = [e["source_type"] for e in result["evidence"]]
        assert "vector_store" in source_types
        # Safety disclaimer should still be attached deterministically
        assert "⚠️ *Research prototype — synthetic data only." in result["pending_response"]
