"""Synthetic research service — always-available deterministic evidence for demo mode.

All results are clearly labelled as synthetic. This is the fallback when
SearxNG and Qdrant are not configured.
"""

from __future__ import annotations

from carepath.research.protocol import EvidenceResult, SearchQuery


class SyntheticResearchService:
    """Deterministic synthetic evidence for demo/test mode."""

    @property
    def name(self) -> str:
        return "synthetic"

    @property
    def available(self) -> bool:
        return True

    async def search(self, query: SearchQuery) -> list[EvidenceResult]:
        results: list[EvidenceResult] = []

        # General guidance based on query
        results.append(EvidenceResult(
            title=f"General guidance: {query.text[:80]}",
            snippet=(
                "⚠️ SYNTHETIC EVIDENCE — This is a research prototype. "
                "In the full system, this would be replaced by curated medical literature. "
                "Always consult a qualified healthcare provider."
            ),
            source_type="synthetic",
            relevance_score=0.5,
        ))

        # Specialty-specific synthetic evidence
        for specialty in query.specialties[:3]:
            results.append(EvidenceResult(
                title=f"Specialty: {specialty}",
                snippet=(
                    f"Patients with these symptoms may benefit from consulting a {specialty} "
                    "specialist. This is synthetic guidance for the research prototype."
                ),
                source_type="synthetic",
                relevance_score=0.4,
                metadata={"specialty": specialty},
            ))

        # Symptom-specific synthetic evidence
        for symptom in query.symptoms[:3]:
            results.append(EvidenceResult(
                title=f"Symptom information: {symptom}",
                snippet=(
                    f"'{symptom}' is a common presentation. Severity, duration, and associated "
                    "symptoms help determine the appropriate level of care. "
                    "This is synthetic evidence."
                ),
                source_type="synthetic",
                relevance_score=0.3,
                metadata={"symptom": symptom},
            ))

        return results[:query.max_results]
