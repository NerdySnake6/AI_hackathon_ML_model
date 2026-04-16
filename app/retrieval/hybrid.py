"""Hybrid candidate retrieval that reranks lexical matches with vector signals."""

from __future__ import annotations

from typing import Any

from app.catalog.repository import InMemoryCatalogRepository
from app.embeddings.encoder import HashingTextEncoder, cosine_similarity
from app.schemas import NormalizedQuery, TitleCandidate
from app.settings import (
    HYBRID_LEXICAL_WEIGHT,
    HYBRID_VECTOR_THRESHOLD,
    HYBRID_VECTOR_WEIGHT,
    MAX_CANDIDATES,
    VECTOR_EMBEDDING_DIMENSION,
)


class HybridCandidateRetriever:
    """Blend lexical candidates with lightweight vector similarity support."""

    def __init__(
        self,
        encoder: HashingTextEncoder | None = None,
    ) -> None:
        """Initialize the encoder used for candidate reranking."""
        self.encoder = encoder or HashingTextEncoder(
            dimension=VECTOR_EMBEDDING_DIMENSION,
        )

    @classmethod
    def from_catalog(
        cls,
        catalog: InMemoryCatalogRepository,
        encoder: HashingTextEncoder | None = None,
    ) -> "HybridCandidateRetriever":
        """Build a retriever from an already loaded in-memory catalog."""
        _ = catalog
        return cls(encoder=encoder)

    def search(
        self,
        query: NormalizedQuery,
        lexical_candidates: list[TitleCandidate] | None = None,
        limit: int = MAX_CANDIDATES,
        min_vector_score: float = HYBRID_VECTOR_THRESHOLD,
    ) -> list[TitleCandidate]:
        """Rerank lexical candidates using vector similarity on a tiny pool."""
        if not lexical_candidates:
            return []

        query_vectors = [self.encoder.encode(variant) for variant in query.variants]
        reranked_candidates: list[TitleCandidate] = []

        for candidate in lexical_candidates[:limit]:
            reranked_candidates.append(
                self._rerank_candidate(
                    candidate=candidate,
                    query_vectors=query_vectors,
                    min_vector_score=min_vector_score,
                )
            )

        return sorted(
            reranked_candidates,
            key=lambda item: item.search_score,
            reverse=True,
        )[:limit]

    def _rerank_candidate(
        self,
        candidate: TitleCandidate,
        query_vectors: list[tuple[float, ...]],
        min_vector_score: float,
    ) -> TitleCandidate:
        """Blend one lexical candidate with vector evidence."""
        texts = [candidate.matched_alias, candidate.canonical_title]
        if candidate.year:
            texts.append(f"{candidate.canonical_title} {candidate.year}")

        best_vector_score = 0.0
        for text in texts:
            candidate_vector = self.encoder.encode(text)
            for query_vector in query_vectors:
                best_vector_score = max(
                    best_vector_score,
                    cosine_similarity(query_vector, candidate_vector),
                )

        if best_vector_score < min_vector_score:
            return candidate.model_copy(deep=True)

        reasons = list(candidate.reasons)
        if "vector_similarity_support" not in reasons:
            reasons.append("vector_similarity_support")
        blended_score = max(
            candidate.search_score,
            candidate.search_score * HYBRID_LEXICAL_WEIGHT
            + best_vector_score * HYBRID_VECTOR_WEIGHT,
        )

        payload: dict[str, Any] = candidate.model_dump()
        payload.update(
            search_score=round(min(blended_score, 1.0), 4),
            rank_score=round(min(candidate.rank_score, 1.0), 4),
            reasons=reasons,
        )
        return TitleCandidate(**payload)
