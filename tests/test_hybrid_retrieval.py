"""Tests for hybrid lexical-plus-vector title retrieval."""

from __future__ import annotations

import unittest

from app.catalog.repository import InMemoryCatalogRepository
from app.embeddings.encoder import HashingTextEncoder, cosine_similarity
from app.preprocessing.normalizer import build_normalized_query
from app.retrieval.hybrid import HybridCandidateRetriever
from app.schemas import CatalogTitle, ContentType


class HybridRetrievalTestCase(unittest.TestCase):
    """Validate the lightweight vector retrieval layer."""

    def setUp(self) -> None:
        """Create a compact catalog for retrieval checks."""
        self.catalog = InMemoryCatalogRepository(
            [
                CatalogTitle(
                    title_id="title_house_of_dragon",
                    canonical_title="Дом дракона",
                    content_type=ContentType.SERIES,
                    aliases=["house of the dragon"],
                    popularity=0.8,
                ),
                CatalogTitle(
                    title_id="title_interstellar",
                    canonical_title="Интерстеллар",
                    content_type=ContentType.FILM,
                    aliases=["interstellar", "интерстелар"],
                    popularity=0.9,
                ),
            ]
        )
        self.retriever = HybridCandidateRetriever.from_catalog(self.catalog)

    def test_encoder_keeps_reordered_title_variants_close(self) -> None:
        """Produce similar vectors for token-reordered title mentions."""
        encoder = HashingTextEncoder()

        direct = encoder.encode("дом дракона")
        reordered = encoder.encode("дракона дом")

        self.assertGreater(cosine_similarity(direct, reordered), 0.6)

    def test_hybrid_retrieval_adds_vector_signal(self) -> None:
        """Support lexical search with vector evidence for reordered input."""
        query = build_normalized_query("дракона дом сериал")
        lexical_candidates = self.catalog.search(query)

        candidates = self.retriever.search(
            query,
            lexical_candidates=lexical_candidates,
        )

        self.assertEqual(candidates[0].title_id, "title_house_of_dragon")
        self.assertIn("vector_similarity_support", candidates[0].reasons)


if __name__ == "__main__":
    unittest.main()
