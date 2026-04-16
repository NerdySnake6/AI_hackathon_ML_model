"""End-to-end tests for the query labeling pipeline."""

from __future__ import annotations

import unittest

from app.pipeline import QueryLabelingPipeline
from app.schemas import Decision


class PipelineTestCase(unittest.TestCase):
    """Validate high-value business scenarios in the MVP pipeline."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the sample catalog once for all pipeline tests."""
        cls.pipeline = QueryLabelingPipeline.from_csv("data/sample_catalog.csv")

    def test_labels_specific_series_query(self) -> None:
        """Resolve a concrete series query to a catalog title."""
        result = self.pipeline.label("1 сезон тьмы")

        self.assertTrue(result.is_prof_video)
        self.assertEqual(result.title_id, "title_001")
        self.assertEqual(result.content_type, "series")
        self.assertEqual(result.decision, Decision.AUTO_ACCEPT)

    def test_rejects_transport_query(self) -> None:
        """Do not force a non-video transport query into a catalog title."""
        result = self.pipeline.label("10 троллейбус ижевск")

        self.assertFalse(result.is_prof_video)
        self.assertEqual(result.decision, Decision.NON_VIDEO)
        self.assertIsNone(result.title_id)

    def test_detects_generic_video_query(self) -> None:
        """Classify a broad video request without inventing a concrete title."""
        result = self.pipeline.label("смотреть фильмы 2025 года онлайн")

        self.assertTrue(result.is_prof_video)
        self.assertEqual(result.content_type, "film")
        self.assertIsNone(result.title_id)
        self.assertEqual(result.decision, Decision.GENERIC_VIDEO)

    def test_rejects_hard_negative_even_with_title_candidate(self) -> None:
        """Prefer hard-negative context over weak catalog matching."""
        result = self.pipeline.label("кухня ремонт")

        self.assertFalse(result.is_prof_video)
        self.assertEqual(result.decision, Decision.NON_VIDEO)
        self.assertIsNone(result.title_id)

    def test_labels_title_without_video_intent_when_catalog_has_it(self) -> None:
        """Resolve a known title even when the query has no video intent words."""
        result = self.pipeline.label("сумерки")

        self.assertTrue(result.is_prof_video)
        self.assertEqual(result.content_type, "film")
        self.assertEqual(result.title_id, "title_009")

    def test_unknown_title_like_query_goes_to_manual_review(self) -> None:
        """Do not treat every out-of-catalog title-like query as non-video."""
        result = self.pipeline.label("пылающий берег")

        self.assertFalse(result.is_prof_video)
        self.assertEqual(result.decision, Decision.MANUAL_REQUIRED)
        self.assertIn("unknown_title_candidate", result.reasons)


if __name__ == "__main__":
    unittest.main()
