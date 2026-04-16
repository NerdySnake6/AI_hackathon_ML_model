"""Tests for in-memory catalog search and typo-tolerant title matching."""

from __future__ import annotations

import unittest

from app.catalog.repository import InMemoryCatalogRepository
from app.preprocessing.normalizer import build_normalized_query
from app.schemas import CatalogTitle, ContentType


class CatalogSearchTestCase(unittest.TestCase):
    """Validate catalog candidate generation for noisy user queries."""

    def setUp(self) -> None:
        """Create a small catalog for search tests."""
        self.repository = InMemoryCatalogRepository(
            [
                CatalogTitle(
                    title_id="title_interstellar",
                    canonical_title="Интерстеллар",
                    content_type=ContentType.FILM,
                    year=2014,
                    aliases=["interstellar", "интерстелар"],
                ),
                CatalogTitle(
                    title_id="title_masha",
                    canonical_title="Маша и Медведь",
                    content_type=ContentType.ANIMATED_SERIES,
                    aliases=["маша медведь", "машаимедведь"],
                ),
            ]
        )

    def test_finds_title_with_typo_alias(self) -> None:
        """Find a title by a misspelled alias."""
        candidates = self.repository.search(
            build_normalized_query("интерстелар смотреть онлайн")
        )

        self.assertEqual(candidates[0].title_id, "title_interstellar")

    def test_finds_title_typed_with_wrong_keyboard_layout(self) -> None:
        """Find a title from a generated keyboard-layout variant."""
        candidates = self.repository.search(build_normalized_query("bynthcntkkfh"))

        self.assertEqual(candidates[0].title_id, "title_interstellar")
        self.assertEqual(candidates[0].matched_variant, "интерстеллар")

    def test_finds_compact_childrens_title(self) -> None:
        """Find a title when words are accidentally glued together."""
        candidates = self.repository.search(
            build_normalized_query("машаи медведь новые серии")
        )

        self.assertEqual(candidates[0].title_id, "title_masha")

    def test_finds_reordered_title_tokens(self) -> None:
        """Find a title when the user swaps token order in the query."""
        repository = InMemoryCatalogRepository(
            [
                CatalogTitle(
                    title_id="title_house_of_dragon",
                    canonical_title="Дом дракона",
                    content_type=ContentType.SERIES,
                    aliases=["house of the dragon"],
                )
            ]
        )

        candidates = repository.search(build_normalized_query("дракона дом сериал"))

        self.assertEqual(candidates[0].title_id, "title_house_of_dragon")


if __name__ == "__main__":
    unittest.main()
