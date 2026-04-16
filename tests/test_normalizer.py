"""Tests for typo-tolerant query normalization."""

from __future__ import annotations

import unittest

from app.preprocessing.normalizer import (
    build_normalized_query,
    fix_keyboard_layout,
    normalize_text,
)


class NormalizerTestCase(unittest.TestCase):
    """Validate text cleanup and query variant generation."""

    def test_normalize_text_handles_tabs_newlines_and_yo(self) -> None:
        """Normalize whitespace, punctuation, and Russian ё variants."""
        result = normalize_text("  Смотреть\tЁлки\nонлайн!!!  ")

        self.assertEqual(result, "смотреть елки онлайн")

    def test_clean_query_removes_video_intent_words(self) -> None:
        """Keep likely title words while removing frequent intent tokens."""
        result = build_normalized_query("Смотреть Интерстелар онлайн HD")

        self.assertEqual(result.clean_text, "интерстелар")
        self.assertIn("интерстелар", result.variants)

    def test_keyboard_layout_variant_is_generated(self) -> None:
        """Include a Russian variant for text typed with English layout."""
        self.assertEqual(fix_keyboard_layout("bynthcntkkfh"), "интерстеллар")

        result = build_normalized_query("bynthcntkkfh")

        self.assertIn("интерстеллар", result.variants)


if __name__ == "__main__":
    unittest.main()

