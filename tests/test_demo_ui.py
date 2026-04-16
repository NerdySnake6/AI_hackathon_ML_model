"""Tests for the browser demo page."""

from __future__ import annotations

from pathlib import Path
import unittest


class DemoPageTestCase(unittest.TestCase):
    """Validate that the demo page exposes the expected UI hooks."""

    def test_demo_page_contains_labeling_controls(self) -> None:
        """Render the browser UI with single, batch, and review controls."""
        html = Path("app/static/index.html").read_text(encoding="utf-8")

        self.assertIn("Query Labeler", html)
        self.assertIn("/static/app.js", html)
        self.assertIn("data-view=\"review\"", html)

    def test_demo_script_calls_labeling_api(self) -> None:
        """Keep the static UI wired to the FastAPI labeling endpoints."""
        script = Path("app/static/app.js").read_text(encoding="utf-8")

        self.assertIn("fetch(\"/label\"", script)
        self.assertIn("fetch(\"/label_batch\"", script)


if __name__ == "__main__":
    unittest.main()
