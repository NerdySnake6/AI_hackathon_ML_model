"""Tests for offline metrics and error categorization."""

from __future__ import annotations

import unittest

from app.evaluation.metrics import (
    EvaluatedExample,
    ExpectedLabel,
    build_error_rows,
    compute_metrics,
)
from app.schemas import Decision, DomainLabel, LabelResult


class EvaluationMetricsTestCase(unittest.TestCase):
    """Validate metric calculations on small controlled examples."""

    def test_compute_metrics_counts_video_and_auto_accept_quality(self) -> None:
        """Compute binary, end-to-end, and auto-accept metrics."""
        examples = [
            make_example(
                expected_video=True,
                expected_content_type="film",
                expected_title_id="title_1",
                predicted_video=True,
                predicted_content_type="film",
                predicted_title_id="title_1",
                decision=Decision.AUTO_ACCEPT,
            ),
            make_example(
                expected_video=False,
                expected_content_type=None,
                expected_title_id=None,
                predicted_video=False,
                predicted_content_type=None,
                predicted_title_id=None,
                decision=Decision.NON_VIDEO,
            ),
            make_example(
                expected_video=True,
                expected_content_type="series",
                expected_title_id="title_2",
                predicted_video=True,
                predicted_content_type="series",
                predicted_title_id="wrong_title",
                decision=Decision.AUTO_ACCEPT,
            ),
        ]

        metrics = compute_metrics(examples)

        self.assertEqual(metrics["video"]["tp"], 2)
        self.assertEqual(metrics["video"]["tn"], 1)
        self.assertEqual(metrics["title_id_accuracy"], 0.5)
        self.assertEqual(metrics["end_to_end_accuracy"], 0.6667)
        self.assertEqual(metrics["auto_accept"]["precision"], 0.5)

    def test_build_error_rows_labels_wrong_title(self) -> None:
        """Return error rows with compact mismatch categories."""
        examples = [
            make_example(
                expected_video=True,
                expected_content_type="series",
                expected_title_id="title_2",
                predicted_video=True,
                predicted_content_type="series",
                predicted_title_id="wrong_title",
                decision=Decision.REVIEW,
            )
        ]

        rows = build_error_rows(examples)

        self.assertEqual(rows[0]["error_type"], "wrong_title")
        self.assertEqual(rows[0]["expected_title_id"], "title_2")
        self.assertEqual(rows[0]["predicted_title_id"], "wrong_title")


def make_example(
    expected_video: bool,
    expected_content_type: str | None,
    expected_title_id: str | None,
    predicted_video: bool,
    predicted_content_type: str | None,
    predicted_title_id: str | None,
    decision: Decision,
) -> EvaluatedExample:
    """Build a minimal evaluated example for metrics tests."""
    expected = ExpectedLabel(
        query_id="query",
        query_text="query",
        is_prof_video=expected_video,
        content_type=expected_content_type,
        title_id=expected_title_id,
    )
    predicted = LabelResult(
        query_id="query",
        query="query",
        normalized_query="query",
        is_prof_video=predicted_video,
        domain_label=DomainLabel.PROF_VIDEO
        if predicted_video
        else DomainLabel.NON_VIDEO,
        content_type=predicted_content_type,
        title_id=predicted_title_id,
        title=predicted_title_id,
        confidence=1.0,
        decision=decision,
    )
    return EvaluatedExample(expected=expected, predicted=predicted)


if __name__ == "__main__":
    unittest.main()

