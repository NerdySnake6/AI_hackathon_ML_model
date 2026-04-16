"""Metrics and error analysis for the query labeling pipeline."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from app.schemas import Decision, LabelResult


@dataclass(frozen=True)
class ExpectedLabel:
    """Human-approved target label for one evaluation query."""

    query_id: str | None
    query_text: str
    is_prof_video: bool
    content_type: str | None
    title_id: str | None
    decision: str | None = None


@dataclass(frozen=True)
class EvaluatedExample:
    """Pair of an expected label and the pipeline prediction."""

    expected: ExpectedLabel
    predicted: LabelResult


def compute_metrics(examples: list[EvaluatedExample]) -> dict[str, object]:
    """Compute classification, linking, routing, and business metrics."""
    total = len(examples)
    if total == 0:
        return {
            "total": 0,
            "error": "no_examples",
        }

    video_counts = compute_binary_counts(examples)
    video_precision = safe_divide(video_counts["tp"], video_counts["tp"] + video_counts["fp"])
    video_recall = safe_divide(video_counts["tp"], video_counts["tp"] + video_counts["fn"])
    video_f1 = safe_divide(
        2 * video_precision * video_recall,
        video_precision + video_recall,
    )

    content_examples = [
        example
        for example in examples
        if example.expected.is_prof_video and example.expected.content_type
    ]
    title_examples = [
        example
        for example in examples
        if example.expected.is_prof_video and example.expected.title_id
    ]
    decision_examples = [
        example for example in examples if example.expected.decision is not None
    ]
    auto_accept_examples = [
        example
        for example in examples
        if example.predicted.decision == Decision.AUTO_ACCEPT
    ]
    end_to_end_correct = [
        example for example in examples if is_end_to_end_correct(example)
    ]

    metrics: dict[str, object] = {
        "total": total,
        "video": {
            **video_counts,
            "precision": round(video_precision, 4),
            "recall": round(video_recall, 4),
            "f1": round(video_f1, 4),
            "accuracy": round(
                safe_divide(video_counts["tp"] + video_counts["tn"], total),
                4,
            ),
        },
        "content_type_accuracy": round(
            safe_divide(
                sum(
                    normalize_label(example.predicted.content_type)
                    == normalize_label(example.expected.content_type)
                    for example in content_examples
                ),
                len(content_examples),
            ),
            4,
        ),
        "title_id_accuracy": round(
            safe_divide(
                sum(
                    normalize_label(example.predicted.title_id)
                    == normalize_label(example.expected.title_id)
                    for example in title_examples
                ),
                len(title_examples),
            ),
            4,
        ),
        "end_to_end_accuracy": round(
            safe_divide(len(end_to_end_correct), total),
            4,
        ),
        "auto_accept": {
            "count": len(auto_accept_examples),
            "coverage": round(safe_divide(len(auto_accept_examples), total), 4),
            "precision": round(
                safe_divide(
                    sum(is_end_to_end_correct(example) for example in auto_accept_examples),
                    len(auto_accept_examples),
                ),
                4,
            ),
        },
        "decision_accuracy": round(
            safe_divide(
                sum(
                    normalize_label(example.predicted.decision)
                    == normalize_label(example.expected.decision)
                    for example in decision_examples
                ),
                len(decision_examples),
            ),
            4,
        ),
        "decision_distribution": dict(
            Counter(example.predicted.decision.value for example in examples)
        ),
    }
    return metrics


def build_error_rows(examples: list[EvaluatedExample]) -> list[dict[str, object]]:
    """Build CSV-friendly rows for examples with an end-to-end mismatch."""
    rows: list[dict[str, object]] = []
    for example in examples:
        if is_end_to_end_correct(example):
            continue

        predicted = example.predicted
        expected = example.expected
        rows.append(
            {
                "query_id": expected.query_id or "",
                "query_text": expected.query_text,
                "error_type": detect_error_type(example),
                "expected_is_prof_video": expected.is_prof_video,
                "predicted_is_prof_video": predicted.is_prof_video,
                "expected_content_type": expected.content_type or "",
                "predicted_content_type": predicted.content_type.value
                if predicted.content_type
                else "",
                "expected_title_id": expected.title_id or "",
                "predicted_title_id": predicted.title_id or "",
                "predicted_title": predicted.title or "",
                "confidence": predicted.confidence,
                "decision": predicted.decision.value,
                "top_candidates": "; ".join(
                    f"{candidate.canonical_title}:{candidate.rank_score}"
                    for candidate in predicted.candidates
                ),
                "reasons": "; ".join(predicted.reasons),
            }
        )
    return rows


def compute_binary_counts(examples: list[EvaluatedExample]) -> dict[str, int]:
    """Compute binary confusion counts for professional video detection."""
    counts = {"tp": 0, "fp": 0, "tn": 0, "fn": 0}
    for example in examples:
        expected = example.expected.is_prof_video
        predicted = example.predicted.is_prof_video
        if expected and predicted:
            counts["tp"] += 1
        elif not expected and predicted:
            counts["fp"] += 1
        elif not expected and not predicted:
            counts["tn"] += 1
        else:
            counts["fn"] += 1
    return counts


def is_end_to_end_correct(example: EvaluatedExample) -> bool:
    """Return whether a prediction matches the target business label."""
    expected = example.expected
    predicted = example.predicted

    if predicted.is_prof_video != expected.is_prof_video:
        return False
    if not expected.is_prof_video:
        return not predicted.is_prof_video

    if normalize_label(predicted.content_type) != normalize_label(expected.content_type):
        return False
    return normalize_label(predicted.title_id) == normalize_label(expected.title_id)


def detect_error_type(example: EvaluatedExample) -> str:
    """Return a compact error category for a failed prediction."""
    expected = example.expected
    predicted = example.predicted

    if expected.is_prof_video and not predicted.is_prof_video:
        return "false_negative_video"
    if not expected.is_prof_video and predicted.is_prof_video:
        return "false_positive_video"
    if normalize_label(predicted.content_type) != normalize_label(expected.content_type):
        return "wrong_content_type"
    if normalize_label(predicted.title_id) != normalize_label(expected.title_id):
        return "wrong_title"
    return "unknown_mismatch"


def normalize_label(value: object) -> str:
    """Normalize enum, string, and empty labels for comparisons."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        return str(value.value)
    return str(value).strip()


def safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers and return 0.0 when denominator is zero."""
    if denominator == 0:
        return 0.0
    return numerator / denominator

