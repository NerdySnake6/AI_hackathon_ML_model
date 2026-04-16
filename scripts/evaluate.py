"""Evaluate the query labeling pipeline against a labeled CSV dataset."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.evaluation.metrics import (
    EvaluatedExample,
    ExpectedLabel,
    build_error_rows,
    compute_metrics,
)
from app.pipeline import QueryLabelingPipeline
from app.schemas import QueryInput
from app.settings import DEFAULT_CATALOG_PATH, MAX_BATCH_SIZE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate query labeling quality.")
    parser.add_argument(
        "--input",
        required=True,
        help="Input CSV with query_text and expected label columns.",
    )
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        help="Catalog CSV path.",
    )
    parser.add_argument(
        "--output",
        default="outputs/metrics_report.json",
        help="Output JSON metrics report path.",
    )
    parser.add_argument(
        "--errors",
        default="outputs/errors.csv",
        help="Output CSV with mismatched examples.",
    )
    return parser.parse_args()


def evaluate(
    input_path: str,
    catalog_path: str,
    output_path: str,
    errors_path: str,
) -> dict[str, object]:
    """Run evaluation and persist metrics plus error rows."""
    pipeline = QueryLabelingPipeline.from_csv(catalog_path)
    examples = load_examples(input_path, pipeline)
    metrics = compute_metrics(examples)
    error_rows = build_error_rows(examples)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    errors = Path(errors_path)
    errors.parent.mkdir(parents=True, exist_ok=True)
    write_error_rows(errors, error_rows)

    return metrics


def load_examples(
    input_path: str,
    pipeline: QueryLabelingPipeline,
) -> list[EvaluatedExample]:
    """Load expected labels from CSV and pair them with model predictions."""
    with Path(input_path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    examples: list[EvaluatedExample] = []
    for row in rows:
        expected = ExpectedLabel(
            query_id=row.get("query_id") or None,
            query_text=row["query_text"],
            is_prof_video=parse_bool(row.get("expected_is_prof_video") or row.get("is_prof_video", "false")),
            content_type=row.get("expected_content_type") or row.get("content_type") or None,
            title_id=row.get("expected_title_id") or row.get("title_id") or None,
            decision=row.get("expected_decision") or row.get("decision") or None,
        )
        prediction = pipeline.label(
            QueryInput(
                query_id=expected.query_id,
                query_text=expected.query_text,
                source=row.get("source") or None,
            )
        )
        examples.append(EvaluatedExample(expected=expected, predicted=prediction))
    return examples


def write_error_rows(path: Path, rows: list[dict[str, object]]) -> None:
    """Write evaluation mismatches to a CSV file."""
    fieldnames = [
        "query_id",
        "query_text",
        "error_type",
        "expected_is_prof_video",
        "predicted_is_prof_video",
        "expected_content_type",
        "predicted_content_type",
        "expected_title_id",
        "predicted_title_id",
        "predicted_title",
        "confidence",
        "decision",
        "top_candidates",
        "reasons",
    ]
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_bool(value: str) -> bool:
    """Parse a CSV boolean value."""
    normalized = value.strip().lower()
    if normalized in {"true", "1", "yes", "y", "да"}:
        return True
    if normalized in {"false", "0", "no", "n", "нет"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value!r}")


def main() -> None:
    """Run the evaluation command."""
    args = parse_args()
    metrics = evaluate(
        input_path=args.input,
        catalog_path=args.catalog,
        output_path=args.output,
        errors_path=args.errors,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

