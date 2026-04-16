"""Run query labeling for a CSV file and write predictions to another CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.pipeline import QueryLabelingPipeline
from app.schemas import QueryInput
from app.settings import DEFAULT_CATALOG_PATH, MAX_BATCH_SIZE


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for batch labeling."""
    parser = argparse.ArgumentParser(description="Run query labeling batch job.")
    parser.add_argument("--input", required=True, help="Input CSV with query_text column.")
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        help="Catalog CSV path.",
    )
    parser.add_argument(
        "--catalog-source",
        choices=("csv", "db"),
        default="csv",
        help="Where to load the title catalog from.",
    )
    parser.add_argument(
        "--database-url",
        help="Database URL used when --catalog-source=db.",
    )
    parser.add_argument("--output", required=True, help="Output predictions CSV path.")
    return parser.parse_args()


def build_pipeline(
    catalog_source: str,
    catalog_path: str,
    database_url: str | None = None,
) -> QueryLabelingPipeline:
    """Build the batch pipeline from the selected catalog source."""
    if catalog_source == "db":
        if database_url:
            return QueryLabelingPipeline.from_db(database_url=database_url)
        return QueryLabelingPipeline.from_db()
    return QueryLabelingPipeline.from_csv(catalog_path)


def run_batch(
    input_path: str,
    output_path: str,
    catalog_path: str = DEFAULT_CATALOG_PATH,
    catalog_source: str = "csv",
    database_url: str | None = None,
) -> int:
    """Label queries from CSV and write structured predictions."""
    pipeline = build_pipeline(
        catalog_source=catalog_source,
        catalog_path=catalog_path,
        database_url=database_url,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with Path(input_path).open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)

    if len(rows) > MAX_BATCH_SIZE:
        raise ValueError(f"Batch size exceeds limit: {len(rows)} > {MAX_BATCH_SIZE}")

    with output.open("w", encoding="utf-8", newline="") as output_file:
        fieldnames = [
            "query_id",
            "query_text",
            "is_prof_video",
            "domain_label",
            "content_type",
            "title",
            "title_id",
            "confidence",
            "decision",
            "top_candidates",
            "reasons",
        ]
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            result = pipeline.label(
                QueryInput(
                    query_id=row.get("query_id") or None,
                    query_text=row["query_text"],
                    source=row.get("source") or None,
                )
            )
            writer.writerow(
                {
                    "query_id": result.query_id,
                    "query_text": result.query,
                    "is_prof_video": result.is_prof_video,
                    "domain_label": result.domain_label.value,
                    "content_type": result.content_type.value
                    if result.content_type
                    else "",
                    "title": result.title or "",
                    "title_id": result.title_id or "",
                    "confidence": result.confidence,
                    "decision": result.decision.value,
                    "top_candidates": "; ".join(
                        f"{candidate.canonical_title}:{candidate.rank_score}"
                        for candidate in result.candidates
                    ),
                    "reasons": "; ".join(result.reasons),
                }
            )

    return len(rows)


def main() -> None:
    """Run the batch command."""
    args = parse_args()
    count = run_batch(
        input_path=args.input,
        output_path=args.output,
        catalog_path=args.catalog,
        catalog_source=args.catalog_source,
        database_url=args.database_url,
    )
    print(f"Processed {count} queries -> {args.output}")


if __name__ == "__main__":
    main()
