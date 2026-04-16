"""Convert IMDb bulk dumps into the local catalog CSV and optional DB import."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.db.models import Base
from app.db.repositories import CatalogDatabaseRepository
from app.db.session import DATABASE_URL, SessionLocal, engine
from app.ingestion.imdb import (
    ImdbImportConfig,
    load_imdb_catalog,
    write_catalog_csv,
)


def parse_csv_list(value: str) -> tuple[str, ...]:
    """Parse a comma-separated CLI option into a tuple of trimmed values."""
    return tuple(item.strip() for item in value.split(",") if item.strip())


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for IMDb bulk import."""
    parser = argparse.ArgumentParser(
        description="Convert IMDb dumps into the local catalog format.",
    )
    parser.add_argument("--basics", required=True, help="Path to title.basics.tsv.gz")
    parser.add_argument("--akas", required=True, help="Path to title.akas.tsv.gz")
    parser.add_argument(
        "--ratings",
        help="Path to title.ratings.tsv.gz. When set, min-votes filtering is applied.",
    )
    parser.add_argument(
        "--output",
        help="Optional debug/export CSV path in the local catalog schema.",
    )
    parser.add_argument(
        "--min-votes",
        type=int,
        default=200,
        help="Keep only titles with at least this many IMDb votes.",
    )
    parser.add_argument(
        "--max-titles",
        type=int,
        help="Optional safety cap for quick smoke runs.",
    )
    parser.add_argument(
        "--max-aliases-per-title",
        type=int,
        default=30,
        help="Upper bound for stored aliases per title.",
    )
    parser.add_argument(
        "--max-titles-per-normalized-alias",
        type=int,
        default=1,
        help="Keep one normalized alias for at most this many titles.",
    )
    parser.add_argument(
        "--title-types",
        default="movie,tvMovie,tvSeries,tvMiniSeries",
        help="Comma-separated IMDb title types to keep.",
    )
    parser.add_argument(
        "--aka-regions",
        default="RU,US,GB",
        help="Comma-separated AKA regions to keep.",
    )
    parser.add_argument(
        "--aka-languages",
        default="ru,en",
        help="Comma-separated AKA languages to keep.",
    )
    parser.add_argument(
        "--import-db",
        action="store_true",
        help="Also upsert the generated catalog into the configured database.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables before DB import; intended for local demo runs.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> ImdbImportConfig:
    """Build an ``ImdbImportConfig`` from parsed CLI arguments."""
    return ImdbImportConfig(
        min_votes=args.min_votes,
        max_titles=args.max_titles,
        max_aliases_per_title=args.max_aliases_per_title,
        max_titles_per_normalized_alias=args.max_titles_per_normalized_alias,
        title_types=parse_csv_list(args.title_types),
        aka_regions=parse_csv_list(args.aka_regions),
        aka_languages=parse_csv_list(args.aka_languages),
    )


def maybe_import_to_db(
    titles: list,
    create_tables: bool = False,
) -> int:
    """Import converted catalog titles into the configured relational database."""
    if create_tables:
        Base.metadata.create_all(engine)
    with SessionLocal() as session:
        repository = CatalogDatabaseRepository(session)
        imported_count = repository.upsert_titles(titles)
        session.commit()
    return imported_count


def main() -> None:
    """Run IMDb conversion and optionally import the result into the DB."""
    args = parse_args()
    if not args.output and not args.import_db:
        raise SystemExit("Provide --output and/or --import-db")

    config = build_config(args)
    catalog = load_imdb_catalog(
        basics_path=args.basics,
        akas_path=args.akas,
        ratings_path=args.ratings,
        config=config,
    )
    if args.output:
        write_catalog_csv(catalog, args.output)
        print(f"Converted {len(catalog)} IMDb titles into {args.output}")
    else:
        print(f"Converted {len(catalog)} IMDb titles")
    if args.import_db:
        imported_count = maybe_import_to_db(
            titles=catalog,
            create_tables=args.create_tables,
        )
        print(f"Imported {imported_count} titles into {DATABASE_URL}")


if __name__ == "__main__":
    main()
