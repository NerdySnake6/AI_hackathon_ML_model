"""Import a CSV title catalog into the configured relational database."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.catalog.repository import InMemoryCatalogRepository
from app.db.models import Base
from app.db.repositories import CatalogDatabaseRepository
from app.db.session import DATABASE_URL, SessionLocal, engine
from app.settings import DEFAULT_CATALOG_PATH


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for catalog import."""
    parser = argparse.ArgumentParser(description="Import catalog CSV into DB.")
    parser.add_argument(
        "--catalog",
        default=DEFAULT_CATALOG_PATH,
        help="Catalog CSV path.",
    )
    parser.add_argument(
        "--create-tables",
        action="store_true",
        help="Create tables directly before import; for local demo only.",
    )
    return parser.parse_args()


def import_catalog(catalog_path: str, create_tables: bool = False) -> int:
    """Import catalog titles into the configured database."""
    if create_tables:
        Base.metadata.create_all(engine)

    catalog = InMemoryCatalogRepository.from_csv(catalog_path)
    titles = catalog.list_titles()
    with SessionLocal() as session:
        repository = CatalogDatabaseRepository(session)
        imported_count = repository.upsert_titles(titles)
        session.commit()
    return imported_count


def main() -> None:
    """Run the catalog import command."""
    args = parse_args()
    count = import_catalog(args.catalog, create_tables=args.create_tables)
    print(f"Imported {count} titles into {DATABASE_URL}")


if __name__ == "__main__":
    main()

