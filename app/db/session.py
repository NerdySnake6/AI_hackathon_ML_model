"""Database engine and session helpers."""

from __future__ import annotations

from collections.abc import Iterator
import os
from pathlib import Path

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

DEFAULT_DATABASE_URL = "sqlite:///outputs/query_labeler.db"
DATABASE_URL = os.getenv("QUERY_LABELER_DATABASE_URL", DEFAULT_DATABASE_URL)


def build_engine(database_url: str = DATABASE_URL) -> Engine:
    """Create a SQLAlchemy engine for SQLite or PostgreSQL-compatible URLs."""
    if database_url.startswith("sqlite:///"):
        db_path = Path(database_url.removeprefix("sqlite:///"))
        if db_path.parent != Path("."):
            db_path.parent.mkdir(parents=True, exist_ok=True)
    return create_engine(database_url)


engine = build_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def get_session() -> Iterator[Session]:
    """Yield a SQLAlchemy session and close it after use."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
