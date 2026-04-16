"""Tests for SQLAlchemy repositories."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from app.db.models import (
    Base,
    CatalogTitleModel,
    Prediction,
    PredictionCandidate,
    ReviewQueueItem,
    SearchQuery,
    TitleAliasModel,
)
from app.db.repositories import CatalogDatabaseRepository, PredictionDatabaseRepository
from app.pipeline import QueryLabelingPipeline
from app.schemas import CatalogTitle, ContentType, Decision


class DatabaseRepositoriesTestCase(unittest.TestCase):
    """Validate catalog import and prediction persistence repositories."""

    def setUp(self) -> None:
        """Create a clean in-memory database for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(bind=self.engine)

    def test_upsert_titles_rebuilds_aliases(self) -> None:
        """Persist catalog titles and their generated searchable aliases."""
        title = CatalogTitle(
            title_id="title_001",
            canonical_title="Интерстеллар",
            content_type=ContentType.FILM,
            year=2014,
            aliases=["interstellar", "интерстелар"],
        )

        with self.session_factory() as session:
            repository = CatalogDatabaseRepository(session)
            imported_count = repository.upsert_titles([title])
            session.commit()

            self.assertEqual(imported_count, 1)
            self.assertEqual(
                session.scalar(select(func.count()).select_from(CatalogTitleModel)),
                1,
            )
            aliases = session.scalars(select(TitleAliasModel)).all()
            self.assertTrue(
                any(alias.normalized_alias == "интерстелар" for alias in aliases)
            )

    def test_save_result_persists_prediction_candidates_and_review_item(self) -> None:
        """Persist a pipeline result and queue review decisions."""
        with self.session_factory() as session:
            catalog_repository = CatalogDatabaseRepository(session)
            catalog_repository.upsert_titles(
                [
                    CatalogTitle(
                        title_id="title_001",
                        canonical_title="Тьма",
                        content_type=ContentType.SERIES,
                        aliases=["dark"],
                    )
                ]
            )
            session.commit()

        pipeline = QueryLabelingPipeline.from_csv("data/sample_catalog.csv")
        result = pipeline.label("тьма")
        result.decision = Decision.REVIEW

        with self.session_factory() as session:
            prediction_repository = PredictionDatabaseRepository(session)
            prediction_id = prediction_repository.save_result(result, source="test")
            session.commit()

            self.assertIsInstance(prediction_id, int)
            self.assertEqual(
                session.scalar(select(func.count()).select_from(SearchQuery)),
                1,
            )
            self.assertEqual(
                session.scalar(select(func.count()).select_from(Prediction)),
                1,
            )
            self.assertEqual(
                session.scalar(select(func.count()).select_from(PredictionCandidate)),
                1,
            )
            self.assertEqual(
                session.scalar(select(func.count()).select_from(ReviewQueueItem)),
                1,
            )

    def test_pipeline_can_load_catalog_directly_from_database(self) -> None:
        """Build an inference pipeline from titles stored in SQLite."""
        with tempfile.TemporaryDirectory() as temp_dir:
            database_url = f"sqlite:///{Path(temp_dir) / 'catalog.db'}"
            engine = create_engine(database_url)
            session_factory = sessionmaker(bind=engine)
            Base.metadata.create_all(engine)

            try:
                with session_factory() as session:
                    repository = CatalogDatabaseRepository(session)
                    repository.upsert_titles(
                        [
                            CatalogTitle(
                                title_id="title_001",
                                canonical_title="Интерстеллар",
                                content_type=ContentType.FILM,
                                year=2014,
                                aliases=["interstellar", "интерстелар"],
                            )
                        ]
                    )
                    session.commit()

                pipeline = QueryLabelingPipeline.from_db(database_url=database_url)
                result = pipeline.label("интерстелар")
            finally:
                engine.dispose()

        self.assertTrue(result.is_prof_video)
        self.assertEqual(result.title_id, "title_001")


if __name__ == "__main__":
    unittest.main()
