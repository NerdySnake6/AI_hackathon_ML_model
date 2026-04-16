"""Database repositories for catalog import and prediction persistence."""

from __future__ import annotations

from sqlalchemy import delete, select
from sqlalchemy.orm import Session, selectinload

from app.catalog.repository import build_alias_records
from app.db.models import (
    CatalogTitleModel,
    Prediction,
    PredictionCandidate,
    ReviewQueueItem,
    SearchQuery,
    TitleAliasModel,
)
from app.preprocessing.normalizer import normalize_text
from app.schemas import CatalogTitle, Decision, LabelResult


class CatalogDatabaseRepository:
    """Persist and load canonical titles in a relational database."""

    def __init__(self, session: Session) -> None:
        """Initialize repository with an active SQLAlchemy session."""
        self.session = session

    def upsert_titles(self, titles: list[CatalogTitle]) -> int:
        """Insert or update catalog titles and rebuild their aliases."""
        for title in titles:
            self.session.merge(
                CatalogTitleModel(
                    id=title.title_id,
                    canonical_title=title.canonical_title,
                    content_type=title.content_type.value,
                    year=title.year,
                    popularity=title.popularity,
                    external_source=title.external_source,
                    external_id=title.external_id,
                )
            )
        self.session.flush()

        title_ids = [title.title_id for title in titles]
        if title_ids:
            self.session.execute(
                delete(TitleAliasModel).where(TitleAliasModel.title_id.in_(title_ids))
            )

        for alias_record in build_alias_records(titles):
            self.session.add(
                TitleAliasModel(
                    title_id=alias_record.title.title_id,
                    alias=alias_record.alias,
                    normalized_alias=alias_record.normalized_alias,
                )
            )
        self.session.flush()
        return len(titles)

    def list_titles(self) -> list[CatalogTitle]:
        """Return catalog titles with aliases stored in the database."""
        rows = self.session.scalars(
            select(CatalogTitleModel)
            .options(selectinload(CatalogTitleModel.aliases))
            .order_by(CatalogTitleModel.id)
        ).all()
        return [
            CatalogTitle(
                title_id=row.id,
                canonical_title=row.canonical_title,
                content_type=row.content_type,
                year=row.year,
                aliases=[alias.alias for alias in row.aliases],
                popularity=row.popularity,
                external_source=row.external_source,
                external_id=row.external_id,
            )
            for row in rows
        ]


class PredictionDatabaseRepository:
    """Persist pipeline predictions and queue uncertain items for review."""

    REVIEW_DECISIONS = {Decision.REVIEW, Decision.MANUAL_REQUIRED}

    def __init__(self, session: Session) -> None:
        """Initialize repository with an active SQLAlchemy session."""
        self.session = session

    def save_result(self, result: LabelResult, source: str | None = None) -> int:
        """Persist one label result and return the prediction id."""
        search_query = SearchQuery(
            raw_text=result.query,
            normalized_text=normalize_text(result.query),
            source=source,
        )
        self.session.add(search_query)
        self.session.flush()

        prediction = Prediction(
            query_id=search_query.id,
            is_prof_video=result.is_prof_video,
            domain_label=result.domain_label.value,
            content_type=result.content_type.value if result.content_type else None,
            title_id=result.title_id,
            confidence=result.confidence,
            decision=result.decision.value,
            model_version=result.model_version,
        )
        self.session.add(prediction)
        self.session.flush()

        for rank, candidate in enumerate(result.candidates, start=1):
            self.session.add(
                PredictionCandidate(
                    prediction_id=prediction.id,
                    title_id=candidate.title_id,
                    rank=rank,
                    score=candidate.rank_score,
                    matched_alias=candidate.matched_alias,
                )
            )

        if result.decision in self.REVIEW_DECISIONS:
            self.session.add(ReviewQueueItem(prediction_id=prediction.id))

        self.session.flush()
        return prediction.id
