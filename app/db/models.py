"""SQLAlchemy models for persistent catalog, predictions, and review queue."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


def utc_now() -> datetime:
    """Return the current UTC time as a timezone-aware datetime."""
    return datetime.now(timezone.utc)


class SearchQuery(Base):
    """Raw user search query stored for labeling and audit."""

    __tablename__ = "search_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_text: Mapped[str] = mapped_column(Text, nullable=False)
    source: Mapped[Optional[str]] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    predictions: Mapped[list[Prediction]] = relationship(back_populates="query")


class CatalogTitleModel(Base):
    """Canonical video title stored in the local catalog."""

    __tablename__ = "catalog_titles"

    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    canonical_title: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(String(50), nullable=False)
    year: Mapped[Optional[int]] = mapped_column(Integer)
    popularity: Mapped[float] = mapped_column(Float, default=0.0)
    external_source: Mapped[Optional[str]] = mapped_column(String(100))
    external_id: Mapped[Optional[str]] = mapped_column(String(100))

    aliases: Mapped[list[TitleAliasModel]] = relationship(back_populates="title")


class TitleAliasModel(Base):
    """Searchable alias for a canonical catalog title."""

    __tablename__ = "title_aliases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    title_id: Mapped[str] = mapped_column(ForeignKey("catalog_titles.id"))
    alias: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_alias: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    title: Mapped[CatalogTitleModel] = relationship(back_populates="aliases")


class Prediction(Base):
    """Model prediction for one stored search query."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("search_queries.id"))
    is_prof_video: Mapped[bool] = mapped_column(Boolean, nullable=False)
    domain_label: Mapped[str] = mapped_column(String(50), nullable=False)
    content_type: Mapped[Optional[str]] = mapped_column(String(50))
    title_id: Mapped[Optional[str]] = mapped_column(ForeignKey("catalog_titles.id"))
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    decision: Mapped[str] = mapped_column(String(50), nullable=False)
    model_version: Mapped[str] = mapped_column(String(100), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)

    query: Mapped[SearchQuery] = relationship(back_populates="predictions")
    candidates: Mapped[list[PredictionCandidate]] = relationship(
        back_populates="prediction"
    )


class PredictionCandidate(Base):
    """Candidate title considered by the model for one prediction."""

    __tablename__ = "prediction_candidates"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    title_id: Mapped[str] = mapped_column(ForeignKey("catalog_titles.id"))
    rank: Mapped[int] = mapped_column(Integer, nullable=False)
    score: Mapped[float] = mapped_column(Float, nullable=False)
    matched_alias: Mapped[str] = mapped_column(Text, nullable=False)

    prediction: Mapped[Prediction] = relationship(back_populates="candidates")


class ReviewQueueItem(Base):
    """Prediction queued for human validation."""

    __tablename__ = "review_queue"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    prediction_id: Mapped[int] = mapped_column(ForeignKey("predictions.id"))
    status: Mapped[str] = mapped_column(String(50), default="pending")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utc_now)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime)
