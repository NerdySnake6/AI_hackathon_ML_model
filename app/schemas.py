"""Pydantic schemas shared by API, batch scripts, and the labeling pipeline."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.settings import MAX_BATCH_SIZE, MAX_QUERY_LENGTH


class ContentType(str, Enum):
    """Supported professional video content types."""

    FILM = "film"
    SERIES = "series"
    CARTOON = "cartoon"
    ANIMATED_SERIES = "animated_series"
    GENERIC = "generic"


class DomainLabel(str, Enum):
    """Domain gate labels for incoming user queries."""

    PROF_VIDEO = "prof_video"
    NON_VIDEO = "non_video"
    UNCERTAIN = "uncertain"


class Decision(str, Enum):
    """Final routing decisions produced by the confidence policy."""

    AUTO_ACCEPT = "auto_accept"
    REVIEW = "review"
    MANUAL_REQUIRED = "manual_required"
    NON_VIDEO = "non_video"
    GENERIC_VIDEO = "generic_video"


class QueryInput(BaseModel):
    """Incoming query payload for single-query labeling."""

    query_id: str | None = None
    query_text: str = Field(..., min_length=1, max_length=MAX_QUERY_LENGTH)
    source: str | None = None

    @field_validator("query_text")
    @classmethod
    def strip_query_text(cls, value: str) -> str:
        """Strip whitespace and reject blank query strings."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("query_text must not be blank")
        return stripped


class BatchQueryInput(BaseModel):
    """Incoming batch request with a bounded number of queries."""

    queries: list[QueryInput] = Field(..., min_length=1, max_length=MAX_BATCH_SIZE)


class NormalizedQuery(BaseModel):
    """Normalized query forms and extracted text signals."""

    raw_text: str
    normalized_text: str
    clean_text: str
    compact_text: str
    variants: list[str]
    signals: dict[str, Any] = Field(default_factory=dict)


class CatalogTitle(BaseModel):
    """Canonical title record loaded from the local catalog."""

    title_id: str
    canonical_title: str
    content_type: ContentType
    year: int | None = None
    aliases: list[str] = Field(default_factory=list)
    popularity: float = Field(default=0.0, ge=0.0, le=1.0)
    external_source: str | None = None
    external_id: str | None = None

    @field_validator("canonical_title")
    @classmethod
    def strip_title(cls, value: str) -> str:
        """Strip canonical title and reject blank values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("canonical_title must not be blank")
        return stripped


class TitleCandidate(BaseModel):
    """Candidate title found in the catalog for a user query."""

    title_id: str
    canonical_title: str
    content_type: ContentType
    year: int | None = None
    matched_alias: str
    matched_variant: str
    search_score: float = Field(ge=0.0, le=1.0)
    rank_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class ClassifierResult(BaseModel):
    """Generic classifier output with confidence and explanation reasons."""

    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class GenericDetectionResult(BaseModel):
    """Result of generic video query detection."""

    is_generic: bool
    content_type: ContentType | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class LabelResult(BaseModel):
    """Final result returned by the query labeling pipeline."""

    query_id: str | None = None
    query: str
    normalized_query: str
    is_prof_video: bool
    domain_label: DomainLabel
    content_type: ContentType | None = None
    title: str | None = None
    title_id: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    decision: Decision
    candidates: list[TitleCandidate] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    model_version: str = "mvp-rules-v1"

