"""Rule-based classifiers for domain and generic query detection."""

from __future__ import annotations

from app import settings
from app.schemas import (
    ClassifierResult,
    ContentType,
    DomainLabel,
    GenericDetectionResult,
    NormalizedQuery,
    TitleCandidate,
)


class RuleBasedDomainClassifier:
    """Rule-based gate that separates video queries from non-video queries."""

    def predict(
        self,
        query: NormalizedQuery,
        candidates: list[TitleCandidate],
    ) -> ClassifierResult:
        """Predict whether a query belongs to professional video content."""
        signals = query.signals
        reasons: list[str] = []
        score = 0.1
        top_score = candidates[0].rank_score if candidates else 0.0

        if signals.get("has_video_intent"):
            score += 0.35
            reasons.append("video_intent_word")
        if signals.get("has_series_word"):
            score += 0.25
            reasons.append("series_marker")
        if signals.get("has_cartoon_word") or signals.get("has_animated_series_word"):
            score += 0.25
            reasons.append("animation_marker")
        if signals.get("has_generic_word"):
            score += 0.2
            reasons.append("generic_video_word")
        if top_score >= 0.9:
            score += 0.5
            reasons.append("strong_catalog_candidate")
        elif top_score >= 0.75:
            score += 0.2
            reasons.append("medium_catalog_candidate")
        elif looks_like_unknown_title_query(query):
            score += 0.32
            reasons.append("unknown_title_candidate")

        if signals.get("has_hard_negative"):
            penalty = 0.2 if top_score >= 0.75 else 0.45
            score -= penalty
            reasons.append("hard_negative_word")

        score = max(0.0, min(score, 1.0))
        if score >= settings.DOMAIN_VIDEO_THRESHOLD:
            label = DomainLabel.PROF_VIDEO
        elif score >= 0.4:
            label = DomainLabel.UNCERTAIN
        else:
            label = DomainLabel.NON_VIDEO

        return ClassifierResult(
            label=label.value,
            confidence=round(score, 4),
            reasons=reasons,
        )


class RuleBasedGenericDetector:
    """Rule-based detector for generic video queries without a concrete title."""

    def predict(
        self,
        query: NormalizedQuery,
        candidates: list[TitleCandidate],
    ) -> GenericDetectionResult:
        """Detect generic video intent and infer a broad content type."""
        signals = query.signals
        top_score = candidates[0].rank_score if candidates else 0.0
        reasons: list[str] = []

        if not signals.get("has_generic_word"):
            return GenericDetectionResult(is_generic=False)
        if top_score >= 0.9:
            return GenericDetectionResult(is_generic=False)

        content_type = infer_generic_content_type(query)
        reasons.append("generic_video_word")
        return GenericDetectionResult(
            is_generic=True,
            content_type=content_type,
            confidence=0.85,
            reasons=reasons,
        )


def infer_generic_content_type(query: NormalizedQuery) -> ContentType:
    """Infer broad content type for a generic video query."""
    tokens = set(query.normalized_text.split())
    if tokens & settings.FILM_WORDS:
        return ContentType.FILM
    if tokens & settings.SERIES_WORDS:
        return ContentType.SERIES
    if tokens & settings.ANIMATED_SERIES_WORDS:
        return ContentType.ANIMATED_SERIES
    if tokens & settings.CARTOON_WORDS:
        return ContentType.CARTOON
    return ContentType.GENERIC


def looks_like_unknown_title_query(query: NormalizedQuery) -> bool:
    """Return whether a query may be an out-of-catalog title mention."""
    signals = query.signals
    if signals.get("has_hard_negative") or signals.get("has_generic_word"):
        return False
    if signals.get("has_video_intent"):
        return False

    tokens = query.clean_text.split()
    if not 1 <= len(tokens) <= 4:
        return False
    if any(len(token) < 3 for token in tokens):
        return False
    return all(any(char.isalpha() for char in token) for token in tokens)
