"""Candidate ranking rules for catalog search results."""

from __future__ import annotations

from app.schemas import ContentType, NormalizedQuery, TitleCandidate


class RuleBasedCandidateRanker:
    """Apply domain-specific boosts to raw catalog search candidates."""

    def rank(
        self,
        query: NormalizedQuery,
        candidates: list[TitleCandidate],
    ) -> list[TitleCandidate]:
        """Return candidates ordered by final rank score."""
        ranked = [self._rank_one(query, candidate) for candidate in candidates]
        return sorted(ranked, key=lambda candidate: candidate.rank_score, reverse=True)

    def _rank_one(
        self,
        query: NormalizedQuery,
        candidate: TitleCandidate,
    ) -> TitleCandidate:
        """Rank one candidate using query signals and catalog metadata."""
        signals = query.signals
        score = candidate.search_score
        reasons = list(candidate.reasons)

        if signals.get("has_series_word") and candidate.content_type == ContentType.SERIES:
            score += 0.08
            reasons.append("series_type_boost")
        if signals.get("has_film_word") and candidate.content_type == ContentType.FILM:
            score += 0.05
            reasons.append("film_type_boost")
        if signals.get("has_cartoon_word") and candidate.content_type == ContentType.CARTOON:
            score += 0.08
            reasons.append("cartoon_type_boost")
        if (
            signals.get("has_animated_series_word")
            and candidate.content_type == ContentType.ANIMATED_SERIES
        ):
            score += 0.08
            reasons.append("animated_series_type_boost")

        years = signals.get("years") or []
        if candidate.year and candidate.year in years:
            score += 0.05
            reasons.append("year_match_boost")

        if signals.get("has_hard_negative"):
            score -= 0.08
            reasons.append("hard_negative_penalty")

        candidate.rank_score = round(max(0.0, min(score, 1.0)), 4)
        candidate.reasons = reasons
        return candidate

