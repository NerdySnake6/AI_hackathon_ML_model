"""Confidence policy for final query labeling decisions."""

from __future__ import annotations

from app import settings
from app.schemas import (
    ClassifierResult,
    Decision,
    DomainLabel,
    GenericDetectionResult,
    TitleCandidate,
)


class ConfidencePolicy:
    """Route labeling results into auto-accept, review, or manual handling."""

    def decide(
        self,
        domain: ClassifierResult,
        generic: GenericDetectionResult,
        candidates: list[TitleCandidate],
    ) -> Decision:
        """Choose the final decision for a labeled query."""
        domain_label = DomainLabel(domain.label)
        top_score = candidates[0].rank_score if candidates else 0.0

        if domain_label == DomainLabel.NON_VIDEO:
            return Decision.NON_VIDEO
        if generic.is_generic:
            return Decision.GENERIC_VIDEO
        if not candidates:
            return Decision.MANUAL_REQUIRED
        if top_score >= settings.AUTO_ACCEPT_THRESHOLD:
            return Decision.AUTO_ACCEPT
        if top_score >= settings.REVIEW_THRESHOLD:
            return Decision.REVIEW
        return Decision.MANUAL_REQUIRED

