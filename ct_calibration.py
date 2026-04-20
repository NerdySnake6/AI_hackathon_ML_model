"""ContentType calibration helpers for combining hints and model outputs."""

from __future__ import annotations

from dataclasses import dataclass


CONTENT_TYPE_LABELS = ["сериал", "фильм", "мультфильм", "мультсериал", "прочее"]


@dataclass
class ContentTypeHint:
    """Container for lexical ContentType hint extraction."""

    label: str
    score: int
    dominance: int


def detect_ct_from_words(query: str) -> str:
    """Return a simple ContentType guess from lexical hints in the query."""
    hint = infer_strong_ct_hint(query)
    return hint.label if hint.score > 0 else ""


def infer_strong_ct_hint(query: str) -> ContentTypeHint:
    """Extract a strong lexical ContentType hint and its confidence proxy."""
    lower = query.lower()
    scores = {
        "сериал": 0,
        "фильм": 0,
        "мультфильм": 0,
        "мультсериал": 0,
        "прочее": 0,
    }

    if "мультфильм" in lower:
        scores["мультфильм"] += 5
    if "мультсериал" in lower:
        scores["мультсериал"] += 5
    if any(token in lower for token in ("аниме", "аниме ", "онгоинг")):
        scores["мультсериал"] += 4
    if "дорама" in lower:
        scores["сериал"] += 4
    if any(token in lower for token in ("сезон", "серия", "эпизод", "season", "episode")):
        scores["сериал"] += 4
    if "сериал" in lower:
        scores["сериал"] += 2
    if any(token in lower for token in ("фильм", "кино", "картина")):
        scores["фильм"] += 2

    has_year = any(str(year) in lower for year in range(1950, 2031))
    if has_year and scores["сериал"] == 0 and scores["мультсериал"] == 0:
        scores["фильм"] += 1

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ordered[0]
    second_score = ordered[1][1] if len(ordered) > 1 else 0
    return ContentTypeHint(
        label=top_label if top_score > 0 else "",
        score=top_score,
        dominance=top_score - second_score,
    )


def calibrate_content_type(
    query: str,
    title_content_type: str,
    model_content_type: str,
    model_margin: float | None,
    title_source: str,
) -> str:
    """Combine title-derived CT, classifier output, and lexical hints."""
    hint = infer_strong_ct_hint(query)
    normalized_title_ct = _normalize_label(title_content_type)
    normalized_model_ct = _normalize_label(model_content_type)

    strong_hint = hint.label if hint.score >= 4 or (hint.score >= 3 and hint.dominance >= 2) else ""
    # Very strong hints (e.g. "мультфильм", "аниме") should almost always win
    is_very_strong_hint = (
        hint.score >= 5 or 
        (hint.label in {"мультфильм", "мультсериал", "сериал"} and hint.score >= 3) or
        (hint.label != "" and hint.dominance >= 4) or
        (any(w in query.lower() for w in ["аниме", "дорама", "т/с", "х/ф"]))
    )
    
    model_is_weak = model_margin is None or model_margin < 0.45
    title_is_strong = title_source in {"franchise_exact", "franchise_substring", "franchise_lemma_match_strong"}
    title_is_weak = title_source in {"raw", "catalog_fuzzy", "aggregate", "franchise_fuzzy", ""}

    if is_very_strong_hint:
        return hint.label

    if strong_hint:
        if normalized_title_ct and normalized_title_ct == strong_hint:
            return normalized_title_ct
        if normalized_model_ct and normalized_model_ct == strong_hint:
            return normalized_model_ct
        # If hint disagrees with title, trust hint if title is weak or hint is about cartoons
        if normalized_title_ct and normalized_title_ct != strong_hint:
            if title_is_weak or strong_hint in {"мультфильм", "мультсериал"} or hint.dominance >= 3:
                return strong_hint
        if normalized_model_ct and normalized_model_ct != strong_hint:
            if model_is_weak or strong_hint in {"мультфильм", "мультсериал"}:
                return strong_hint
        return strong_hint

    if normalized_title_ct and (title_is_strong or not normalized_model_ct or model_is_weak):
        return normalized_title_ct

    if normalized_model_ct:
        return normalized_model_ct

    if normalized_title_ct:
        return normalized_title_ct

    if hint.label:
        return hint.label

    return "прочее"


def _normalize_label(label: str) -> str:
    """Normalize a ContentType label into the leaderboard label space."""
    normalized = (label or "").strip().lower()
    return normalized if normalized in CONTENT_TYPE_LABELS else ""
