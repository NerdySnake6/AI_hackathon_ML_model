"""Text normalization and query variant generation utilities."""

from __future__ import annotations

import re
from collections.abc import Iterable

from app import settings
from app.schemas import NormalizedQuery

_DASH_RE = re.compile(r"[-‐‑‒–—―]+")
_PUNCT_RE = re.compile(r"[\"'`“”„«».,!?;:()\[\]{}<>/\\|]+")
_SPACE_RE = re.compile(r"\s+")
_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2})\b")
_SEASON_RE = re.compile(r"\b(\d+)\s*(сезон|сезона|сезоне|серия|серии)\b")

_EN_TO_RU = str.maketrans(
    {
        "q": "й",
        "w": "ц",
        "e": "у",
        "r": "к",
        "t": "е",
        "y": "н",
        "u": "г",
        "i": "ш",
        "o": "щ",
        "p": "з",
        "[": "х",
        "]": "ъ",
        "a": "ф",
        "s": "ы",
        "d": "в",
        "f": "а",
        "g": "п",
        "h": "р",
        "j": "о",
        "k": "л",
        "l": "д",
        ";": "ж",
        "'": "э",
        "z": "я",
        "x": "ч",
        "c": "с",
        "v": "м",
        "b": "и",
        "n": "т",
        "m": "ь",
        ",": "б",
        ".": "ю",
    }
)

_RU_TO_LATIN = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def normalize_text(text: str) -> str:
    """Normalize case, whitespace, punctuation, and Russian letter variants."""
    normalized = text.replace("\u00a0", " ").replace("ё", "е").replace("Ё", "е")
    normalized = normalized.lower()
    normalized = _DASH_RE.sub(" ", normalized)
    normalized = _PUNCT_RE.sub(" ", normalized)
    normalized = _SPACE_RE.sub(" ", normalized)
    return normalized.strip()


def compact_text(text: str) -> str:
    """Return a normalized text form without whitespace."""
    return normalize_text(text).replace(" ", "")


def fix_keyboard_layout(text: str) -> str:
    """Convert text typed with an English keyboard layout into Russian letters."""
    return normalize_text(text.translate(_EN_TO_RU))


def transliterate_ru_to_latin(text: str) -> str:
    """Transliterate Russian text into a simple Latin representation."""
    normalized = normalize_text(text)
    return "".join(_RU_TO_LATIN.get(char, char) for char in normalized)


def clean_title_query(normalized_text: str) -> str:
    """Remove frequent video intent words while preserving likely title tokens."""
    tokens = normalized_text.split()
    result: list[str] = []
    skip_indexes: set[int] = set()

    for index, token in enumerate(tokens):
        next_token = tokens[index + 1] if index + 1 < len(tokens) else ""
        prev_token = tokens[index - 1] if index > 0 else ""
        if token.isdigit() and next_token in {"сезон", "сезона", "серия", "серии"}:
            skip_indexes.add(index)
            skip_indexes.add(index + 1)
        if token in {"сезон", "сезона", "серия", "серии"} and prev_token.isdigit():
            skip_indexes.add(index)

    for index, token in enumerate(tokens):
        if index in skip_indexes:
            continue
        if token in settings.TITLE_NOISE_WORDS:
            continue
        result.append(token)

    return " ".join(result).strip()


def extract_signals(normalized_text: str) -> dict[str, object]:
    """Extract lightweight lexical signals used by rules and rankers."""
    tokens = set(normalized_text.split())
    years = [int(match) for match in _YEAR_RE.findall(normalized_text)]
    return {
        "has_video_intent": bool(tokens & settings.VIDEO_INTENT_WORDS),
        "has_generic_word": bool(tokens & settings.GENERIC_VIDEO_WORDS),
        "has_hard_negative": bool(tokens & settings.HARD_NEGATIVE_WORDS),
        "has_film_word": bool(tokens & settings.FILM_WORDS),
        "has_series_word": bool(tokens & settings.SERIES_WORDS)
        or bool(_SEASON_RE.search(normalized_text)),
        "has_cartoon_word": bool(tokens & settings.CARTOON_WORDS),
        "has_animated_series_word": bool(tokens & settings.ANIMATED_SERIES_WORDS),
        "years": years,
    }


def rough_case_variants(text: str) -> list[str]:
    """Generate conservative Russian case variants for short title fragments."""
    tokens = text.split()
    if not tokens:
        return []

    variants: list[str] = []
    for index, token in enumerate(tokens):
        if len(token) <= 3:
            continue
        replacements = []
        if token.endswith(("ы", "и")):
            replacements.append(token[:-1] + "а")
        if token.endswith("е"):
            replacements.append(token[:-1] + "а")
        for replacement in replacements:
            changed = tokens.copy()
            changed[index] = replacement
            variants.append(" ".join(changed))
    return variants


def unique_non_empty(values: Iterable[str]) -> list[str]:
    """Return non-empty strings in insertion order without duplicates."""
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_text(value)
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def build_normalized_query(raw_text: str) -> NormalizedQuery:
    """Build normalized query forms and query variants for robust matching."""
    normalized = normalize_text(raw_text)
    clean = clean_title_query(normalized)
    layout_fixed = fix_keyboard_layout(normalized)
    layout_fixed_clean = clean_title_query(layout_fixed)

    variants = unique_non_empty(
        [
            normalized,
            clean,
            compact_text(clean),
            layout_fixed,
            layout_fixed_clean,
            compact_text(layout_fixed_clean),
            *rough_case_variants(clean),
            *rough_case_variants(layout_fixed_clean),
        ]
    )

    return NormalizedQuery(
        raw_text=raw_text,
        normalized_text=normalized,
        clean_text=clean,
        compact_text=compact_text(clean),
        variants=variants,
        signals=extract_signals(normalized),
    )

