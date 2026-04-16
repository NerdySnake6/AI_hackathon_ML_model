"""Catalog repositories and typo-tolerant candidate search."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from heapq import nlargest
from pathlib import Path

from app.preprocessing.normalizer import (
    compact_text,
    normalize_text,
    transliterate_ru_to_latin,
)
from rapidfuzz import fuzz
from app.schemas import CatalogTitle, NormalizedQuery, TitleCandidate
from app.settings import (
    MAX_CANDIDATES,
    SEARCH_MAX_ALIAS_CANDIDATES,
    SEARCH_PREFIX_LENGTH,
    SEARCH_TOKEN_MIN_LENGTH,
    TITLE_NOISE_WORDS,
)


@dataclass(frozen=True)
class AliasRecord:
    """Normalized alias bound to a catalog title."""

    title: CatalogTitle
    alias: str
    normalized_alias: str
    compact_alias: str


class InMemoryCatalogRepository:
    """In-memory catalog repository suitable for MVP and tests."""

    def __init__(self, titles: Sequence[CatalogTitle]) -> None:
        """Initialize repository and precompute searchable alias records."""
        self._titles = list(titles)
        self._aliases = build_alias_records(self._titles)
        self._exact_alias_index = build_exact_alias_index(self._aliases)
        self._compact_alias_index = build_compact_alias_index(self._aliases)
        self._token_index = build_token_index(self._aliases)
        self._prefix_index = build_prefix_index(self._aliases)
        self._compact_prefix_index = build_compact_prefix_index(self._aliases)

    @classmethod
    def from_csv(cls, path: str | Path) -> "InMemoryCatalogRepository":
        """Load catalog titles from a CSV file."""
        titles: list[CatalogTitle] = []
        with Path(path).open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                aliases = [
                    alias.strip()
                    for alias in row.get("aliases", "").split(";")
                    if alias.strip()
                ]
                year_value = row.get("year", "").strip()
                popularity_value = row.get("popularity", "").strip()
                titles.append(
                    CatalogTitle(
                        title_id=row["title_id"].strip(),
                        canonical_title=row["canonical_title"].strip(),
                        content_type=row["content_type"].strip(),
                        year=int(year_value) if year_value else None,
                        aliases=aliases,
                        popularity=float(popularity_value)
                        if popularity_value
                        else 0.0,
                        external_source=row.get("external_source") or None,
                        external_id=row.get("external_id") or None,
                    )
                )
        return cls(titles)

    def list_titles(self) -> list[CatalogTitle]:
        """Return all catalog titles."""
        return list(self._titles)

    def search(
        self,
        query: NormalizedQuery,
        limit: int = MAX_CANDIDATES,
        min_score: float = 0.55,
    ) -> list[TitleCandidate]:
        """Find candidate titles for normalized query variants."""
        best_by_title: dict[str, TitleCandidate] = {}

        for variant in query.variants:
            normalized_variant = normalize_text(variant)
            variant_compact = compact_text(variant)
            candidate_alias_ids = self._collect_candidate_alias_ids(
                normalized_variant=normalized_variant,
                variant_compact=variant_compact,
            )

            for alias_id in candidate_alias_ids:
                alias = self._aliases[alias_id]
                score, reasons = score_alias_match(
                    normalized_variant=normalized_variant,
                    variant_compact=variant_compact,
                    alias=alias,
                )
                if score < min_score:
                    continue
                current = best_by_title.get(alias.title.title_id)
                if current is None or score > current.search_score:
                    best_by_title[alias.title.title_id] = TitleCandidate(
                        title_id=alias.title.title_id,
                        canonical_title=alias.title.canonical_title,
                        content_type=alias.title.content_type,
                        year=alias.title.year,
                        matched_alias=alias.alias,
                        matched_variant=variant,
                        search_score=score,
                        rank_score=score,
                        reasons=reasons,
                    )

        return sorted(
            best_by_title.values(),
            key=lambda candidate: candidate.search_score,
            reverse=True,
        )[:limit]

    def _collect_candidate_alias_ids(
        self,
        normalized_variant: str,
        variant_compact: str,
    ) -> list[int]:
        """Collect a bounded alias candidate pool for one query variant."""
        alias_hits: Counter[int] = Counter()

        for alias_id in self._exact_alias_index.get(normalized_variant, ()):
            alias_hits[alias_id] += 100
        for alias_id in self._compact_alias_index.get(variant_compact, ()):
            alias_hits[alias_id] += 90

        tokens = extract_search_tokens(normalized_variant)
        for token in tokens:
            for alias_id in self._token_index.get(token, ()):
                alias_hits[alias_id] += 10
            prefix = token[:SEARCH_PREFIX_LENGTH]
            for alias_id in self._prefix_index.get(prefix, ()):
                alias_hits[alias_id] += 3

        compact_prefix = variant_compact[:SEARCH_PREFIX_LENGTH]
        if compact_prefix:
            for alias_id in self._compact_prefix_index.get(compact_prefix, ()):
                alias_hits[alias_id] += 4

        if not alias_hits:
            return []

        return nlargest(
            SEARCH_MAX_ALIAS_CANDIDATES,
            alias_hits,
            key=alias_hits.get,
        )


def build_alias_records(titles: Iterable[CatalogTitle]) -> list[AliasRecord]:
    """Build normalized aliases, including transliteration and year variants."""
    records: list[AliasRecord] = []
    seen: set[tuple[str, str]] = set()

    for title in titles:
        aliases = [title.canonical_title, *title.aliases]
        if title.year:
            aliases.append(f"{title.canonical_title} {title.year}")
        translit_aliases = [
            transliterate_ru_to_latin(alias)
            for alias in aliases
            if contains_cyrillic(alias)
        ]
        aliases.extend(translit_aliases)

        for alias in aliases:
            normalized_alias = normalize_text(alias)
            if not normalized_alias:
                continue
            key = (title.title_id, normalized_alias)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                AliasRecord(
                    title=title,
                    alias=alias,
                    normalized_alias=normalized_alias,
                    compact_alias=compact_text(normalized_alias),
                )
            )
    return records


def build_exact_alias_index(aliases: Sequence[AliasRecord]) -> dict[str, list[int]]:
    """Index aliases by exact normalized form."""
    index: dict[str, list[int]] = defaultdict(list)
    for alias_id, alias in enumerate(aliases):
        index[alias.normalized_alias].append(alias_id)
    return dict(index)


def build_compact_alias_index(aliases: Sequence[AliasRecord]) -> dict[str, list[int]]:
    """Index aliases by compact normalized form."""
    index: dict[str, list[int]] = defaultdict(list)
    for alias_id, alias in enumerate(aliases):
        if alias.compact_alias:
            index[alias.compact_alias].append(alias_id)
    return dict(index)


def build_token_index(aliases: Sequence[AliasRecord]) -> dict[str, list[int]]:
    """Index aliases by significant normalized tokens."""
    index: dict[str, list[int]] = defaultdict(list)
    for alias_id, alias in enumerate(aliases):
        for token in extract_search_tokens(alias.normalized_alias):
            index[token].append(alias_id)
    return dict(index)


def build_prefix_index(aliases: Sequence[AliasRecord]) -> dict[str, list[int]]:
    """Index aliases by token prefixes for typo-tolerant candidate selection."""
    index: dict[str, list[int]] = defaultdict(list)
    for alias_id, alias in enumerate(aliases):
        seen_prefixes: set[str] = set()
        for token in extract_search_tokens(alias.normalized_alias):
            prefix = token[:SEARCH_PREFIX_LENGTH]
            if not prefix or prefix in seen_prefixes:
                continue
            seen_prefixes.add(prefix)
            index[prefix].append(alias_id)
    return dict(index)


def build_compact_prefix_index(aliases: Sequence[AliasRecord]) -> dict[str, list[int]]:
    """Index aliases by compact prefix for glued or no-space user queries."""
    index: dict[str, list[int]] = defaultdict(list)
    for alias_id, alias in enumerate(aliases):
        prefix = alias.compact_alias[:SEARCH_PREFIX_LENGTH]
        if prefix:
            index[prefix].append(alias_id)
    return dict(index)


def extract_search_tokens(text: str) -> tuple[str, ...]:
    """Return significant tokens that should participate in alias retrieval."""
    tokens = []
    seen: set[str] = set()
    for token in normalize_text(text).split():
        if token in seen:
            continue
        if len(token) < SEARCH_TOKEN_MIN_LENGTH:
            continue
        if token in TITLE_NOISE_WORDS:
            continue
        if token.isdigit():
            continue
        seen.add(token)
        tokens.append(token)
    return tuple(tokens)


def contains_cyrillic(text: str) -> bool:
    """Return whether a string contains Cyrillic letters."""
    return any("а" <= char <= "я" or char == "ё" for char in text.lower())


def score_alias_match(
    normalized_variant: str,
    variant_compact: str,
    alias: AliasRecord,
) -> tuple[float, list[str]]:
    """Score a query variant against one catalog alias."""
    reasons: list[str] = []

    if normalized_variant == alias.normalized_alias:
        return 1.0, ["exact_alias_match"]

    if variant_compact and variant_compact == alias.compact_alias:
        return 0.98, ["compact_alias_match"]

    # fuzz.ratio returns a score between 0 and 100, so we divide by 100
    score = fuzz.ratio(
        normalized_variant,
        alias.normalized_alias,
    ) / 100.0
    compact_score = fuzz.ratio(
        variant_compact,
        alias.compact_alias,
    ) / 100.0
    score = max(score, compact_score)

    if len(alias.normalized_alias) >= 4 and alias.normalized_alias in normalized_variant:
        score = max(score, 0.93)
        reasons.append("alias_inside_query")
    if len(normalized_variant) >= 4 and normalized_variant in alias.normalized_alias:
        score = max(score, 0.88)
        reasons.append("query_inside_alias")
    if len(alias.compact_alias) >= 4 and alias.compact_alias in variant_compact:
        score = max(score, 0.92)
        reasons.append("compact_alias_inside_query")
    if len(variant_compact) >= 4 and variant_compact in alias.compact_alias:
        score = max(score, 0.87)
        reasons.append("compact_query_inside_alias")

    if score >= 0.85:
        reasons.append("high_fuzzy_similarity")
    elif score >= 0.7:
        reasons.append("medium_fuzzy_similarity")
    else:
        reasons.append("low_fuzzy_similarity")

    return round(min(score, 1.0), 4), reasons
