"""Title retrieval helpers for robust query-to-title fallback handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from collections import defaultdict

from rapidfuzz import fuzz, process

from preprocessing import normalize
from title_extraction import TITLE_FUNCTION_WORDS, TITLE_STOP_WORDS, extract_title_candidate

GENERIC_CANDIDATE_WORDS = {
    "фильм",
    "фильмы",
    "сериал",
    "сериалы",
    "кино",
    "мультфильм",
    "мультфильмы",
    "мультсериал",
    "мультсериалы",
    "аниме",
    "дорама",
    "новинки",
    "новинка",
    "лучшие",
    "лучший",
    "русское",
    "русский",
    "русские",
    "корейские",
    "турецкие",
    "американские",
    "онлайн",
    "смотреть",
    "скачать",
    "бесплатно",
}
LEADING_CONTEXT_WORDS = {
    "смотреть",
    "скачать",
    "фильм",
    "фильма",
    "фильмы",
    "кино",
    "сериал",
    "сериала",
    "сериалы",
    "мультфильм",
    "мультсериал",
    "аниме",
    "дорама",
    "дораму",
}
BROAD_CONTEXT_WORDS = {
    "фильм",
    "фильмы",
    "сериал",
    "сериалы",
    "кино",
    "мультфильм",
    "мультфильмы",
    "мультсериал",
    "мультсериалы",
    "аниме",
    "дорама",
    "дорамы",
    "новинки",
    "новинка",
    "лучшие",
    "лучший",
    "русские",
    "русское",
    "корейские",
    "турецкие",
    "американские",
}
TRAILING_CONTEXT_WORDS = {
    "смотреть",
    "скачать",
    "онлайн",
    "бесплатно",
    "торрент",
    "magnet",
    "hd",
    "hdrip",
    "bdrip",
    "webrip",
    "телефон",
    "смартфон",
    "русский",
    "русская",
    "русское",
    "озвучка",
    "озвучкой",
    "озвучке",
    "перевод",
    "субтитры",
    "субтитрами",
    "качестве",
    "качествe",
    "узбек",
    "узбекча",
    "тилида",
    "вк",
    "tv",
    "ru",
    "net",
}

TRAILING_CONTEXT_PATTERN = re.compile(
    r"(?:\b(?:сезон|серия|эпизод|season|episode|tv|тв|torrent|magnet)\b|\b\d+\b)$",
    re.IGNORECASE,
)


@dataclass
class TitleRetrievalResult:
    """Container for retrieval-first title candidate resolution."""

    title: str
    content_type: str
    confidence: float
    source: str
    raw_candidate: str


class TitleRetriever:
    """Resolve raw title candidates and optional catalog corrections."""

    def __init__(self, title_dict: dict[str, dict]) -> None:
        """Build lightweight lookup tables from a franchise-style dictionary."""
        self._variant_to_title: dict[str, str] = {}
        self._catalog_to_title: dict[str, str] = {}
        self._title_to_content_type: dict[str, str] = {}
        self._token_to_catalog: dict[str, set[str]] = defaultdict(set)

        for title, data in title_dict.items():
            normalized_title = normalize(title)
            if not normalized_title:
                continue

            self._title_to_content_type[title] = str(data.get("contentType", "")).strip().lower()
            self._catalog_to_title.setdefault(normalized_title, title)
            self._variant_to_title.setdefault(normalized_title, title)

            for variant in data.get("variants", []):
                normalized_variant = normalize(variant)
                if normalized_variant:
                    self._catalog_to_title.setdefault(normalized_variant, title)
                    self._variant_to_title.setdefault(normalized_variant, title)

        self._catalog_strings = list(self._catalog_to_title.keys())
        for catalog_string in self._catalog_strings:
            for token in catalog_string.split():
                if token and token not in GENERIC_CANDIDATE_WORDS and not token.isdigit():
                    self._token_to_catalog[token].add(catalog_string)

    def retrieve(self, query: str) -> TitleRetrievalResult:
        """Return a raw candidate plus optional exact/fuzzy catalog correction."""
        candidate_variants = self.generate_candidate_variants(query)
        if not candidate_variants:
            return TitleRetrievalResult("", "", 0.0, "", "")
        has_broad_context = self._has_broad_context(query)

        for raw_candidate in candidate_variants:
            if raw_candidate in self._variant_to_title:
                title = self._variant_to_title[raw_candidate]
                return TitleRetrievalResult(
                    title=title,
                    content_type=self._title_to_content_type.get(title, ""),
                    confidence=0.99,
                    source="catalog_exact",
                    raw_candidate=raw_candidate,
                )

            best_match = self._find_fuzzy_catalog_match(raw_candidate)
            if best_match:
                return best_match

        best_raw_candidate = candidate_variants[0]
        if has_broad_context and len(best_raw_candidate.split()) == 1:
            return TitleRetrievalResult("", "", 0.0, "", "")

        return TitleRetrievalResult("", "", 0.0, "raw", best_raw_candidate)

    def extract_candidate(self, query: str) -> str:
        """Extract the cleanest raw title span available from a query."""
        candidate = extract_title_candidate(query)
        if not candidate:
            candidate = normalize(query)
        candidate = self._strip_leading_context(candidate)
        candidate = self._strip_trailing_context(candidate)
        candidate = self._strip_leading_context(candidate)
        candidate = re.sub(r"\s+", " ", candidate).strip()
        return candidate

    def generate_candidate_variants(self, query: str) -> list[str]:
        """Generate plausible title prefixes from the extracted query span."""
        candidate = self.extract_candidate(query)
        if not self.is_plausible_candidate(candidate):
            return []

        tokens = candidate.split()
        variants: list[str] = []
        seen: set[str] = set()

        for end in range(len(tokens), 0, -1):
            prefix_tokens = tokens[:end]
            while prefix_tokens and (
                prefix_tokens[-1].isdigit()
                or prefix_tokens[-1] in TITLE_STOP_WORDS
                or prefix_tokens[-1] in TITLE_FUNCTION_WORDS
            ):
                prefix_tokens = prefix_tokens[:-1]

            if not prefix_tokens:
                continue

            prefix = " ".join(prefix_tokens).strip()
            if prefix and prefix not in seen and self.is_plausible_candidate(prefix):
                variants.append(prefix)
                seen.add(prefix)

        return variants

    def is_plausible_candidate(self, candidate: str) -> bool:
        """Check whether a raw span is specific enough to be used as a title."""
        if not candidate:
            return False

        normalized_candidate = normalize(candidate)
        if len(normalized_candidate) < 2:
            return False

        tokens = [token for token in normalized_candidate.split() if token]
        if not tokens:
            return False

        meaningful_tokens = [
            token
            for token in tokens
            if token not in GENERIC_CANDIDATE_WORDS and not token.isdigit()
        ]
        if not meaningful_tokens:
            return False

        if len(meaningful_tokens) == 1 and len(meaningful_tokens[0]) < 3:
            return False

        return True

    def is_match_consistent(self, raw_candidate: str, proposed_title: str) -> bool:
        """Reject weak catalog matches that drift too far from the query span."""
        normalized_candidate = normalize(raw_candidate)
        normalized_title = normalize(proposed_title)
        if not normalized_candidate or not normalized_title:
            return False

        if normalized_candidate == normalized_title:
            return True

        candidate_tokens = set(normalized_candidate.split())
        title_tokens = set(normalized_title.split())
        overlap = candidate_tokens & title_tokens
        strong_overlap = {
            token
            for token in overlap
            if token not in GENERIC_CANDIDATE_WORDS and len(token) >= 4
        }
        if strong_overlap:
            return True
        if overlap:
            if len(candidate_tokens) == 1 and len(title_tokens) == 1:
                return True

        similarity = fuzz.ratio(normalized_candidate, normalized_title)
        if len(candidate_tokens) == 1:
            token = next(iter(candidate_tokens))
            return len(token) >= 6 and similarity >= 72
        return similarity >= 84

    def is_query_compatible(self, query: str, proposed_title: str) -> bool:
        """Check whether a proposed title is sufficiently supported by the query."""
        normalized_query = normalize(query)
        normalized_title = normalize(proposed_title)
        if not normalized_query or not normalized_title:
            return False

        query_tokens = [
            token
            for token in normalized_query.split()
            if token and token not in TITLE_STOP_WORDS
        ]
        title_tokens = [
            token
            for token in normalized_title.split()
            if token and token not in TITLE_STOP_WORDS and not token.isdigit()
        ]
        if not query_tokens or not title_tokens:
            return False

        exact_hits = sum(token in query_tokens for token in title_tokens)
        strong_exact_hits = sum(token in query_tokens and len(token) >= 4 for token in title_tokens)
        coverage = exact_hits / max(len(title_tokens), 1)
        if coverage >= 0.6 or strong_exact_hits >= min(2, len(title_tokens)):
            return True

        phrase_score = fuzz.token_set_ratio(normalized_title, normalized_query)
        partial_phrase_score = fuzz.partial_ratio(normalized_title, normalized_query)
        if len(title_tokens) >= 2 and phrase_score >= 78 and partial_phrase_score >= 82:
            return True

        if len(title_tokens) == 1:
            token = title_tokens[0]
            best_token_score = max(
                fuzz.ratio(token, query_token)
                for query_token in query_tokens
            )
            return len(token) >= 5 and best_token_score >= 70

        fuzzy_hits = 0
        for title_token in title_tokens:
            best_score = max(fuzz.ratio(title_token, query_token) for query_token in query_tokens)
            if best_score >= 86:
                fuzzy_hits += 1

        return fuzzy_hits / max(len(title_tokens), 1) >= 0.5 and partial_phrase_score >= 74

    def should_accept_raw_candidate(self, query: str, raw_candidate: str) -> bool:
        """Return True when a raw fallback candidate is specific enough to emit."""
        normalized_candidate = normalize(raw_candidate)
        if not self.is_plausible_candidate(normalized_candidate):
            return False

        tokens = normalized_candidate.split()
        if len(tokens) > 5:
            return False

        meaningful_tokens = [
            token
            for token in tokens
            if token not in GENERIC_CANDIDATE_WORDS
            and token not in TITLE_STOP_WORDS
            and token not in TRAILING_CONTEXT_WORDS
            and not token.isdigit()
        ]
        if not meaningful_tokens:
            return False

        if tokens[-1] in TRAILING_CONTEXT_WORDS or tokens[-1] in GENERIC_CANDIDATE_WORDS:
            return False

        if any(token.isdigit() for token in tokens[:-1]):
            return False

        if len(meaningful_tokens) == 1 and len(meaningful_tokens[0]) < 4:
            return False

        if self._has_broad_context(query) and len(meaningful_tokens) == 1 and len(meaningful_tokens[0]) < 5:
            return False

        return True

    def _find_fuzzy_catalog_match(self, raw_candidate: str) -> TitleRetrievalResult | None:
        """Find a conservative fuzzy catalog correction for a raw candidate."""
        shortlist = self._build_catalog_shortlist(raw_candidate)
        if not shortlist:
            return None

        score_cutoff = 90 if len(raw_candidate.split()) == 1 else 84
        match = process.extractOne(
            raw_candidate,
            shortlist,
            scorer=fuzz.WRatio,
            score_cutoff=score_cutoff,
        )
        if not match:
            return None

        matched_string, matched_score, _ = match
        title = self._catalog_to_title.get(matched_string, "")
        if not title or not self.is_match_consistent(raw_candidate, title):
            return None

        return TitleRetrievalResult(
            title=title,
            content_type=self._title_to_content_type.get(title, ""),
            confidence=float(matched_score) / 100.0,
            source="catalog_fuzzy",
            raw_candidate=raw_candidate,
        )

    def _build_catalog_shortlist(self, raw_candidate: str) -> list[str]:
        """Select a small fuzzy-match shortlist using overlapping meaningful tokens."""
        candidate_tokens = [
            token
            for token in normalize(raw_candidate).split()
            if token and token not in GENERIC_CANDIDATE_WORDS and not token.isdigit()
        ]
        if not candidate_tokens:
            return []

        shortlist: set[str] = set()
        for token in candidate_tokens:
            shortlist.update(self._token_to_catalog.get(token, set()))

        if shortlist:
            return sorted(shortlist)

        # For typo-heavy single-word candidates, use only entries with the same first letter.
        if len(candidate_tokens) == 1:
            token = candidate_tokens[0]
            return [
                catalog_string
                for catalog_string in self._catalog_strings
                if catalog_string[:1] == token[:1]
            ]

        return []

    @staticmethod
    def _strip_trailing_context(candidate: str) -> str:
        """Remove common trailing non-title context such as season numbers."""
        cleaned = normalize(candidate)
        if not cleaned:
            return ""

        tokens = cleaned.split()
        while len(tokens) > 1:
            tail = tokens[-1]
            joined_tail = " ".join(tokens[-2:])
            if tail.isdigit() or TRAILING_CONTEXT_PATTERN.search(tail) or TRAILING_CONTEXT_PATTERN.search(joined_tail):
                tokens.pop()
                continue
            if tail in TITLE_STOP_WORDS or tail in GENERIC_CANDIDATE_WORDS or tail in TRAILING_CONTEXT_WORDS:
                tokens.pop()
                continue
            break

        return " ".join(tokens).strip()

    @staticmethod
    def _strip_leading_context(candidate: str) -> str:
        """Remove generic leading query context before the actual title span."""
        cleaned = normalize(candidate)
        if not cleaned:
            return ""

        tokens = cleaned.split()
        while len(tokens) > 1:
            head = tokens[0]
            if (
                head in LEADING_CONTEXT_WORDS
                or head in TITLE_STOP_WORDS
                or head.startswith(("дорам", "фильм", "сериал", "аниме", "мульт"))
            ):
                tokens.pop(0)
                continue
            break

        while len(tokens) > 1 and tokens[0].isdigit():
            tokens.pop(0)

        return " ".join(tokens).strip()

    @staticmethod
    def _has_broad_context(query: str) -> bool:
        """Return True when the full query looks like a broad category search."""
        query_words = set(normalize(query).split())
        return bool(query_words & BROAD_CONTEXT_WORDS)
