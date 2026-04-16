"""Helpers for converting IMDb bulk dumps into the local catalog format."""

from __future__ import annotations

import csv
import gzip
import math
import sys
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from app.preprocessing.normalizer import normalize_text
from app.schemas import CatalogTitle, ContentType

DEFAULT_IMDB_TITLE_TYPES = ("movie", "tvMovie", "tvSeries", "tvMiniSeries")
DEFAULT_IMDB_AKA_REGIONS = ("RU", "US", "GB")
DEFAULT_IMDB_AKA_LANGUAGES = ("ru", "en")
DEFAULT_IMDB_ALLOWED_AKA_TYPES = ("imdbDisplay", "original", "alternative")
DEFAULT_IMDB_BLOCKED_AKA_TYPES = ("working", "tv", "dvd", "video", "festival")
DEFAULT_IMDB_BLOCKED_ATTRIBUTE_KEYWORDS = (
    "review",
    "poster",
    "video box",
    "dvd box",
    "premiere",
    "pre-release",
    "copyright",
    "subtitle",
    "long title",
    "complete title",
    "informal",
)


@dataclass(slots=True)
class ImdbImportConfig:
    """Configuration for importing IMDb titles into the local catalog."""

    min_votes: int = 200
    max_titles: int | None = None
    max_aliases_per_title: int = 30
    min_alias_length: int = 3
    max_alias_tokens: int = 8
    max_titles_per_normalized_alias: int = 1
    title_types: tuple[str, ...] = DEFAULT_IMDB_TITLE_TYPES
    aka_regions: tuple[str, ...] = DEFAULT_IMDB_AKA_REGIONS
    aka_languages: tuple[str, ...] = DEFAULT_IMDB_AKA_LANGUAGES
    allowed_aka_types: tuple[str, ...] = DEFAULT_IMDB_ALLOWED_AKA_TYPES
    blocked_aka_types: tuple[str, ...] = DEFAULT_IMDB_BLOCKED_AKA_TYPES
    blocked_attribute_keywords: tuple[str, ...] = DEFAULT_IMDB_BLOCKED_ATTRIBUTE_KEYWORDS
    include_original_title_akas: bool = True


@dataclass(slots=True)
class ImdbTitleRecord:
    """Intermediate IMDb title record used before conversion to CatalogTitle."""

    title_id: str
    canonical_title: str
    content_type: ContentType
    year: int | None
    votes: int
    aliases: list[str] = field(default_factory=list)
    _alias_index: set[str] = field(default_factory=set)

    def add_alias(self, alias: str, max_aliases_per_title: int) -> None:
        """Add one alias while preserving order and avoiding duplicates."""
        cleaned_alias = clean_tsv_value(alias)
        if not cleaned_alias:
            return
        alias_key = cleaned_alias.casefold()
        if alias_key in self._alias_index:
            return
        if len(self.aliases) >= max_aliases_per_title:
            return
        self.aliases.append(cleaned_alias)
        self._alias_index.add(alias_key)


def clean_tsv_value(value: str | None) -> str:
    """Normalize one IMDb TSV cell, treating ``\\N`` as an empty value."""
    if value is None:
        return ""
    cleaned = value.strip()
    if cleaned == "\\N":
        return ""
    return cleaned


def parse_optional_int(value: str | None) -> int | None:
    """Parse an optional IMDb integer cell."""
    cleaned = clean_tsv_value(value)
    if not cleaned:
        return None
    return int(cleaned)


def map_imdb_content_type(title_type: str, genres: str | None) -> ContentType | None:
    """Map IMDb title types and genres into the local content taxonomy."""
    normalized_type = clean_tsv_value(title_type)
    genre_values = {
        genre.strip()
        for genre in clean_tsv_value(genres).split(",")
        if genre.strip()
    }
    is_animation = "Animation" in genre_values

    if normalized_type in {"tvSeries", "tvMiniSeries"}:
        return ContentType.ANIMATED_SERIES if is_animation else ContentType.SERIES
    if normalized_type in {"movie", "tvMovie"}:
        return ContentType.CARTOON if is_animation else ContentType.FILM
    return None


def iter_gzip_tsv_rows(path: str | Path) -> Iterator[dict[str, str]]:
    """Yield parsed rows from a gzipped TSV file."""
    configure_csv_field_limit()
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        for row in reader:
            yield row


def configure_csv_field_limit() -> None:
    """Raise the CSV field limit high enough for large IMDb AKA rows."""
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


def load_imdb_votes(
    ratings_path: str | Path | None,
    min_votes: int,
) -> tuple[dict[str, int], int]:
    """Load IMDb vote counts filtered by the configured threshold."""
    if ratings_path is None:
        return {}, 0

    selected_votes: dict[str, int] = {}
    max_votes = 0
    for row in iter_gzip_tsv_rows(ratings_path):
        votes = parse_optional_int(row.get("numVotes"))
        if votes is None or votes < min_votes:
            continue
        title_id = clean_tsv_value(row.get("tconst"))
        if not title_id:
            continue
        selected_votes[title_id] = votes
        max_votes = max(max_votes, votes)
    return selected_votes, max_votes


def load_imdb_catalog(
    basics_path: str | Path,
    akas_path: str | Path,
    ratings_path: str | Path | None = None,
    config: ImdbImportConfig | None = None,
) -> list[CatalogTitle]:
    """Load IMDb dumps and convert them into local ``CatalogTitle`` records."""
    import_config = config or ImdbImportConfig()
    selected_votes, max_votes = load_imdb_votes(
        ratings_path=ratings_path,
        min_votes=import_config.min_votes,
    )
    titles = load_imdb_title_records(
        basics_path=basics_path,
        selected_votes=selected_votes,
        use_vote_filter=ratings_path is not None,
        config=import_config,
    )
    enrich_imdb_aliases(
        titles=titles,
        akas_path=akas_path,
        config=import_config,
    )

    catalog = [
        CatalogTitle(
            title_id=title.title_id,
            canonical_title=title.canonical_title,
            content_type=title.content_type,
            year=title.year,
            popularity=build_popularity_score(title.votes, max_votes),
            aliases=title.aliases[1:],
            external_source="imdb",
            external_id=title.title_id,
        )
        for title in titles.values()
    ]
    filter_ambiguous_aliases(catalog, import_config)
    return catalog


def load_imdb_title_records(
    basics_path: str | Path,
    selected_votes: dict[str, int],
    use_vote_filter: bool,
    config: ImdbImportConfig,
) -> dict[str, ImdbTitleRecord]:
    """Load and filter IMDb ``title.basics`` rows into intermediate records."""
    titles: dict[str, ImdbTitleRecord] = {}

    for row in iter_gzip_tsv_rows(basics_path):
        title_id = clean_tsv_value(row.get("tconst"))
        title_type = clean_tsv_value(row.get("titleType"))
        if title_type not in config.title_types:
            continue
        if clean_tsv_value(row.get("isAdult")) == "1":
            continue
        if use_vote_filter and title_id not in selected_votes:
            continue

        content_type = map_imdb_content_type(title_type, row.get("genres"))
        if content_type is None:
            continue

        canonical_title = clean_tsv_value(row.get("primaryTitle"))
        if not canonical_title:
            continue
        votes = selected_votes.get(title_id, 0)
        record = ImdbTitleRecord(
            title_id=title_id,
            canonical_title=canonical_title,
            content_type=content_type,
            year=parse_optional_int(row.get("startYear")),
            votes=votes,
        )
        record.add_alias(canonical_title, config.max_aliases_per_title)
        original_title = clean_tsv_value(row.get("originalTitle"))
        if original_title:
            record.add_alias(original_title, config.max_aliases_per_title)
        titles[title_id] = record

        if config.max_titles is not None and len(titles) >= config.max_titles:
            break

    return titles


def enrich_imdb_aliases(
    titles: dict[str, ImdbTitleRecord],
    akas_path: str | Path,
    config: ImdbImportConfig,
) -> None:
    """Attach selected IMDb AKA rows to the already filtered title set."""
    if not titles:
        return

    allowed_regions = {value.casefold() for value in config.aka_regions}
    allowed_languages = {value.casefold() for value in config.aka_languages}

    for row in iter_gzip_tsv_rows(akas_path):
        title_id = clean_tsv_value(row.get("titleId"))
        if title_id not in titles:
            continue

        title = titles[title_id]
        alias = clean_tsv_value(row.get("title"))
        if not alias:
            continue
        if not should_keep_imdb_alias(alias, row, title, config):
            continue

        region = clean_tsv_value(row.get("region")).casefold()
        language = clean_tsv_value(row.get("language")).casefold()
        is_original = clean_tsv_value(row.get("isOriginalTitle")) == "1"
        keep_alias = (
            region in allowed_regions
            or language in allowed_languages
            or (config.include_original_title_akas and is_original)
        )
        if not keep_alias:
            continue

        title.add_alias(alias, config.max_aliases_per_title)


def should_keep_imdb_alias(
    alias: str,
    row: dict[str, str],
    title: ImdbTitleRecord,
    config: ImdbImportConfig,
) -> bool:
    """Return whether one IMDb AKA row looks useful for query matching."""
    normalized_alias = normalize_text(alias)
    if not normalized_alias:
        return False
    if len(normalized_alias) < config.min_alias_length:
        return False

    tokens = normalized_alias.split()
    if len(tokens) > config.max_alias_tokens:
        return False
    if sum(char.isalpha() for char in normalized_alias) < config.min_alias_length:
        return False

    types = parse_imdb_list_field(row.get("types"))
    attributes = parse_imdb_list_field(row.get("attributes"))
    allowed_types = {value.casefold() for value in config.allowed_aka_types}
    blocked_types = {value.casefold() for value in config.blocked_aka_types}
    blocked_attribute_keywords = {
        value.casefold() for value in config.blocked_attribute_keywords
    }

    if types and not any(value in allowed_types for value in types):
        return False
    if any(value in blocked_types for value in types):
        return False
    for attribute in attributes:
        if any(keyword in attribute for keyword in blocked_attribute_keywords):
            return False

    canonical_normalized = normalize_text(title.canonical_title)
    if normalized_alias == canonical_normalized:
        return True
    if len(tokens) == 1 and tokens[0] == str(title.year or ""):
        return False
    return True


def parse_imdb_list_field(value: str | None) -> tuple[str, ...]:
    """Parse IMDb list-like cells such as ``types`` and ``attributes``."""
    cleaned = clean_tsv_value(value).replace("\x02", ",")
    if not cleaned:
        return ()
    return tuple(
        part.strip().casefold()
        for part in cleaned.split(",")
        if part.strip()
    )


def filter_ambiguous_aliases(
    titles: list[CatalogTitle],
    config: ImdbImportConfig,
) -> None:
    """Drop highly ambiguous aliases from less popular titles."""
    if config.max_titles_per_normalized_alias < 1:
        return

    alias_to_titles: dict[str, list[int]] = defaultdict(list)
    for index, title in enumerate(titles):
        for alias in title.aliases:
            normalized_alias = normalize_text(alias)
            if normalized_alias:
                alias_to_titles[normalized_alias].append(index)

    keep_aliases: list[set[str]] = [set() for _ in titles]
    for normalized_alias, title_indexes in alias_to_titles.items():
        if len(title_indexes) <= config.max_titles_per_normalized_alias:
            for title_index in title_indexes:
                keep_aliases[title_index].add(normalized_alias)
            continue

        sorted_indexes = sorted(
            title_indexes,
            key=lambda title_index: (
                titles[title_index].popularity,
                titles[title_index].year or 0,
            ),
            reverse=True,
        )
        for title_index in sorted_indexes[: config.max_titles_per_normalized_alias]:
            keep_aliases[title_index].add(normalized_alias)

    for title, allowed_aliases in zip(titles, keep_aliases, strict=True):
        title.aliases = [
            alias
            for alias in title.aliases
            if normalize_text(alias) in allowed_aliases
        ]


def build_popularity_score(votes: int, max_votes: int) -> float:
    """Convert raw IMDb votes into a normalized ``0..1`` popularity score."""
    if votes <= 0 or max_votes <= 0:
        return 0.0
    popularity = math.log1p(votes) / math.log1p(max_votes)
    return round(min(max(popularity, 0.0), 1.0), 4)


def write_catalog_csv(titles: list[CatalogTitle], output_path: str | Path) -> None:
    """Write the converted IMDb catalog into the local CSV schema."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "title_id",
                "canonical_title",
                "content_type",
                "year",
                "popularity",
                "aliases",
                "external_source",
                "external_id",
            ],
        )
        writer.writeheader()
        for title in titles:
            writer.writerow(
                {
                    "title_id": title.title_id,
                    "canonical_title": title.canonical_title,
                    "content_type": title.content_type.value,
                    "year": title.year or "",
                    "popularity": title.popularity,
                    "aliases": ";".join(title.aliases),
                    "external_source": title.external_source or "",
                    "external_id": title.external_id or "",
                }
            )
