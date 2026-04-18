"""Supplemental curated title sources used to expand title coverage."""

import csv
import re
from pathlib import Path

from preprocessing import normalize


def clean_title_raw(text: str) -> str:
    """Strip noise from raw titles (e.g. years in parens, trailing junk)."""
    if not text: return ""
    # Remove (2024), (фильм), etc.
    t = re.sub(r'\s*\(.*?\)', '', text)
    # Remove year if it's at the end without parens
    t = re.sub(r'\s+\d{4}$', '', t)
    # Strip common prefixes/suffixes
    t = re.sub(r'^(?:фильм|сериал|мультфильм|аниме)\s+', '', t, flags=re.IGNORECASE)
    return t.strip()


def load_kinopoisk_top250(path: str | Path = "kinopoisk-top250.csv") -> dict:
    """Load Kinopoisk Top 250 as a franchise-style title dictionary."""
    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    title_dict: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_title = clean_title_raw(str(row.get("movie", "")).strip().lower())
            if not raw_title:
                continue

            year = str(row.get("year", "")).strip()
            description = str(row.get("description", "")).lower()
            
            # Simple content type detection from description
            if "мультсериал" in description:
                ct = "мультсериал"
            elif "мультфильм" in description:
                ct = "мультфильм"
            elif "сериал" in description:
                ct = "сериал"
            else:
                ct = "фильм"

            item_season = None
            season_match = re.search(r'(\d+)\s*(?:сезон|season|s\b)', description, re.IGNORECASE)
            if season_match:
                item_season = season_match.group(1)

            variants = _build_title_variants(raw_title, year)
            title_dict[raw_title] = {
                "contentType": ct,
                "variants": sorted(variants),
                "count": 1,
                "year": year if year else None,
                "season": item_season,
            }

    return title_dict
def load_enriched_dataset(path: str | Path = "artifacts/enriched_films.csv") -> dict:
    """Load enriched film dataset with Russian and English titles."""
    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    title_dict: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            ru_title = clean_title_raw(str(row.get("title_ru", "")).strip().lower())
            en_title = clean_title_raw(str(row.get("title_en", "")).strip().lower())
            
            if not ru_title and not en_title:
                continue

            title = ru_title if ru_title else en_title

            # FILTER: avoid noisy short titles
            words_ru = ru_title.split()
            if len(words_ru) < 2 and len(ru_title) <= 5:
                continue

            # Determine content type
            genres = str(row.get("Genres", "")).lower()
            description = str(row.get("Description Kinopoisk", "")).lower()
            is_series = "сериал" in genres or "сериал" in description or "многосерийный" in description
            content_type = "сериал" if is_series else "фильм"

            year = str(row.get("Year", "")).replace(".0", "").strip()
            
            # Base variants from Russian title
            variants = _build_title_variants(ru_title, year)
            
            # Add English title as a variant if available
            if en_title:
                variants.update(_build_title_variants(en_title, year))
                
            # Extract year/season
            year_match = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', f"{ru_title} {description}")
            item_year = year_match.group(1) if year_match else None
            season_match = re.search(r'(\d+)\s*(?:сезон|season|s\b)', f"{ru_title} {description}", re.IGNORECASE)
            item_season = season_match.group(1) if season_match else None

            title_dict[ru_title] = {
                "contentType": content_type,
                "variants": sorted(list(variants)),
                "count": 1,
                "year": item_year,
                "season": item_season,
            }

    return title_dict


def merge_title_dicts(primary: dict, supplemental: dict) -> dict:
    """Merge two franchise-style dictionaries by normalized title."""
    if not supplemental:
        return dict(primary)

    merged = {
        title: {
            "contentType": data.get("contentType", ""),
            "variants": list(data.get("variants", [])),
            "count": int(data.get("count", 0)),
        }
        for title, data in primary.items()
    }
    normalized_index = {normalize(title): title for title in merged}

    for title, data in supplemental.items():
        normalized_title = normalize(title)
        target_title = normalized_index.get(normalized_title, title)
        preferred_title = _choose_canonical_title(target_title, title)

        if preferred_title != target_title and target_title in merged:
            merged[preferred_title] = merged.pop(target_title)
            normalized_index[normalized_title] = preferred_title
            target_title = preferred_title

        if target_title not in merged:
            merged[target_title] = {
                "contentType": data.get("contentType", ""),
                "variants": list(data.get("variants", [])),
                "count": int(data.get("count", 0)),
                "year": data.get("year"),
                "season": data.get("season"),
            }
            normalized_index[normalized_title] = target_title
            continue

        existing = merged[target_title]
        variant_set = set(existing.get("variants", []))
        variant_set.update(data.get("variants", []))
        existing["variants"] = sorted(variant_set)
        existing["count"] = int(existing.get("count", 0)) + int(data.get("count", 0))
        
        if not existing.get("year"): existing["year"] = data.get("year")
        if not existing.get("season"): existing["season"] = data.get("season")

        existing_ct = existing.get("contentType", "")
        incoming_ct = data.get("contentType", "")
        if (not existing_ct or existing_ct == "прочее") and incoming_ct:
            existing["contentType"] = incoming_ct

    return merged


def _build_title_variants(title: str, year: str) -> set[str]:
    """Build normalized aliases for a curated title."""
    normalized = normalize(title)
    variants = {title, normalized}

    if "ё" in title:
        variants.add(title.replace("ё", "е"))
    if "ё" in normalized:
        variants.add(normalized.replace("ё", "е"))

    if year.isdigit():
        variants.add(f"{title} {year}")
        variants.add(f"{normalized} {year}")

    return {variant.strip() for variant in variants if variant.strip()}


def _choose_canonical_title(current_title: str, incoming_title: str) -> str:
    """Choose the better display form for the same normalized title."""
    if current_title == incoming_title:
        return current_title

    current_normalized = normalize(current_title)
    incoming_normalized = normalize(incoming_title)
    if current_normalized != incoming_normalized:
        return current_title

    current_is_plain = current_title == current_normalized
    incoming_is_plain = incoming_title == incoming_normalized

    if current_is_plain and not incoming_is_plain:
        return incoming_title
    if len(incoming_title) > len(current_title) and incoming_title.replace(" ", "") != current_title.replace(" ", ""):
        return incoming_title
    return current_title
