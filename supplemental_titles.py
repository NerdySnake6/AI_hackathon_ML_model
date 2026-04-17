"""Supplemental curated title sources used to expand title coverage."""

import csv
from pathlib import Path

from preprocessing import normalize


def load_kinopoisk_top250(path: str | Path = "kinopoisk-top250.csv") -> dict:
    """Load Kinopoisk Top 250 as a franchise-style title dictionary."""
    csv_path = Path(path)
    if not csv_path.exists():
        return {}

    title_dict: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_title = str(row.get("movie", "")).strip().lower()
            if not raw_title:
                continue

            year = str(row.get("year", "")).strip()
            variants = _build_title_variants(raw_title, year)
            title_dict[raw_title] = {
                "contentType": "фильм",
                "variants": sorted(variants),
                "count": 1,
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
            }
            normalized_index[normalized_title] = target_title
            continue

        existing = merged[target_title]
        variant_set = set(existing.get("variants", []))
        variant_set.update(data.get("variants", []))
        existing["variants"] = sorted(variant_set)
        existing["count"] = int(existing.get("count", 0)) + int(data.get("count", 0))

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
