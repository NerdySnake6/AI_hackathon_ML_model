"""
Franchise dictionary module: build canonical title→metadata mapping,
cluster variant spellings, and perform fuzzy matching.
"""
import os
import json
import re
from collections import Counter
from rapidfuzz import process, fuzz

from preprocessing import normalize


def build_franchise_dict(df) -> dict:
    """
    Build {canonical_title: {contentType, variants, count, queries}} from train data.
    Variants are clustered using fuzzy similarity.
    """
    # Group by Title where Title is not null
    title_data = {}
    for _, row in df[df['Title'].notna()].iterrows():
        title = row['Title'].strip().lower()
        content_type = row.get('ContentType', '')
        query = str(row.get('QueryText', '')).lower()

        if title not in title_data:
            title_data[title] = {'variants': set(), 'content_types': Counter(), 'count': 0, 'queries': []}

        title_data[title]['count'] += 1
        title_data[title]['content_types'][content_type] += 1
        title_data[title]['queries'].append(query)

        # Add query as variant if different from title
        norm_query = re.sub(r'[^a-zа-яё0-9\s]', '', query).strip()
        if norm_query and norm_query != title:
            # Check if query is a variant (contains key words of title)
            title_words = set(title.split())
            query_words = set(norm_query.split())
            if title_words & query_words:  # some overlap
                title_data[title]['variants'].add(norm_query)

        # Also add the title itself as variant
        title_data[title]['variants'].add(title)

    # Merge similar titles using fuzzy clustering
    titles = list(title_data.keys())
    merged = set()
    final = {}

    for i, t1 in enumerate(titles):
        if t1 in merged:
            continue
        cluster = {t1}
        for j in range(i + 1, len(titles)):
            t2 = titles[j]
            if t2 in merged:
                continue
            # Check fuzzy similarity
            score = fuzz.ratio(t1, t2)
            if score >= 85 or (t1 in t2 or t2 in t1) and score >= 70:
                cluster.add(t2)
                merged.add(t2)

        # Merge cluster into canonical (most frequent) title
        best_title = max(cluster, key=lambda t: title_data[t]['count'])
        all_variants = set()
        all_content_types = Counter()
        all_queries = []
        total_count = 0

        for t in cluster:
            merged.add(t)
            all_variants |= title_data[t]['variants']
            all_content_types += title_data[t]['content_types']
            all_queries.extend(title_data[t]['queries'])
            total_count += title_data[t]['count']

        final[best_title] = {
            'contentType': all_content_types.most_common(1)[0][0] if all_content_types else '',
            'variants': list(all_variants),
            'count': total_count,
        }

    return final


def save_artifacts(franchise_dict: dict, path: str = "artifacts/franchise_dict.json"):
    """Save the franchise dictionary artifact to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert sets to lists for JSON
    serializable = {}
    for title, data in franchise_dict.items():
        serializable[title] = {
            'contentType': data['contentType'],
            'variants': data['variants'],
            'count': data['count'],
        }
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)


def load_artifacts(path: str = "artifacts/franchise_dict.json") -> dict:
    """Load the franchise dictionary artifact from disk."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class FranchiseMatcher:
    """Fuzzy matching against canonical franchise dictionary."""

    def __init__(self, franchise_dict: dict):
        self.franchise_dict = franchise_dict
        self.variant_map = {}
        self.canonical_titles = []
        self.normalized_titles = {}
        self.normalized_title_map = {}

        for title, data in franchise_dict.items():
            self.canonical_titles.append(title)
            normalized_title = normalize(title)
            self.normalized_titles[title] = normalized_title
            existing_title = self.normalized_title_map.get(normalized_title)
            if not existing_title or self._is_better_title(title, existing_title):
                self.normalized_title_map[normalized_title] = title

            for variant in data.get('variants', []):
                normalized_variant = normalize(variant)
                if normalized_variant:
                    existing_variant_title = self.variant_map.get(normalized_variant)
                    if not existing_variant_title or self._is_better_title(title, existing_variant_title):
                        self.variant_map[normalized_variant] = title

            if normalized_title:
                existing_variant_title = self.variant_map.get(normalized_title)
                if not existing_variant_title or self._is_better_title(title, existing_variant_title):
                    self.variant_map[normalized_title] = title

    def _is_better_title(self, candidate: str, current: str) -> bool:
        """Return True when the candidate is a better canonical display title."""
        candidate_score = (
            self.franchise_dict.get(candidate, {}).get('count', 0),
            len(self.normalized_titles.get(candidate, candidate).split()),
            len(candidate),
        )
        current_score = (
            self.franchise_dict.get(current, {}).get('count', 0),
            len(self.normalized_titles.get(current, current).split()),
            len(current),
        )
        return candidate_score > current_score

    def match(self, query: str) -> tuple:
        """
        Match query against franchise dictionary.
        Returns (canonical_title, contentType, confidence, match_type)
        match_type: 'exact' | 'fuzzy' | 'substring' | None
        """
        norm = normalize(query)
        words = set(norm.split())

        # 1. Exact match on variants
        if norm in self.variant_map:
            canonical = self.variant_map[norm]
            data = self.franchise_dict[canonical]
            return canonical, data['contentType'], 0.95, 'exact'

        # 2. Exact match on normalized canonical title
        if norm in self.normalized_title_map:
            canonical = self.normalized_title_map[norm]
            data = self.franchise_dict[canonical]
            return canonical, data['contentType'], 0.95, 'exact'

        # 3. Substring match: check if query contains a canonical title
        for title in self.canonical_titles:
            normalized_title = self.normalized_titles.get(title, "")
            if (
                normalized_title
                and normalized_title in norm
                and (len(normalized_title) > 3 or len(normalized_title.split()) > 1)
            ):
                data = self.franchise_dict[title]
                return title, data['contentType'], 0.75, 'substring'

        # 4. Fuzzy match on canonical titles
        if len(norm) > 3:
            best = process.extractOne(
                norm,
                list(self.normalized_title_map.keys()),
                scorer=fuzz.partial_ratio,
                score_cutoff=65,
            )
            if best:
                normalized_canonical, score, _ = best
                canonical = self.normalized_title_map[normalized_canonical]
                # Reject very short titles unless strong overlap
                if len(normalized_canonical) < 4:
                    if score < 85:
                        return '', '', 0.0, None
                # Require meaningful word overlap for fuzzy matches
                title_words = set(normalized_canonical.split())
                query_words = set(norm.split())
                word_overlap = len(title_words & query_words)
                n_title_words = len(title_words)

                if n_title_words >= 2 and word_overlap == 0:
                    # No shared words at all — reject unless very high score
                    if score < 80:
                        return '', '', 0.0, None
                elif n_title_words >= 2 and word_overlap == 1:
                    # Only 1 shared word — require higher score
                    # Reject if the shared word is a common adjective (country/language)
                    common_adj = {'английский', 'американский', 'российский', 'турецкий',
                                  'корейский', 'китайский', 'японский', 'французский',
                                  'немецкий', 'итальянский', 'испанский', 'советский'}
                    shared = title_words & query_words
                    if shared & common_adj and score < 85:
                        return '', '', 0.0, None
                    if score < 75:
                        return '', '', 0.0, None

                conf = score / 100.0 * 0.8
                data = self.franchise_dict[canonical]
                return canonical, data['contentType'], conf, 'fuzzy'

        # 5. Word overlap heuristic
        best_overlap = 0
        best_title = None
        for title in self.canonical_titles:
            title_words = set(self.normalized_titles.get(title, "").split())
            if not title_words:
                continue
            overlap = len(words & title_words) / max(len(title_words), 1)
            if overlap > best_overlap and overlap >= 0.4:
                # Reject very short single-word titles unless strong overlap
                if len(title) < 5 and len(title_words) == 1:
                    continue
                best_overlap = overlap
                best_title = title

        if best_title and len(best_title.split()) >= 2:
            data = self.franchise_dict[best_title]
            return best_title, data['contentType'], best_overlap * 0.7, 'word_overlap'

        return '', '', 0.0, None
