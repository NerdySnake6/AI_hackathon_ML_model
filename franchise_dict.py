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

        # Extract year/season from all queries in cluster
        years = []
        seasons = []
        for q in all_queries:
            y = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', q)
            if y: years.append(y.group(1))
            s = re.search(r'(\d+)\s*(?:сезон|season|s\b)', q, re.IGNORECASE)
            if s: seasons.append(s.group(1))
            
        final[best_title] = {
            'contentType': all_content_types.most_common(1)[0][0] if all_content_types else '',
            'variants': list(all_variants),
            'count': total_count,
            'year': Counter(years).most_common(1)[0][0] if years else None,
            'season': Counter(seasons).most_common(1)[0][0] if seasons else None,
        }

    return final


def save_artifacts(franchise_dict: dict, path: str = "artifacts/franchise_dict.json"):
    """Save the franchise dictionary and pre-computed maps to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # 1. Store the basic dict
    serializable = {}
    for title, data in franchise_dict.items():
        serializable[title] = {
            'contentType': data['contentType'],
            'variants': list(data['variants']),
            'count': data['count'],
        }
    
    # 2. Pre-compute maps to avoid runtime normalization overhead
    variant_map = {}
    normalized_title_map = {}
    canonical_to_norm_map = {}
    lemma_variant_map = {} # (title, lemmas_list)
    
    # Simple helper for map building (copied from FranchiseMatcher logic)
    for title, data in franchise_dict.items():
        norm_t = normalize(title, use_lemmatization=False)
        if norm_t:
            normalized_title_map[norm_t] = title
            canonical_to_norm_map[title] = norm_t
        
        for v in data['variants']:
            v_norm = normalize(v, use_lemmatization=False)
            if v_norm:
                variant_map[v_norm] = title
            
            v_lemma_str = normalize(v, use_lemmatization=True)
            if v_lemma_str:
                v_lemma_list = v_lemma_str.split()
                sorted_v_lemma = " ".join(sorted(v_lemma_list))
                lemma_variant_map[sorted_v_lemma] = (title, v_lemma_list)

    output = {
        'franchise_dict': serializable,
        'variant_map': variant_map,
        'normalized_title_map': normalized_title_map,
        'canonical_to_norm_map': canonical_to_norm_map,
        'lemma_variant_map': lemma_variant_map
    }
    
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)


def load_artifacts(path: str = "artifacts/franchise_dict.json") -> dict:
    """Load the franchise artifacts from disk."""
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


class FranchiseMatcher:
    """Fuzzy matching against canonical franchise dictionary."""

    def __init__(self, artifacts: dict):
        self.franchise_dict = artifacts.get('franchise_dict', {})
        self.variant_map = artifacts.get('variant_map', {})
        self.normalized_title_map = artifacts.get('normalized_title_map', {})
        self.lemma_variant_map = artifacts.get('lemma_variant_map', {})
        
        self.normalized_titles_list = list(self.normalized_title_map.keys())
        self.normalized_titles = artifacts.get('canonical_to_norm_map', {})

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

    SPAN_TRIM_WORDS = {
        "смотреть", "онлайн", "бесплатно", "скачать", "торрент",
        "хорошем", "качестве", "hd", "720p", "1080p", "4k",
        "фильм", "сериал", "мультфильм", "мультсериал", "аниме",
        "все", "серии", "сезон", "серия", "эпизод",
        "в", "и", "с", "а", "но", "на", "для",
        "ютуб", "youtube", "вк", "vk", "дата", "выхода", "актеры",
        "сюжет", "отзывы", "кино", "видео", "трейлер",
    }

    def match(self, query: str) -> tuple:
        """
        Match query against franchise dictionary.
        Returns (canonical_title, matched_span, contentType, confidence, match_type)
        Fast: orig_lemmas computed once, no slow subset pass, extractOne with cutoff.
        """
        norm_raw = normalize(query, use_lemmatization=False)
        norm_lemma = normalize(query, use_lemmatization=True)
        query_tokens_raw = norm_raw.split()
        query_tokens_lemma = norm_lemma.split()

        # Pre-compute ONCE — used by find_best_span
        orig_tokens = query.split()
        orig_lemmas = [normalize(t, use_lemmatization=True) for t in orig_tokens]

        # Metadata extraction for year-based disambiguation
        query_year = None
        _ym = re.search(r'\b(19[5-9]\d|20[0-2]\d)\b', query)
        if _ym:
            query_year = _ym.group(1)

        # Pre-compute clean query for fuzzy once
        _noise = {"смотреть", "онлайн", "бесплатно", "скачать", "торрент",
                  "хорошем", "качестве", "hd", "720p", "1080p", "4k",
                  "фильм", "сериал", "все", "серии", "сезон", "в", "и", "с"}
        clean_query = " ".join(w for w in query_tokens_raw if w not in _noise) or norm_raw

        def trim_span(span: str) -> str:
            tokens = span.split()
            while tokens and tokens[0].lower() in self.SPAN_TRIM_WORDS:
                tokens.pop(0)
            while tokens and tokens[-1].lower() in self.SPAN_TRIM_WORDS:
                tokens.pop()
            return " ".join(tokens)

        def find_best_span(target_tokens_lemma: list) -> str:
            """Robust span extractor with translit and fuzzy support."""
            if not target_tokens_lemma: return ""
            target_set = set(target_tokens_lemma)
            n = len(target_tokens_lemma)
            
            # Pre-compute transliterated lemmas for the whole query once
            from preprocessing import transliterate, detect_translit
            query_lemmas_translit = []
            for t in orig_tokens:
                if detect_translit(t):
                    t_cyr = transliterate(t)
                    query_lemmas_translit.append(normalize(t_cyr, use_lemmatization=True))
                else:
                    query_lemmas_translit.append(normalize(t, use_lemmatization=True))

            best_span_str = ""
            max_overlap = 0

            # Try windows of size n to n+2
            for ws in range(n, min(n + 3, len(orig_lemmas) + 1)):
                for i in range(len(orig_lemmas) - ws + 1):
                    # Combine regular lemmas and translit lemmas for this window
                    window_lemmas = set()
                    for ls in orig_lemmas[i:i + ws]:
                        window_lemmas.update(ls.split())
                    for ls in query_lemmas_translit[i:i + ws]:
                        window_lemmas.update(ls.split())
                    
                    overlap = len(window_lemmas & target_set)
                    if overlap == n: # Perfect match (even with translit)
                        return trim_span(" ".join(orig_tokens[i:i + ws]))
                    
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_span_str = " ".join(orig_tokens[i:i + ws])
            
            # Fuzzy fallback: if we matched most of the title, return it
            if n > 1 and max_overlap >= (n * 0.7):
                return trim_span(best_span_str)
            
            return ""

        # 1. Exact match on variants
        if norm_raw in self.variant_map:
            canonical = self.variant_map[norm_raw]
            data = self.franchise_dict[canonical]
            span = find_best_span(normalize(canonical, use_lemmatization=True).split())
            return canonical, span, data['contentType'], 0.95, 'exact'

        # 2. Lemma match (exact sequence of lemmas)
        sorted_lemma = " ".join(sorted(query_tokens_lemma))
        if sorted_lemma in self.lemma_variant_map:
            canonical, target_lemmas = self.lemma_variant_map[sorted_lemma]
            data = self.franchise_dict[canonical]
            span = find_best_span(target_lemmas)
            return canonical, span, data['contentType'], 0.90, 'lemma_match'

        # 3. Lemma overlap (only for multi-word queries)
        if len(query_tokens_lemma) >= 2:
            best_title, best_len, best_lemmas = None, 0, []
            query_lemma_set = set(query_tokens_lemma)
            for _, (title, tl) in self.lemma_variant_map.items():
                tls = set(tl)
                if len(tls) >= 2 and tls.issubset(query_lemma_set) and len(tls) > best_len:
                    best_len, best_title, best_lemmas = len(tls), title, tl
            if best_title:
                data = self.franchise_dict[best_title]
                span = find_best_span(best_lemmas)
                return best_title, span, data['contentType'], 0.85, 'lemma_overlap'

        # 4. Fuzzy match — extractOne with score_cutoff (fastest path)
        if len(clean_query) > 3:
            best = process.extractOne(
                clean_query,
                self.normalized_titles_list,
                scorer=fuzz.token_set_ratio,
                score_cutoff=78,
            )
            if best:
                norm_can, score, _ = best
                if len(norm_can) < 4 and score < 88:
                    return '', '', '', 0.0, None
                canonical = self.normalized_title_map[norm_can]
                data = self.franchise_dict[canonical]
                # Year boost
                if query_year and data.get('year') == query_year:
                    score = min(score + 10, 100)
                conf = score / 100.0 * 0.8
                # Fast span: token set intersection
                tw = set(norm_can.split())
                best_span = ""
                for i in range(len(query_tokens_raw)):
                    for j in range(i + 1, len(query_tokens_raw) + 1):
                        if set(query_tokens_raw[i:j]) == tw:
                            best_span = trim_span(" ".join(orig_tokens[i:j]))
                            break
                    if best_span:
                        break
                return canonical, best_span, data['contentType'], conf, 'fuzzy'

        return '', '', '', 0.0, None



