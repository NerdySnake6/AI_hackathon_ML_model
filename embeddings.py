"""
Embedding-based title detection using TF-IDF + cosine similarity.
Lightweight alternative to transformer embeddings that works within sandbox constraints.
"""
import os
import json
import pickle
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_embedding_index(df, franchise_dict=None, n_features: int = 50000, ngram_range=(1, 2)):
    """
    Build TF-IDF vectors for all labeled queries and compute prototype vectors per franchise.
    Also includes all titles from franchise_dict as prototypes.
    Returns (vectorizer, prototypes, prototype_info).
    """
    from preprocessing import normalize
    labeled = df[df['Title'].notna()].copy()
    labeled['norm_title'] = labeled['Title'].str.strip().str.lower()
    labeled = labeled.reset_index(drop=True)

    # Initial queries from training data
    train_queries = [normalize(q, use_lemmatization=True) for q in labeled['QueryText'].tolist()]
    
    # Add synthetic queries for each canonical title in franchise_dict to ensure coverage
    synthetic_data = []
    if franchise_dict:
        for title, data in franchise_dict.items():
            # Use title and its variants as synthetic queries
            norm_title = normalize(title, use_lemmatization=True)
            synthetic_data.append({'title': title, 'query': norm_title, 'contentType': data['contentType']})
            for variant in data.get('variants', []):
                norm_variant = normalize(variant, use_lemmatization=True)
                if norm_variant and norm_variant != norm_title:
                    synthetic_data.append({'title': title, 'query': norm_variant, 'contentType': data['contentType']})

    all_queries = train_queries + [d['query'] for d in synthetic_data]

    # Build TF-IDF on all queries
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        ngram_range=ngram_range,
        analyzer='char_wb',
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(all_queries)

    # Compute prototype vector per franchise
    prototypes = {}
    prototype_info = {}

    # Map titles to their vector indices
    title_to_indices = {}
    
    # 1. From training data
    for idx, row in labeled.iterrows():
        title = row['norm_title']
        if title not in title_to_indices: title_to_indices[title] = []
        title_to_indices[title].append(idx)
        
    # 2. From synthetic data
    train_count = len(train_queries)
    for i, d in enumerate(synthetic_data):
        title = d['title']
        if title not in title_to_indices: title_to_indices[title] = []
        title_to_indices[title].append(train_count + i)

    for title, indices in title_to_indices.items():
        # Mean TF-IDF vector
        prototype = tfidf_matrix[indices].mean(axis=0)
        prototypes[title] = np.asarray(prototype).flatten()
        
        # Determine content type (prefer train data if available)
        if franchise_dict and title in franchise_dict:
            ct = franchise_dict[title]['contentType']
        else:
            # Fallback to train data
            group = labeled[labeled['norm_title'] == title]
            ct_counter = Counter(group['ContentType'].tolist())
            ct = ct_counter.most_common(1)[0][0] if ct_counter else ''
            
        prototype_info[title] = {
            'contentType': ct,
            'count': len(indices),
        }

    return vectorizer, prototypes, prototype_info


def save_artifacts(vectorizer, prototypes, prototype_info, path: str = "artifacts/embeddings.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # Convert numpy arrays to lists for serialization
    proto_serializable = {k: v.tolist() for k, v in prototypes.items()}
    with open(path, 'wb') as f:
        pickle.dump({
            'vectorizer': vectorizer,
            'prototypes': proto_serializable,
            'prototype_info': prototype_info,
        }, f)


def load_artifacts(path: str = "artifacts/embeddings.pkl"):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        # Convert lists back to numpy arrays
        prototypes = {k: np.array(v) for k, v in data['prototypes'].items()}
        return data['vectorizer'], prototypes, data['prototype_info']
    return None, {}, {}


class EmbeddingMatcher:
    """Match queries to franchise prototypes using TF-IDF + cosine similarity."""

    def __init__(self, vectorizer, prototypes, prototype_info):
        self.vectorizer = vectorizer
        self.prototypes = prototypes
        self.prototype_info = prototype_info
        self._proto_matrix = None
        self._title_order = None

        if prototypes:
            self._title_order = list(prototypes.keys())
            proto_array = np.array([prototypes[t] for t in self._title_order])
            self._proto_matrix = proto_array

    @staticmethod
    def _is_generic_query(query: str) -> bool:
        """Check if query is too generic for embedding-based title detection.

        Generic queries like "смотреть фильмы 2025" or "новинки сериалов"
        should NOT return specific titles from embedding matching.
        """
        from preprocessing import normalize
        norm = normalize(query)
        words = set(norm.split())

        # Generic content words (these don't identify a specific franchise)
        generic_words = {
            'смотреть', 'онлайн', 'бесплатно', 'фильм', 'фильмы', 'кино',
            'сериал', 'сериалы', 'мультфильм', 'мультфильмы', 'мультсериал',
            'аниме', 'дорама', 'новинки', 'новинк', 'лучшие', 'топ',
            'год', 'года', '2023', '2024', '2025', '2026', '2020',
            '2021', '2022', 'hd', 'full', 'online', 'смотреть',
            'качестве', 'русском', 'русский', 'озвучк', 'перевод',
            'все', 'подряд', 'все серии', 'мы', 'мой', 'мне', 'меня',
            'наш', 'ваш', 'свой', 'твой',
        }
        broad_generic_words = {
            'фильмы', 'сериалы', 'мультфильмы', 'мультсериалы',
            'аниме', 'дорамы', 'новинки', 'топ', 'лучшие', 'лучший',
            'русские', 'корейские', 'турецкие', 'американские',
        }

        # If query contains ONLY generic words + maybe season/episode numbers
        non_generic = {
            word
            for word in words
            if word not in generic_words and not word.isdigit()
        }

        if not non_generic:
            return True

        has_broad_context = bool(words & broad_generic_words)
        if len(non_generic) == 1:
            # One-word queries can still be exact titles ("крик", "теща").
            return has_broad_context

        return has_broad_context and len(non_generic) <= 2

    def match(self, query: str, top_k: int = 3, threshold: float = 0.15):
        """
        Match query against franchise prototypes.
        Returns list of (title, contentType, confidence) sorted by confidence.
        """
        results = self.match_batch([query], top_k=top_k, threshold=threshold)
        return results[0]

    def match_batch(self, queries: list[str], top_k: int = 3, threshold: float = 0.15) -> list[list[tuple[str, str, float]]]:
        """
        Match a batch of queries against franchise prototypes using vectorized operations.
        Returns a list (one per query) of lists of (title, contentType, confidence).
        """
        from preprocessing import normalize

        if self._proto_matrix is None or self.vectorizer is None or not queries:
            return [[] for _ in queries]

        # Prepare results placeholder
        batch_results = [[] for _ in queries]
        valid_indices = []
        normalized_queries = []

        for i, q in enumerate(queries):
            if not self._is_generic_query(q):
                valid_indices.append(i)
                normalized_queries.append(normalize(q, use_lemmatization=True))

        if not valid_indices:
            return batch_results

        # Vectorized transform and similarity
        try:
            query_matrix = self.vectorizer.transform(normalized_queries)
            # query_matrix is (len(valid_indices), n_features) sparse
            # self._proto_matrix is (n_titles, n_features) dense
            sim_matrix = cosine_similarity(query_matrix, self._proto_matrix)
            # sim_matrix is (len(valid_indices), n_titles) dense
        except Exception:
            return batch_results

        # Process results for each valid query
        for i, v_idx in enumerate(valid_indices):
            sims = sim_matrix[i]
            top_indices = np.argsort(sims)[::-1][:top_k]
            
            res = []
            for t_idx in top_indices:
                sim = float(sims[t_idx])
                if sim >= threshold:
                    title = self._title_order[t_idx]
                    info = self.prototype_info.get(title, {})
                    res.append((title, info.get('contentType', ''), sim))
            batch_results[v_idx] = res

        return batch_results
