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


def build_embedding_index(df, n_features: int = 50000, ngram_range=(1, 2)):
    """
    Build TF-IDF vectors for all labeled queries and compute prototype vectors per franchise.
    Returns (vectorizer, prototypes, prototype_info).
    """
    labeled = df[df['Title'].notna()].copy()
    labeled['norm_title'] = labeled['Title'].str.strip().str.lower()
    labeled = labeled.reset_index(drop=True)  # Reset to positional indices

    # Group by title
    title_groups = labeled.groupby('norm_title')

    queries_list = labeled['QueryText'].tolist()

    # Build TF-IDF on all labeled queries
    vectorizer = TfidfVectorizer(
        max_features=n_features,
        ngram_range=ngram_range,
        analyzer='char_wb',
        min_df=2,
        max_df=0.95,
    )
    tfidf_matrix = vectorizer.fit_transform(queries_list)

    # Compute prototype vector per franchise (mean of TF-IDF vectors)
    prototypes = {}
    prototype_info = {}

    for title, group in title_groups:
        pos_indices = group.index.tolist()  # Now positional (0-based)
        if not pos_indices:
            continue
        # Mean TF-IDF vector
        prototype = tfidf_matrix[pos_indices].mean(axis=0)
        prototypes[title] = np.asarray(prototype).flatten()
        ct = Counter(group['ContentType'].tolist())
        prototype_info[title] = {
            'contentType': ct.most_common(1)[0][0] if ct else '',
            'count': len(group),
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

        # If query contains ONLY generic words + maybe season/episode numbers
        non_generic = words - generic_words
        # Remove pure numbers (seasons/episodes)
        non_generic = {w for w in non_generic if not w.isdigit()}

        # If 1 or fewer non-generic words, it's generic
        return len(non_generic) <= 1

    def match(self, query: str, top_k: int = 3, threshold: float = 0.15):
        """
        Match query against franchise prototypes.
        Returns list of (title, contentType, confidence) sorted by confidence.
        """
        from preprocessing import normalize

        if self._proto_matrix is None or self.vectorizer is None:
            return []

        # Skip embedding matching for generic queries
        if self._is_generic_query(query):
            return []

        norm = normalize(query)
        # Transform query
        try:
            query_vec = self.vectorizer.transform([norm])
            # query_vec is (1, n_features) sparse matrix
            query_dense = np.asarray(query_vec.todense()).flatten()
        except Exception:
            return []

        # Compute cosine similarity to all prototypes
        sims = cosine_similarity(query_dense.reshape(1, -1), self._proto_matrix)[0]

        # Get top-k
        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            title = self._title_order[idx]
            sim = float(sims[idx])
            if sim >= threshold:
                info = self.prototype_info.get(title, {})
                results.append((title, info.get('contentType', ''), sim))

        return results
