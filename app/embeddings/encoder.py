"""Lightweight deterministic encoders for hybrid title retrieval."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Sequence

import numpy as np

from app.preprocessing.normalizer import normalize_text


def cosine_similarity(left: np.ndarray | Sequence[float], right: np.ndarray | Sequence[float]) -> float:
    """Return cosine similarity for two NumPy vectors."""
    u = np.asarray(left, dtype=np.float32)
    v = np.asarray(right, dtype=np.float32)
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0.0 or norm_v == 0.0:
        return 0.0
    return float(np.dot(u, v) / (norm_u * norm_v))


class HashingTextEncoder:
    """Build deterministic dense vectors from tokens and character n-grams using NumPy."""

    def __init__(
        self,
        dimension: int = 256,
        min_n: int = 3,
        max_n: int = 4,
        token_weight: float = 2.0,
        ngram_weight: float = 1.0,
    ) -> None:
        """Initialize the encoder with stable hashing parameters."""
        self.dimension = dimension
        self.min_n = min_n
        self.max_n = max_n
        self.token_weight = token_weight
        self.ngram_weight = ngram_weight

    def encode(self, text: str) -> np.ndarray:
        """Encode text into a normalized dense vector optimized with NumPy."""
        vector = np.zeros(self.dimension, dtype=np.float32)
        normalized = normalize_text(text)
        if not normalized:
            return vector

        tokens = normalized.split()
        for token in tokens:
            self._add_feature(vector, f"tok:{token}", self.token_weight)

        compact_text = normalized.replace(" ", "")
        for ngram in self._iter_char_ngrams(normalized):
            self._add_feature(vector, f"chr:{ngram}", self.ngram_weight)
        for ngram in self._iter_char_ngrams(compact_text):
            self._add_feature(vector, f"cmp:{ngram}", self.ngram_weight * 0.8)

        norm = np.linalg.norm(vector)
        if norm > 0.0:
            vector /= norm
        return vector

    def _iter_char_ngrams(self, text: str) -> Iterable[str]:
        """Yield character n-grams for normalized title matching."""
        padded_text = f" {text} "
        for n_size in range(self.min_n, self.max_n + 1):
            if len(padded_text) < n_size:
                continue
            for start in range(len(padded_text) - n_size + 1):
                ngram = padded_text[start : start + n_size]
                if ngram.strip():
                    yield ngram

    def _add_feature(self, vector: np.ndarray, feature: str, weight: float) -> None:
        """Project one symbolic feature into a dense hashing vector."""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight
