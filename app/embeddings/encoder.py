"""Lightweight deterministic encoders for hybrid title retrieval."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Sequence

from app.preprocessing.normalizer import normalize_text


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Return cosine similarity for two already aligned vectors."""
    if len(left) != len(right):
        raise ValueError("Vectors must have the same dimension")

    dot_product = sum(lhs * rhs for lhs, rhs in zip(left, right, strict=True))
    if dot_product == 0.0:
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot_product / (left_norm * right_norm)


class HashingTextEncoder:
    """Build deterministic dense vectors from tokens and character n-grams."""

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

    def encode(self, text: str) -> tuple[float, ...]:
        """Encode text into a normalized dense vector."""
        normalized = normalize_text(text)
        if not normalized:
            return tuple(0.0 for _ in range(self.dimension))

        vector = [0.0] * self.dimension
        tokens = normalized.split()
        for token in tokens:
            self._add_feature(vector, f"tok:{token}", self.token_weight)

        compact_text = normalized.replace(" ", "")
        for ngram in self._iter_char_ngrams(normalized):
            self._add_feature(vector, f"chr:{ngram}", self.ngram_weight)
        for ngram in self._iter_char_ngrams(compact_text):
            self._add_feature(vector, f"cmp:{ngram}", self.ngram_weight * 0.8)

        return tuple(self._normalize(vector))

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

    def _add_feature(self, vector: list[float], feature: str, weight: float) -> None:
        """Project one symbolic feature into a dense hashing vector."""
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dimension
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign * weight

    @staticmethod
    def _normalize(vector: Sequence[float]) -> list[float]:
        """Normalize a vector to unit length."""
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return list(vector)
        return [value / norm for value in vector]
