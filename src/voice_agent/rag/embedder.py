from __future__ import annotations

import hashlib
import re

import numpy as np

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "for", "in", "is", "it", "me", "of", "on", "or",
    "the", "to", "what", "who", "with", "you", "your",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]


class HashingEmbedder:
    """Deterministic bag-of-words embedder. No model download, stable in CI."""

    def __init__(self, dim: int = 96) -> None:
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._one(t) for t in texts])

    def _one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float64)
        for tok in tokenize(text):
            digest = hashlib.md5(tok.encode()).digest()
            idx = int.from_bytes(digest[:4], "little") % self.dim
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm:
            vec /= norm
        return vec
