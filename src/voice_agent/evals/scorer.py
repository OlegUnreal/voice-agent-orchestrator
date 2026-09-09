from __future__ import annotations

from ..rag.embedder import HashingEmbedder


def keyword_pass(text: str, keywords: list[str]) -> list[str]:
    lower = text.lower()
    return [k for k in keywords if k.lower() not in lower]


def cosine_to_expected(response: str, expected: str) -> float:
    embedder = HashingEmbedder()
    a, b = embedder.embed([response, expected])
    return float(a @ b)
