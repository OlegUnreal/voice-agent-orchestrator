"""Legacy import shim.

v1 kept the embedder here. The implementation moved to
:mod:`voice_agent.rag.embeddings` (where the TF-IDF+LSA and OpenAI backends live)
and this module only re-exports the deterministic hashing path so old imports —
``from voice_agent.rag.embedder import HashingEmbedder, tokenize`` — keep working.

New code should import from ``voice_agent.rag.embeddings``.
"""

from __future__ import annotations

from .embeddings import (
    DEFAULT_DIM,
    Embedder,
    HashingEmbedder,
    HashingVectorizer,
    STOPWORDS,
    TfidfSvdEmbedder,
    build_embedder,
    tokenize,
)

__all__ = [
    "DEFAULT_DIM",
    "Embedder",
    "HashingEmbedder",
    "HashingVectorizer",
    "STOPWORDS",
    "TfidfSvdEmbedder",
    "build_embedder",
    "tokenize",
]
