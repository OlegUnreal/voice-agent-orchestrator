"""Retrieval stack: corpus -> embeddings -> hybrid retriever -> learned reranker."""

from __future__ import annotations

from .corpus import Chunk, Document, chunk_document, chunk_text, load_corpus, parse_document
from .embedder import HashingEmbedder, tokenize
from .embeddings import (
    Embedder,
    HashingVectorizer,
    OpenAIEmbedder,
    TfidfSvdEmbedder,
    build_embedder,
)
from .labels import GoldenQuery, RelevanceRecord, load_golden_set, load_records
from .reranker import FEATURE_NAMES, CrossFeatureReranker, RerankerLabel, extract_features
from .retriever import DEFAULT_RRF_K, Breakdown, Hit, HybridRetriever, RetrievalStats, rrf_score
from .seed import DEFAULT_DOCS
from .store import KnowledgeStore, load_knowledge, rebuild_artifacts

__all__ = [
    "Breakdown",
    "Chunk",
    "CrossFeatureReranker",
    "DEFAULT_DOCS",
    "DEFAULT_RRF_K",
    "Document",
    "Embedder",
    "FEATURE_NAMES",
    "GoldenQuery",
    "HashingEmbedder",
    "HashingVectorizer",
    "Hit",
    "HybridRetriever",
    "KnowledgeStore",
    "OpenAIEmbedder",
    "RelevanceRecord",
    "RerankerLabel",
    "RetrievalStats",
    "TfidfSvdEmbedder",
    "build_embedder",
    "chunk_document",
    "chunk_text",
    "extract_features",
    "load_corpus",
    "load_golden_set",
    "load_knowledge",
    "load_records",
    "parse_document",
    "rebuild_artifacts",
    "rrf_score",
    "tokenize",
]
