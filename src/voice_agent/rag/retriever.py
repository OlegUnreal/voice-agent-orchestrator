"""Hybrid retrieval: BM25 + dense LSA fused with Reciprocal Rank Fusion,
optionally rescored by the learned cross-feature reranker.

Pipeline (one ``search()``)::

    query ─┬─ BM25Okapi over chunk tokens ─────┐
           │                                   ├─ RRF(k=60) ─ top `rerank_pool`
           └─ dense cosine over LSA vectors ───┘        │
                                                        ├─ reranker P(relevant)  [optional]
                                                        ├─ evidence gate (refusal)
                                                        └─ per-doc diversity ─ top_k hits

Every candidate carries a :class:`Breakdown` (lexical rank/score, dense rank/score,
rrf, feature row, final) so traces and the demo API can show *why* a chunk was
selected, not just that it was. ``search_stats()`` returns the per-stage candidate
counts the observability layer records.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Any, Sequence

import numpy as np

from .corpus import Chunk
from .embeddings import Embedder, tokenize
from .reranker import FEATURE_NAMES, CrossFeatureReranker, build_idf, extract_features

__all__ = [
    "Breakdown",
    "Hit",
    "Refusal",
    "RetrievalStats",
    "HybridRetriever",
    "rrf_score",
    "snippet",
    "DEFAULT_RRF_K",
]

#: Rank-fusion constant from the original RRF paper (Cormack et al. 2009).
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class Hit:
    """A retrieved chunk. Field order/signature is unchanged from v1; the extras
    are defaulted so existing callers and tests keep working."""

    doc_id: str
    title: str
    quote: str
    score: float
    chunk_id: str = ""
    breakdown: dict[str, Any] = field(default_factory=dict)

    def why(self) -> str:
        """Compact justification, e.g. ``lex#1 dense#2 p=0.913 rrf=0.0328``."""
        b = self.breakdown
        parts = [f"lex#{b.get('lexical_rank', '-')}", f"dense#{b.get('dense_rank', '-')}"]
        if b.get("probability") is not None:
            parts.append(f"p={b['probability']:.3f}")
        parts.append(f"rrf={b.get('rrf', 0.0):.4f}")
        return " ".join(parts)


@dataclass(frozen=True)
class Breakdown:
    lexical_rank: int | None
    lexical_score: float
    lexical_norm: float
    dense_rank: int | None
    dense_score: float
    rrf: float
    rrf_norm: float
    final: float
    gate: float
    probability: float | None = None
    features: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalStats:
    """Per-query observability record: what each stage contributed and what it cost."""

    query: str
    mode: str
    latency_ms: float = 0.0
    lexical_candidates: int = 0
    dense_candidates: int = 0
    fused_candidates: int = 0
    rerank_pool: int = 0
    returned: int = 0
    refused: bool = False
    top_gate: float = 0.0
    min_score: float = 0.0
    embedding_backend: str = ""
    embedding_dim: int = 0
    context_chars: int = 0
    dropped_no_overlap: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Refusal:
    """Returned instead of hits when the fused evidence is too weak to answer."""

    query: str
    reason: str
    top_gate: float
    min_score: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _rank_map(scores: Sequence[float], *, cutoff: int, threshold: float = 0.0) -> list[tuple[int, float]]:
    """Deterministic descending ranking with a stable tie-break on index."""
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return [(i, float(scores[i])) for i in order[:cutoff] if float(scores[i]) > threshold]


def snippet(text: str, query: str, width: int = 220) -> str:
    """Best window of ``text`` around the first query token that actually appears."""
    lower = text.lower()
    positions = [lower.find(word) for word in query.lower().split() if word]
    pos = next((p for p in positions if p >= 0), -1)
    if pos < 0:
        pos = 0
    start = max(0, pos - 40)
    return text[start : start + width].strip()


class HybridRetriever:
    """Lexical + dense retrieval with RRF fusion, a diversity filter, a learned
    reranker, and an explicit evidence gate that produces refusals."""

    def __init__(
        self,
        chunks: Sequence[Chunk],
        embedder: Embedder,
        *,
        rrf_k: int = DEFAULT_RRF_K,
        top_k: int = 3,
        min_score: float = 0.10,
        reranker: CrossFeatureReranker | None = None,
        lexical_pool: int = 20,
        dense_pool: int = 20,
        rerank_pool: int = 8,
        max_chunks_per_doc: int = 1,
        require_lexical_overlap: bool = True,
    ) -> None:
        self.chunks: list[Chunk] = list(chunks)
        self.embedder = embedder
        self.rrf_k = int(rrf_k)
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.reranker = reranker
        self.lexical_pool = int(lexical_pool)
        self.dense_pool = int(dense_pool)
        self.rerank_pool = int(rerank_pool)
        self.max_chunks_per_doc = int(max_chunks_per_doc)
        self.require_lexical_overlap = bool(require_lexical_overlap)
        self._bm25: Any = None
        self._idf: dict[str, float] = {}
        self._corpus_stems: frozenset[str] = frozenset()
        self._matrix: np.ndarray | None = None
        if self.chunks:
            self._build_lexical_index()
            self._build_dense_index()

    # ------------------------------------------------------------------ indexes
    def _build_lexical_index(self) -> None:
        from rank_bm25 import BM25Okapi

        token_sets = [set(c.tokens) for c in self.chunks]
        self._bm25 = BM25Okapi([list(c.tokens) for c in self.chunks], k1=1.4, b=0.7)
        self._idf = build_idf(token_sets)
        # Query-side collection coverage for the reranker's corpus_coverage feature:
        # a term that appears in no document is exactly what makes a near-miss
        # question refusable.
        self._corpus_stems = frozenset().union(*token_sets) if token_sets else frozenset()

    def _build_dense_index(self) -> None:
        if not self.chunks:
            self._matrix = np.zeros((0, self.embedder.dim or 1), dtype=np.float32)
            return
        self._matrix = self.embedder.embed([c.search_text for c in self.chunks])

    @property
    def size(self) -> int:
        return len(self.chunks)

    def add_chunks(self, chunks: Sequence[Chunk]) -> None:
        """Append and reindex. Index rebuild is O(corpus); fine at this scale."""
        self.chunks.extend(chunks)
        if self.chunks:
            self._build_lexical_index()
            self._build_dense_index()

    # ------------------------------------------------------------------ signals
    def lexical_scores(self, query: str) -> np.ndarray:
        tokens = tokenize(query)
        if self._bm25 is None or not tokens:
            return np.zeros(len(self.chunks), dtype=np.float64)
        return np.asarray(self._bm25.get_scores(tokens), dtype=np.float64)

    def dense_scores(self, query: str) -> np.ndarray:
        if self._matrix is None or self._matrix.size == 0:
            return np.zeros(len(self.chunks), dtype=np.float64)
        qv = self.embedder.embed([query])[0]
        return np.asarray(self._matrix @ qv, dtype=np.float64)

    def feature_rows(self, query: str) -> list[tuple[Chunk, np.ndarray, float, float]]:
        """(chunk, feature row, bm25, dense) for *every* chunk — the training view.

        Uses the corpus-wide BM25 max as the normalisation denominator so a feature
        value means the same thing during training and at query time.
        """
        bm25 = self.lexical_scores(query)
        dense = self.dense_scores(query)
        best = float(bm25.max()) if bm25.size else 0.0
        rows = []
        for i, chunk in enumerate(self.chunks):
            row = extract_features(
                query,
                chunk,
                dense=float(dense[i]),
                bm25=float(bm25[i]),
                bm25_best=best,
                idf=self._idf,
                corpus_stems=self._corpus_stems,
            )
            rows.append((chunk, row, float(bm25[i]), float(dense[i])))
        return rows

    # ------------------------------------------------------------------ search
    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_reranker: bool | None = None,
    ) -> list[Hit]:
        hits, _stats = self.search_stats(
            query,
            top_k=top_k,
            min_score=min_score,
            use_lexical=use_lexical,
            use_dense=use_dense,
            use_reranker=use_reranker,
        )
        return hits

    def fuse(
        self,
        query: str,
        bm25: np.ndarray,
        dense: np.ndarray,
        *,
        use_lexical: bool,
        use_dense: bool,
    ) -> tuple[list[tuple[int, float, Breakdown]], int, RetrievalStats]:
        """Reciprocal Rank Fusion over the two ranked lists.

        Returns ``(candidates, dropped, partial_stats)`` with candidates ordered by
        fused RRF (best first) and every rank/score kept in the breakdown.
        """
        lexical_ranked = _rank_map(bm25, cutoff=self.lexical_pool) if use_lexical else []
        dense_ranked = _rank_map(dense, cutoff=self.dense_pool, threshold=-1.0) if use_dense else []
        lexical_ranks = {i: r for r, (i, _s) in enumerate(lexical_ranked, start=1)}
        dense_ranks = {i: r for r, (i, _s) in enumerate(dense_ranked, start=1)}
        best_bm25 = float(bm25.max()) if bm25.size else 0.0
        lists_used = int(bool(lexical_ranked)) + int(bool(dense_ranked)) or 1
        rrf_ceiling = lists_used / (self.rrf_k + 1.0)

        q_tokens = set(tokenize(query))
        fused: list[tuple[int, float, Breakdown]] = []
        dropped = 0
        for index, chunk in enumerate(self.chunks):
            l_rank = lexical_ranks.get(index)
            d_rank = dense_ranks.get(index)
            if l_rank is None and d_rank is None:
                continue
            rrf = rrf_score([r for r in (l_rank, d_rank) if r], k=self.rrf_k)
            if self.require_lexical_overlap and use_lexical and not (q_tokens & set(chunk.tokens)):
                dropped += 1
                continue
            features = extract_features(
                query,
                chunk,
                dense=float(dense[index]),
                bm25=float(bm25[index]),
                bm25_best=best_bm25,
                idf=self._idf,
                corpus_stems=self._corpus_stems,
            )
            rrf_norm = min(1.0, rrf / rrf_ceiling) if rrf_ceiling else 0.0
            breakdown = Breakdown(
                lexical_rank=l_rank,
                lexical_score=round(float(bm25[index]), 4),
                lexical_norm=round(float(bm25[index]) / best_bm25, 4) if best_bm25 else 0.0,
                dense_rank=d_rank,
                dense_score=round(float(dense[index]), 4),
                rrf=round(rrf, 6),
                rrf_norm=round(rrf_norm, 4),
                final=0.0,
                gate=0.0,
                features={n: round(float(v), 4) for n, v in zip(FEATURE_NAMES, features)},
            )
            fused.append((index, rrf, breakdown))
        fused.sort(key=lambda t: (-t[1], -t[2].rrf_norm, self.chunks[t[0]].chunk_id))
        stats = RetrievalStats(
            query=query,
            mode=_mode_name(use_lexical, use_dense, False),
            lexical_candidates=len(lexical_ranked),
            dense_candidates=len(dense_ranked),
            fused_candidates=len(fused),
            dropped_no_overlap=dropped,
            embedding_backend=getattr(self.embedder, "backend", self.embedder.name),
            embedding_dim=int(self.embedder.dim),
        )
        return fused, dropped, stats

    def search_stats(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_reranker: bool | None = None,
    ) -> tuple[list[Hit], RetrievalStats]:
        started = perf_counter()
        top_k = self.top_k if top_k is None else int(top_k)
        min_score = self.min_score if min_score is None else float(min_score)
        rerank = (self.reranker is not None and self.reranker.fitted) if use_reranker is None else bool(use_reranker)
        if use_reranker and not (self.reranker is not None and self.reranker.fitted):
            rerank = False

        bm25 = self.lexical_scores(query) if use_lexical else np.zeros(len(self.chunks))
        dense = self.dense_scores(query) if use_dense else np.zeros(len(self.chunks))
        fused, _dropped, stats = self.fuse(query, bm25, dense, use_lexical=use_lexical, use_dense=use_dense)
        stats.mode = _mode_name(use_lexical, use_dense, rerank)
        stats.min_score = min_score

        pool = fused[: self.rerank_pool] if rerank else fused
        stats.rerank_pool = len(pool) if rerank else 0

        scored: list[tuple[float, int, Breakdown]] = []
        for index, _rrf, breakdown in pool:
            features = np.array(
                [breakdown.features[name] for name in FEATURE_NAMES], dtype=np.float64
            )
            if rerank:
                assert self.reranker is not None
                probability = self.reranker.probability(features)
                final = probability
                gate = probability
            else:
                probability = None
                # Uncalibrated mode: gate on the absolute dense signal (a cosine, so
                # the documented floor means the same thing as it did in v1) and rank
                # on fused RRF. Ordering never mixes the two scales.
                final = breakdown.rrf_norm
                gate = max(0.0, breakdown.dense_score) if use_dense else breakdown.rrf_norm
            scored.append(
                (
                    final,
                    index,
                    Breakdown(
                        **{
                            **breakdown.as_dict(),
                            "final": round(final, 6),
                            "gate": round(gate, 4),
                            "probability": None if probability is None else round(probability, 4),
                        }
                    ),
                )
            )
        scored.sort(key=lambda t: (-t[0], -t[2].rrf, self.chunks[t[1]].chunk_id))

        if not scored:
            stats.refused = True
            stats.latency_ms = round((perf_counter() - started) * 1000, 3)
            return [], stats
        stats.top_gate = round(scored[0][2].gate, 4)
        if stats.top_gate < min_score:
            stats.refused = True
            stats.latency_ms = round((perf_counter() - started) * 1000, 3)
            return [], stats

        selected = _limit_per_doc(scored, self.chunks, self.max_chunks_per_doc)[:top_k]
        hits = [
            Hit(
                doc_id=self.chunks[index].doc_id,
                title=self.chunks[index].title,
                quote=snippet(self.chunks[index].text, query),
                score=round(breakdown.gate, 4),
                chunk_id=self.chunks[index].chunk_id,
                breakdown=breakdown.as_dict(),
            )
            for _final, index, breakdown in selected
        ]
        stats.returned = len(hits)
        stats.context_chars = sum(len(h.quote) for h in hits)
        stats.latency_ms = round((perf_counter() - started) * 1000, 3)
        return hits, stats

    # ------------------------------------------------------- reranker training
    def labelled_rows(
        self,
        queries: Sequence[str],
        relevant: Sequence[set[str]],
    ) -> tuple[list[np.ndarray], list[int], list[str]]:
        """Corpus-wide feature rows for ``(query, relevant doc_ids)`` pairs.

        Every non-relevant document contributes negatives, so the fitted weights
        are not an artifact of the candidate pool.
        """
        rows: list[np.ndarray] = []
        labels: list[int] = []
        keys: list[str] = []
        for query, docs in zip(queries, relevant):
            for chunk, features, _bm25, _dense in self.feature_rows(query):
                rows.append(features)
                labels.append(1 if chunk.doc_id in set(docs) else 0)
                keys.append(query)
        return rows, labels, keys


def rrf_score(ranks: Sequence[int], *, k: int = DEFAULT_RRF_K) -> float:
    """``sum(1 / (k + rank))`` over 1-based ranks — the RRF fusion function."""
    return float(sum(1.0 / (k + r) for r in ranks if r and r > 0))


def _limit_per_doc(
    candidates: Sequence[tuple[float, int, Breakdown]],
    chunks: Sequence[Chunk],
    limit: int,
) -> list[tuple[float, int, Breakdown]]:
    """Near-duplicate suppression: at most ``limit`` chunks per document."""
    if limit <= 0:
        return list(candidates)
    seen: dict[str, int] = {}
    out: list[tuple[float, int, Breakdown]] = []
    for item in candidates:
        doc_id = chunks[item[1]].doc_id
        if seen.get(doc_id, 0) >= limit:
            continue
        seen[doc_id] = seen.get(doc_id, 0) + 1
        out.append(item)
    return out


def _mode_name(use_lexical: bool, use_dense: bool, rerank: bool) -> str:
    if use_lexical and not use_dense:
        base = "lexical"
    elif use_dense and not use_lexical:
        base = "dense"
    else:
        base = "hybrid"
    return f"{base}+rerank" if rerank else base
