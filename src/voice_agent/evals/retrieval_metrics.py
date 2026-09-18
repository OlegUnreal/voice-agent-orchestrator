"""Ranking metrics over the graded, document-level golden set.

What is measured, and why these definitions
-------------------------------------------
Recall@{1,3,5}
    Fraction of *known relevant documents* that appear in the top k. Denominator is
    the query's own relevant-doc count, so a query with three graded documents is
    not satisfied by a system that keeps returning the same one.
Precision@{1,3}
    Share of the top k that is relevant. ``precision@1`` is the "would the agent
    cite the right policy?" number, and it is what a voice turn can actually use.
MRR
    Reciprocal rank of the first relevant document. Rank-aware but not
    grade-aware, so it is reported next to nDCG rather than instead of it.
nDCG@{5,10}
    Graded, position-discounted gain normalised by the ideal ordering of the same
    qrels. Exponential gain (``2**grade - 1``) means a grade-3 document is worth
    seven grade-1 documents, which matches how the grades are defined in
    ``data/eval/retrieval_golden.json``: only the grade-3 document answers alone.
Hit@{1,3}
    "At least one relevant document". Cheapest metric to explain on a call.

Relevance lives at document level (see the ``notes`` in the golden file) while the
retriever ranks *chunks*, so a chunk run is collapsed to a document run by keeping
the **best** rank of each document (``best_ranks``). That is exactly the product
question: "did the right document get into the prompt, and how early".

Queries with empty qrels are *unanswerable* by construction. Their recall is
undefined, so they are excluded from every mean (``graded=False``) and are scored
separately as a refusal rate by :mod:`voice_agent.evals.retrieval`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ..rag.labels import GoldenQuery

__all__ = [
    "K_RECALL",
    "K_PRECISION",
    "K_NDCG",
    "METRIC_KEYS",
    "QueryResult",
    "aggregate",
    "best_ranks",
    "by_group",
    "dcg_at_k",
    "evaluate_run",
    "gain",
    "hit_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]

#: Cutoffs reported for each family. ``max(K_NDCG)`` is the depth a run must have.
K_RECALL: tuple[int, ...] = (1, 3, 5)
K_PRECISION: tuple[int, ...] = (1, 3)
K_NDCG: tuple[int, ...] = (5, 10)

#: Flat metric names, in report order.
METRIC_KEYS: tuple[str, ...] = (
    "recall@1",
    "recall@3",
    "recall@5",
    "precision@1",
    "precision@3",
    "mrr",
    "ndcg@5",
    "ndcg@10",
    "hit@1",
    "hit@3",
    "hit@5",
)


def gain(grade: int) -> float:
    """Exponential relevance gain: grade 3 -> 7, grade 2 -> 3, grade 1 -> 1."""
    return float(2 ** max(0, int(grade)) - 1)


def discounted_gain(grade: int, rank: int) -> float:
    """``gain / log2(rank + 1)`` with 1-based ranks: rank 1 is undiscounted."""
    return gain(grade) / math.log2(rank + 1)


def dcg_at_k(ranked_grades: Sequence[int], k: int) -> float:
    return float(sum(discounted_gain(g, r) for r, g in enumerate(ranked_grades[:k], start=1)))


def ndcg_at_k(ranked_grades: Sequence[int], ideal_grades: Iterable[int], k: int) -> float:
    """nDCG@k in ``[0, 1]``; ``0.0`` when the query has no relevant documents.

    Callers are expected to skip ungraded queries rather than read that ``0.0`` as
    a failure — :func:`evaluate_run` does exactly that.
    """
    ideal = sorted((int(g) for g in ideal_grades if g > 0), reverse=True)[:k]
    if not ideal:
        return 0.0
    best = dcg_at_k(ideal, k)
    if best <= 0.0:  # pragma: no cover - ideal is non-empty and graded, so > 0
        return 0.0
    return dcg_at_k(ranked_grades[:k], k) / best


def best_ranks(ranked: Sequence[str]) -> dict[str, int]:
    """Collapse a chunk run to a document run, keeping each document's best rank.

    Accepts either ``doc_id`` or ``doc_id#chunk`` forms. Ties are resolved by the
    first occurrence, i.e. ranking order is preserved.
    """
    ranks: dict[str, int] = {}
    for position, identifier in enumerate(ranked, start=1):
        doc_id = str(identifier).split("#", 1)[0]
        if doc_id and doc_id not in ranks:
            ranks[doc_id] = position
    return ranks


def recall_at_k(ranks: dict[str, int], relevant: dict[str, int], k: int) -> float:
    positives = [d for d, g in relevant.items() if g > 0]
    if not positives:
        return 0.0
    found = sum(1 for d in positives if ranks.get(d, 10**9) <= k)
    return found / len(positives)


def precision_at_k(ranks: dict[str, int], relevant: dict[str, int], k: int) -> float:
    if k <= 0:
        return 0.0
    hits = sum(1 for doc_id, rank in ranks.items() if rank <= k and relevant.get(doc_id, 0) > 0)
    return hits / float(k)


def hit_at_k(ranks: dict[str, int], relevant: dict[str, int], k: int) -> float:
    return 1.0 if any(
        rank <= k and relevant.get(doc_id, 0) > 0 for doc_id, rank in ranks.items()
    ) else 0.0


def reciprocal_rank(ranks: dict[str, int], relevant: dict[str, int]) -> tuple[float, int | None]:
    """``(1 / first relevant rank, that rank)`` — ``(0.0, None)`` when absent."""
    positives = [rank for doc_id, rank in ranks.items() if relevant.get(doc_id, 0) > 0]
    if not positives:
        return 0.0, None
    best = min(positives)
    return 1.0 / best, best


@dataclass(frozen=True)
class QueryResult:
    """Per-query metrics plus enough state to re-check the numbers by hand."""

    qid: str
    query: str
    group: str
    graded: bool
    n_relevant: int
    first_relevant_rank: int | None
    ranked: tuple[str, ...]
    metrics: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "qid": self.qid,
            "group": self.group,
            "query": self.query,
            "n_relevant": self.n_relevant,
            "first_relevant_rank": self.first_relevant_rank,
            "top_docs": list(self.ranked[:5]),
        }
        row.update({k: round(v, 4) for k, v in self.metrics.items()})
        return row


def evaluate_run(ranked: Sequence[str], golden: GoldenQuery, *, k_depth: int | None = None) -> QueryResult:
    """Score one retrieved ranking (best-first doc or chunk ids) against one qrel."""
    relevant = {d: g for d, g in golden.relevant.items() if g > 0}
    ranks = best_ranks(ranked)
    depth = max(K_NDCG) if k_depth is None else int(k_depth)
    # Grades per rank position, for the discounted measures.
    graded_by_rank = [relevant.get(doc_id, 0) for doc_id, _rank in sorted(ranks.items(), key=lambda kv: kv[1])]
    rr, first = reciprocal_rank(ranks, relevant)
    metrics: dict[str, float] = {}
    if relevant:
        for k in K_RECALL:
            metrics[f"recall@{k}"] = recall_at_k(ranks, relevant, k)
        for k in K_PRECISION:
            metrics[f"precision@{k}"] = precision_at_k(ranks, relevant, k)
        for k in K_NDCG:
            metrics[f"ndcg@{k}"] = ndcg_at_k(graded_by_rank, golden.ideal_grades(), min(k, depth))
        for k in (1, 3, 5):
            metrics[f"hit@{k}"] = hit_at_k(ranks, relevant, k)
        metrics["mrr"] = rr
    return QueryResult(
        qid=golden.qid,
        query=golden.query,
        group=golden.group,
        graded=bool(relevant),
        n_relevant=len(relevant),
        first_relevant_rank=first,
        ranked=tuple(sorted(ranks, key=lambda d: ranks[d])[:depth]),
        metrics=metrics,
    )


def evaluate_runs(runs: dict[str, Sequence[str]], golden: Sequence[GoldenQuery]) -> list[QueryResult]:
    """Score ``{qid: ranked ids}`` against a golden set; unknown qids score empty."""
    by_qid = {g.qid: g for g in golden}
    results: list[QueryResult] = []
    for qid, ranked in runs.items():
        query = by_qid.get(qid)
        if query is None:  # pragma: no cover - guarded by the caller
            continue
        results.append(evaluate_run(ranked, query))
    return results


def aggregate(results: Sequence[QueryResult]) -> dict[str, float]:
    """Mean of every metric over *graded* queries (unanswerable ones excluded).

    A missing metric for a query counts as 0.0 — a system that returns nothing has
    recall 0, and silently averaging over only the queries it answered is how
    retrieval evals lie.
    """
    graded = [r for r in results if r.graded]
    n = len(graded)
    out: dict[str, float] = {"queries": float(n), "unanswerable": float(len(results) - n)}
    for key in METRIC_KEYS:
        values = [r.metrics.get(key, 0.0) for r in graded]
        out[key] = round(sum(values) / n, 4) if n else 0.0
    return out


def by_group(results: Sequence[QueryResult]) -> dict[str, dict[str, float]]:
    """Same aggregation, bucketed by the golden set's ``group`` labels."""
    buckets: dict[str, list[QueryResult]] = {}
    for result in results:
        buckets.setdefault(result.group, []).append(result)
    return {
        group: {
            key: value
            for key, value in aggregate(items).items()
            if key in {"queries", "recall@5", "mrr", "ndcg@10", "precision@1"}
        }
        for group, items in sorted(buckets.items())
    }
