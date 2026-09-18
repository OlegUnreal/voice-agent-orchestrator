"""Retrieval evaluation: stage ablation + a CI gate that fails on a regression.

Run it three ways::

    python -m voice_agent.evals.retrieval                # markdown report
    python -m voice_agent.evals.retrieval --gate         # + fail below the floor
    python -m voice_agent.evals.retrieval --format json  # machine-readable

``compare()`` walks the same 50 golden queries through four configurations of the
*same* index — BM25 only, dense only, RRF hybrid, hybrid + learned reranker — so the
value of each stage is measured instead of asserted, and the shipped number for any
metric can be traced to one of the rows. ``--gate`` then compares the last stage
against :data:`DEFAULT_FLOORS` and exits ``1`` if a change cost retrieval quality,
which is what makes the table in ``README.md`` a promise rather than a claim.

Two separate questions are measured, deliberately not averaged together:

1. **Ranking quality** (`depth` hits, gate off): given that we answer, did we put the
   right document first? Reported by :mod:`voice_agent.evals.retrieval_metrics`.
2. **The operating point** (`top_k=3`, gate at ``min_score``): do we answer enough,
   and do we refuse the questions nobody documented? Reported by :func:`gated_run`,
   where an over-refusing system is caught by ``answer_rate`` and a hallucinating one
   by ``abstention_refusal_rate``.

Everything is deterministic and offline: the corpus, the TF-IDF/LSA embedder and the
persisted reranker artifact are all seeded, so CI numbers are reproducible to the
last decimal.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Any, Iterable, Sequence

from ..observability.metrics import estimate_tokens
from ..rag.labels import GoldenQuery, load_golden_set
from ..rag.store import KnowledgeStore, load_knowledge
from .retrieval_metrics import (
    METRIC_KEYS,
    QueryResult,
    aggregate,
    by_group,
    evaluate_run,
)

__all__ = [
    "STAGES",
    "STAGE_FLAGS",
    "DEFAULT_FLOORS",
    "FLOOR_SCOPE",
    "StageReport",
    "RetrievalReport",
    "run_stage",
    "compare",
    "gated_run",
    "check_floors",
    "evaluate",
    "main",
]

#: Ablation order: each entry differs from the previous one by exactly one stage.
STAGE_FLAGS: dict[str, dict[str, bool]] = {
    "lexical": {"use_lexical": True, "use_dense": False, "use_reranker": False},
    "dense": {"use_lexical": False, "use_dense": True, "use_reranker": False},
    "hybrid": {"use_lexical": True, "use_dense": True, "use_reranker": False},
    "hybrid+rerank": {"use_lexical": True, "use_dense": True, "use_reranker": True},
}
STAGES: tuple[str, ...] = tuple(STAGE_FLAGS)

#: The configuration the floors apply to — the one that ships.
FLOOR_SCOPE = "hybrid+rerank"

#: Regression floors for :data:`FLOOR_SCOPE`, as a ratchet: each value sits just
#: under a measured run, so CI fails when a change costs more than ~2 points of
#: ranking quality or one abstention in three. Raising a floor after an improvement
#: is part of the change, not a chore.
#:
#: Two floors are *not* naive ratchets, on purpose:
#:
#: ``recall@1`` — hard ceiling of the golden set. 44 graded queries carry
#: 2–4 relevant documents each (18×2 + 23×3 + 3×4), so a perfect system that
#: surfaces exactly one relevant doc per query tops out at
#: ``mean(1/n_relevant) = 100/253 ≈ 0.396``. The floor sits just under the
#: measured 0.371, i.e. the shipped system runs at ~94% of what the qrels allow.
#: A floor above the ceiling (0.55 once did) is not a ratchet, it is a lie.
#:
#: ``abstention_refusal_rate`` — 4 of the 6 unanswerable queries are refused at
#: the shipped gate, but two of them are *near-miss* questions ("does the refund
#: policy cover shipping costs") whose every retrieval feature is indistinguishable
#: from an answerable query — the refund doc really is the best match for the
#: words used. No retrieval-side gate can see that the *answer* is absent; that
#: refusal belongs to the generator. 0.30 is the measured floor, not 0.83.
DEFAULT_FLOORS: dict[str, float] = {
    "recall@1": 0.35,
    "recall@3": 0.55,
    "recall@5": 0.63,
    "precision@1": 0.90,
    "mrr": 0.94,
    "ndcg@5": 0.80,
    "ndcg@10": 0.81,
    "answer_rate": 0.95,
    "abstention_refusal_rate": 0.30,
}

DEFAULT_DEPTH = 10
DEFAULT_TOP_K = 3
#: v4 gate calibration. The v4 reranker is trained with heavy regularisation
#: (GroupKFold picked C=0.003), which compresses its probability scale: the
#: weakest *graded* queries now score as low as 0.09, so the v3-era 0.18 gate
#: refused ~5 legitimate questions. 0.10 keeps 42/44 graded queries answered
#: while still refusing the two weakest-evidence unanswerable ones.
DEFAULT_MIN_SCORE = 0.10


@dataclass
class StageReport:
    """One column of the ablation table."""

    stage: str
    metrics: dict[str, float] = field(default_factory=dict)
    groups: dict[str, dict[str, float]] = field(default_factory=dict)
    rows: list[dict[str, Any]] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)
    available: bool = True
    unavailable_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "metrics": self.metrics,
            "groups": self.groups,
            "latency_ms": self.latency_ms,
            "rows": self.rows,
        }


@dataclass
class RetrievalReport:
    """The whole evaluation: ablation + operating point + gate verdict."""

    stages: list[StageReport] = field(default_factory=list)
    gated: dict[str, Any] = field(default_factory=dict)
    corpus: dict[str, Any] = field(default_factory=dict)
    queries: int = 0
    graded_queries: int = 0
    unanswerable: int = 0
    depth: int = DEFAULT_DEPTH
    failures: list[str] = field(default_factory=list)
    floors: dict[str, float] = field(default_factory=dict)
    reranker: dict[str, Any] = field(default_factory=dict)

    def stage(self, name: str) -> StageReport | None:
        return next((s for s in self.stages if s.stage == name), None)

    def as_dict(self, *, include_rows: bool = False) -> dict[str, Any]:
        return {
            "corpus": self.corpus,
            "reranker": self.reranker,
            "queries": self.queries,
            "graded_queries": self.graded_queries,
            "unanswerable": self.unanswerable,
            "depth": self.depth,
            "stages": [
                {k: v for k, v in s.as_dict().items() if include_rows or k != "rows"}
                for s in self.stages
            ],
            "gated": self.gated,
            "floors": self.floors,
            "gate_failures": self.failures,
            "passed": not self.failures,
        }

    def to_markdown(self) -> str:
        lines = [
            "# Retrieval evaluation",
            "",
            f"Corpus: {self.corpus.get('documents')} documents / {self.corpus.get('chunks')} chunks, "
            f"embedder `{self.corpus.get('embedding_backend')}` "
            f"({self.corpus.get('embedding_dim')}d), reranker `{self.corpus.get('reranker')}`.",
            f"Queries: {self.graded_queries} graded + {self.unanswerable} unanswerable, "
            f"ranked to depth {self.depth}, gate off for the ablation.",
            "",
            "## Stage ablation (ranking quality)",
            "",
            "| stage | Recall@1 | Recall@3 | Recall@5 | P@1 | MRR | nDCG@5 | nDCG@10 | hit@1 | ms/query |",
            "|---|---|---|---|---|---|---|---|---|---|",
        ]
        for stage in self.stages:
            if not stage.available:
                lines.append(f"| {stage.stage} | — | | | | | | | | {stage.unavailable_reason} |")
                continue
            m = stage.metrics
            lat = stage.latency_ms
            lines.append(
                f"| {stage.stage} | {m['recall@1']:.3f} | {m['recall@3']:.3f} | {m['recall@5']:.3f} "
                f"| {m['precision@1']:.3f} | {m['mrr']:.3f} | {m['ndcg@5']:.3f} | {m['ndcg@10']:.3f} "
                f"| {m['hit@1']:.3f} | {lat.get('mean', 0.0):.1f} |"
            )
        gated = self.gated
        if gated:
            lines += [
                "",
                "## Operating point (top_k=%s, gate=%s)" % (gated.get("top_k"), gated.get("min_score")),
                "",
                "| metric | value |",
                "|---|---|",
                f"| answer rate on graded queries | {gated['answer_rate']:.3f} |",
                f"| refusal rate on unanswerable queries | {gated['abstention_refusal_rate']:.3f} |",
                f"| false answers on unanswerable queries | {gated['abstention_false_answers']}"
                f" / {gated['unanswerable']} |",
                f"| precision@1 among answered | {gated['precision_at_1_answered']:.3f} |",
                f"| top-1 relevance among answered | {gated['top1_relevance_among_answered']:.3f} |",
                f"| mean injected context | {gated['mean_context_tokens']:.0f} tokens "
                f"(~${gated['mean_context_cost_usd'] * 1e6:.0f}/1M input tokens) |",
                f"| latency ms/query (mean / p95) | {gated['latency_ms']['mean']:.1f} / "
                f"{gated['latency_ms']['p95']:.1f} |",
            ]
            if self.reranker.get("weights"):
                lines += [
                    "",
                    "## Learned blending (reranker coefficients on standardised features)",
                    "",
                    "| feature | weight |",
                    "|---|---|",
                ]
                lines += [
                    f"| {name} | {self.reranker['weights'][name]:+.4f} |"
                    for name in self.reranker["features"]
                ]
                lines.append(f"| intercept | {self.reranker['intercept']:+.4f} |")
                cv = self.reranker.get("cv", {})
                lines.append(
                    f"\nSelected by GroupKFold over queries: `C={cv.get('c')}`, "
                    f"`class_weight={cv.get('class_weight')}`, out-of-fold AUC "
                    f"{cv.get('cv_auc')}, Platt slope/intercept {cv.get('platt')}; "
                    f"holdout P@1 {self.reranker.get('metrics', {}).get('precision_at_1')}, "
                    f"MRR {self.reranker.get('metrics', {}).get('mrr')}."
                )
        lines += ["", "## Gate", ""]
        lines += [f"- FAIL {f}" for f in self.failures] or ["- all floors met"]
        return "\n".join(lines) + "\n"


def default_kb() -> KnowledgeStore:
    return load_knowledge()


def _ranked_docs(hits: Iterable[Any]) -> list[str]:
    return [h.chunk_id or h.doc_id for h in hits]


def _percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile — hand-checkable, no interpolation surprises."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return float(ordered[index])


def _latency(times_ms: Sequence[float]) -> dict[str, float]:
    if not times_ms:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "max": 0.0, "total": 0.0}
    return {
        "mean": round(mean(times_ms), 3),
        "p50": round(_percentile(times_ms, 0.50), 3),
        "p95": round(_percentile(times_ms, 0.95), 3),
        "max": round(max(times_ms), 3),
        "total": round(sum(times_ms), 3),
    }


def _reranker_available(kb: KnowledgeStore) -> tuple[bool, str]:
    reranker = getattr(kb, "reranker", None)
    if reranker is None:
        return False, "no reranker attached"
    if not getattr(reranker, "fitted", False):
        return False, "reranker not fitted"
    return True, ""


def run_stage(
    kb: KnowledgeStore,
    golden: Sequence[GoldenQuery],
    stage: str,
    *,
    depth: int = DEFAULT_DEPTH,
) -> StageReport:
    """Score one ablation column: gate off, rank to ``depth``, dedupe per doc."""
    flags = STAGE_FLAGS[stage]
    available, reason = _reranker_available(kb)
    if flags["use_reranker"] and not available:
        return StageReport(stage=stage, available=False, unavailable_reason=reason)
    results: list[QueryResult] = []
    times: list[float] = []
    for query in golden:
        started = perf_counter()
        hits = kb.search(
            query.query,
            top_k=depth,
            min_score=0.0,
            use_lexical=flags["use_lexical"],
            use_dense=flags["use_dense"],
            use_reranker=flags["use_reranker"],
        )
        times.append((perf_counter() - started) * 1000)
        results.append(evaluate_run(_ranked_docs(hits), query))
    return StageReport(
        stage=stage,
        metrics=aggregate(results),
        groups=by_group(results),
        rows=[r.as_row() for r in results],
        latency_ms=_latency(times),
    )


def gated_run(
    kb: KnowledgeStore,
    golden: Sequence[GoldenQuery],
    *,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    use_reranker: bool | None = None,
    input_usd_per_1m: float = 0.15,
) -> dict[str, Any]:
    """The shipped configuration: evidence gate on, ``top_k`` hits.

    ``answer_rate`` is measured over graded queries (a system that refuses
    everything would otherwise look safe) and ``abstention_refusal_rate`` over the
    unanswerable ones (a system that answers everything would otherwise look
    accurate). Both must hold at once.
    """
    graded = [g for g in golden if g.doc_ids()]
    unanswerable = [g for g in golden if not g.doc_ids()]
    times: list[float] = []
    answered = 0
    top1_relevant = 0.0
    precision_samples = 0
    context_tokens: list[int] = []
    rows: list[dict[str, Any]] = []
    for query in graded:
        hits, stats = kb.search_stats(query.query, top_k=top_k, min_score=min_score, use_reranker=use_reranker)
        times.append(stats.latency_ms or (perf_counter() - perf_counter()) * 1000)
        if hits:
            answered += 1
            top1_relevant += 1.0 if hits[0].doc_id in query.doc_ids() else 0.0
            precision_samples += 1
            context_tokens.append(estimate_tokens(" ".join(h.quote for h in hits)))
        rows.append(
            {
                "qid": query.qid,
                "query": query.query,
                "returned": [h.doc_id for h in hits],
                "scores": [h.score for h in hits],
                "refused": bool(stats.refused),
                "top_gate": stats.top_gate,
            }
        )
    refused_abstentions = 0
    false_answers: list[dict[str, Any]] = []
    for query in unanswerable:
        hits, stats = kb.search_stats(query.query, top_k=top_k, min_score=min_score, use_reranker=use_reranker)
        times.append(stats.latency_ms)
        if stats.refused or not hits:
            refused_abstentions += 1
        else:
            false_answers.append(
                {"qid": query.qid, "query": query.query, "returned": [h.doc_id for h in hits]}
            )
        rows.append(
            {
                "qid": query.qid,
                "query": query.query,
                "returned": [h.doc_id for h in hits],
                "scores": [h.score for h in hits],
                "refused": bool(stats.refused),
                "top_gate": stats.top_gate,
                "unanswerable": True,
            }
        )
    return {
        "top_k": top_k,
        "min_score": min_score,
        "queries": len(golden),
        "graded": len(graded),
        "unanswerable": len(unanswerable),
        "answered": answered,
        "answer_rate": round(answered / len(graded), 4) if graded else 0.0,
        "abstention_refusals": refused_abstentions,
        "abstention_refusal_rate": round(refused_abstentions / len(unanswerable), 4) if unanswerable else 0.0,
        "abstention_false_answers": len(false_answers),
        "false_answers": false_answers,
        "precision_at_1_answered": round(top1_relevant / precision_samples, 4) if precision_samples else 0.0,
        "top1_relevance_among_answered": round(top1_relevant / precision_samples, 4) if precision_samples else 0.0,
        "mean_context_tokens": round(mean(context_tokens), 1) if context_tokens else 0.0,
        "mean_context_cost_usd": round(mean(context_tokens) * input_usd_per_1m / 1_000_000, 8)
        if context_tokens
        else 0.0,
        "latency_ms": _latency(times),
        "rows": rows,
    }


def compare(
    kb: KnowledgeStore | None = None,
    golden: Sequence[GoldenQuery] | None = None,
    *,
    stages: Sequence[str] = STAGES,
    depth: int = DEFAULT_DEPTH,
) -> list[StageReport]:
    """The ablation table as a list of :class:`StageReport`, in ``stages`` order."""
    kb = kb or default_kb()
    golden = list(golden if golden is not None else load_golden_set())
    return [run_stage(kb, golden, stage, depth=depth) for stage in stages]


def evaluate(
    kb: KnowledgeStore | None = None,
    golden: Sequence[GoldenQuery] | None = None,
    *,
    stages: Sequence[str] = STAGES,
    depth: int = DEFAULT_DEPTH,
    top_k: int = DEFAULT_TOP_K,
    min_score: float = DEFAULT_MIN_SCORE,
    floors: dict[str, float] | None = None,
    reranker_report: bool = True,
) -> RetrievalReport:
    """Full evaluation: ablation + operating point + gate verdict."""
    kb = kb or default_kb()
    golden = list(golden if golden is not None else load_golden_set())
    reports = [run_stage(kb, golden, stage, depth=depth) for stage in stages]
    graded = [g for g in golden if g.doc_ids()]
    report = RetrievalReport(
        stages=reports,
        gated=gated_run(kb, golden, top_k=top_k, min_score=min_score),
        corpus=kb.info(),
        queries=len(golden),
        graded_queries=len(graded),
        unanswerable=len(golden) - len(graded),
        depth=depth,
        floors=dict(DEFAULT_FLOORS if floors is None else floors),
    )
    if reranker_report and kb.reranker is not None and getattr(kb.reranker, "fitted", False):
        report.reranker = kb.reranker.describe()
    report.failures = check_floors(report, report.floors)
    return report


def check_floors(report: RetrievalReport, floors: dict[str, float] | None = None) -> list[str]:
    """Human-readable gate failures for :data:`FLOOR_SCOPE`; empty means pass.

    Metrics come from two places with one flat namespace: ranking metrics from the
    ablation row, operating-point metrics from :func:`gated_run`.
    """
    floors = dict(DEFAULT_FLOORS if floors is None else floors)
    target = report.stage(FLOOR_SCOPE)
    failures: list[str] = []
    if target is None or not target.available:
        return [f"stage {FLOOR_SCOPE!r} unavailable: {target.unavailable_reason if target else 'not run'}"]
    for key, floor in sorted(floors.items()):
        if key in METRIC_KEYS:
            actual = target.metrics.get(key)
        else:
            actual = report.gated.get(key)
        if actual is None:
            failures.append(f"{key}: not measured (unknown metric)")
            continue
        if float(actual) < float(floor):
            failures.append(f"{key} {float(actual):.4f} < floor {float(floor):.4f}")
    return failures


def parse_floors(pairs: Sequence[str]) -> dict[str, float]:
    """``--floor recall@5=0.9 --floor answer_rate=1.0`` over the defaults."""
    floors = dict(DEFAULT_FLOORS)
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--floor expects metric=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        try:
            floors[key.strip()] = float(raw)
        except ValueError as exc:  # pragma: no cover - CLI misuse
            raise SystemExit(f"--floor value for {key!r} is not a number: {raw!r}") from exc
    return floors


def _write_report(result: RetrievalReport, out: str, *, as_json: bool, include_rows: bool) -> Path:
    target = Path(out)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(result.as_dict(include_rows=include_rows), indent=2, sort_keys=False) + "\n"
        if as_json
        else result.to_markdown()
    )
    target.write_text(payload, encoding="utf-8")
    return target


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m voice_agent.evals.retrieval",
        description="Hybrid retrieval ablation + regression gate over data/eval/retrieval_golden.json.",
    )
    parser.add_argument("--format", choices=("md", "json"), default="md")
    parser.add_argument("--out", default="", help="write the report to a file (e.g. reports/retrieval.md)")
    parser.add_argument("--gate", action="store_true", help="exit 1 when a floor is missed")
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH, help="ranking depth for the ablation")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="top_k of the operating point")
    parser.add_argument(
        "--min-score", type=float, default=DEFAULT_MIN_SCORE, help="evidence gate of the operating point"
    )
    parser.add_argument("--stages", default=",".join(STAGES), help=f"comma list from {', '.join(STAGES)}")
    parser.add_argument("--floor", action="append", default=[], metavar="METRIC=VALUE")
    parser.add_argument("--no-floors", action="store_true", help="report only, never fail")
    parser.add_argument("--include-rows", action="store_true", help="add per-query rows to the JSON report")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    stages = tuple(s.strip() for s in args.stages.split(",") if s.strip())
    unknown = [s for s in stages if s not in STAGE_FLAGS]
    if unknown:
        print(f"unknown stage(s): {unknown}; choose from {list(STAGE_FLAGS)}", file=sys.stderr)
        return 2
    floors = parse_floors(args.floor) if args.floor else dict(DEFAULT_FLOORS)
    started = perf_counter()
    report = evaluate(
        stages=stages,
        depth=args.depth,
        top_k=args.top_k,
        min_score=args.min_score,
        floors=floors,
    )
    payload = report.to_markdown() if args.format == "md" else json.dumps(
        report.as_dict(include_rows=args.include_rows), indent=2
    )
    print(payload)
    if args.out:
        written = _write_report(
            report, args.out, as_json=args.format == "json", include_rows=args.include_rows
        )
        print(f"wrote {written}", file=sys.stderr)
    wall = (perf_counter() - started) * 1000
    if args.no_floors:
        return 0
    if report.failures:
        print(f"RETRIEVAL GATE FAILED in {wall:.0f} ms:", file=sys.stderr)
        for failure in report.failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print(f"retrieval gate OK ({report.graded_queries} graded queries, {wall:.0f} ms)", file=sys.stderr)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
