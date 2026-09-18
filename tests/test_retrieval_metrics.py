"""Hand-computed checks for the ranking metrics in ``evals.retrieval_metrics``.

Every expected number below is worked out by hand in a comment, so a reader can
verify the metric definitions without trusting the implementation.
"""

from __future__ import annotations

import pytest

from voice_agent.evals.retrieval_metrics import (
    aggregate,
    best_ranks,
    dcg_at_k,
    evaluate_run,
    evaluate_runs,
    gain,
    hit_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from voice_agent.rag.labels import GoldenQuery


def test_best_ranks_collapses_chunks_to_best_document_rank():
    # positions:        1     2     3     4
    ranked = ["a#2", "b#1", "a#1", "c#3"]
    # "a" appears at ranks 1 and 3 -> best rank 1 wins
    assert best_ranks(ranked) == {"a": 1, "b": 2, "c": 4}


def test_recall_denominator_is_the_querys_own_relevant_count():
    ranks = {"a": 1, "b": 2, "c": 5}
    relevant = {"a": 3, "b": 1, "d": 2}  # d is relevant but never retrieved
    # positives = {a, b, d}; found within k:
    assert recall_at_k(ranks, relevant, 1) == pytest.approx(1 / 3)  # a
    assert recall_at_k(ranks, relevant, 2) == pytest.approx(2 / 3)  # a, b
    assert recall_at_k(ranks, relevant, 5) == pytest.approx(2 / 3)  # d still missing
    assert recall_at_k(ranks, {}, 5) == 0.0  # no positives -> defined as 0


def test_precision_counts_relevant_within_top_k():
    ranks = {"a": 1, "b": 2, "c": 5}
    relevant = {"a": 3, "b": 1, "d": 2}
    assert precision_at_k(ranks, relevant, 1) == pytest.approx(1.0)  # a
    assert precision_at_k(ranks, relevant, 3) == pytest.approx(2 / 3)  # a, b in top 3
    assert precision_at_k(ranks, relevant, 0) == 0.0


def test_hit_is_one_when_any_relevant_is_within_k():
    ranks = {"a": 4}
    relevant = {"a": 1}
    assert hit_at_k(ranks, relevant, 3) == 0.0
    assert hit_at_k(ranks, relevant, 4) == 1.0


def test_reciprocal_rank_of_first_relevant():
    ranks = {"x": 1, "a": 2, "y": 3}
    relevant = {"a": 3}
    assert reciprocal_rank(ranks, relevant) == (0.5, 2)
    assert reciprocal_rank({"x": 1}, {"a": 3}) == (0.0, None)


def test_exponential_gain_matches_grade_definitions():
    # grade 3 is worth seven grade-1 documents: 2**3 - 1 == 7
    assert gain(3) == 7.0
    assert gain(2) == 3.0
    assert gain(1) == 1.0
    assert gain(0) == 0.0


def test_dcg_undiscounted_at_rank_one():
    # 7/log2(1+1) + 0/log2(2+1) = 7.0
    assert dcg_at_k([3, 0], 2) == pytest.approx(7.0)


def test_ndcg_normalises_by_the_ideal_ordering():
    # ideal grades [3, 1]: dcg = 7/1 + 1/log2(3) = 7.63093
    # ranked grades [3, 0]: dcg = 7.0  -> ndcg = 7/7.63093 = 0.9173
    assert ndcg_at_k([3, 0], [3, 1], 2) == pytest.approx(0.9173, abs=1e-3)
    assert ndcg_at_k([3, 1], [3, 1], 2) == 1.0
    assert ndcg_at_k([3], [], 5) == 0.0  # nothing graded -> 0, callers must skip


def test_evaluate_run_scores_one_ranking_end_to_end():
    golden = GoldenQuery(
        qid="q1",
        query="refund prepaid minutes",
        relevant={"a": 3, "b": 1},
        group="billing",
    )
    result = evaluate_run(["a#1", "c#2"], golden)
    # doc ranks: a -> 1, c -> 2; only a is relevant
    assert result.graded is True
    assert result.n_relevant == 2
    assert result.first_relevant_rank == 1
    assert result.metrics["recall@1"] == pytest.approx(0.5)
    assert result.metrics["precision@1"] == pytest.approx(1.0)
    assert result.metrics["precision@3"] == pytest.approx(1 / 3)
    assert result.metrics["mrr"] == pytest.approx(1.0)
    assert result.metrics["hit@3"] == 1.0
    # grades by rank [3, 0] vs ideal [3, 1] -> 0.9173
    assert result.metrics["ndcg@5"] == pytest.approx(0.9173, abs=1e-3)
    assert result.ranked == ("a", "c")


def test_evaluate_run_marks_empty_qrels_unanswerable():
    golden = GoldenQuery(qid="q2", query="ceo favorite color", relevant={})
    result = evaluate_run(["a#1"], golden)
    assert result.graded is False
    assert result.metrics == {}


def test_aggregate_excludes_unanswerable_but_counts_them():
    graded = evaluate_run(["a#1"], GoldenQuery(qid="q1", query="x", relevant={"a": 3}))
    ungraded = evaluate_run([], GoldenQuery(qid="q2", query="y", relevant={}))
    totals = aggregate([graded, ungraded])
    assert totals["queries"] == 1.0
    assert totals["unanswerable"] == 1.0
    # the unanswerable query must not dilute the mean
    assert totals["recall@1"] == pytest.approx(1.0)


def test_aggregate_treats_missing_metrics_as_zero_not_skips_them():
    answered = evaluate_run(["a#1"], GoldenQuery(qid="q1", query="x", relevant={"a": 3}))
    empty = evaluate_run([], GoldenQuery(qid="q2", query="y", relevant={"b": 1}))
    totals = aggregate([answered, empty])
    # a silent skip would report recall@1 = 1.0; honesty reports 0.5
    assert totals["recall@1"] == pytest.approx(0.5)


def test_evaluate_runs_ignores_unknown_qids():
    golden = [GoldenQuery(qid="q1", query="x", relevant={"a": 3})]
    results = evaluate_runs({"q1": ["a#1"], "ghost": ["b#1"]}, golden)
    assert [r.qid for r in results] == ["q1"]
