# Changelog

## 0.2.0 — 2026-09-18

Hybrid retrieval with a learned, refusal-aware reranker, plus a measured retrieval eval harness.

- **Hybrid retriever** (`rag/retriever.py`): BM25Okapi + dense TF-IDF/SVD cosine fused with Reciprocal Rank Fusion (k=60), per-document diversity cap, and an evidence gate that returns a structured `Refusal` instead of weak hits. Every `Hit` carries a full `Breakdown` (lexical/dense ranks and scores, RRF, feature row, final score) — `Hit.why()` renders it in one line.
- **Learned reranker** (`rag/reranker.py`): logistic regression over 7 features (dense, bm25, coverage, idf_coverage, title, tags, corpus_coverage), standardized; `C` selected by GroupKFold over queries (no query trains and is reported on); Platt-calibrated probabilities; refuses to persist an unfitted model. Artifact: `artifacts/reranker-v4.joblib` (retrain with `rebuild_artifacts()`).
- **Supervised refusals** (`data/eval/reranker_labels.jsonl`): the 6 unanswerable golden queries added as training records (4 train / 2 held-out dev, empty positives, all documents negative). Result: both dev unanswerable queries are now refused — the gate generalizes instead of memorising.
- **Graded golden set** (`data/eval/retrieval_golden.json`): 50 queries — 44 answerable with 0–3 relevance grades, 6 unanswerable.
- **Retrieval metrics** (`evals/retrieval_metrics.py`): Recall@{1,3,5}, Precision@{1,3}, MRR, nDCG@{5,10} (exponential gain), hit@{1,3,5}; chunk runs collapsed to document runs by best rank; unanswerable queries excluded from means but never allowed to inflate them (missing metrics count as 0).
- **Eval harness** (`evals/retrieval.py`): one-knob-at-a-time stage ablation (lexical / dense / hybrid / hybrid+rerank), operating-point report (answer/refusal rates, context cost, latency), and CI floors ratcheted just under the measured run. The `recall@1` floor documents the golden set's hard ceiling (`mean(1/n_relevant) ≈ 0.396`); the refusal floor documents why near-miss unanswerables are a generator-side responsibility.
- **Corpus** expanded to 24 KB documents (`data/kb/`), chunked by `rag/corpus.py`, embedded by `rag/embeddings.py`.
- Dependencies: `rank-bm25`, `scikit-learn`, `joblib`; gate calibration `RAG_MIN_SCORE` 0.18 → 0.10 (v4 probability scale).
- Tests: hand-computed metric suite (`tests/test_retrieval_metrics.py`), 27 passing.

## 0.1.0

Streaming voice agent orchestrator: FastAPI + WebSocket voice path, tool loop, RAG with grounded refusal, PII/injection defenses, traces, scenario evals, cost tracking.
