# Voice Agent Orchestrator

[![CI](https://github.com/OlegUnreal/voice-agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/voice-agent-orchestrator/actions/workflows/ci.yml)

Production-style voice agent backend: provider abstraction, streaming tool loop, RAG with grounded refusal, PII/injection defenses, traces, and evals that actually fail.

Offline mode (empty `PROVIDER_API_KEY`) uses a scripted model so `pytest`, golden evals, and the demo UI run without network.

## Stack

- Python 3.11+
- FastAPI + WebSocket
- OpenAI-compatible chat, STT, and TTS (optional)
- Hybrid retrieval: BM25 (rank-bm25) + dense TF-IDF/SVD cosine, RRF-fused, rescored by a learned reranker (scikit-learn) with calibrated refusal — no PyTorch, no vector-DB service
- pytest + GitHub Actions as the eval gate

## Layout

```
src/voice_agent/
  agent/           # tool loop, parallel tools, budgets, traces
  voice/           # STT → agent → TTS, barge-in cancel
  rag/             # corpus chunking, embeddings, hybrid retriever, learned reranker
  safety/          # PII redaction, injection flags
  observability/    # latency, tokens, cost
  evals/           # golden scenarios, retrieval metrics + ablation harness
  api/             # HTTP + /demo
data/kb/           # retrieval corpus (24 documents)
data/eval/         # graded golden set + reranker training labels
artifacts/         # trained reranker (joblib)
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
PYTHONPATH=src uvicorn voice_agent.api.main:app --reload
```

Windows PowerShell: `$env:PYTHONPATH="src"; uvicorn voice_agent.api.main:app --reload`

- Health: `GET /health`
- Metrics: `GET /metrics`
- Demo UI: `http://localhost:8000/demo/`
- WebSocket: `ws://localhost:8000/ws/voice`

```bash
pytest
python -m voice_agent.evals              # scenario evals
python -m voice_agent.evals.retrieval    # retrieval ablation + gate
```

## Retrieval and evaluation

One `search()` runs: BM25 and dense cosine in parallel → Reciprocal Rank Fusion → optional learned reranker → evidence gate → per-document diversity cap.

- **Reranker**: logistic regression over 7 features (dense, bm25, query-coverage, idf-coverage, title, tags, corpus-coverage), standardized, hyperparameter `C` picked by GroupKFold *over queries* so no query trains and is reported on, Platt-calibrated to a probability, trained from `data/eval/reranker_labels.jsonl` (positives / hard negatives / weak negatives per query). Refusals are supervised: the labels include unanswerable questions where every document is a negative.
- **Evaluation** (`python -m voice_agent.evals.retrieval`): a 50-query graded golden set (44 answerable with graded relevance, 6 unanswerable) scored with Recall@{1,3,5}, Precision@{1,3}, MRR, nDCG@{5,10}, hit@{1,3,5}, refusal and answer rates. The report includes a one-knob-at-a-time stage ablation (lexical / dense / hybrid / hybrid+rerank) and floors that fail CI on regression.
- **Measured operating point** (top_k=3, gate=0.10): P@1 0.93, MRR 0.96, nDCG@10 0.83, ~6 ms/query, 42/44 graded queries answered, 4/6 unanswerable refused.
- **Honest ceilings**: the qrels give each query 2–4 relevant documents, so Recall@1 tops out at `mean(1/n_relevant) ≈ 0.396` — the shipped system runs at ~94% of that ceiling. Two unanswerable queries are *near-miss* (every retrieval feature looks answerable); refusing those belongs to the generator, not the retriever. Both facts are documented next to the CI floors.

## What this is meant to show

1. **Evals as a gate** — golden scenarios drive the real loop; CI fails if tools, keywords, or PII checks regress.
2. **Grounded RAG** — `knowledge_search` plus refuse-when-empty, not parametric guessing.
3. **Hybrid retrieval with a learned reranker** — every `Hit` carries a breakdown (lexical rank, dense rank, RRF, feature values, calibrated probability) via `Hit.why()`, so ranking decisions are auditable, and the eval harness measures what each stage adds.
4. **Voice path** — audio frames, passthrough or Whisper-compatible STT, TTS chunks, interrupt cancels the turn.
5. **Safety** — PII stripped before the model; injection wrapped as untrusted data; refunds need approval.
6. **Unit economics** — every turn records latency, token estimates, and USD.

## License

Private. All rights reserved.
