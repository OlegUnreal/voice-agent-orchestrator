# Voice Agent Orchestrator

[![CI](https://github.com/OlegUnreal/voice-agent-orchestrator/actions/workflows/ci.yml/badge.svg)](https://github.com/OlegUnreal/voice-agent-orchestrator/actions/workflows/ci.yml)

Production-style voice agent backend: provider abstraction, streaming tool loop, RAG with grounded refusal, PII/injection defenses, traces, and evals that actually fail.

Offline mode (empty `PROVIDER_API_KEY`) uses a scripted model so `pytest`, golden evals, and the demo UI run without network.

> **Not the same as [voice-agent-orchestration (Helix)](https://github.com/OlegUnreal/voice-agent-orchestration).** This repo is the standalone Python backend that proves the retrieval + safety pattern independently. Helix is the full tool-first supervisor with TypeScript voice UI, MCP integration, and quant tools. This repo focuses on the retrieval pipeline, eval harness, and safety layer.

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
git clone https://github.com/OlegUnreal/voice-agent-orchestrator.git
cd voice-agent-orchestrator

python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
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

## Libraries used and why

| Library | Version | Why it is here |
|---|---|---|
| `fastapi` | `>=0.110` | ASGI framework for the HTTP + WebSocket API. Sync handlers dispatched to Starlette's threadpool; no connection-state bugs. |
| `uvicorn[standard]` | `>=0.27` | ASGI server. The `[standard]` extra pulls in `uvloop`/`httptools` for better throughput. |
| `pydantic` | `>=2.7` | Request/response models, tool schemas, eval rubrics. Strict validation catches malformed tool output before it reaches the user. |
| `openai` | `>=1.40` | Typed client for chat, STT, TTS. Provider abstraction means the same code works with any OpenAI-compatible endpoint. |
| `rank-bm25` | `>=0.2` | Okapi BM25 for lexical retrieval. Term-frequency saturation + IDF weighting, unlike raw TF which has no ceiling. |
| `numpy` | `>=1.26` | Dense embeddings, cosine similarity, RRF score blending, feature arrays for the reranker. |
| `scikit-learn` | `>=1.4` | TF-IDF + TruncatedSVD for dense embeddings, logistic regression reranker with Platt calibration, GroupKFold for hyperparameter selection. |
| `scipy` | `>=1.11` | Sparse matrix ops for TF-IDF, statistical tests for eval metrics. |
| `python-dotenv` | `>=1.0` | Loads `.env` so the API key never lands in source control. |
| `pytest` | `>=8.0` | (dev) Test runner for agent loop, retrieval, safety, evals. Offline mode means CI stays green without network. |
| `httpx` | `>=0.27` | Async HTTP client for testing FastAPI endpoints without spinning up a real server. |

Why no vector DB: the whole retrieval layer is `TfidfVectorizer` + `TruncatedSVD` + BM25 + RRF. A corpus of 24 documents doesn't need pgvector or FAISS — an in-process index removes the network dependency, keeps the suite deterministic and offline, and makes the retrieval math reviewable in one file.

Why scikit-learn over PyTorch for the reranker: a 7-feature logistic regression trains in milliseconds, ships as a joblib file, and needs no GPU. PyTorch would be overkill for a linear model. The fine-tuned BERT reranker lives in voice-agent-orchestration (Helix), not here.

## Retrieval and evaluation

One `search()` runs: BM25 and dense cosine in parallel → Reciprocal Rank Fusion → optional learned reranker → evidence gate → per-document diversity cap.

- **Reranker**: logistic regression over 7 features (dense, bm25, query-coverage, idf-coverage, title, tags, corpus-coverage), standardized, hyperparameter `C` picked by GroupKFold *over queries* so no query trains and is reported on, Platt-calibrated to a probability, trained from `data/eval/reranker_labels.jsonl` (positives / hard negatives / weak negatives per query). Refusals are supervised: the labels include unanswerable questions where every document is a negative.
- **Evaluation** (`python -m voice_agent.evals.retrieval`): a 50-query graded golden set (44 answerable with graded relevance, 6 unanswerable) scored with Recall@{1,3,5}, Precision@{1,3}, MRR, nDCG@{5,10}, hit@{1,3,5}, refusal and answer rates. The report includes a one-knob-at-a-time stage ablation (lexical / dense / hybrid / hybrid+rerank) and floors that fail CI on regression.
- **Measured operating point** (top_k=3, gate=0.10): P@1 0.93, MRR 0.96, nDCG@10 0.83, ~6 ms/query, 42/44 graded questions answered, 4/6 unanswerable refused.
- **Honest ceilings**: the qrels give each query 2–4 relevant documents, so Recall@1 tops out at `mean(1/n_relevant) ≈ 0.396` — the shipped system runs at ~94% of that ceiling. Two unanswerable queries are *near-miss* (every retrieval feature looks answerable); refusing those belongs to the generator, not the retriever. Both facts are documented next to the CI floors.

## What this is meant to show

1. **Evals as a gate** — golden scenarios drive the real loop; CI fails if tools, keywords, or PII checks regress.
2. **Grounded RAG** — `knowledge_search` plus refuse-when-empty, not parametric guessing.
3. **Hybrid retrieval with a learned reranker** — every `Hit` carries a breakdown (lexical rank, dense rank, RRF, feature values, calibrated probability) via `Hit.why()`, so ranking decisions are auditable, and the eval harness measures what each stage adds.
4. **Voice path** — audio frames, passthrough or Whisper-compatible STT, TTS chunks, interrupt cancels the turn.
5. **Safety** — PII stripped before the model; injection wrapped as untrusted data; refunds need approval.
6. **Unit economics** — every turn records latency, token estimates, and USD.

## Design decisions (interview notes)

1. **Why hybrid retrieval (BM25 + dense) instead of just dense?**
   Dense embeddings capture semantic similarity but miss exact keyword matches. BM25 catches "BTC" when the query says "Bitcoin" but the document says "BTC" — exact match matters for ticker symbols. RRF fuses on rank, not score, so the two incomparable scales don't need normalization.

2. **Why a learned reranker instead of just BM25 + dense?**
   First-stage retrieval is necessarily coarse — it scores each document independently. The reranker sees the query-document pair and learns interaction features (does the title match? do the tags align? is the corpus coverage broad?). A 7-feature logistic regression is interpretable and trains in milliseconds.

3. **Why Platt calibration on the reranker?**
   Raw logistic regression output is a logit, not a probability. Platt calibration maps it to a calibrated probability so the evidence gate threshold (0.10) means something: "if the probability is below 0.10, refuse to answer." Without calibration, the threshold is arbitrary.

4. **Why GroupKFold over queries for hyperparameter selection?**
   Standard KFold would let the same query appear in both train and test, leaking information. GroupKFold ensures no query trains and is reported on, so the hyperparameter `C` generalizes to unseen questions.

5. **Why offline mode with a scripted model?**
   CI should never depend on an API key or network. The scripted model returns deterministic output for known inputs, so the eval harness runs in CI and catches regressions. Live mode with a real LLM is for development and production.

6. **Why PII redaction before the model?**
   Once PII reaches the model, it's in the prompt, the logs, and potentially the training data. Redacting before the model means PII never enters the system. Injection defenses wrap user input as untrusted data so the model can't be tricked into executing actions.

7. **Why a separate eval harness instead of unit tests?**
   Unit tests check individual functions. The eval harness checks end-to-end behavior: does the retrieval pipeline return the right documents? Does the agent answer correctly? Does it refuse when it should? The eval harness is the gate that catches regressions in the full pipeline, not just the parts.

## Testing

```bash
pytest                                # 27 tests
python -m voice_agent.evals           # golden scenario evals
python -m voice_agent.evals.retrieval # retrieval ablation + gate
```

Covered: agent loop, tool dispatch, PII redaction, injection detection, retrieval metrics (Recall@k, Precision@k, MRR, nDCG), refusal behavior, WebSocket streaming, offline mode.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `PROVIDER_API_KEY` | — | OpenAI-compatible API key (empty = offline mode) |
| `PROVIDER_BASE_URL` | `https://api.openai.com/v1` | API endpoint |
| `PROVIDER_MODEL` | `gpt-4o-mini` | Chat model |
| `PROVIDER_STT_MODEL` | `whisper-1` | STT model |
| `PROVIDER_TTS_MODEL` | `tts-1` | TTS model |
| `RERANKER_MODEL_PATH` | `artifacts/reranker.joblib` | Trained reranker artifact |
| `EVAL_GATE_MRR` | `0.90` | MRR floor for CI |
| `EVAL_GATE_NDCG10` | `0.75` | nDCG@10 floor for CI |

## Project status

Working prototype with real retrieval pipeline, learned reranker, eval harness, safety layer, and offline mode. Not production — no distributed retrieval, no multi-tenancy. Strong portfolio piece for retrieval + safety interviews.

## License

Private. All rights reserved.

*Last updated: 2026-09-19*
