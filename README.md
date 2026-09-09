# Voice Agent Orchestrator

Production-style voice agent backend: provider abstraction, streaming tool loop, RAG with grounded refusal, PII/injection defenses, traces, and evals that actually fail.

Offline mode (empty `PROVIDER_API_KEY`) uses a scripted model so `pytest`, golden evals, and the demo UI run without network.

## Stack

- Python 3.11+
- FastAPI + WebSocket
- OpenAI-compatible chat, STT, and TTS (optional)
- Hashing embeddings + lexical gate for RAG (numpy, no PyTorch)
- pytest + GitHub Actions as the eval gate

## Layout

```
src/voice_agent/
  agent/           # tool loop, parallel tools, budgets, traces
  voice/           # STT → agent → TTS, barge-in cancel
  rag/             # embeddings, knowledge store
  safety/          # PII redaction, injection flags
  observability/    # latency, tokens, cost
  evals/           # golden scenarios + scorer
  api/             # HTTP + /demo
data/kb/           # retrieval corpus
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
python -m voice_agent.evals
```

## What this is meant to show

1. **Evals as a gate** — golden scenarios drive the real loop; CI fails if tools, keywords, or PII checks regress.
2. **Grounded RAG** — `knowledge_search` plus refuse-when-empty, not parametric guessing.
3. **Voice path** — audio frames, passthrough or Whisper-compatible STT, TTS chunks, interrupt cancels the turn.
4. **Safety** — PII stripped before the model; injection wrapped as untrusted data; refunds need approval.
5. **Unit economics** — every turn records latency, token estimates, and USD.

## License

Private. All rights reserved.
