# Voice Agent Orchestrator

A production-style voice agent backend in Python. Not a toy demo — real architecture: provider abstraction, streaming, tool-calling loop, session memory, evals.

## Why this exists

Most AI agent tutorials are a single `openai.chat.completions.create` call. This project separates concerns the way a real system does:

- **Providers** are swappable (OpenAI, local vLLM, any OpenAI-compatible endpoint) behind one interface.
- **The tool loop** is a first-class citizen: the model can call tools, get results, and continue — with a hard iteration cap and structured error recovery.
- **Streaming** is end-to-end: tokens flow to the client as they are produced, not after the full response.
- **Sessions** persist conversation + tool traces so you can resume or audit.
- **Evals** run the agent against a golden set of scenarios and score it.

## Stack

- Python 3.11+
- FastAPI + Uvicorn (WebSocket + SSE)
- Pydantic v2 for config and validation
- httpx for provider calls (no hard dependency on the OpenAI SDK — keeps it provider-agnostic)
- structlog for structured logging
- pytest + pytest-asyncio

PyTorch is intentionally **not** a runtime dependency. It shows up in `evals/`, where a small embedding-based scorer ranks agent responses — the same pattern you'd use to evaluate a fine-tuned model.

## Layout

```
src/voice_agent/
  config.py          # pydantic settings, env-driven
  providers/         # LLM provider abstraction + OpenAI-compatible impl
  tools/             # tool registry, base class, example tools
  agent/             # the tool-calling loop (the heart)
  session/           # in-memory session store + trace
  api/               # FastAPI app, websocket endpoint, SSE
  evals/             # golden scenarios + scoring
tests/               # unit + integration tests
```

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # set PROVIDER_API_KEY, PROVIDER_BASE_URL
uvicorn voice_agent.api.main:app --reload
```

WebSocket: `ws://localhost:8000/ws/voice`

## Design decisions worth defending in an interview

1. **Provider interface, not SDK lock-in.** Swapping OpenAI for a local model is a config change, not a rewrite.
2. **Tool loop with a budget.** Unbounded tool loops are how agents burn money and hang. We cap iterations and surface partial results on failure.
3. **Streaming by default.** Voice latency is the product. Buffering the full response before speaking is a bug.
4. **Evals as a gate.** No prompt change ships without the golden set going green.
5. **Trace everything.** Every tool call, token count, and latency is logged — you can't improve what you can't see.

## License

Private. All rights reserved.
