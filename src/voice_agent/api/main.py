from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from ..agent.loop import AgentLoop
from ..config import get_settings
from ..providers.openai_compatible import OpenAICompatibleProvider
from ..session.store import SessionStore
from ..tools.examples import calculator, current_time
from ..tools.registry import ToolRegistry

log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    store = SessionStore()
    registry = ToolRegistry()
    registry.register(current_time)
    registry.register(calculator)
    provider = OpenAICompatibleProvider(
        api_key=settings.provider_api_key,
        base_url=settings.provider_base_url,
        model=settings.provider_model,
    )
    app.state.store = store
    app.state.agent = AgentLoop(provider, registry, settings)
    app.state.provider = provider
    log.info("agent_ready", model=settings.provider_model)
    yield
    await provider.aclose()


app = FastAPI(title="Voice Agent Orchestrator", lifespan=lifespan)


@app.websocket("/ws/voice")
async def voice_ws(ws: WebSocket) -> None:
    await ws.accept()
    session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
    store: SessionStore = app.state.store
    agent: AgentLoop = app.state.agent
    store.get_or_create(session_id)

    try:
        while True:
            raw = await ws.receive_text()
            frame = json.loads(raw)
            if frame.get("type") != "user_text":
                continue
            text = frame.get("text", "")
            if not text:
                continue

            await ws.send_text(json.dumps({"type": "assistant.start", "session_id": session_id}))
            full = ""
            async for event in agent.run(session_id, store, text):
                # The loop currently returns a final result; stream deltas via a
                # thin adapter. For brevity the scaffold emits the assembled text.
                pass
            result = await agent.run(session_id, store, text)
            full = result.text
            await ws.send_text(
                json.dumps(
                    {
                        "type": "assistant.delta",
                        "session_id": session_id,
                        "text": full,
                    }
                )
            )
            await ws.send_text(
                json.dumps(
                    {
                        "type": "assistant.done",
                        "session_id": session_id,
                        "tool_trace": result.tool_trace,
                        "iterations": result.iterations,
                    }
                )
            )
    except WebSocketDisconnect:
        log.info("ws_disconnected", session_id=session_id)
    except Exception as exc:  # noqa: BLE001
        log.exception("ws_error", session_id=session_id)
        await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok"}
