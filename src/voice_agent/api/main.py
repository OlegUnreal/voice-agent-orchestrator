from __future__ import annotations

import asyncio
import base64
import json
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles

from ..agent.loop import AgentLoop
from ..config import Settings, get_settings
from ..evals.scripted import ScriptedEvalProvider
from ..observability.metrics import MetricsRegistry
from ..providers.openai_compatible import OpenAICompatibleProvider
from ..rag.seed import load_knowledge
from ..session.store import SessionStore
from ..tools.examples import calculator, current_time
from ..tools.knowledge import KnowledgeSearchTool
from ..tools.registry import ToolRegistry
from ..tools.sensitive import IssueRefundTool
from ..voice.media import EchoTTS, PassthroughSTT
from ..voice.openai_media import OpenAICompatibleSTT, OpenAICompatibleTTS
from ..voice.pipeline import VoicePipeline, encode_ws

log = structlog.get_logger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def build_runtime(settings: Settings):
    metrics = MetricsRegistry()
    store = SessionStore()
    kb = load_knowledge(Path(__file__).resolve().parents[3] / "data" / "kb")
    registry = ToolRegistry()
    registry.register(current_time)
    registry.register(calculator)
    registry.register(
        KnowledgeSearchTool(kb, top_k=settings.rag_top_k, min_score=settings.rag_min_score)
    )
    registry.register(IssueRefundTool(auto_approve=settings.auto_approve_sensitive))

    if settings.provider_api_key:
        provider = OpenAICompatibleProvider(
            api_key=settings.provider_api_key,
            base_url=settings.provider_base_url,
            model=settings.provider_model,
        )
        stt = OpenAICompatibleSTT(
            settings.provider_api_key, settings.provider_base_url, settings.stt_model
        )
        tts = OpenAICompatibleTTS(
            settings.provider_api_key,
            settings.provider_base_url,
            settings.tts_model,
            settings.tts_voice,
        )
    else:
        provider = ScriptedEvalProvider()
        stt = PassthroughSTT()
        tts = EchoTTS()

    agent = AgentLoop(provider, registry, settings, metrics=metrics)
    pipeline = VoicePipeline(agent, store, stt, tts)
    return store, agent, provider, stt, tts, metrics, pipeline


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store, agent, provider, stt, tts, metrics, pipeline = build_runtime(settings)
        app.state.settings = settings
        app.state.store = store
        app.state.agent = agent
        app.state.provider = provider
        app.state.metrics = metrics
        app.state.pipeline = pipeline
        app.state.closers = [provider, stt, tts]
        log.info("agent_ready", model=getattr(provider, "model", "unknown"), live=bool(settings.provider_api_key))
        yield
        for closer in app.state.closers:
            close = getattr(closer, "aclose", None)
            if close:
                await close()

    app = FastAPI(title="Voice Agent Orchestrator", lifespan=lifespan)

    @app.websocket("/ws/voice")
    async def voice_ws(ws: WebSocket) -> None:
        await ws.accept()
        session_id = ws.query_params.get("session_id") or str(uuid.uuid4())
        pipeline: VoicePipeline = app.state.pipeline
        current: asyncio.Task | None = None

        async def pump(events) -> None:
            async for event in events:
                await ws.send_text(encode_ws(event))

        try:
            while True:
                raw = await ws.receive_text()
                frame = json.loads(raw)
                kind = frame.get("type")
                if kind == "interrupt":
                    if current and not current.done():
                        current.cancel()
                        try:
                            await current
                        except asyncio.CancelledError:
                            pass
                        await ws.send_text(
                            json.dumps({"type": "assistant.interrupted", "session_id": session_id})
                        )
                    continue
                if kind == "user_text":
                    text = frame.get("text", "")
                    if not text:
                        continue
                    source = pipeline.handle_text(session_id, text)
                elif kind == "user_audio":
                    audio = base64.b64decode(frame.get("audio_b64") or "")
                    source = pipeline.handle_audio(
                        session_id, audio, frame.get("mime") or "audio/wav"
                    )
                else:
                    continue
                current = asyncio.create_task(pump(source))
                try:
                    await current
                except asyncio.CancelledError:
                    log.info("turn_cancelled", session_id=session_id)
                finally:
                    current = None
        except WebSocketDisconnect:
            log.info("ws_disconnected", session_id=session_id)
        except Exception as exc:  # noqa: BLE001
            log.exception("ws_error", session_id=session_id)
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))

    @app.get("/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "model": getattr(app.state.provider, "model", None),
            "live_provider": bool(settings.provider_api_key),
        }

    @app.get("/metrics")
    async def metrics() -> dict[str, Any]:
        return app.state.metrics.snapshot()

    @app.get("/traces/{session_id}")
    async def traces(session_id: str) -> dict[str, Any]:
        items = [t for t in app.state.metrics.traces if t.session_id == session_id]
        payload = {
            "session_id": session_id,
            "turns": [
                {
                    "latency_ms": t.latency_ms,
                    "cost_usd": t.cost_usd,
                    "tools": t.tools,
                    "iterations": t.iterations,
                    "stopped_reason": t.stopped_reason,
                }
                for t in items
            ],
        }
        return payload

    if STATIC_DIR.exists():
        app.mount("/demo", StaticFiles(directory=STATIC_DIR, html=True), name="demo")
    return app


app = create_app()
