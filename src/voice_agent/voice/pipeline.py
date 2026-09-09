from __future__ import annotations

import json
from typing import Any

from ..agent.loop import AgentLoop, AgentResult
from ..session.store import SessionStore
from ..voice.media import STTProvider, TTSProvider


class VoicePipeline:
    """STT → agent (cancellable) → TTS chunks. Text frames skip STT."""

    def __init__(
        self,
        agent: AgentLoop,
        store: SessionStore,
        stt: STTProvider,
        tts: TTSProvider,
    ) -> None:
        self._agent = agent
        self._store = store
        self._stt = stt
        self._tts = tts

    async def handle_text(self, session_id: str, text: str):
        async for event in self._run(session_id, text):
            yield event

    async def handle_audio(self, session_id: str, audio: bytes, mime: str = "audio/wav"):
        transcript = await self._stt.transcribe(audio, mime)
        yield {"type": "transcript.final", "text": transcript, "session_id": session_id}
        if not transcript:
            return
        async for event in self._run(session_id, transcript):
            yield event

    async def _run(self, session_id: str, text: str):
        yield {"type": "assistant.start", "session_id": session_id}
        result: AgentResult | None = None
        spoken: list[str] = []
        async for event in self._agent.events(session_id, self._store, text):
            if event["type"] == "text.delta":
                spoken.append(event["text"])
                yield {**event, "session_id": session_id}
            elif event["type"] == "tool.executed":
                yield {**event, "session_id": session_id}
            elif event["type"] == "interrupted":
                yield {**event, "session_id": session_id}
                return
            elif event["type"] == "done":
                result = event["result"]
        if result is None:
            return
        full = "".join(spoken) or result.text
        if full:
            async for chunk in self._tts.synthesize(full):
                yield {
                    "type": "assistant.audio",
                    "session_id": session_id,
                    "audio_b64": _b64(chunk),
                }
        yield {
            "type": "assistant.done",
            "session_id": session_id,
            "tool_trace": result.tool_trace,
            "iterations": result.iterations,
            "latency_ms": result.latency_ms,
            "cost_usd": result.cost_usd,
            "stopped_reason": result.stopped_reason,
        }


def _b64(data: bytes) -> str:
    import base64

    return base64.b64encode(data).decode("ascii")


def encode_ws(event: dict[str, Any]) -> str:
    return json.dumps(event)
