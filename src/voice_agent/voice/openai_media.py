from __future__ import annotations

from typing import AsyncIterator

import httpx

from .media import STTProvider, TTSProvider


class OpenAICompatibleSTT(STTProvider):
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    async def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        files = {"file": ("audio.wav", audio, mime)}
        data = {"model": self._model}
        resp = await self._client.post("/audio/transcriptions", files=files, data=data)
        resp.raise_for_status()
        payload = resp.json()
        return str(payload.get("text") or "").strip()

    async def aclose(self) -> None:
        await self._client.aclose()


class OpenAICompatibleTTS(TTSProvider):
    def __init__(self, api_key: str, base_url: str, model: str, voice: str) -> None:
        self._model = model
        self._voice = voice
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=60.0,
        )

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        resp = await self._client.post(
            "/audio/speech",
            json={"model": self._model, "voice": self._voice, "input": text},
        )
        resp.raise_for_status()
        yield resp.content

    async def aclose(self) -> None:
        await self._client.aclose()
