from __future__ import annotations

from abc import ABC, abstractmethod
from typing import AsyncIterator


class STTProvider(ABC):
    @abstractmethod
    async def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str: ...


class TTSProvider(ABC):
    @abstractmethod
    def synthesize(self, text: str) -> AsyncIterator[bytes]: ...


class PassthroughSTT(STTProvider):
    """Tests and local demos: UTF-8 bytes in the audio frame are the transcript."""

    async def transcribe(self, audio: bytes, mime: str = "audio/wav") -> str:
        return audio.decode("utf-8", errors="replace").strip()


class EchoTTS(TTSProvider):
    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        yield text.encode("utf-8")
