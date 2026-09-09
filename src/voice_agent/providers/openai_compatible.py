from __future__ import annotations

import json
from typing import Any, AsyncIterator

import httpx
import structlog

from .base import ChatMessage, LLMProvider, ToolSpec

log = structlog.get_logger(__name__)


class OpenAICompatibleProvider(LLMProvider):
    """Talks to any OpenAI-compatible /v1/chat/completions endpoint.

    Works with OpenAI, vLLM, LM Studio, Ollama, Together, etc. No SDK lock-in.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout,
        )

    @property
    def model(self) -> str:
        return self._model

    async def stream(
        self, messages, tools, temperature) -> AsyncIterator[dict[str, Any]]:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [self._to_payload(m) for m in messages],
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"

        async with self._client.stream("POST", "/chat/completions", json=payload) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    yield {"type": "done"}
                    return
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices", []):
                    delta = choice.get("delta", {})
                    if content := delta.get("content"):
                        yield {"type": "text.delta", "text": content}
                    if tcs := delta.get("tool_calls"):
                        for tc in tcs:
                            yield {
                                "type": "tool_call.delta",
                                "index": tc.get("index", 0),
                                "id": tc.get("id"),
                                "name": (tc.get("function") or {}).get("name"),
                                "arguments": (tc.get("function") or {}).get("arguments")
                                or "",
                            }
                    if choice.get("finish_reason") == "tool_calls":
                        yield {"type": "tool_call.finish"}

    @staticmethod
    def _to_payload(m: ChatMessage) -> dict[str, Any]:
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if m.tool_calls:
            d["tool_calls"] = m.tool_calls
        if m.tool_call_id:
            d["tool_call_id"] = m.tool_call_id
        if m.name:
            d["name"] = m.name
        return d

    async def aclose(self) -> None:
        await self._client.aclose()
