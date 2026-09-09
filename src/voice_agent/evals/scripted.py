from __future__ import annotations

import json
from typing import Any, AsyncIterator

from ..providers.base import ChatMessage, LLMProvider, ToolSpec


class ScriptedEvalProvider(LLMProvider):
    """Offline stand-in that exercises the real tool loop for golden evals."""

    @property
    def model(self) -> str:
        return "eval-scripted"

    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        temperature: float,
    ) -> AsyncIterator[dict[str, Any]]:
        last = messages[-1]
        if last.role == "tool":
            async for event in self._after_tool(last):
                yield event
            return
        user = _last_user(messages)
        async for event in self._from_user(user):
            yield event

    async def _after_tool(self, last: ChatMessage) -> AsyncIterator[dict[str, Any]]:
        name = last.name or ""
        raw = last.content or ""
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            payload = {"_raw": raw}
        if name == "current_time":
            yield {"type": "text.delta", "text": f"The time is {payload} UTC"}
        elif name == "calculator":
            yield {"type": "text.delta", "text": str(payload)}
        elif name == "knowledge_search":
            hits = payload.get("hits") if isinstance(payload, dict) else None
            if not hits:
                yield {
                    "type": "text.delta",
                    "text": "I don't have evidence in the knowledge base for that.",
                }
            else:
                top = hits[0]
                quote = top.get("quote", "")
                title = top.get("title", "")
                yield {
                    "type": "text.delta",
                    "text": f"According to {title}: {quote}",
                }
        elif name == "issue_refund":
            yield {
                "type": "text.delta",
                "text": "A human must approve this refund before it is issued.",
            }
        else:
            yield {"type": "text.delta", "text": "Done."}
        yield {"type": "done"}

    async def _from_user(self, user: str) -> AsyncIterator[dict[str, Any]]:
        lower = user.lower()
        if "issue a refund" in lower or "account a-99" in lower:
            yield _tool("issue_refund", {"account_id": "A-99", "amount_usd": 40})
            yield {"type": "done"}
            return
        if "favorite color" in lower or "ceo" in lower:
            yield _tool("knowledge_search", {"query": "CEO favorite color"})
            yield {"type": "done"}
            return
        if "refund policy" in lower or "prepaid minutes" in lower:
            yield _tool("knowledge_search", {"query": "refund policy prepaid minutes"})
            yield {"type": "done"}
            return
        if "time" in lower:
            yield _tool("current_time", {})
            yield {"type": "done"}
            return
        if "calculate" in lower or "2 + 3" in lower:
            yield _tool("calculator", {"expression": "(2 + 3) * 4"})
            yield {"type": "done"}
            return
        yield {"type": "text.delta", "text": "hello"}
        yield {"type": "done"}


def _last_user(messages: list[ChatMessage]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content or ""
    return ""


def _tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "tool_call.delta",
        "index": 0,
        "id": f"call_{name}",
        "name": name,
        "arguments": json.dumps(args),
    }
