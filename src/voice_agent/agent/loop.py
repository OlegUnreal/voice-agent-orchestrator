from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import structlog

from ..config import Settings
from ..providers.base import ChatMessage, LLMProvider, ToolSpec
from ..session.store import SessionStore
from ..tools.registry import ToolRegistry

log = structlog.get_logger(__name__)


@dataclass
class AgentResult:
    text: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "completed"


class AgentLoop:
    """The heart of the agent: stream text, execute tool calls, repeat.

    Hard cap on iterations prevents runaway loops. On tool failure the error is
    fed back to the model as a tool message so it can recover or apologise.
    """

    def __init__(
        self, provider: LLMProvider, tools: ToolRegistry, settings: Settings) -> None:
        self._provider = provider
        self._tools = tools
        self._settings = settings

    async def run(self, session_id: str, store: SessionStore, user_text: str) -> AgentResult:
        messages = store.get_messages(session_id)
        messages.append(ChatMessage(role="user", content=user_text))
        store.append(session_id, messages[-1])

        trace: list[dict[str, Any]] = []
        iterations = 0
        max_iter = self._settings.agent_max_tool_iterations

        while iterations < max_iter:
            iterations += 1
            text_parts: list[str] = []
            pending_calls: dict[int, dict[str, Any]] = {}

            async for event in self._provider.stream(
                messages, self._tool_specs(), self._settings.agent_temperature
            ):
                et = event["type"]
                if et == "text.delta":
                    text_parts.append(event["text"])
                    yield_text = event["text"]  # streamed to client by the API layer
                    log.debug("text.delta", text=yield_text)
                elif et == "tool_call.delta":
                    idx = event["index"]
                    slot = pending_calls.setdefault(
                        idx, {"id": None, "name": None, "arguments": ""}
                    )
                    if event.get("id"):
                        slot["id"] = event["id"]
                    if event.get("name"):
                        slot["name"] = event["name"]
                    slot["arguments"] += event.get("arguments", "")
                elif et == "tool_call.finish":
                    pass
                elif et == "done":
                    break

            assistant_text = "".join(text_parts)
            if assistant_text:
                messages.append(ChatMessage(role="assistant", content=assistant_text))
                store.append(session_id, messages[-1])

            if not pending_calls:
                return AgentResult(text=assistant_text, tool_trace=trace, iterations=iterations)

            # Execute every tool call the model requested in this turn.
            tool_messages: list[ChatMessage] = []
            for slot in pending_calls.values():
                name = slot["name"] or ""
                args = self._parse_args(slot["arguments"])
                result, err = await self._execute(name, args)
                trace.append({"tool": name, "args": args, "result": result, "error": err})
                tool_messages.append(
                    ChatMessage(
                        role="tool",
                        content=json.dumps(result if not err else {"error": err}),
                        name=name,
                        tool_call_id=slot["id"],
                    )
                )
            messages.extend(tool_messages)
            store.append(session_id, *tool_messages)

        return AgentResult(
            text=assistant_text,
            tool_trace=trace,
            iterations=iterations,
            stopped_reason="max_iterations",
        )

    def _tool_specs(self) -> list[ToolSpec]:
        return [
            ToolSpec(name=t.name, description=t.description, parameters=t.parameters)
            for t in self._tools.all()
        ]

    async def _execute(self, name: str, args: dict[str, Any]) -> tuple[Any, str | None]:
        tool = self._tools.get(name)
        if tool is None:
            return None, f"Unknown tool: {name}"
        try:
            return await tool.run(**args), None
        except Exception as exc:  # noqa: BLE001 — surface to the model
            log.warning("tool_failed", tool=name, error=str(exc))
            return None, str(exc)

    @staticmethod
    def _parse_args(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
