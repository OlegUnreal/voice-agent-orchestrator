from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from ..config import Settings
from ..observability.metrics import (
    MetricsRegistry,
    Timer,
    TurnTrace,
    estimate_cost,
    estimate_tokens,
)
from ..prompts import SYSTEM_PROMPT
from ..providers.base import ChatMessage, LLMProvider, ToolSpec
from ..safety.injection import detect_injection, wrap_untrusted
from ..safety.pii import redact_pii
from ..session.memory import compact_messages
from ..session.store import SessionStore
from ..tools.registry import ToolRegistry
from ..tools.schema import validate_tool_args

log = structlog.get_logger(__name__)


@dataclass
class AgentResult:
    text: str
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    iterations: int = 0
    stopped_reason: str = "completed"
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    injection_flagged: bool = False
    pii_redactions: int = 0


class AgentLoop:
    """Stream text, execute tool calls, repeat — with a hard iteration cap."""

    def __init__(
        self,
        provider: LLMProvider,
        tools: ToolRegistry,
        settings: Settings,
        metrics: MetricsRegistry | None = None,
    ) -> None:
        self._provider = provider
        self._tools = tools
        self._settings = settings
        self._metrics = metrics

    async def run(self, session_id: str, store: SessionStore, user_text: str) -> AgentResult:
        result: AgentResult | None = None
        async for event in self.events(session_id, store, user_text):
            if event["type"] == "done":
                result = event["result"]
        if result is None:
            return AgentResult(text="", stopped_reason="empty")
        return result

    async def events(
        self, session_id: str, store: SessionStore, user_text: str
    ) -> AsyncIterator[dict[str, Any]]:
        timer = Timer()
        redacted = redact_pii(user_text)
        flagged = detect_injection(redacted.text)
        safe_text = wrap_untrusted(redacted.text, flagged=flagged)

        self._ensure_system(store, session_id)
        messages = compact_messages(
            store.get_messages(session_id), self._settings.memory_max_messages
        )
        store.get_or_create(session_id).messages = list(messages)

        user_msg = ChatMessage(role="user", content=safe_text)
        messages.append(user_msg)
        store.append(session_id, user_msg)

        if estimate_tokens(" ".join(m.content or "" for m in messages)) > self._settings.session_token_budget:
            messages = compact_messages(messages, max(6, self._settings.memory_max_messages // 2))
            store.get_or_create(session_id).messages = list(messages)

        trace: list[dict[str, Any]] = []
        iterations = 0
        max_iter = self._settings.agent_max_tool_iterations
        assistant_text = ""
        output_tokens = 0

        try:
            while iterations < max_iter:
                iterations += 1
                text_parts: list[str] = []
                pending_calls: dict[int, dict[str, Any]] = {}

                async for event in self._provider.stream(
                    messages, self._tool_specs(), self._settings.agent_temperature
                ):
                    et = event["type"]
                    if et == "text.delta":
                        chunk = event.get("text") or ""
                        text_parts.append(chunk)
                        yield {"type": "text.delta", "text": chunk}
                    elif et == "tool_call.delta":
                        idx = event["index"]
                        slot = pending_calls.setdefault(
                            idx, {"id": None, "name": None, "arguments": ""}
                        )
                        if event.get("id"):
                            slot["id"] = event["id"]
                        if event.get("name"):
                            slot["name"] = event["name"]
                        slot["arguments"] += event.get("arguments") or ""
                    elif et == "done":
                        break

                assistant_text = "".join(text_parts)
                output_tokens += estimate_tokens(assistant_text)
                if not pending_calls:
                    if assistant_text:
                        messages.append(ChatMessage(role="assistant", content=assistant_text))
                        store.append(session_id, messages[-1])
                    result = self._finish(
                        session_id,
                        assistant_text,
                        trace,
                        iterations,
                        "completed",
                        timer,
                        messages,
                        output_tokens,
                        flagged,
                        len(redacted.found),
                    )
                    yield {"type": "done", "result": result}
                    return

                tool_calls = [
                    {
                        "id": slot["id"],
                        "type": "function",
                        "function": {
                            "name": slot["name"] or "",
                            "arguments": slot["arguments"] or "{}",
                        },
                    }
                    for slot in pending_calls.values()
                ]
                assistant = ChatMessage(
                    role="assistant",
                    content=assistant_text or None,
                    tool_calls=tool_calls,
                )
                messages.append(assistant)
                store.append(session_id, assistant)

                executed = await asyncio.gather(
                    *[self._execute_slot(slot) for slot in pending_calls.values()]
                )
                tool_messages: list[ChatMessage] = []
                for slot, (payload, err) in zip(pending_calls.values(), executed):
                    name = slot["name"] or ""
                    trace.append(
                        {"tool": name, "args": slot.get("parsed") or {}, "result": payload, "error": err}
                    )
                    yield {
                        "type": "tool.executed",
                        "tool": name,
                        "error": err,
                    }
                    safe_payload = payload if not err else {"error": err}
                    body = redact_pii(json.dumps(safe_payload)).text
                    tool_messages.append(
                        ChatMessage(
                            role="tool",
                            content=body,
                            name=name,
                            tool_call_id=slot["id"],
                        )
                    )
                messages.extend(tool_messages)
                store.append(session_id, *tool_messages)
        except asyncio.CancelledError:
            yield {"type": "interrupted"}
            raise

        result = self._finish(
            session_id,
            assistant_text,
            trace,
            iterations,
            "max_iterations",
            timer,
            messages,
            output_tokens,
            flagged,
            len(redacted.found),
        )
        yield {"type": "done", "result": result}

    def _finish(
        self,
        session_id: str,
        text: str,
        trace: list[dict[str, Any]],
        iterations: int,
        reason: str,
        timer: Timer,
        messages: list[ChatMessage],
        output_tokens: int,
        flagged: bool,
        pii_count: int,
    ) -> AgentResult:
        input_tokens = estimate_tokens(" ".join(m.content or "" for m in messages))
        cost = estimate_cost(
            input_tokens,
            output_tokens,
            self._settings.input_token_usd_per_1m,
            self._settings.output_token_usd_per_1m,
        )
        result = AgentResult(
            text=text,
            tool_trace=trace,
            iterations=iterations,
            stopped_reason=reason,
            latency_ms=round(timer.ms(), 2),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
            injection_flagged=flagged,
            pii_redactions=pii_count,
        )
        if self._metrics:
            self._metrics.record(
                TurnTrace(
                    session_id=session_id,
                    latency_ms=result.latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    cost_usd=result.cost_usd,
                    iterations=iterations,
                    tools=[t["tool"] for t in trace],
                    stopped_reason=reason,
                    pii_redactions=pii_count,
                    injection_flagged=flagged,
                )
            )
        return result

    def _ensure_system(self, store: SessionStore, session_id: str) -> None:
        session = store.get_or_create(session_id)
        if not any(m.role == "system" for m in session.messages):
            store.append(session_id, ChatMessage(role="system", content=SYSTEM_PROMPT))

    def _tool_specs(self) -> list[ToolSpec]:
        allow = self._settings.allowlisted_tools()
        specs = []
        for t in self._tools.all():
            if allow is not None and t.name not in allow:
                continue
            specs.append(ToolSpec(name=t.name, description=t.description, parameters=t.parameters))
        return specs

    async def _execute_slot(self, slot: dict[str, Any]) -> tuple[Any, str | None]:
        name = slot["name"] or ""
        allow = self._settings.allowlisted_tools()
        if allow is not None and name not in allow:
            return None, f"Tool not allowlisted: {name}"
        tool = self._tools.get(name)
        if tool is None:
            return None, f"Unknown tool: {name}"
        try:
            args = validate_tool_args(tool.parameters, self._parse_args(slot["arguments"]))
            slot["parsed"] = args
        except ValueError as exc:
            return None, str(exc)
        if tool.requires_approval and not self._settings.auto_approve_sensitive:
            pending = {
                "status": "pending_approval",
                "tool": name,
                "args": args,
            }
            slot["parsed"] = args
            return pending, None
        try:
            result = await asyncio.wait_for(
                tool.run(**args), timeout=self._settings.tool_timeout_s
            )
            return result, None
        except TimeoutError:
            return None, f"Tool timed out: {name}"
        except Exception as exc:  # noqa: BLE001
            log.warning("tool_failed", tool=name, error=str(exc))
            return None, str(exc)

    @staticmethod
    def _parse_args(raw: str) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        return parsed if isinstance(parsed, dict) else {"_raw": raw}
