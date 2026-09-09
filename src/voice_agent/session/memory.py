from __future__ import annotations

from ..providers.base import ChatMessage


def compact_messages(messages: list[ChatMessage], max_messages: int) -> list[ChatMessage]:
    if len(messages) <= max_messages:
        return messages
    system = [m for m in messages if m.role == "system"][:1]
    keep = max_messages - 1
    recent = [m for m in messages if m.role != "system"][-keep:]
    dropped = [m for m in messages if m not in system and m not in recent]
    blob = " ".join((m.content or "")[:120] for m in dropped if m.content)
    summary = ChatMessage(role="system", content=f"[memory_summary] {blob[:800]}")
    return [*system, summary, *recent]
