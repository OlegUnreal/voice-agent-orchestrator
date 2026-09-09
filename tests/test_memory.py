from __future__ import annotations

from voice_agent.providers.base import ChatMessage
from voice_agent.session.memory import compact_messages


def test_compact_keeps_system_and_recent_turns():
    system = ChatMessage(role="system", content="rules")
    old = [ChatMessage(role="user", content=f"old {i}") for i in range(10)]
    recent = [ChatMessage(role="user", content="latest"), ChatMessage(role="assistant", content="ok")]
    compacted = compact_messages([system, *old, *recent], max_messages=4)
    assert compacted[0].role == "system"
    assert compacted[0].content == "rules"
    assert any(m.content and m.content.startswith("[memory_summary]") for m in compacted)
    assert compacted[-1].content == "ok"
    assert len(compacted) <= 5
