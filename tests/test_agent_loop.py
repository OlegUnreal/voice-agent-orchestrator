import pytest

from voice_agent.agent.loop import AgentLoop
from voice_agent.config import Settings
from voice_agent.providers.base import ChatMessage
from voice_agent.session.store import SessionStore
from voice_agent.tools.examples import calculator, current_time
from voice_agent.tools.registry import ToolRegistry


class FakeProvider:
    """Deterministic stand-in for the real LLM — no network in unit tests."""

    def __init__(self, script): self._script = list(script)
    @property
    def model(self): return "fake"
    async def stream(self, messages, tools, temperature):
        while self._script:
            yield self._script.pop(0)


@pytest.mark.asyncio
async def test_loop_calls_tool_then_answers():
    provider = FakeProvider([
        {"type": "tool_call.delta", "index": 0, "id": "c1", "name": "calculator", "arguments": '{"expression": "2+2"}'},
        {"type": "tool_call.finish"},
        {"type": "done"},
    ])
    reg = ToolRegistry(); reg.register(calculator); reg.register(current_time)
    settings = Settings(agent_max_tool_iterations=3, agent_temperature=0.0)
    loop = AgentLoop(provider, reg, settings)
    store = SessionStore()
    result = await loop.run("s1", store, "what is 2+2?")
    assert result.iterations == 1
    assert result.tool_trace and result.tool_trace[0]["tool"] == "calculator"
