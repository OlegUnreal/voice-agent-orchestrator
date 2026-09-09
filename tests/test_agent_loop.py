import pytest

from voice_agent.agent.loop import AgentLoop
from voice_agent.config import Settings
from voice_agent.session.store import SessionStore
from voice_agent.tools.examples import calculator, current_time
from voice_agent.tools.registry import ToolRegistry


class FakeProvider:
    """Deterministic stand-in for the real LLM — no network in unit tests."""

    def __init__(self, turns):
        self._turns = [list(turn) for turn in turns]

    @property
    def model(self):
        return "fake"

    async def stream(self, messages, tools, temperature):
        if not self._turns:
            return
        for event in self._turns.pop(0):
            yield event


@pytest.mark.asyncio
async def test_loop_calls_tool_then_answers():
    provider = FakeProvider(
        [
            [
                {
                    "type": "tool_call.delta",
                    "index": 0,
                    "id": "c1",
                    "name": "calculator",
                    "arguments": '{"expression": "2+2"}',
                },
                {"type": "tool_call.finish"},
                {"type": "done"},
            ],
            [
                {"type": "text.delta", "text": "4"},
                {"type": "done"},
            ],
        ]
    )
    reg = ToolRegistry()
    reg.register(calculator)
    reg.register(current_time)
    settings = Settings(agent_max_tool_iterations=3, agent_temperature=0.0)
    loop = AgentLoop(provider, reg, settings)
    store = SessionStore()
    result = await loop.run("s1", store, "what is 2+2?")
    assert result.iterations == 2
    assert result.text == "4"
    assert result.tool_trace and result.tool_trace[0]["tool"] == "calculator"
    assert result.tool_trace[0]["result"] == 4
    roles = [m.role for m in store.get_messages("s1")]
    assert roles == ["system", "user", "assistant", "tool", "assistant"]
    assert store.get_messages("s1")[2].tool_calls
