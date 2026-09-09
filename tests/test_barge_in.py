import asyncio

import pytest

from voice_agent.agent.loop import AgentLoop
from voice_agent.config import Settings
from voice_agent.session.store import SessionStore
from voice_agent.tools.registry import ToolRegistry


class SlowProvider:
    model = "slow"

    async def stream(self, messages, tools, temperature):
        await asyncio.sleep(5)
        yield {"type": "text.delta", "text": "late"}
        yield {"type": "done"}


@pytest.mark.asyncio
async def test_barge_in_cancels_generation():
    loop = AgentLoop(SlowProvider(), ToolRegistry(), Settings(agent_max_tool_iterations=1))
    store = SessionStore()
    task = asyncio.create_task(loop.run("s", store, "hello"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
