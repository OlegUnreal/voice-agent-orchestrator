import pytest

from voice_agent.providers.base import ChatMessage
from voice_agent.providers.openai_compatible import OpenAICompatibleProvider
from voice_agent.tools.examples import calculator


@pytest.mark.asyncio
async def test_calculator_tool():
    result = await calculator.run(expression="(2 + 3) * 4")
    assert result == 20


@pytest.mark.asyncio
async def test_calculator_rejects_injection():
    with pytest.raises(ValueError):
        await calculator.run(expression="__import__('os').system('rm -rf /')")


def test_provider_payload_roundtrip():
    msg = ChatMessage(
        role="tool",
        content='{"ok": true}',
        name="calculator",
        tool_call_id="c1",
    )
    assert OpenAICompatibleProvider._to_payload(msg) == {
        "role": "tool",
        "content": '{"ok": true}',
        "tool_call_id": "c1",
        "name": "calculator",
    }

    assistant = ChatMessage(
        role="assistant",
        content=None,
        tool_calls=[
            {
                "id": "c1",
                "type": "function",
                "function": {"name": "calculator", "arguments": "{}"},
            }
        ],
    )
    payload = OpenAICompatibleProvider._to_payload(assistant)
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["id"] == "c1"
    assert "content" not in payload
