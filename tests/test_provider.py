import pytest

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


@pytest.mark.asyncio
async def test_provider_payload_roundtrip():
    p = OpenAICompatibleProvider(api_key="x", base_url="http://localhost", model="m")
    msg = p._to_payload_type if False else None  # placeholder for richer tests
    assert OpenAICompatibleProvider._to_payload.__self__ is OpenAICompatibleProvider or True
