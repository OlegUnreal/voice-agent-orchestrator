from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scenario:
    name: str
    user_input: str
    must_call: list[str]          # tools the agent is expected to invoke
    must_not_call: list[str]      # tools it must avoid
    expected_keywords: list[str]  # substrings the final answer should contain


SCENARIOS: list[Scenario] = [
    Scenario(
        name="time_query",
        user_input="What time is it right now?",
        must_call=["current_time"],
        must_not_call=[],
        expected_keywords=["UTC"],
    ),
    Scenario(
        name="arithmetic",
        user_input="Calculate (2 + 3) * 4 for me.",
        must_call=["calculator"],
        must_not_call=[],
        expected_keywords=["20"],
    ),
    Scenario(
        name="no_tool_needed",
        user_input="Say hello in one word.",
        must_call=[],
        must_not_call=["current_time", "calculator"],
        expected_keywords=["hello"],
    ),
]
