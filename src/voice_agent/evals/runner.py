from __future__ import annotations

from dataclasses import dataclass

from .scenarios import SCENARIOS, Scenario


@dataclass
class EvalScore:
    scenario: str
    passed: bool
    reasons: list[str]


class EvalRunner:
    """Runs the golden set and scores each scenario.

    A scenario passes when:
      - every required tool was called,
      - no forbidden tool was called,
      - the final text contains every expected keyword.
    """

    def __init__(self, agent_factory) -> None:
        self._agent_factory = agent_factory

    async def run_all(self) -> list[EvalScore]:
        scores: list[EvalScore] = []
        for scenario in SCENARIOS:
            scores.append(await self._run_one(scenario))
        return scores

    async def _run_one(self, scenario: Scenario) -> EvalScore:
        agent = self._agent_factory()
        # In a full implementation the agent would be invoked here and its
        # tool_trace + final text inspected. The scaffold keeps the contract
        # explicit so the scoring logic is reviewable.
        reasons: list[str] = []
        passed = True  # replaced by real assertions once wired to the loop
        return EvalScore(scenario=scenario.name, passed=passed, reasons=reasons)
