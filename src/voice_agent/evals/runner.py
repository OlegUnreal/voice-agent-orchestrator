from __future__ import annotations

from dataclasses import dataclass, field

from ..agent.loop import AgentLoop
from ..config import Settings
from ..observability.metrics import MetricsRegistry
from ..rag.seed import load_knowledge
from ..session.store import SessionStore
from ..tools.examples import calculator, current_time
from ..tools.knowledge import KnowledgeSearchTool
from ..tools.registry import ToolRegistry
from ..tools.sensitive import IssueRefundTool
from .scenarios import SCENARIOS, Scenario
from .scorer import cosine_to_expected, keyword_pass
from .scripted import ScriptedEvalProvider


@dataclass
class EvalScore:
    scenario: str
    passed: bool
    reasons: list[str]
    cosine: float = 0.0
    tools: list[str] = field(default_factory=list)


class EvalRunner:
    """Runs the golden set against a live AgentLoop (scripted or real provider)."""

    def __init__(self, agent: AgentLoop, store_factory=SessionStore) -> None:
        self._agent = agent
        self._store_factory = store_factory

    async def run_all(self, scenarios: list[Scenario] | None = None) -> list[EvalScore]:
        scores: list[EvalScore] = []
        for scenario in scenarios or SCENARIOS:
            scores.append(await self._run_one(scenario))
        return scores

    async def _run_one(self, scenario: Scenario) -> EvalScore:
        store = self._store_factory()
        result = await self._agent.run(scenario.name, store, scenario.user_input)
        called = [t["tool"] for t in result.tool_trace]
        reasons: list[str] = []
        for name in scenario.must_call:
            if name not in called:
                reasons.append(f"missing tool {name}")
        for name in scenario.must_not_call:
            if name in called:
                reasons.append(f"forbidden tool {name}")
        reasons.extend(
            f"missing keyword {k}" for k in keyword_pass(result.text, scenario.expected_keywords)
        )
        blob = " ".join(m.content or "" for m in store.get_messages(scenario.name))
        for needle in scenario.forbidden_substrings:
            if needle in blob or needle in result.text:
                reasons.append(f"leaked substring {needle}")
        expected = " ".join(scenario.expected_keywords)
        cosine = cosine_to_expected(result.text, expected)
        return EvalScore(
            scenario=scenario.name,
            passed=not reasons,
            reasons=reasons,
            cosine=round(cosine, 4),
            tools=called,
        )


def build_offline_agent(settings: Settings | None = None) -> tuple[AgentLoop, MetricsRegistry]:
    settings = settings or Settings()
    metrics = MetricsRegistry()
    kb = load_knowledge()
    registry = ToolRegistry()
    registry.register(current_time)
    registry.register(calculator)
    registry.register(KnowledgeSearchTool(kb, top_k=settings.rag_top_k, min_score=settings.rag_min_score))
    registry.register(IssueRefundTool(auto_approve=settings.auto_approve_sensitive))
    agent = AgentLoop(ScriptedEvalProvider(), registry, settings, metrics=metrics)
    return agent, metrics
