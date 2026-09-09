from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


def estimate_cost(input_tokens: int, output_tokens: int, in_rate: float, out_rate: float) -> float:
    return (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000


@dataclass
class TurnTrace:
    session_id: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    iterations: int
    tools: list[str]
    stopped_reason: str
    pii_redactions: int = 0
    injection_flagged: bool = False


@dataclass
class MetricsRegistry:
    turns: int = 0
    tool_calls: int = 0
    cost_usd: float = 0.0
    latency_ms_total: float = 0.0
    pii_redactions: int = 0
    injections: int = 0
    traces: list[TurnTrace] = field(default_factory=list)

    def record(self, trace: TurnTrace) -> None:
        self.turns += 1
        self.tool_calls += len(trace.tools)
        self.cost_usd += trace.cost_usd
        self.latency_ms_total += trace.latency_ms
        self.pii_redactions += trace.pii_redactions
        self.injections += int(trace.injection_flagged)
        self.traces.append(trace)

    def snapshot(self) -> dict[str, Any]:
        avg = self.latency_ms_total / self.turns if self.turns else 0.0
        return {
            "turns": self.turns,
            "tool_calls": self.tool_calls,
            "cost_usd": round(self.cost_usd, 6),
            "avg_latency_ms": round(avg, 2),
            "pii_redactions": self.pii_redactions,
            "injection_flags": self.injections,
        }


class Timer:
    def __init__(self) -> None:
        self._start = perf_counter()

    def ms(self) -> float:
        return (perf_counter() - self._start) * 1000
