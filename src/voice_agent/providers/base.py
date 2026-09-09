from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator


@dataclass
class ChatMessage:
    """A single turn in the conversation."""

    role: str  # system | user | assistant | tool
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ToolSpec:
    """Describes a tool the model is allowed to call."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class LLMProvider(ABC):
    """Swappable LLM backend. Implementations must be async and streaming."""

    @abstractmethod
    async def stream(
        self,
        messages: list[ChatMessage],
        tools: list[ToolSpec],
        temperature: float,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield raw provider events: {type: text.delta|tool_call.*|done, ...}."""

    @property
    @abstractmethod
    def model(self) -> str: ...
