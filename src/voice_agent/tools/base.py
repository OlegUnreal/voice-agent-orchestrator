from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """A capability the agent can invoke. Each tool is pure data + an async run()."""

    name: str
    description: str
    parameters: dict[str, Any]
    requires_approval: bool = False

    @abstractmethod
    async def run(self, **kwargs: Any) -> Any: ...
