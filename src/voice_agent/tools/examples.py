from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from .base import Tool


class CurrentTimeTool(Tool):
    name = "current_time"
    description = "Return the current UTC time."
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": []}

    async def run(self, **kwargs) -> Any:
        return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ UTC")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a simple arithmetic expression. Supports + - * / and parentheses."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "e.g. '(2 + 3) * 4'",
            }
        },
        "required": ["expression"],
    }

    async def run(self, expression: str, **kwargs) -> Any:
        allowed = set("0123456789+-*/(). ")
        if not set(expression) <= allowed:
            raise ValueError("Expression contains disallowed characters")
        return eval(expression, {"__builtins__": {}}, {})


current_time = CurrentTimeTool()
calculator = CalculatorTool()
