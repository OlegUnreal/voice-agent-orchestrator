from __future__ import annotations

from typing import Any


def validate_tool_args(schema: dict[str, Any], args: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(args, dict):
        raise ValueError("Tool arguments must be an object")
    if "_raw" in args:
        raise ValueError("Tool arguments were not valid JSON")
    required = schema.get("required") or []
    for key in required:
        if key not in args:
            raise ValueError(f"Missing required argument: {key}")
    props = schema.get("properties") or {}
    extra = set(args) - set(props)
    if extra and schema.get("additionalProperties") is False:
        raise ValueError(f"Unexpected arguments: {sorted(extra)}")
    for key, value in args.items():
        spec = props.get(key) or {}
        expected = spec.get("type")
        if expected == "string" and not isinstance(value, str):
            raise ValueError(f"{key} must be a string")
        if expected == "number" and not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        if expected == "integer" and not isinstance(value, int):
            raise ValueError(f"{key} must be an integer")
        if expected == "boolean" and not isinstance(value, bool):
            raise ValueError(f"{key} must be a boolean")
    return args
