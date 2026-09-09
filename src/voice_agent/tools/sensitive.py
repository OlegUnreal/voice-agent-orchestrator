from __future__ import annotations

from typing import Any

from .base import Tool


class IssueRefundTool(Tool):
    name = "issue_refund"
    description = "Issue a customer refund. Requires human approval unless auto-approve is enabled."
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "account_id": {"type": "string"},
            "amount_usd": {"type": "number"},
        },
        "required": ["account_id", "amount_usd"],
        "additionalProperties": False,
    }
    requires_approval = True

    def __init__(self, auto_approve: bool = False) -> None:
        self._auto_approve = auto_approve

    async def run(self, account_id: str, amount_usd: float, **kwargs: Any) -> Any:
        if not self._auto_approve:
            return {
                "status": "pending_approval",
                "account_id": account_id,
                "amount_usd": amount_usd,
            }
        return {"status": "refunded", "account_id": account_id, "amount_usd": amount_usd}
