from __future__ import annotations

from typing import Any

from ..rag.store import KnowledgeStore
from .base import Tool


class KnowledgeSearchTool(Tool):
    name = "knowledge_search"
    description = (
        "Search company documents. Use for policies, product facts, and support hours. "
        "If hits are empty, tell the user you do not have evidence."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, store: KnowledgeStore, top_k: int = 3, min_score: float = 0.10) -> None:
        self._store = store
        self._top_k = top_k
        self._min_score = min_score

    async def run(self, query: str, **kwargs: Any) -> Any:
        hits = self._store.search(query, top_k=self._top_k, min_score=self._min_score)
        if not hits:
            return {
                "hits": [],
                "note": "No supporting documents. Do not guess. Say you do not have evidence.",
            }
        return {
            "hits": [
                {
                    "id": h.doc_id,
                    "title": h.title,
                    "quote": h.quote,
                    "score": h.score,
                }
                for h in hits
            ]
        }
