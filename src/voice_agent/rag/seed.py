from __future__ import annotations

from pathlib import Path

from .store import KnowledgeStore

DEFAULT_DOCS = [
    (
        "refunds",
        "Refund policy",
        "Northwind Voice Labs refunds unused prepaid minutes within 14 days of purchase. "
        "Refunds after 14 days require manager approval. Partial months are not prorated.",
    ),
    (
        "hours",
        "Support hours",
        "Northwind Voice Labs support is available Monday to Friday 09:00-18:00 UTC. "
        "Severity-1 voice outages are paged 24/7.",
    ),
    (
        "product",
        "Product overview",
        "The orchestrator streams speech-to-text, runs a tool-calling agent with a hard "
        "iteration budget, then speaks the answer with low-latency TTS. Providers are OpenAI-compatible.",
    ),
]


def load_knowledge(path: Path | None = None) -> KnowledgeStore:
    store = KnowledgeStore()
    for doc_id, title, text in DEFAULT_DOCS:
        store.add(doc_id, title, text)
    if path and path.exists():
        for file in sorted(path.glob("*.txt")):
            store.add(file.stem, file.stem.replace("_", " "), file.read_text(encoding="utf-8"))
    return store
