"""Bundled fallback corpus.

``data/kb/*.txt`` is the source of truth. These three documents exist so the store
still works if the corpus directory is missing (a trimmed install, a container that
forgot to copy ``data/``); they mirror ``refunds.txt``, ``hours.txt`` and
``product.txt`` verbatim, and ``tests/test_rag_embeddings.py::test_seed_docs_match_files``
fails if the two ever drift.

``load_knowledge`` now lives in :mod:`voice_agent.rag.store` — it is re-exported
here because the API and the eval runner import it from this module.
"""

from __future__ import annotations

from .store import KnowledgeStore, load_knowledge

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

__all__ = ["DEFAULT_DOCS", "KnowledgeStore", "load_knowledge"]
