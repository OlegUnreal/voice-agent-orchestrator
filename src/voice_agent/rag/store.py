from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .embedder import HashingEmbedder, tokenize


@dataclass(frozen=True)
class Hit:
    doc_id: str
    title: str
    quote: str
    score: float


@dataclass
class Document:
    doc_id: str
    title: str
    text: str


class KnowledgeStore:
    def __init__(self, embedder: HashingEmbedder | None = None) -> None:
        self._embedder = embedder or HashingEmbedder()
        self._docs: list[Document] = []
        self._matrix: np.ndarray | None = None

    def add(self, doc_id: str, title: str, text: str) -> None:
        self._docs.append(Document(doc_id=doc_id, title=title, text=text))
        self._matrix = None

    def search(self, query: str, *, top_k: int = 3, min_score: float = 0.18) -> list[Hit]:
        if not self._docs:
            return []
        if self._matrix is None:
            self._matrix = self._embedder.embed([d.text for d in self._docs])
        q = self._embedder.embed([query])[0]
        q_terms = set(tokenize(query))
        scores = self._matrix @ q
        order = np.argsort(-scores)
        hits: list[Hit] = []
        for idx in order:
            doc = self._docs[int(idx)]
            if q_terms and not q_terms.intersection(tokenize(doc.text)):
                continue
            score = float(scores[idx])
            if score < min_score:
                continue
            hits.append(
                Hit(
                    doc_id=doc.doc_id,
                    title=doc.title,
                    quote=_snippet(doc.text, query),
                    score=round(score, 4),
                )
            )
            if len(hits) >= top_k:
                break
        return hits


def _snippet(text: str, query: str, width: int = 220) -> str:
    lower = text.lower()
    q = query.lower().split()
    pos = -1
    for word in q:
        pos = lower.find(word)
        if pos >= 0:
            break
    if pos < 0:
        pos = 0
    start = max(0, pos - 40)
    return text[start : start + width].strip()
