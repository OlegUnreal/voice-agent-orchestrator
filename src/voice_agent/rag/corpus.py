"""Corpus loading and chunking for the knowledge store.

Files in ``data/kb/`` are plain UTF-8 text with an optional header block::

    Title: Refund policy
    Tags: refunds, billing
    Updated: 2026-05-01

    Body paragraph one.

    Body paragraph two.

Anything before the first blank line that is not a ``Key: value`` header is
treated as body, so the three legacy files (no headers) still load — they just
get a filename-derived title, no tags, and ``updated=None``.

Chunking is paragraph-driven with a character cap: retrieval over small chunks
sharpens the lexical signal, while the doc-level ``title``/``tags``/``updated``
metadata is copied onto every chunk so the reranker can score on it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Sequence

from .embeddings import tokenize

__all__ = ["Chunk", "Document", "chunk_text", "load_corpus", "parse_document"]

_REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_KB_DIR = _REPO_ROOT / "data" / "kb"

_HEADER_KEYS = ("title", "tags", "updated", "owner")
_HEADER = re.compile(rf"^(?P<key>{'|'.join(_HEADER_KEYS)})\s*:\s*(?P<value>.*?)\s*$", re.IGNORECASE)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")

DEFAULT_CHUNK_CHARS = 520


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    body: str
    tags: tuple[str, ...] = ()
    updated: date | None = None
    source: str = ""

    def metadata_text(self) -> str:
        return f"{self.title} {' '.join(self.tags)}".strip()


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. ``chunk_id`` is stable: ``<doc_id>#<ordinal>``."""

    chunk_id: str
    doc_id: str
    title: str
    text: str
    tags: tuple[str, ...] = ()
    updated: date | None = None
    source: str = ""
    ordinal: int = 0
    tokens: tuple[str, ...] = field(default_factory=tuple, compare=False)

    def __post_init__(self) -> None:
        if not self.tokens:
            object.__setattr__(
                self,
                "tokens",
                tuple(tokenize(f"{self.title} {self.text} {' '.join(self.tags)}")),
            )

    @property
    def search_text(self) -> str:
        """What gets embedded: title + tags are part of the vector, like a real index."""
        parts = [self.text, self.title]
        if self.tags:
            parts.append(" ".join(self.tags))
        return " \n ".join(p for p in parts if p)

    def year(self) -> int | None:
        return self.updated.year if self.updated else None


def parse_document(doc_id: str, raw: str, *, source: str = "") -> Document:
    """Split an optional ``Key: value`` header block from the body."""
    lines = raw.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    title = tags_raw = updated_raw = ""
    tags: tuple[str, ...] = ()
    updated: date | None = None
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            break
        match = _HEADER.match(line)
        if not match:
            break
        key = match.group("key").lower()
        value = match.group("value").strip()
        if key == "title":
            title = value
        elif key == "tags":
            tags = tuple(t.strip().lower() for t in value.split(",") if t.strip())
        elif key == "updated":
            updated = _parse_date(value)
        index += 1
    body = "\n".join(lines[index:]).strip()
    return Document(
        doc_id=doc_id,
        title=title or doc_id.replace("_", " ").replace("-", " ").strip().title(),
        body=body or raw.strip(),
        tags=tags or _derive_tags(body),
        updated=updated,
        source=source,
    )


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _derive_tags(body: str, limit: int = 6) -> tuple[str, ...]:
    """Cheap fallback tags: the most frequent non-stopword tokens in the body.

    Keeps the title/tag reranking feature meaningful for legacy documents that
    ship without a header block.
    """
    counts: dict[str, int] = {}
    for token in tokenize(body):
        counts[token] = counts.get(token, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(t for t, _ in ranked[:limit])


def chunk_text(
    text: str,
    *,
    max_chars: int = DEFAULT_CHUNK_CHARS,
    min_chars: int = 80,
) -> list[str]:
    """Paragraph-first chunking with a character cap and sentence fallback.

    Deterministic; no overlap by default because the reranker sees the fused
    candidate list and overlap only inflates the index.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]
    chunks: list[str] = []
    buffer = ""
    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())
        if len(paragraph) > max_chars:
            if buffer:
                chunks.append(buffer)
                buffer = ""
            chunks.extend(_split_sentences(paragraph, max_chars))
            continue
        candidate = f"{buffer} {paragraph}".strip()
        if len(candidate) > max_chars and buffer:
            chunks.append(buffer)
            buffer = paragraph
        else:
            buffer = candidate
    if buffer:
        chunks.append(buffer)
    return _merge_tiny(chunks, min_chars)


def _split_sentences(paragraph: str, max_chars: int) -> list[str]:
    sentences = [s.strip() for s in _SENTENCE.split(paragraph) if s.strip()]
    out: list[str] = []
    buffer = ""
    for sentence in sentences:
        candidate = f"{buffer} {sentence}".strip()
        if len(candidate) > max_chars and buffer:
            out.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        out.append(buffer)
    return out


def _merge_tiny(chunks: Sequence[str], min_chars: int) -> list[str]:
    if len(chunks) <= 1:
        return list(chunks)
    merged: list[str] = [chunks[0]]
    for chunk in chunks[1:]:
        if len(chunk) < min_chars and len(merged[-1]) + len(chunk) + 1 <= min_chars * 3:
            merged[-1] = f"{merged[-1]} {chunk}".strip()
        else:
            merged.append(chunk)
    if len(merged) > 1 and len(merged[-1]) < min_chars:
        merged[-2] = f"{merged[-2]} {merged[-1]}".strip()
        merged.pop()
    return merged


def chunk_document(doc: Document, *, max_chars: int = DEFAULT_CHUNK_CHARS) -> list[Chunk]:
    pieces = chunk_text(doc.body, max_chars=max_chars)
    return [
        Chunk(
            chunk_id=f"{doc.doc_id}#{i:02d}",
            doc_id=doc.doc_id,
            title=doc.title,
            text=piece,
            tags=doc.tags,
            updated=doc.updated,
            source=doc.source,
            ordinal=i,
        )
        for i, piece in enumerate(pieces)
    ]


def iter_documents(path: Path | None = None) -> Iterable[Document]:
    """Read ``*.txt`` from the corpus directory in stable (sorted) order."""
    directory = Path(path) if path else DEFAULT_KB_DIR
    if not directory.exists():
        return []
    docs: list[Document] = []
    seen: set[str] = set()
    for file in sorted(directory.glob("*.txt")):
        doc_id = file.stem.lower()
        if doc_id in seen:
            continue
        seen.add(doc_id)
        docs.append(parse_document(doc_id, file.read_text(encoding="utf-8"), source=str(file)))
    return docs


def load_corpus(
    path: Path | None = None,
    *,
    extra_docs: Sequence[Document] = (),
    max_chars: int = DEFAULT_CHUNK_CHARS,
) -> list[Chunk]:
    """Corpus as a flat chunk list. ``extra_docs`` lets tests inject synthetic docs."""
    chunks: list[Chunk] = []
    for doc in [*iter_documents(path), *extra_docs]:
        chunks.extend(chunk_document(doc, max_chars=max_chars))
    return chunks
