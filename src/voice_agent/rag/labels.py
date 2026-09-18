"""Fixture loading for the two labelled data sets retrieval depends on.

Two files, two purposes — deliberately disjoint query sets:

``data/eval/retrieval_golden.json``
    The **evaluation** qrels: >=20 queries with graded relevance used for
    Recall@k / MRR / nDCG. Never read by the trainer.

``data/eval/reranker_labels.jsonl``
    The **training** fixture: conversational paraphrases (different surface forms
    from the golden queries) expanded into ``(query, chunk, relevant)`` rows and
    split into ``train`` / ``dev`` per query, so nothing trains on a query it is
    later scored against.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .reranker import RerankerLabel

__all__ = [
    "GOLDEN_PATH",
    "LABELS_PATH",
    "GoldenQuery",
    "RelevanceRecord",
    "expand_record",
    "load_golden_set",
    "load_records",
    "load_reranker_labels",
    "REPO_ROOT",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
GOLDEN_PATH = REPO_ROOT / "data" / "eval" / "retrieval_golden.json"
LABELS_PATH = REPO_ROOT / "data" / "eval" / "reranker_labels.jsonl"


@dataclass(frozen=True)
class GoldenQuery:
    """One qrel row: ``relevant`` maps ``doc_id -> grade`` (0 = known non-relevant)."""

    qid: str
    query: str
    relevant: dict[str, int] = field(default_factory=dict)
    group: str = "general"
    note: str = ""

    def ideal_grades(self) -> list[int]:
        return sorted(self.relevant.values(), reverse=True)

    def doc_ids(self) -> set[str]:
        return {d for d, g in self.relevant.items() if g > 0}


@dataclass(frozen=True)
class RelevanceRecord:
    """Training fixture record: positives, hard negatives, optional weak negatives."""

    query: str
    split: str
    relevant: tuple[str, ...]
    negatives: tuple[str, ...]
    weak_negatives: bool

    @property
    def is_train(self) -> bool:
        return self.split == "train"


def expand_record(record: RelevanceRecord) -> list[RerankerLabel]:
    """Flatten a record into labelled rows (the trainer's input shape)."""
    rows = [RerankerLabel(record.query, d, 1, record.split) for d in record.relevant]
    rows += [RerankerLabel(record.query, d, 0, record.split) for d in record.negatives]
    return rows


def _read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_golden_set(path: Path | str | None = None) -> list[GoldenQuery]:
    target = Path(path) if path else GOLDEN_PATH
    raw = _read_json(target)
    items: Iterable[dict[str, Any]] = raw["queries"] if isinstance(raw, dict) else raw
    return [
        GoldenQuery(
            qid=item.get("qid") or f"q{index:02d}",
            query=item["query"].strip(),
            relevant={k: int(v) for k, v in (item.get("relevant") or {}).items()},
            group=item.get("group", "general"),
            note=item.get("note", ""),
        )
        for index, item in enumerate(items, start=1)
    ]


def load_records(path: Path | str | None = None) -> list[RelevanceRecord]:
    target = Path(path) if path else LABELS_PATH
    records: list[RelevanceRecord] = []
    for line in Path(target).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        item = json.loads(line)
        records.append(
            RelevanceRecord(
                query=item["query"].strip(),
                split=item.get("split", "train"),
                relevant=tuple(item.get("relevant") or ()),
                negatives=tuple(item.get("negatives") or ()),
                weak_negatives=bool(item.get("weak_negatives", False)),
            )
        )
    return records


def load_reranker_labels(path: Path | str | None = None) -> list[RerankerLabel]:
    """The literal ``(query, chunk, relevant)`` triples of the fixture.

    Explicit rows only — the trainer consumes :func:`load_records` instead so it
    can widen the negative pool with corpus knowledge it has and this module has
    not. :func:`load_reranker_labels` exists for reporting and tests.
    """
    rows: list[RerankerLabel] = []
    for record in load_records(path):
        rows.extend(expand_record(record))
    return rows
