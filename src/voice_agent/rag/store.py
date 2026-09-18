"""Knowledge store: corpus -> embedder -> hybrid retriever -> reranker.

The public surface (``add``, ``search``, ``Hit``) is the v1 one, so the tool layer,
evals and tests keep working; the extras are defaulted parameters.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any, Sequence

from .corpus import Chunk, DEFAULT_KB_DIR, Document, chunk_document, parse_document
from .embeddings import DEFAULT_DIM, Embedder, HashingEmbedder, build_embedder
from .reranker import CrossFeatureReranker, RerankerLabel
from .retriever import DEFAULT_RRF_K, Hit, HybridRetriever, RetrievalStats

__all__ = [
    "Document",
    "HashingEmbedder",
    "Hit",
    "KnowledgeStore",
    "RerankerLabel",
    "RetrievalStats",
    "load_knowledge",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "artifacts"


class KnowledgeStore:
    """Chunks + indexes + the retrieval stack over them."""

    def __init__(
        self,
        embedder: Embedder | None = None,
        *,
        reranker: CrossFeatureReranker | None = None,
        top_k: int = 3,
        min_score: float = 0.18,
        rrf_k: int = DEFAULT_RRF_K,
        rerank_pool: int = 8,
        max_chunks_per_doc: int = 1,
        require_lexical_overlap: bool = True,
        embedding_dim: int = DEFAULT_DIM,
        embedding_backend: str | None = None,
    ) -> None:
        self._embedder = embedder
        # ``None`` means "resolve from RAG_EMBEDDING_BACKEND, else auto"; passing the
        # literal "auto" here used to make the env gate unreachable from the store.
        self._embedding_backend = embedding_backend
        self._embedding_dim = embedding_dim
        self.reranker = reranker
        self.top_k = int(top_k)
        self.min_score = float(min_score)
        self.rrf_k = int(rrf_k)
        self.rerank_pool = int(rerank_pool)
        self.max_chunks_per_doc = int(max_chunks_per_doc)
        self.require_lexical_overlap = bool(require_lexical_overlap)
        self._docs: dict[str, Document] = {}
        self._chunks: list[Chunk] = []
        self._retriever: HybridRetriever | None = None

    # ------------------------------------------------------------------- ingest
    @property
    def embedder(self) -> Embedder:
        self._ensure_index()
        assert self._embedder is not None
        return self._embedder

    def add(
        self,
        doc_id: str,
        title: str,
        text: str,
        *,
        tags: Sequence[str] = (),
        updated: date | None = None,
        source: str = "",
    ) -> None:
        """Upsert a document. Re-adding a ``doc_id`` replaces it instead of
        duplicating chunks (v1 silently indexed duplicates)."""
        doc = Document(
            doc_id=doc_id,
            title=title or doc_id,
            body=text,
            tags=tuple(tags),
            updated=updated,
            source=source,
        )
        if doc_id in self._docs and self._docs[doc_id] == doc:
            return
        self._docs[doc_id] = doc
        self._retriever = None

    def add_document(self, doc: Document) -> None:
        self._docs[doc.doc_id] = doc
        self._retriever = None

    def add_raw(self, doc_id: str, raw: str, *, source: str = "") -> Document:
        doc = parse_document(doc_id, raw, source=source)
        self.add_document(doc)
        return doc

    def load_dir(self, path: Path | None = None) -> "KnowledgeStore":
        """Bulk-load ``*.txt`` (headers optional). Idempotent per ``doc_id``."""
        directory = Path(path) if path else DEFAULT_KB_DIR
        for doc in _iter_docs(directory):
            self.add_document(doc)
        return self

    @property
    def doc_ids(self) -> list[str]:
        return sorted(self._docs)

    @property
    def documents(self) -> list[Document]:
        return [self._docs[k] for k in sorted(self._docs)]

    @property
    def chunks(self) -> list[Chunk]:
        self._ensure_index()
        return list(self._chunks)

    # ------------------------------------------------------------------ indexing
    def _ensure_index(self) -> None:
        if self._retriever is not None:
            return
        self._chunks = _chunk_all(self.documents)
        if self._embedder is None:
            self._embedder = build_embedder(
                backend=self._embedding_backend,
                corpus=[c.search_text for c in self._chunks],
                dim=self._embedding_dim,
            )
        self._retriever = self._build_retriever()

    def _build_retriever(self) -> HybridRetriever:
        assert self._embedder is not None
        return HybridRetriever(
            self._chunks,
            self._embedder,
            rrf_k=self.rrf_k,
            top_k=self.top_k,
            min_score=self.min_score,
            reranker=self.reranker,
            rerank_pool=self.rerank_pool,
            max_chunks_per_doc=self.max_chunks_per_doc,
            require_lexical_overlap=self.require_lexical_overlap,
        )

    @property
    def retriever(self) -> HybridRetriever:
        self._ensure_index()
        assert self._retriever is not None
        return self._retriever

    def info(self) -> dict[str, Any]:
        retriever = self.retriever
        return {
            "documents": len(self._docs),
            "chunks": retriever.size,
            "embedding_backend": getattr(retriever.embedder, "backend", retriever.embedder.name),
            "embedding_dim": retriever.embedder.dim,
            "reranker": "fitted" if (self.reranker and self.reranker.fitted) else "off",
            "rrf_k": retriever.rrf_k,
            "max_chunks_per_doc": retriever.max_chunks_per_doc,
        }

    # ------------------------------------------------------------------- search
    def search(
        self,
        query: str,
        *,
        top_k: int = 3,
        min_score: float = 0.18,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_reranker: bool | None = None,
    ) -> list[Hit]:
        hits, _stats = self.search_stats(
            query,
            top_k=top_k,
            min_score=min_score,
            use_lexical=use_lexical,
            use_dense=use_dense,
            use_reranker=use_reranker,
        )
        return hits

    def search_stats(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
        use_lexical: bool = True,
        use_dense: bool = True,
        use_reranker: bool | None = None,
    ) -> tuple[list[Hit], RetrievalStats]:
        retriever = self.retriever
        return retriever.search_stats(
            query,
            top_k=self.top_k if top_k is None else top_k,
            min_score=self.min_score if min_score is None else min_score,
            use_lexical=use_lexical,
            use_dense=use_dense,
            use_reranker=use_reranker,
        )

    def explain(self, query: str, *, top_k: int = 5, min_score: float = 0.0) -> dict[str, Any]:
        """Everything the pipeline knows about one query — for ``/retrieval/explain``."""
        hits, stats = self.search_stats(query, top_k=top_k, min_score=min_score)
        return {
            "query": query,
            "stats": stats.as_dict(),
            "hits": [
                {
                    "doc_id": h.doc_id,
                    "chunk_id": h.chunk_id,
                    "title": h.title,
                    "score": h.score,
                    "why": h.why(),
                    "breakdown": h.breakdown,
                }
                for h in hits
            ],
        }

    # ---------------------------------------------------------- reranker fitting
    def train_reranker(
        self,
        records: Sequence[Any],
        *,
        save_to: Path | str | None = None,
        c: float | None = None,
        cv: int = 4,
        weak_negatives: bool = True,
    ) -> CrossFeatureReranker:
        """Fit the cross-feature weights on the labelled fixture.

        Splitting happens *per query* (``record.split``) so no query trains and is
        then reported on, and the reported metrics come from the ``dev`` rows only.
        ``c=None`` (default) lets grouped cross-validation pick the regularisation
        strength instead of hard-coding it. ``weak_negatives=True`` also labels
        unmentioned documents as negatives -- honest for this corpus, where the
        fixture author reviewed every document.
        """
        train = [r for r in records if getattr(r, "split", "train") == "train"]
        dev = [r for r in records if getattr(r, "split", "train") != "train"]
        if not train:
            raise ValueError("reranker needs at least one split='train' record")

        def rows_for(items: Sequence[Any]):
            return self.training_rows(items, weak_negatives=weak_negatives)

        x_train, y_train, keys_train = rows_for(train)
        if dev:
            x_dev, y_dev, keys_dev = rows_for(dev)
        else:
            x_dev, y_dev, keys_dev = x_train, y_train, keys_train
        reranker = CrossFeatureReranker(c=c, cv=cv).fit_rows(
            x_train,
            y_train,
            keys=keys_train,
            holdout=x_dev,
            holdout_labels=y_dev,
            holdout_keys=keys_dev,
        )
        reranker.fingerprint = corpus_fingerprint(self, records)
        reranker.trained_on = (
            f"{len(x_train)} train rows / {len(train)} queries"
            + (f"; metrics on {len(x_dev)} rows / {len(dev)} held-out queries" if dev else "; no holdout")
        )
        self.attach_reranker(reranker)
        if save_to is not None:
            reranker.save(save_to)
        return reranker

    def training_rows(
        self,
        records: Sequence[Any],
        *,
        weak_negatives: bool = True,
    ) -> tuple[list[Any], list[int], list[str]]:
        """Corpus-wide feature rows for a set of fixture records.

        Positives are every chunk of a relevant document, hard negatives every
        chunk of an explicitly irrelevant one, and (optionally) the remaining
        documents as weak negatives.
        """
        from .labels import RelevanceRecord  # local import keeps module graph flat

        rows: list[Any] = []
        labels: list[int] = []
        keys: list[str] = []
        for record in records:
            if not isinstance(record, RelevanceRecord):
                raise TypeError(f"expected RelevanceRecord, got {type(record)!r}")
            positives = set(record.relevant)
            hard = set(record.negatives)
            for chunk, features, _bm25, _dense in self.retriever.feature_rows(record.query):
                if chunk.doc_id in positives:
                    label = 1
                elif chunk.doc_id in hard or weak_negatives:
                    label = 0
                else:
                    continue
                rows.append(features)
                labels.append(label)
                keys.append(record.query)
        return rows, labels, keys

    def attach_reranker(self, reranker: CrossFeatureReranker | None) -> None:
        self.reranker = reranker
        if self._retriever is not None:
            self._retriever.reranker = reranker


def corpus_fingerprint(kb: "KnowledgeStore", records: Sequence[Any]) -> str:
    """Stable id for "this corpus + this fixture", stored on the artifact."""
    import hashlib

    chunks = kb.chunks
    digest = hashlib.blake2b(digest_size=12)
    digest.update(f"{kb.info()['embedding_backend']}:{kb.info()['embedding_dim']}".encode())
    for chunk in chunks:
        digest.update(f"{chunk.chunk_id}:{len(chunk.text)}".encode())
    for record in sorted(records, key=lambda r: r.query):
        digest.update(
            f"|{record.query}:{','.join(record.relevant)}:{','.join(record.negatives)}:{record.split}".encode()
        )
    return f"fp-{digest.hexdigest()[:16]}"


def _chunk_all(docs: Sequence[Document]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks


def _iter_docs(directory: Path) -> list[Document]:
    from .corpus import iter_documents

    return list(iter_documents(directory))


def load_knowledge(
    path: Path | None = None,
    *,
    store: KnowledgeStore | None = None,
    with_reranker: bool = True,
    **kwargs: Any,
) -> KnowledgeStore:
    """Build a store from ``data/kb`` (or ``path``) and attach the reranker.

    ``path=None`` keeps the historical behaviour of falling back to the bundled
    seed documents when the corpus directory is absent. v1 added the seed docs
    *and* the files with the same ids, indexing every refund/hours/product chunk
    twice; the corpus is now a single source of truth keyed by ``doc_id``.
    """
    kb = store or KnowledgeStore(**kwargs)
    directory = Path(path) if path is not None else DEFAULT_KB_DIR
    loaded = False
    if directory and directory.exists():
        if sorted(directory.glob("*.txt")):
            kb.load_dir(directory)
            loaded = True
    if not loaded:
        from .seed import DEFAULT_DOCS

        for doc_id, title, text in DEFAULT_DOCS:
            kb.add(doc_id, title, text)
    if with_reranker and kb.reranker is None:
        kb.attach_reranker(default_reranker(kb))
    return kb


def default_reranker(kb: KnowledgeStore) -> CrossFeatureReranker | None:
    """Load the persisted reranker, or fit it from the shipped labels fixture.

    Fitting takes milliseconds at this corpus size, so a missing artifact is not a
    failure mode. The artifact carries a corpus+fixture fingerprint; a stale one is
    refitted instead of silently degrading ranking quality.
    """
    from .labels import load_records

    path = reranker_path()
    try:
        records = load_records()
    except FileNotFoundError:
        return None
    fingerprint = corpus_fingerprint(kb, records)
    if path.exists():
        try:
            existing = CrossFeatureReranker.load(path)
            if existing.fingerprint == fingerprint:
                return existing
        except Exception:  # noqa: BLE001 - a broken artifact must not break startup
            pass
    kb.train_reranker(records, save_to=path)
    assert kb.reranker is not None
    return kb.reranker


def reranker_path() -> Path:
    import os

    return Path(os.getenv("RAG_RERANKER_PATH") or ARTIFACT_DIR / "reranker-v4.joblib")


def rebuild_artifacts() -> dict[str, Any]:
    """CI/dev helper: force a reranker retrain and report what it learned."""
    from .labels import load_records

    kb = load_knowledge(with_reranker=False)
    records = load_records()
    reranker = kb.train_reranker(records, save_to=reranker_path())
    described = reranker.describe()
    described["selection_table"] = reranker.selection.get("table", [])
    described["info"] = kb.info()
    return described
