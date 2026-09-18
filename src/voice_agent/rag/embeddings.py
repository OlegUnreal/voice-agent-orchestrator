"""Embedding backends for the knowledge store.

Three backends share one interface (``embed(list[str]) -> (n, dim) float32`` with
L2-normalised rows) so the retriever never cares which one produced a vector:

``tfidf-svd`` (default, "LSA-lite")
    TF-IDF over word 1-2 grams with ``sublinear_tf`` *plus* a ``char_wb`` 3-5 gram
    view, hstacked into one sparse matrix and reduced with ``TruncatedSVD``
    (randomized solver == probabilistic LSA). The char view is what gives the
    model robustness to morphology/typos ("refunds" / "refunded") that a pure
    bag-of-words gate throws away.

``hash``
    Deterministic signed blake2b hashing trick over the same 1-2 gram features.
    Used automatically when the corpus is too small to support a meaningful SVD
    (``n_components`` is bounded by ``min(n_docs, n_features) - 1``), and as the
    zero-dependency fallback. No fitting, so it cannot leak corpus statistics.

``openai``
    Real semantic embeddings, opt-in via ``RAG_EMBEDDING_BACKEND=openai`` *and* a
    non-empty ``PROVIDER_API_KEY``. ``openai`` is imported lazily through
    ``importlib`` so the offline test suite never touches it and the package
    keeps working without the extra dependency.

Everything except the OpenAI path is pure CPU, seeded, and reproducible: the same
corpus in gives byte-identical vectors out, so eval numbers are stable in CI.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = [
    "Embedder",
    "HashingEmbedder",
    "HashingVectorizer",
    "TfidfSvdEmbedder",
    "OpenAIEmbedder",
    "build_embedder",
    "tokenize",
    "stem",
    "STOPWORDS",
    "DEFAULT_DIM",
]

DEFAULT_DIM = 32
#: Below this many reduced dimensions LSA is noise, so ``build_embedder`` falls back.
MIN_COMPONENTS = 8

_TOKEN = re.compile(r"[a-z0-9]+")
#: ``sender's`` / ``don't`` / ``we'll`` — the apostrophe tail is a function word,
#: not a term. Leaving it in emits one-letter tokens that collide with queries.
_CONTRACTION = re.compile(r"'[a-z]{1,3}\b")
STOPWORDS = {
    "a", "an", "and", "are", "for", "in", "is", "it", "me", "of", "on", "or",
    "the", "to", "what", "who", "with", "you", "your",
}
_VOWELS = "aeiou"

#: Suffixes stripped in one pass, longest first. Deliberately narrower than Porter:
#: every family here is exercised by a unit test, and the ones Porter handles with a
#: measure function (``-er``, ``-or``, ``-ise``, ``-ify``) are omitted because they
#: over-stem short English words (``prefer`` -> ``pref``, ``issue`` -> ``iss``).
_STEM_SUFFIXES: tuple[str, ...] = (
    "izations", "ization", "isations", "isation",
    "acements", "acement",
    "ations", "ation", "itions", "ition",
    "ements", "ement", "ments", "ment",
    "ables", "able", "ibles", "ible",
    "ances", "ance", "ences", "ence",
    "ically", "ical", "ally", "ings", "ing", "ness", "less",
    "edly", "ies", "ied", "ily", "ed", "ly", "al",
)
#: Doubled consonants that collapse after stripping (``cancelled`` -> ``cancell``).
#: ``ss``/``zz`` are excluded so ``pass`` and ``buzz`` stay intact.
_DOUBLE_CONSONANTS = frozenset("bcdfgmnprtl")


def _strip_plural(word: str) -> str:
    """Step 1: regular plurals and the ``-ies`` / ``-es`` alternations.

    ``-es`` is not special-cased: dropping the bare ``s`` and then the silent
    final ``e`` (see :func:`stem`) lands ``boxes`` and ``classes`` on the same
    stem as ``box`` and ``class`` anyway.
    """
    if len(word) < 4 or word[-1] != "s":
        return word
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-3] + "y"
    if word.endswith(("ss", "us")):
        return word
    stem = word[:-1]
    return stem if len(stem) >= 3 else word


def stem(word: str) -> str:
    """Normalise one already-lowercased token to a stable English stem.

    Pure, deterministic and dependency-free. Five surface forms of one verb must
    share a BM25 term or the lexical channel silently loses to the dense channel
    on exactly the queries ("how do I cancel my contract") it is supposed to win.
    """
    if len(word) < 4 or not word.isalpha():
        return word
    out = _strip_plural(word)
    if len(out) >= 5:
        for suffix in _STEM_SUFFIXES:
            if out.endswith(suffix) and len(out) - len(suffix) >= 3:
                out = out[: -len(suffix)]
                break
    if len(out) >= 3 and out[-1] == out[-2] and out[-1] in _DOUBLE_CONSONANTS:
        out = out[:-1]
    # A silent final "e" is the difference between ``invoice`` and ``invoiced``.
    if len(out) > 4 and out.endswith("e") and out[-2] not in _VOWELS:
        out = out[:-1]
    return out


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens, contractions dropped, stopwords removed, stemmed.

    Shared by every stage (BM25, TF-IDF, hashing, reranker overlap features) so
    there is exactly one notion of a term in the whole retrieval stack.
    """
    cleaned = _CONTRACTION.sub("", text.lower())
    return [stem(t) for t in _TOKEN.findall(cleaned) if t not in STOPWORDS]


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32)


class Embedder:
    """Interface actually implemented by the backends below (kept explicit so the
    retriever can be typed without a ``Protocol`` import at runtime)."""

    name: str = "base"
    dim: int = 0

    def fit(self, corpus: Sequence[str]) -> "Embedder":
        return self

    def embed(self, texts: Sequence[str]) -> np.ndarray:  # pragma: no cover - iface
        raise NotImplementedError


class HashingVectorizer(Embedder):
    """Signed blake2b hashing trick: unigrams + bigrams, ``dim`` buckets.

    Deterministic by construction (blake2b is not affected by ``PYTHONHASHSEED``)
    and stateless, so it can embed a query that was never in the corpus without
    any fit step. Sign bit flips collisions from ``+1/-1`` instead of letting
    them pile up.
    """

    name = "hash"

    def __init__(self, dim: int = DEFAULT_DIM, *, use_bigrams: bool = True) -> None:
        if dim < 8:
            raise ValueError("dim must be >= 8")
        self.dim = int(dim)
        self.use_bigrams = use_bigrams

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = tokenize(text)
            feats: Iterable[str] = tokens
            if self.use_bigrams:
                feats = tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
            for feat in feats:
                digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
                bucket = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, bucket] += sign
        return _l2_normalize(out)


class HashingEmbedder(HashingVectorizer):
    """Backwards-compatible name for the v1 embedder (same behaviour, new impl)."""

    def __init__(self, dim: int = DEFAULT_DIM) -> None:
        super().__init__(dim)


def _tfidf_vectorizers() -> tuple[Any, Any]:
    """Word 1-2 gram + char_wb 3-5 gram TF-IDF pair, sublinear TF."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    word = TfidfVectorizer(
        tokenizer=tokenize,
        token_pattern=None,
        ngram_range=(1, 2),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float64,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        sublinear_tf=True,
        min_df=1,
        dtype=np.float64,
    )
    return word, char


def _make_svd(n_components: int, seed: int) -> Any:
    from sklearn.decomposition import TruncatedSVD

    try:
        return TruncatedSVD(n_components=n_components, algorithm="randomized", random_state=seed)
    except TypeError:  # pragma: no cover - newer/older sklearn kwarg rename
        return TruncatedSVD(n_components=n_components, svd_solver="randomized", random_state=seed)


class TfidfSvdEmbedder(Embedder):
    """TF-IDF (word 1-2g + char_wb 3-5g) reduced by TruncatedSVD, L2-normalised.

    ``fit`` must run on the corpus before vectors are comparable. If the corpus is
    too small for a meaningful reduction the embedder degrades to
    :class:`HashingVectorizer` and reports ``backend='hash'`` rather than emitting
    a degenerate 2-dimensional space.
    """

    name = "tfidf-svd"

    def __init__(
        self,
        dim: int = DEFAULT_DIM,
        *,
        min_components: int = MIN_COMPONENTS,
        seed: int = 0,
    ) -> None:
        self.requested_dim = int(dim)
        self.dim = int(dim)
        self.min_components = int(min_components)
        self.seed = int(seed)
        self.backend = "tfidf-svd"
        self.fit_reason = ""
        self._word: Any = None
        self._char: Any = None
        self._svd: Any = None
        self._fallback: HashingVectorizer | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, corpus: Sequence[str]) -> "TfidfSvdEmbedder":
        docs = [c for c in corpus if c and c.strip()]
        if len(docs) < 2:
            return self._use_hashing("corpus has fewer than 2 documents")
        try:
            from scipy.sparse import hstack
        except ImportError:  # pragma: no cover - scipy is a hard dep in practice
            return self._use_hashing("scipy unavailable")

        self._word, self._char = _tfidf_vectorizers()
        x_word = self._word.fit_transform(docs)
        x_char = self._char.fit_transform(docs)
        x = hstack([x_word, x_char], format="csr")
        n_components = int(min(self.requested_dim, x.shape[0] - 1, x.shape[1] - 1))
        if n_components < self.min_components:
            return self._use_hashing(
                f"corpus supports only {n_components} SVD components "
                f"(< min_components={self.min_components})"
            )
        self._svd = _make_svd(n_components, self.seed).fit(x)
        self.dim = int(n_components)
        self.backend = "tfidf-svd"
        self._fallback = None
        return self

    def _use_hashing(self, reason: str) -> "TfidfSvdEmbedder":
        self._fallback = HashingVectorizer(self.requested_dim)
        self._word = self._char = self._svd = None
        self.dim = self.requested_dim
        self.backend = "hash"
        self.fit_reason = reason
        return self

    @property
    def fitted(self) -> bool:
        return self._svd is not None or self._fallback is not None

    # --------------------------------------------------------------- embedding
    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self._fallback is not None:
            return self._fallback.embed(texts)
        if self._svd is None:
            raise RuntimeError("TfidfSvdEmbedder.embed() called before fit(); use build_embedder()")
        from scipy.sparse import hstack

        x = hstack([self._word.transform(texts), self._char.transform(texts)], format="csr")
        return _l2_normalize(self._svd.transform(x))

    # ------------------------------------------------------------ persistence
    def save(self, path: Path | str) -> Path:
        import joblib

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "format": 1,
                "backend": self.backend,
                "requested_dim": self.requested_dim,
                "dim": self.dim,
                "min_components": self.min_components,
                "seed": self.seed,
                "word": self._word,
                "char": self._char,
                "svd": self._svd,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "TfidfSvdEmbedder":
        import joblib

        blob = joblib.load(Path(path))
        embedder = cls(
            dim=blob["requested_dim"],
            min_components=blob.get("min_components", MIN_COMPONENTS),
            seed=blob.get("seed", 0),
        )
        embedder.dim = int(blob["dim"])
        embedder.backend = blob["backend"]
        embedder._word, embedder._char, embedder._svd = (
            blob["word"],
            blob["char"],
            blob["svd"],
        )
        if embedder.backend == "hash":
            embedder._fallback = HashingVectorizer(embedder.dim)
        return embedder


class OpenAIEmbedder(Embedder):
    """Optional semantic backend. Never constructed by tests or offline runs.

    Gated twice on purpose: ``RAG_EMBEDDING_BACKEND=openai`` *and* a non-empty
    ``PROVIDER_API_KEY``. The ``openai`` SDK is imported through ``importlib``
    inside :meth:`embed` so the module imports fine without it installed.
    """

    name = "openai"

    def __init__(
        self,
        *,
        model: str | None = None,
        dim: int = 256,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        self.model = model or os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")
        self.dim = int(dim)
        self.api_key = api_key if api_key is not None else os.getenv("PROVIDER_API_KEY", "")
        self.base_url = base_url or os.getenv("PROVIDER_BASE_URL") or None
        self._client: Any = None
        self._cache: dict[str, np.ndarray] = {}
        if not self.api_key:
            raise RuntimeError(
                "OpenAI embeddings need PROVIDER_API_KEY; unset RAG_EMBEDDING_BACKEND "
                "to use the deterministic offline embedder."
            )

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                openai = importlib.import_module("openai")
            except ImportError as exc:  # pragma: no cover - optional dependency
                raise ImportError(
                    "RAG_EMBEDDING_BACKEND=openai needs the 'openai' package installed"
                ) from exc
            kwargs: dict[str, Any] = {"api_key": self.api_key}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = openai.OpenAI(**kwargs)
        return self._client

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if t not in self._cache]
        if missing:
            client = self._get_client()
            for start in range(0, len(missing), 64):
                batch = missing[start : start + 64]
                response = client.embeddings.create(model=self.model, input=batch)
                for text, item in zip(batch, response.data):
                    self._cache[text] = np.asarray(item.embedding, dtype=np.float32)
        return _l2_normalize(np.stack([self._cache[t] for t in texts]))


def resolve_backend(backend: str | None = None) -> str:
    """Explicit argument > ``RAG_EMBEDDING_BACKEND`` > ``'auto'`` (deterministic LSA).

    The OpenAI path is reachable *only* when someone asks for it by name, which is
    what keeps the offline suite from ever making a network call.
    """
    return (backend or os.getenv("RAG_EMBEDDING_BACKEND", "auto") or "auto").strip().lower()


def build_embedder(
    backend: str | None = None,
    corpus: Sequence[str] | None = None,
    dim: int = DEFAULT_DIM,
    **kwargs: Any,
) -> Embedder:
    """Factory used by :class:`~voice_agent.rag.store.KnowledgeStore`.

    ``backend='auto'`` (default) picks the OpenAI backend only when the operator
    explicitly asked for it via env, otherwise the deterministic TF-IDF+LSA path,
    which itself degrades to the hashing trick for tiny corpora.
    """
    chosen = resolve_backend(backend)
    if chosen == "openai":
        return OpenAIEmbedder(dim=kwargs.pop("openai_dim", 256), **kwargs)
    if chosen in {"hash", "blake2b"}:
        return HashingVectorizer(dim)
    if chosen in {"auto", "lsa", "tfidf-svd", "tfidf_svd"}:
        embedder = TfidfSvdEmbedder(dim, **kwargs)
        return embedder.fit(list(corpus or []))
    raise ValueError(f"Unknown embedding backend: {backend!r}")
