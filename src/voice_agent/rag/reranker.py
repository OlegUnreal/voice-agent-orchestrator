"""Cross-feature reranker with *learned* weights and a calibrated score.

Why not hand-tuned blending
---------------------------
A hybrid retriever normally ends up with constants like ``0.6*lexical + 0.4*dense``
that nobody can justify. Here the blend is a logistic model fitted on a labelled
``(query, chunk, relevant)`` fixture, so every weight in the final ranking has a
measured sign and magnitude, the regularisation strength is chosen by grouped
cross-validation (never by hand), and the same model emits the probability that the
refusal gate consumes. One number, ``P(relevant)``, with a meaning instead of three
magic knobs.

Features (all in ``[0, 1]``, computed by :func:`extract_features`)
------------------------------------------------------------------
``dense``         cosine between query and chunk vectors (clipped at 0).
``bm25``          BM25 score normalised by the best BM25 score in the corpus.
``coverage``      fraction of query stems present in the chunk.
``idf_coverage``  the same overlap weighted by inverse document frequency, so a
                  chunk matching ``sev1``+``page`` beats one matching ``call``+``time``.
``title``         fraction of query stems present in the document title.
``tags``          fraction of query stems present in the document tags.
``corpus_coverage``
                  IDF-weighted fraction of the query that the *collection* covers,
                  with out-of-vocabulary terms priced at the maximum observed IDF.
                  Constant per query, so it acts as a query-level prior: a
                  near-miss question ("does the refund policy cover shipping
                  costs") contains real policy vocabulary plus terms that appear
                  nowhere in the KB, and this feature is what lets the calibrated
                  gate push the whole query below the refusal threshold instead of
                  confidently citing the refunds document. Boilerplate matches
                  (the company name on every page) barely move it because their
                  IDF is near-minimal. Classic query-performance-predictor
                  territory (collection coverage / SCQ): measured before it
                  existed, that query scored P=0.99.

Recency was a seventh feature and is **not** any more: on this corpus every
document is dated inside a 137-day window, so an exponentially decayed age scored
AUC 0.439 against the labels (worse than a coin flip) and its fitted weight was
noise. A feature that cannot discriminate only spends model capacity, so the fit
is better off without it -- see ``docs/retrieval.md``.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .corpus import Chunk
from .embeddings import tokenize

__all__ = [
    "FEATURE_NAMES",
    "C_GRID",
    "CLASS_WEIGHT_GRID",
    "CrossFeatureReranker",
    "RerankerLabel",
    "RerankerMetrics",
    "build_idf",
    "extract_features",
]

FEATURE_NAMES: tuple[str, ...] = (
    "dense",
    "bm25",
    "coverage",
    "idf_coverage",
    "title",
    "tags",
    "corpus_coverage",
)

#: Regularisation constants tried by grouped cross-validation. ``C`` is the inverse
#: of the L2 penalty, so the small end is strong shrinkage -- which is what a
#: 2.2k-row fixture with collinear lexical signals can actually support.
C_GRID: tuple[float, ...] = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
CLASS_WEIGHT_GRID: tuple[str | None, ...] = (None, "balanced")
#: Used when the fixture has too few distinct queries to cross-validate over.
FALLBACK_C = 0.01


def build_idf(token_sets: Sequence[set[str]]) -> dict[str, float]:
    """Okapi-form IDF over chunk token sets: ``log((N-df+0.5)/(df+0.5)+1)``.

    Computed from the same stems BM25 indexes, so the reranker's rarity weighting
    and the lexical channel cannot drift apart.
    """
    n = max(1, len(token_sets))
    df: Counter[str] = Counter()
    for tokens in token_sets:
        df.update(tokens)
    return {
        term: math.log((n - count + 0.5) / (count + 0.5) + 1.0)
        for term, count in df.items()
    }


@dataclass(frozen=True)
class RerankerLabel:
    """One row of the training fixture: is ``doc_id`` relevant to ``query``?"""

    query: str
    doc_id: str
    label: int
    split: str = "train"

    @property
    def is_train(self) -> bool:
        return self.split == "train"


@dataclass
class RerankerMetrics:
    precision_at_1: float
    mrr: float
    n_queries: int
    n_rows: int
    brier: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "precision_at_1": round(self.precision_at_1, 4),
            "mrr": round(self.mrr, 4),
            "brier": round(self.brier, 4),
            "queries": self.n_queries,
            "rows": self.n_rows,
        }


def extract_features(
    query: str,
    chunk: Chunk,
    *,
    dense: float,
    bm25: float,
    bm25_best: float,
    idf: dict[str, float] | None = None,
    corpus_stems: set[str] | frozenset[str] | None = None,
) -> np.ndarray:
    """One row of ``len(FEATURE_NAMES)`` floats in :data:`FEATURE_NAMES` order.

    ``corpus_stems`` is the union of chunk token sets, computed once per index by
    the retriever; without it ``corpus_coverage`` is 0.0, which the standardised
    model treats as "no information" rather than "no coverage".
    """
    q_tokens = tokenize(query)
    denom = len(q_tokens) or 1
    chunk_tokens = set(chunk.tokens)
    matched = set(q_tokens) & chunk_tokens
    coverage = len(matched) / denom
    idf_coverage = _weighted_coverage(q_tokens, matched, idf)
    title_overlap = len(set(q_tokens) & set(tokenize(chunk.title))) / denom
    tag_overlap = len(set(q_tokens) & set(tokenize(" ".join(chunk.tags)))) / denom
    bm25_norm = (bm25 / bm25_best) if bm25_best > 0 else 0.0
    corpus_coverage = _collection_coverage(set(q_tokens), corpus_stems, idf)
    return np.array(
        [
            max(0.0, min(1.0, dense)),
            max(0.0, min(1.0, bm25_norm)),
            coverage,
            idf_coverage,
            title_overlap,
            tag_overlap,
            corpus_coverage,
        ],
        dtype=np.float64,
    )


def _collection_coverage(
    query_terms: set[str],
    corpus_stems: set[str] | frozenset[str] | None,
    idf: dict[str, float] | None,
) -> float:
    """IDF-weighted fraction of the query that the *collection* can cover.

    ``sum(idf(t) for t in query & vocab) / (that sum + n_oov * max_idf)``, with
    out-of-vocabulary terms priced at the maximum observed IDF — an unseen term is
    rarer than every observed one by construction. 1.0 means every query stem
    occurs somewhere in the corpus; a boilerplate match like the company name
    barely moves it because such terms have near-minimal IDF. Without
    ``corpus_stems`` there is no information, so the feature is 0.0 (dead after
    standardisation) rather than a fake 0 coverage.
    """
    if not corpus_stems:
        return 0.0
    oov = [t for t in query_terms if t not in corpus_stems]
    if not oov:
        return 1.0
    if not idf:
        return 0.0
    idf_max = max(idf.values())
    matched_mass = sum(idf.get(t, 0.0) for t in query_terms if t in corpus_stems)
    total = matched_mass + len(oov) * idf_max
    return matched_mass / total if total > 0.0 else 0.0


def _weighted_coverage(query_tokens: Sequence[str], matched: set[str], idf: dict[str, float] | None) -> float:
    if idf is None or not query_tokens:
        return 0.0
    total = sum(idf.get(t, 0.0) for t in set(query_tokens))
    if total <= 0.0:
        return 0.0
    return sum(idf.get(t, 0.0) for t in matched) / total


def _sigmoid(z: float) -> float:
    z = max(-30.0, min(30.0, z))
    return float(1.0 / (1.0 + math.exp(-z)))


class CrossFeatureReranker:
    """Logistic scoring model over :data:`FEATURE_NAMES`.

    ``fit_rows`` with query keys selects ``C`` and ``class_weight`` by
    ``GroupKFold`` over *queries* (a query's chunks never appear in both folds) and
    Platt-scales the resulting log-odds on out-of-fold scores, so
    :meth:`probability` is a calibrated ``P(relevant)`` that the refusal gate can
    use as an absolute threshold regardless of how strong the shrinkage is.
    """

    name = "cross-feature-logistic"

    def __init__(
        self,
        *,
        c: float | None = None,
        class_weight: str | None = None,
        seed: int = 0,
        cv: int = 4,
        fitted: bool = False,
        fingerprint: str = "",
    ) -> None:
        self.c = c
        self.class_weight = class_weight
        self.seed = int(seed)
        self.cv = int(cv)
        self.fitted = fitted
        self.fingerprint = fingerprint
        self._coef: np.ndarray | None = None
        self._intercept: float = 0.0
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._calibration: tuple[float, float] | None = None
        self.metrics: RerankerMetrics | None = None
        self.selection: dict[str, object] = {}
        self.trained_on: str = ""

    # ----------------------------------------------------------------- scoring
    def _standardize(self, row: np.ndarray) -> np.ndarray:
        if self._mean is None or self._scale is None:
            return np.asarray(row, dtype=np.float64)
        return (np.asarray(row, dtype=np.float64) - self._mean) / self._scale

    def log_odds(self, row: np.ndarray) -> float:
        if not self.fitted or self._coef is None:
            raise RuntimeError("reranker not fitted; call fit_rows() or load()")
        return float(np.dot(self._coef, self._standardize(row)) + self._intercept)

    def probability(self, row: np.ndarray) -> float:
        z = self.log_odds(row)
        if self._calibration is not None:
            a, b = self._calibration
            z = a * z + b
        return _sigmoid(z)

    def score_row(self, features: Iterable[float]) -> float:
        return self.probability(np.asarray(list(features), dtype=np.float64))

    def rank(self, rows: Sequence[np.ndarray]) -> list[tuple[int, float]]:
        """Return ``(index, probability)`` sorted best-first, stable on ties."""
        scored = [(i, self.probability(row)) for i, row in enumerate(rows)]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored

    @property
    def weights(self) -> dict[str, float]:
        if self._coef is None:
            return {}
        return {name: round(float(w), 4) for name, w in zip(FEATURE_NAMES, self._coef)}

    @property
    def intercept(self) -> float:
        return float(self._intercept)

    @property
    def calibrated(self) -> bool:
        return self._calibration is not None

    # ------------------------------------------------------- hyper-parameter CV
    def _fits_model(self, x: np.ndarray, y: np.ndarray, *, c: float, class_weight: str | None):
        from sklearn.linear_model import LogisticRegression

        model = LogisticRegression(
            C=float(c),
            solver="liblinear",
            class_weight=class_weight,
            max_iter=1000,
            random_state=self.seed,
        )
        model.fit(x, y)
        return model

    @staticmethod
    def _rank_metrics(scores: Sequence[float], labels: Sequence[int], keys: Sequence[str]) -> tuple[float, float]:
        """P@1 and MRR grouped by query, ignoring the refusal gate."""
        groups: dict[str, list[tuple[int, float, int]]] = {}
        for key, score, label in zip(keys, scores, labels):
            bucket = groups.setdefault(key, [])
            bucket.append((len(bucket), float(score), int(label)))
        hits = 0
        reciprocal = 0.0
        for items in groups.values():
            ranked = sorted(items, key=lambda t: (-t[1], t[0]))
            hits += 1 if ranked[0][2] == 1 else 0
            for position, (_, _, label) in enumerate(ranked, start=1):
                if label == 1:
                    reciprocal += 1.0 / position
                    break
        n = len(groups) or 1
        return hits / n, reciprocal / n

    def _splits(self, x: np.ndarray, y: np.ndarray, keys: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        """GroupKFold over queries: every chunk of a query moves together."""
        from sklearn.model_selection import GroupKFold

        groups = np.unique(keys)
        n_splits = max(2, min(self.cv, len(groups)))
        if len(groups) < 2:
            return []
        return list(GroupKFold(n_splits=n_splits).split(x, y, groups=keys))

    def _fit_scaled(self, x: np.ndarray, y: np.ndarray, *, c: float, class_weight: str | None):
        """Standardise on ``x`` and fit. Returns (model, mean, scale)."""
        mean = x.mean(axis=0)
        var = x.var(axis=0)
        scale = np.where(var > 1e-12, np.sqrt(var), 1.0)
        model = self._fits_model((x - mean) / scale, y, c=c, class_weight=class_weight)
        return model, mean, scale

    def _cross_val_oof(
        self,
        x: np.ndarray,
        y: np.ndarray,
        keys: np.ndarray,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
        *,
        c: float,
        class_weight: str | None,
    ) -> np.ndarray:
        """Out-of-fold log-odds for one config, aligned with the input rows."""
        z = np.zeros(len(x), dtype=np.float64)
        for train_idx, test_idx in splits:
            model, mean, scale = self._fit_scaled(x[train_idx], y[train_idx], c=c, class_weight=class_weight)
            z[test_idx] = model.decision_function((x[test_idx] - mean) / scale)
        return z

    @staticmethod
    def _log_loss(y: np.ndarray, z: np.ndarray) -> float:
        eps = 1e-6
        p = np.clip(1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0))), eps, 1.0 - eps)
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def _select_hyperparameters(
        self,
        x: np.ndarray,
        y: np.ndarray,
        keys: np.ndarray,
        splits: Sequence[tuple[np.ndarray, np.ndarray]],
    ) -> dict[str, object]:
        """Grid search scored on pooled out-of-fold predictions.

        Ranking is what a reranker is for, so the primary criterion is out-of-fold
        ROC AUC: it is invariant to monotone transforms, which keeps the choice
        independent of the Platt head fitted afterwards. The secondary rule is what
        makes the shipped weights defensible -- a configuration that puts a negative
        coefficient on any of these features is rejected, because "more BM25 overlap
        makes this document less relevant" is not a claim worth shipping, and among
        the surviving ones the least shrunk (largest ``C``) wins because it
        discriminates best. If nothing survives the constraint the best unconstrained
        config is used and ``constraint_relaxed`` is recorded.
        """
        from sklearn.metrics import roc_auc_score

        if not splits:
            return {"c": FALLBACK_C, "class_weight": None, "skipped": "fewer than 2 distinct queries"}

        table: list[dict[str, object]] = []
        oof_cache: dict[tuple[float, str | None], np.ndarray] = {}
        for c in C_GRID:
            for class_weight in CLASS_WEIGHT_GRID:
                z = self._cross_val_oof(x, y, keys, splits, c=c, class_weight=class_weight)
                try:
                    auc = float(roc_auc_score(y, z))
                except ValueError:  # pragma: no cover - degenerate label set
                    auc = 0.5
                p1, mrr = self._rank_metrics(z, y.tolist(), keys.tolist())
                model, _mean, _scale = self._fit_scaled(x, y, c=c, class_weight=class_weight)
                non_negative = bool(np.all(model.coef_ >= 0.0))
                oof_cache[(c, class_weight)] = z
                table.append(
                    {
                        "c": c,
                        "class_weight": class_weight,
                        "cv_auc": round(auc, 4),
                        "cv_mrr": round(mrr, 4),
                        "cv_precision_at_1": round(p1, 4),
                        "cv_log_loss": round(self._log_loss(y, z), 4),
                        "non_negative_weights": non_negative,
                    }
                )
        constrained = [row for row in table if row["non_negative_weights"]]
        pool = constrained or table
        best = max(pool, key=lambda row: (float(row["cv_auc"]), float(row["c"])))
        unconstrained = max(table, key=lambda row: (float(row["cv_auc"]), float(row["c"])))
        return {
            "c": best["c"],
            "class_weight": best["class_weight"],
            "cv_auc": best["cv_auc"],
            "cv_mrr": best["cv_mrr"],
            "cv_log_loss": best["cv_log_loss"],
            "cv_splits": len(splits),
            "non_negative_constraint": bool(constrained),
            "constraint_relaxed": not constrained,
            "best_unconstrained_auc": unconstrained["cv_auc"],
            "table": table,
            "oof": oof_cache.get((float(best["c"]), best["class_weight"])),
        }

    def _platt(self, y: np.ndarray, z: np.ndarray) -> tuple[float, float] | None:
        """Fit ``sigmoid(a*z+b)`` on out-of-fold log-odds (Platt scaling).

        Without this the gate threshold would inherit the scale of the penalty term:
        the same evidence would pass or refuse depending on ``C``.
        """
        from sklearn.linear_model import LogisticRegression

        if z is None or len(z) < 4 or len(set(y.tolist())) < 2:
            return None
        head = LogisticRegression(C=1e6, solver="liblinear", max_iter=1000, random_state=self.seed)
        head.fit(np.asarray(z, dtype=np.float64).reshape(-1, 1), y)
        return (float(head.coef_[0][0]), float(head.intercept_[0]))

    # ---------------------------------------------------------------- training
    def fit_rows(
        self,
        rows: Sequence[np.ndarray],
        labels: Sequence[int],
        *,
        keys: Sequence[str] | None = None,
        holdout: Sequence[np.ndarray] | None = None,
        holdout_labels: Sequence[int] | None = None,
        holdout_keys: Sequence[str] | None = None,
    ) -> "CrossFeatureReranker":
        """Fit on standardised feature rows; optionally report holdout P@1 / MRR.

        With ``keys`` (the query each row belongs to) ``C`` and ``class_weight`` are
        chosen by grouped cross-validation and the score is Platt-calibrated.
        Reporting metrics on held-out rows rather than the training rows is the
        point: the numbers in ``README.md`` come from here.
        """
        x = np.asarray(rows, dtype=np.float64)
        y = np.asarray(labels, dtype=int)
        if x.ndim != 2 or x.shape[1] != len(FEATURE_NAMES):
            raise ValueError(f"expected (n, {len(FEATURE_NAMES)}) feature rows, got {x.shape}")
        if len(set(y.tolist())) < 2:
            raise ValueError("need both relevant and irrelevant rows to fit")

        self.selection = {}
        self._calibration = None
        keys_arr = np.asarray(keys, dtype=object) if keys is not None else None
        splits = self._splits(x, y, keys_arr) if keys_arr is not None else []
        oof: np.ndarray | None = None
        if keys_arr is not None:
            self.selection = self._select_hyperparameters(x, y, keys_arr, splits)
            # Out-of-fold scores are bookkeeping for the calibration head, not part
            # of the model, so they never go near the persisted artifact.
            oof = self.selection.pop("oof", None)
        chosen_c = self.selection.get("c", self.c)
        c = float(FALLBACK_C if chosen_c is None else chosen_c)
        self.c = c
        if "class_weight" in self.selection:
            self.class_weight = self.selection.get("class_weight")  # type: ignore[assignment]

        self._mean = x.mean(axis=0)
        var = x.var(axis=0)
        self._scale = np.where(var > 1e-12, np.sqrt(var), 1.0)
        model = self._fits_model((x - self._mean) / self._scale, y, c=c, class_weight=self.class_weight)
        self._coef = model.coef_[0]
        self._intercept = float(model.intercept_[0])
        self.fitted = True

        self._calibration = self._platt(y, oof) if oof is not None else None
        self.selection["platt"] = None if self._calibration is None else [round(v, 4) for v in self._calibration]

        if holdout is not None and holdout_labels is not None:
            keys_out = list(holdout_keys) if holdout_keys is not None else [""] * len(holdout)
            self.metrics = self._holdout_metrics(list(holdout), list(holdout_labels), keys_out)
        return self

    def _holdout_metrics(
        self, rows: Sequence[np.ndarray], labels: Sequence[int], keys: Sequence[str]
    ) -> "RerankerMetrics":
        """Precision@1, MRR and Brier per query group, *before* the refusal gate."""
        probabilities = [self.probability(row) for row in rows]
        p1, mrr = self._rank_metrics(probabilities, labels, keys)
        brier = float(np.mean((np.asarray(probabilities) - np.asarray(labels, dtype=float)) ** 2)) if rows else 0.0
        return RerankerMetrics(
            precision_at_1=p1,
            mrr=mrr,
            n_queries=len(set(keys)) or 1,
            n_rows=len(rows),
            brier=brier,
        )

    # ------------------------------------------------------------- persistence
    def describe(self) -> dict[str, object]:
        """Human-readable summary used by ``rebuild_artifacts`` and the README."""
        return {
            "features": list(FEATURE_NAMES),
            "weights": self.weights,
            "intercept": round(self.intercept, 4) if self.fitted else None,
            "c": self.c,
            "class_weight": self.class_weight,
            "calibrated": self.calibrated,
            "platt": None if self._calibration is None else [round(v, 4) for v in self._calibration],
            "cv": {k: v for k, v in self.selection.items() if k != "table"},
            "metrics": self.metrics.as_dict() if self.metrics else {},
            "trained_on": self.trained_on,
            "fingerprint": self.fingerprint,
        }

    def save(self, path: Path | str) -> Path:
        import joblib

        if not self.fitted:
            raise RuntimeError("refusing to persist an unfitted reranker")
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                # The feature tuple is validated on load, so this bump is what makes
                # a pre-corpus_coverage artifact fail loudly instead of scoring
                # misaligned feature rows.
                "format": 3,
                "features": FEATURE_NAMES,
                "coef": self._coef,
                "intercept": self._intercept,
                "mean": self._mean,
                "scale": self._scale,
                "calibration": self._calibration,
                "c": self.c,
                "class_weight": self.class_weight,
                "cv": self.cv,
                "seed": self.seed,
                "selection": self.selection,
                "metrics": self.metrics,
                "trained_on": self.trained_on,
                "fingerprint": self.fingerprint,
            },
            target,
        )
        return target

    @classmethod
    def load(cls, path: Path | str) -> "CrossFeatureReranker":
        import joblib

        blob = joblib.load(Path(path))
        if tuple(blob["features"]) != FEATURE_NAMES:
            raise ValueError(
                f"reranker artifact features {tuple(blob['features'])} != {FEATURE_NAMES}"
            )
        reranker = cls(
            c=blob["c"],
            class_weight=blob.get("class_weight"),
            seed=blob.get("seed", 0),
            cv=blob.get("cv", 4),
            fitted=True,
        )
        reranker._coef = blob["coef"]
        reranker._intercept = float(blob["intercept"])
        reranker._mean = blob["mean"]
        reranker._scale = blob["scale"]
        reranker._calibration = blob.get("calibration")
        reranker.selection = blob.get("selection", {})
        reranker.metrics = blob.get("metrics")
        reranker.trained_on = blob.get("trained_on", "")
        reranker.fingerprint = blob.get("fingerprint", "")
        return reranker
