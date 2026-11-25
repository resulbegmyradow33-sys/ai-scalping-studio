"""Deterministic chunked ``partial_fit`` for incremental model training.

The chunk order is the caller's row order (chronological for a dataset bundle),
never shuffled inside this module. Chunks are contiguous and the first blocks
absorb the division remainder, so the split is reproducible for a given
``(rows, chunks)`` pair. The class set is computed once from the full ``y`` and
passed to every ``partial_fit`` call, so a chunk that happens to contain a
single class cannot shrink the model's output columns.

Honest limitation: this is one ordered pass per chunk, not a drop-in replacement
for a converged batch fit. A batch ``fit`` visits every row once per epoch in
row order; ``chunks=N`` visits chunk 1, then chunk 2, ... and only returns to
chunk 1 in a second pass. The two optimize the same objective over the same rows
but in a different order, so the fitted coefficients are not expected to match
bit-for-bit. What this module guarantees is reproducibility: the same seed, rows
and chunk count give the same model, and a one-pass batch fit with
``max_iter=1``, ``tol=None`` and ``shuffle=False`` matches one chunked pass
exactly. Metric parity on real data must be measured (CV or a paired run) rather
than assumed; the synthetic test case only pins the deterministic behaviour.

``SGDClassifier.partial_fit`` refuses ``class_weight="balanced"`` because each
chunk's class frequencies would drift. When an estimator declares that setting,
this module computes the balanced weights once from the full label vector, clears
the parameter and passes the weights as ``sample_weight`` on every chunk, so the
weighting cannot depend on chunk boundaries.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

__all__ = ["incremental_fit", "supports_partial_fit"]


def supports_partial_fit(estimator: Any) -> bool:
    """Whether ``estimator`` implements scikit-learn's ``partial_fit``."""
    return callable(getattr(estimator, "partial_fit", None))


def _chunk_edges(rows: int, chunks: int) -> list[int]:
    """Contiguous ``[start, end)`` edges; the first blocks absorb the remainder."""
    base, extra = divmod(rows, chunks)
    sizes = [base + 1 if index < extra else base for index in range(chunks)]
    edges = [0]
    for size in sizes:
        edges.append(edges[-1] + size)
    if edges[-1] != rows:  # pragma: no cover - guard only
        raise RuntimeError("chunk edges do not cover the rows")
    return edges


def incremental_fit(
    estimator: Any,
    X: np.ndarray,
    y: np.ndarray,
    *,
    chunks: int,
    classes: np.ndarray | None = None,
) -> Any:
    """Fit ``estimator`` in ``chunks`` contiguous, ordered passes.

    ``chunks == 1`` is a normal batch ``fit``. ``chunks > 1`` requires
    ``partial_fit``; a tree ensemble without it raises instead of silently
    falling back to a batch fit. ``classes`` defaults to the sorted labels in
    ``y``; when given it must cover every label in ``y`` and is passed unchanged
    to every partial pass.
    """
    if chunks < 1:
        raise ValueError("chunks must be >= 1")
    features = np.ascontiguousarray(X)
    labels = np.asarray(y).reshape(-1)
    if features.shape[0] != labels.size:
        raise ValueError(f"X has {features.shape[0]} rows but y has {labels.size} labels")
    if labels.size == 0:
        raise ValueError("incremental_fit needs at least one row")

    class_set = np.unique(labels) if classes is None else np.unique(np.asarray(classes))
    if class_set.size < 2:
        raise ValueError(f"incremental_fit needs at least two classes, got {class_set.tolist()}")
    missing = np.setdiff1d(np.unique(labels), class_set)
    if missing.size:
        raise ValueError(
            f"rows contain classes {missing.tolist()} outside classes={class_set.tolist()}"
        )

    if chunks == 1:
        return estimator.fit(features, labels)
    if not supports_partial_fit(estimator):
        raise ValueError(
            f"{type(estimator).__name__} has no partial_fit; "
            "choose a linear model (sgd) or use the batch path"
        )
    if chunks > labels.size:
        raise ValueError(
            f"chunks={chunks} exceeds the {labels.size} available rows; "
            "empty chunks are not allowed"
        )

    # ``SGDClassifier.partial_fit`` refuses ``class_weight="balanced"`` because
    # per-chunk class frequencies would drift between passes. Compute the
    # balanced weights once from the full training labels instead, which is
    # both deterministic and independent of the chunk boundaries.
    sample_weight: np.ndarray | None = None
    if getattr(estimator, "class_weight", None) == "balanced":
        estimator.set_params(class_weight=None)
        sample_weight = _balanced_weight(labels, class_set)

    edges = _chunk_edges(labels.size, chunks)
    for start, end in itertools.pairwise(edges):
        fit_kwargs: dict[str, Any] = {"classes": class_set}
        if sample_weight is not None:
            fit_kwargs["sample_weight"] = sample_weight[start:end]
        estimator.partial_fit(features[start:end], labels[start:end], **fit_kwargs)
    return estimator


def _balanced_weight(labels: np.ndarray, class_set: np.ndarray) -> np.ndarray:
    """Per-row balanced weight from the full label vector."""
    encoded = np.searchsorted(class_set, labels)
    counts = np.bincount(encoded, minlength=class_set.size).astype(np.float64)
    if (counts == 0).any():  # pragma: no cover - guarded by the class check
        raise ValueError("a declared class has no rows")
    per_class = labels.size / (class_set.size * counts)
    return per_class[encoded]
