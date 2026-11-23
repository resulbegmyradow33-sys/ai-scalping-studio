"""Distribution drift between a reference sample and newer data.

Three questions, three descriptive answers:

* did the *features* move (per-column PSI and two-sample KS)?
* did the *prediction confidence* move (PSI/KS over per-row maximum probability)?
* did *calibration* move (mean ECE over the classes that occur)?

Nothing here decides that a model is broken. Drift is a reason to look, not a
proof that the world changed: a quiet market and a genuine regime shift both
move these numbers. The champion gate is where policy lives; this module only
reports what the two samples say.

PSI uses equal-frequency bins cut on the **reference** sample (deciles by
default), so it is directional: swapping the arguments can change the value.
Empty bins are floored at :data:`_EPSILON` before the logarithm, the standard
guard that keeps ``0 * log(0)`` finite; a bin empty on both sides contributes
zero. A constant reference has no interior quantile edges, so it is compared as
"equal to that constant" versus "anything else": a shifted current sample lands
on the other side and the PSI is large instead of silently zero. Identical
samples return exactly ``0.0``.

KS is the standard two-sample statistic, dependency-free (stdlib ``math`` on
top of numpy). The p-value uses the asymptotic Kolmogorov distribution with the
effective sample size ``sqrt(n*m/(n+m))`` and the common small-sample
correction; it is a descriptive tail probability, not an exact permutation
test, and is not corrected for the number of features compared.

Non-finite values (NaN and infinities) are dropped before either statistic.
:func:`drift_from_datasets` counts how many rows each feature lost, so a column
that is mostly missing cannot look merely "stable".
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Final

import numpy as np

from scalping.ml.calibration import expected_calibration_error

__all__ = [
    "DEFAULT_BINS",
    "DEFAULT_PSI_THRESHOLD",
    "DriftReport",
    "FeatureDrift",
    "PredictionDrift",
    "drift_from_datasets",
    "ks_statistic",
    "population_stability_index",
    "prediction_drift",
]

#: Deciles, matching the calibration tables.
DEFAULT_BINS: Final[int] = 10
#: A PSI above this is usually called a significant shift; below 0.1 is noise.
DEFAULT_PSI_THRESHOLD: Final[float] = 0.2
#: Floor for an empty bin share, so ``log(0)`` never happens.
_EPSILON: Final[float] = 1e-6


def _finite(values: Any, name: str) -> np.ndarray:
    """A 1-D float array with non-finite values removed; empty input raises."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{name} has no finite values")
    return finite


def _finite_or_none(values: Any, name: str) -> tuple[np.ndarray | None, int]:
    """As :func:`_finite`, but an empty result is ``(None, dropped)``."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {array.shape}")
    mask = np.isfinite(array)
    dropped = int((~mask).sum())
    finite = array[mask]
    return (finite if finite.size else None), dropped


def _psi_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Interior bin edges from reference quantiles, deduplicated."""
    quantiles = np.quantile(reference, np.linspace(0.0, 1.0, bins + 1))
    return np.unique(quantiles[1:-1])


def _psi_counts(
    reference: np.ndarray, current: np.ndarray, bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Per-bin counts for both samples.

    A constant reference has no interior quantile edges. It is compared as
    "equal to the constant" versus "anything else", so a different constant or
    a shifted sample cannot land in the same single bin.
    """
    if np.unique(reference).size == 1:
        constant = reference[0]
        ref_counts = np.asarray([np.count_nonzero(reference == constant), 0], dtype=np.float64)
        cur_counts = np.asarray(
            [
                np.count_nonzero(current == constant),
                np.count_nonzero(current != constant),
            ],
            dtype=np.float64,
        )
        return ref_counts, cur_counts

    edges = _psi_edges(reference, bins)
    slots = edges.size + 1
    ref_counts = np.bincount(
        np.searchsorted(edges, reference, side="left"), minlength=slots
    ).astype(np.float64)
    cur_counts = np.bincount(np.searchsorted(edges, current, side="left"), minlength=slots).astype(
        np.float64
    )
    return ref_counts, cur_counts


def _check_bins(bins: int) -> None:
    if isinstance(bins, bool) or not isinstance(bins, int):
        raise TypeError("bins must be an int")
    if bins < 2:
        raise ValueError("bins must be >= 2")


def population_stability_index(
    reference: Any,
    current: Any,
    *,
    bins: int = DEFAULT_BINS,
) -> float:
    """Population Stability Index of ``current`` against ``reference``.

    Bins are quantiles of the reference sample. Shares are floored at
    ``1e-6`` before the log; identical samples return exactly ``0.0``.
    """
    _check_bins(bins)
    ref = _finite(reference, "reference")
    cur = _finite(current, "current")
    if ref.size == cur.size and np.array_equal(np.sort(ref), np.sort(cur)):
        return 0.0

    ref_counts, cur_counts = _psi_counts(ref, cur, bins)
    ref_share = np.maximum(ref_counts / ref.size, _EPSILON)
    cur_share = np.maximum(cur_counts / cur.size, _EPSILON)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def _kolmogorov_survival(lam: float) -> float:
    """``P(D_nm >= d)`` under the asymptotic Kolmogorov distribution."""
    if lam <= 0.0:
        return 1.0
    # Below the first useful term the series is numerically 1.0 anyway.
    if lam < 0.2:
        return 1.0
    total = 0.0
    for k in range(1, 101):
        term = 2.0 * (-1.0) ** (k - 1) * math.exp(-2.0 * k * k * lam * lam)
        total += term
        if abs(term) < 1e-12:
            break
    return min(1.0, max(0.0, total))


def ks_statistic(reference: Any, current: Any) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov ``(D, p)`` with no scipy dependency.

    ``D`` is the largest gap between the empirical CDFs; ``p`` is the
    asymptotic tail probability, which is why it is reported only as a
    descriptive value.
    """
    ref = np.sort(_finite(reference, "reference"))
    cur = np.sort(_finite(current, "current"))
    data = np.concatenate([ref, cur])
    cdf_ref = np.searchsorted(ref, data, side="right") / ref.size
    cdf_cur = np.searchsorted(cur, data, side="right") / cur.size
    d = float(np.max(np.abs(cdf_ref - cdf_cur)))

    effective = math.sqrt(ref.size * cur.size / (ref.size + cur.size))
    lam = (effective + 0.12 + 0.11 / effective) * d
    return d, _kolmogorov_survival(lam)


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """PSI and KS for one column; ``None`` means the comparison could not run.

    ``note`` explains a skipped comparison (for example, every value in one
    sample was non-finite), so absence is visible instead of looking stable.
    """

    feature: str
    psi: float | None
    ks: float | None
    ks_pvalue: float | None
    note: str | None = None
    dropped_reference: int = 0
    dropped_current: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "psi": self.psi,
            "ks": self.ks,
            "ks_pvalue": self.ks_pvalue,
            "note": self.note,
            "dropped_reference": self.dropped_reference,
            "dropped_current": self.dropped_current,
        }


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Per-feature drift plus the threshold and the features that crossed it.

    ``drifted`` is sorted by feature name and uses a strict ``psi >
    threshold``; a PSI exactly at the threshold is not drift. A feature whose
    PSI could not be computed appears in :attr:`features` with ``psi=None`` and
    never in ``drifted``.
    """

    psi_threshold: float = DEFAULT_PSI_THRESHOLD
    features: Mapping[str, FeatureDrift] = field(default_factory=dict)
    drifted: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        compared = sum(1 for item in self.features.values() if item.psi is not None)
        skipped = len(self.features) - compared
        if self.drifted:
            head = (
                f"{compared} features compared, {len(self.drifted)} drifted "
                f"(psi > {self.psi_threshold:g}): {', '.join(self.drifted)}"
            )
        else:
            head = f"{compared} features compared, none drifted (psi <= {self.psi_threshold:g})"
        if skipped:
            head += f"; {skipped} skipped"
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi_threshold": self.psi_threshold,
            "drifted": list(self.drifted),
            "summary": self.summary,
            "notes": list(self.notes),
            "features": {name: item.to_dict() for name, item in self.features.items()},
        }


def _bundle_columns(bundle: Any) -> tuple[tuple[str, ...], np.ndarray]:
    """Feature names and the ``(rows, columns)`` matrix of a dataset bundle.

    A ``FeatureFrame`` (``X`` + ``columns``) is the normal input; a mapping of
    name to 1-D array is accepted so callers can compare a subset without
    rebuilding a frame.
    """
    if isinstance(bundle, Mapping):
        names = tuple(str(name) for name in bundle)
        columns = [np.asarray(bundle[name], dtype=np.float64) for name in names]
        if not columns:
            raise ValueError("dataset bundle has no columns")
        lengths = {column.shape[0] for column in columns}
        if len(lengths) != 1:
            raise ValueError("dataset bundle columns have different lengths")
        return names, np.column_stack(columns)

    names = tuple(str(name) for name in bundle.columns)
    matrix = np.asarray(bundle.X, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"dataset bundle X must be two-dimensional, got {matrix.shape}")
    if matrix.shape[1] != len(names):
        raise ValueError(f"dataset bundle has {matrix.shape[1]} columns but {len(names)} names")
    return names, matrix


def drift_from_datasets(
    reference_bundle: Any,
    current_bundle: Any,
    *,
    columns: Sequence[str] | None = None,
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    bins: int = DEFAULT_BINS,
) -> DriftReport:
    """Per-feature PSI/KS between two feature bundles.

    ``columns`` restricts the comparison and must exist in both bundles;
    a missing name is a schema mismatch, not drift, so it raises. Rows with a
    non-finite value in a column are dropped for that column only and counted
    in the per-feature note.
    """
    _check_bins(bins)
    if not math.isfinite(psi_threshold) or psi_threshold < 0.0:
        raise ValueError("psi_threshold must be finite and >= 0")
    ref_names, ref_matrix = _bundle_columns(reference_bundle)
    cur_names, cur_matrix = _bundle_columns(current_bundle)
    ref_index = {name: index for index, name in enumerate(ref_names)}
    cur_index = {name: index for index, name in enumerate(cur_names)}
    selected = list(ref_names if columns is None else columns)
    missing = [name for name in selected if name not in ref_index or name not in cur_index]
    if missing:
        raise ValueError(f"columns missing from a bundle: {missing}")

    features: dict[str, FeatureDrift] = {}
    notes: list[str] = []
    drifted: list[str] = []
    for name in selected:
        ref, dropped_ref = _finite_or_none(ref_matrix[:, ref_index[name]], f"reference[{name}]")
        cur, dropped_cur = _finite_or_none(cur_matrix[:, cur_index[name]], f"current[{name}]")
        note_parts: list[str] = []
        if dropped_ref or dropped_cur:
            note_parts.append(
                f"dropped {dropped_ref} reference / {dropped_cur} current non-finite rows"
            )
        if ref is None or cur is None:
            note_parts.append("no finite values in one sample; psi/ks not computed")
            note = "; ".join(note_parts)
            features[name] = FeatureDrift(
                feature=name,
                psi=None,
                ks=None,
                ks_pvalue=None,
                note=note,
                dropped_reference=dropped_ref,
                dropped_current=dropped_cur,
            )
            notes.append(f"{name}: {note}")
            continue
        psi = population_stability_index(ref, cur, bins=bins)
        ks, pvalue = ks_statistic(ref, cur)
        feature_note = "; ".join(note_parts) or None
        if feature_note:
            notes.append(f"{name}: {feature_note}")
        features[name] = FeatureDrift(
            feature=name,
            psi=psi,
            ks=ks,
            ks_pvalue=pvalue,
            note=feature_note,
            dropped_reference=dropped_ref,
            dropped_current=dropped_cur,
        )
        if psi > psi_threshold:
            drifted.append(name)

    return DriftReport(
        psi_threshold=psi_threshold,
        features=features,
        drifted=tuple(sorted(drifted)),
        notes=tuple(notes),
    )


@dataclass(frozen=True, slots=True)
class PredictionDrift:
    """Whether the model's confidence and calibration moved between two samples.

    ``psi`` and ``ks`` describe the per-row maximum class probability, the
    usual uncertainty score. ``mean_ece_reference``/``mean_ece_current`` are
    the averages of the per-class ECE over the classes that actually occur;
    they are ``None`` when outcomes were not supplied. ``drifted`` is set by
    PSI alone — ECE is reported for context, not used as a second gate.
    """

    psi: float
    ks: float
    ks_pvalue: float
    psi_threshold: float
    drifted: bool
    mean_ece_reference: float | None = None
    mean_ece_current: float | None = None
    ece_delta: float | None = None
    notes: tuple[str, ...] = ()

    @property
    def summary(self) -> str:
        head = (
            f"prediction confidence psi={self.psi:.3f} (threshold "
            f"{self.psi_threshold:g}), ks={self.ks:.3f}"
        )
        head += ", drifted" if self.drifted else ", stable"
        if self.ece_delta is not None:
            head += f", mean ECE {self.mean_ece_reference:.3f} -> {self.mean_ece_current:.3f}"
        elif self.notes:
            head += "; " + "; ".join(self.notes)
        return head

    def to_dict(self) -> dict[str, Any]:
        return {
            "psi": self.psi,
            "ks": self.ks,
            "ks_pvalue": self.ks_pvalue,
            "psi_threshold": self.psi_threshold,
            "drifted": self.drifted,
            "mean_ece_reference": self.mean_ece_reference,
            "mean_ece_current": self.mean_ece_current,
            "ece_delta": self.ece_delta,
            "summary": self.summary,
            "notes": list(self.notes),
        }


def _probability_matrix(values: Any, name: str, classes: int) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"{name} probabilities must be two-dimensional, got {matrix.shape}")
    if matrix.shape[1] != classes:
        raise ValueError(f"{name} probabilities must have {classes} classes, got {matrix.shape[1]}")
    if matrix.shape[0] == 0:
        raise ValueError(f"{name} probabilities are empty")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} probabilities contain non-finite values")
    if (matrix < 0.0).any() or (matrix > 1.0).any():
        raise ValueError(f"{name} probabilities must lie in [0, 1]")
    totals = matrix.sum(axis=1)
    if not np.allclose(totals, 1.0, atol=1e-6):
        raise ValueError(f"{name} probabilities must sum to 1 per row")
    return matrix


def _mean_ece(
    outcomes: Any,
    probabilities: np.ndarray,
    classes: Sequence[int],
    bins: int,
) -> float | None:
    """Average per-class ECE over the classes that occur in ``outcomes``."""
    y = np.asarray(outcomes)
    if y.ndim != 1 or y.shape[0] != probabilities.shape[0]:
        raise ValueError("outcomes must be one-dimensional and match the probability rows")
    present = [value for value in classes if bool(np.any(y == value))]
    if not present:
        return None
    errors = [
        expected_calibration_error(y, probabilities, tuple(classes), positive=value, bins=bins)
        for value in present
    ]
    return float(np.mean(errors))


def prediction_drift(
    reference_probabilities: Any,
    current_probabilities: Any,
    *,
    reference_outcomes: Any | None = None,
    current_outcomes: Any | None = None,
    classes: Sequence[int] = (-1, 0, 1),
    psi_threshold: float = DEFAULT_PSI_THRESHOLD,
    bins: int = DEFAULT_BINS,
) -> PredictionDrift:
    """Compare prediction confidence histograms and mean calibration error.

    Both probability matrices are ``(rows, len(classes))`` and each row sums
    to 1. Outcomes are optional but must be supplied for both samples or
    neither; a class absent from an outcome vector is excluded from that
    sample's mean ECE, because a calibration error for a class the model never
    sees is not a fact.
    """
    _check_bins(bins)
    if not math.isfinite(psi_threshold) or psi_threshold < 0.0:
        raise ValueError("psi_threshold must be finite and >= 0")
    if len(classes) < 2:
        raise ValueError("classes must contain at least two labels")
    if (reference_outcomes is None) != (current_outcomes is None):
        raise ValueError("outcomes must be supplied for both samples or neither")

    ref = _probability_matrix(reference_probabilities, "reference", len(classes))
    cur = _probability_matrix(current_probabilities, "current", len(classes))
    ref_score = ref.max(axis=1)
    cur_score = cur.max(axis=1)

    psi = population_stability_index(ref_score, cur_score, bins=bins)
    ks, pvalue = ks_statistic(ref_score, cur_score)

    ece_ref: float | None = None
    ece_cur: float | None = None
    delta: float | None = None
    notes: list[str] = []
    if reference_outcomes is not None and current_outcomes is not None:
        ece_ref = _mean_ece(reference_outcomes, ref, classes, bins)
        ece_cur = _mean_ece(current_outcomes, cur, classes, bins)
        if ece_ref is not None and ece_cur is not None:
            delta = ece_cur - ece_ref
        else:
            notes.append("no outcome classes present; mean ECE not compared")
    else:
        notes.append("outcomes not supplied; mean ECE not compared")

    return PredictionDrift(
        psi=psi,
        ks=ks,
        ks_pvalue=pvalue,
        psi_threshold=psi_threshold,
        drifted=psi > psi_threshold,
        mean_ece_reference=ece_ref,
        mean_ece_current=ece_cur,
        ece_delta=delta,
        notes=tuple(notes),
    )
