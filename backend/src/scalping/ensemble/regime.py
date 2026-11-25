"""Regime-dependent member weights learned from out-of-fold predictions.

Phase 9, slice 2. :mod:`scalping.ensemble.core` combines members with one set of
weights. This module allows the weights to depend on the market regime described
by :func:`scalping.regime.rules.classify`, while keeping the evidence base the
same: every weight is derived from an :class:`~scalping.ensemble.core.OOFResult`,
so the rows used to choose the weights were scored by fits that never saw them.
Validation and test rows are never read here.

The regime label attached to a row is a *description of the past*. It answers
"what did the market look like up to this bar", never "what comes next". Labels
are joined to bundle rows with a backward as-of rule, so a row can only inherit a
label whose timestamp is at or before it.

What this does not claim: regime labels are heuristic, buckets are small, and the
weights are both fitted and scored on the same OOF rows. The report is
descriptive. A higher score than the global ensemble or the best member is not a
validated improvement and is not a trading edge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final, Literal

import numpy as np
import polars as pl

from scalping.datasets.builder import DatasetBundle
from scalping.ensemble.core import OOFResult, _check_members, _probability_matrix
from scalping.nn.evaluate import classification_metrics
from scalping.regime.strategy import UNKNOWN_REGIME

__all__ = [
    "RegimeSelectionConfig",
    "RegimeWeightTable",
    "RegimeWeightedEnsemble",
    "RegimeWeights",
    "WeightSource",
    "align_regime_labels",
    "fit_regime_weights",
    "regime_selection_report",
]

#: Where a row's weights came from.
WeightSource = Literal["regime", "global"]

#: Inverse-loss weights are undefined for a zero loss; clamp instead of dividing
#: by zero. The clamp is far below any probability a classifier emits in
#: practice, so it only guards a perfect synthetic member.
_MIN_LOG_LOSS: Final[float] = 1e-9

#: Reserved names in the report's overall comparison.
_COMPARISON_NAMES: Final[tuple[str, ...]] = ("regime_ensemble", "global_ensemble")

_MISSING_LABELS: Final[tuple[str, ...]] = ("", "nan", "NaN", "null", "None")


@dataclass(frozen=True, slots=True)
class RegimeSelectionConfig:
    """How much evidence a regime bucket needs before it gets its own weights.

    ``min_rows_per_regime`` is a research choice, not a law: a bucket with fewer
    out-of-fold rows falls back to the global weights instead of fitting a
    handful of rows. The fallback is reported, never silent.
    """

    min_rows_per_regime: int = 30

    def __post_init__(self) -> None:
        if self.min_rows_per_regime < 1:
            raise ValueError("min_rows_per_regime must be >= 1")


@dataclass(frozen=True, slots=True)
class RegimeWeights:
    """Validated weights plus the loss evidence they were derived from.

    ``rows`` is the bucket size used for the decision. It is ``0`` for a regime
    that was never seen in the OOF rows (the decision-time unknown bucket), which
    always uses the global weights. ``loss`` is the mean OOF log loss per member
    **of the bucket** for ``source="regime"`` and the global loss for
    ``source="global"``, so the numbers behind every weight are inspectable.
    """

    regime: str
    rows: int
    source: WeightSource
    weights: Mapping[str, float]
    loss: Mapping[str, float]

    def __post_init__(self) -> None:
        if not self.regime:
            raise ValueError("regime weights need a non-empty regime label")
        if self.rows < 0:
            raise ValueError("regime rows must be non-negative")
        if self.source not in ("regime", "global"):
            raise ValueError(f"unknown weight source {self.source!r}")
        if not self.weights:
            raise ValueError("regime weights need at least one member")
        values = np.asarray([float(value) for value in self.weights.values()], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("regime weights must be finite")
        if (values < 0.0).any():
            raise ValueError("regime weights must be non-negative")
        if not np.isclose(float(values.sum()), 1.0, atol=1e-9):
            raise ValueError("regime weights must sum to 1")
        if set(self.weights) != set(self.loss):
            raise ValueError("regime weights and losses must name the same members")
        losses = np.asarray(list(self.loss.values()), dtype=np.float64)
        if not np.isfinite(losses).all():
            raise ValueError("regime losses must be finite")
        if (losses < 0.0).any():
            raise ValueError("regime losses must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime": self.regime,
            "rows": int(self.rows),
            "source": self.source,
            "weights": {name: float(value) for name, value in self.weights.items()},
            "loss": {name: float(value) for name, value in self.loss.items()},
        }


@dataclass(frozen=True, slots=True)
class RegimeWeightTable:
    """Global weights and, per fixed regime label, the weights used for it.

    ``per_regime`` only contains labels that appear in the OOF rows. A label at
    decision time that never appeared there is not an error: it resolves to the
    global weights with ``rows=0``, because refusing to trade on an unseen regime
    would turn a coverage gap into a silent NO TRADE.
    """

    members: tuple[str, ...]
    classes: tuple[int, ...]
    rows: int
    min_rows_per_regime: int
    global_weights: Mapping[str, float]
    global_loss: Mapping[str, float]
    per_regime: Mapping[str, RegimeWeights]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.members)
        if not names:
            raise ValueError("a regime weight table needs at least one member")
        if len(set(names)) != len(names):
            raise ValueError(f"member names must be unique, got {names}")
        if len(self.classes) < 2:
            raise ValueError("a regime weight table needs at least two classes")
        if self.rows < 1:
            raise ValueError("a regime weight table needs at least one OOF row")
        if self.min_rows_per_regime < 1:
            raise ValueError("min_rows_per_regime must be >= 1")
        if set(self.global_weights) != set(names):
            raise ValueError("global weights must name exactly the members")
        if set(self.global_loss) != set(names):
            raise ValueError("global losses must name exactly the members")
        _check_unit_weights("global", self.global_weights)
        for label, entry in self.per_regime.items():
            if not label:
                raise ValueError("per_regime keys must be non-empty labels")
            if set(entry.weights) != set(names):
                raise ValueError(f"regime {label!r}: weights do not name exactly the members")
            if entry.rows > self.rows:
                raise ValueError(f"regime {label!r}: more rows than the OOF result has")
            if entry.source == "regime" and entry.rows < self.min_rows_per_regime:
                raise ValueError(
                    f"regime {label!r}: {entry.rows} rows is below min_rows_per_regime="
                    f"{self.min_rows_per_regime} but has its own weights"
                )

    @property
    def regimes(self) -> tuple[str, ...]:
        return tuple(sorted(self.per_regime))

    def for_regime(self, regime: str) -> RegimeWeights:
        """Weights for one label; unseen labels fall back to the global weights."""
        if not isinstance(regime, str) or not regime:
            raise ValueError("regime must be a non-empty string")
        entry = self.per_regime.get(regime)
        if entry is not None:
            return entry
        return RegimeWeights(
            regime=regime,
            rows=0,
            source="global",
            weights=self.global_weights,
            loss=self.global_loss,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rows": int(self.rows),
            "min_rows_per_regime": int(self.min_rows_per_regime),
            "classes": list(self.classes),
            "global_weights": {name: float(value) for name, value in self.global_weights.items()},
            "global_loss": {name: float(value) for name, value in self.global_loss.items()},
            "per_regime": {label: self.per_regime[label].to_dict() for label in self.regimes},
        }


def _check_unit_weights(scope: str, weights: Mapping[str, float]) -> None:
    values = np.asarray([float(value) for value in weights.values()], dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError(f"{scope} weights must be finite")
    if (values < 0.0).any():
        raise ValueError(f"{scope} weights must be non-negative")
    if not np.isclose(float(values.sum()), 1.0, atol=1e-9):
        raise ValueError(f"{scope} weights must sum to 1")


def _regime_vector(regimes: Sequence[str] | np.ndarray, *, rows: int) -> np.ndarray:
    """Validate a per-row regime label vector against the expected row count."""
    values = np.asarray(regimes, dtype=object)
    if values.ndim != 1:
        raise ValueError(f"regimes must be 1-D, got shape {values.shape}")
    if values.size != rows:
        raise ValueError(f"{values.size} regime labels do not match the {rows} rows they describe")
    labels = np.empty(values.size, dtype=object)
    for index, value in enumerate(values.tolist()):
        if not isinstance(value, str) or not value:
            raise ValueError(f"regime label at row {index} is not a non-empty string")
        if value in _MISSING_LABELS:
            raise ValueError(f"regime label at row {index} is a missing-value spelling")
        labels[index] = value
    return labels


def align_regime_labels(
    bundle: DatasetBundle,
    labels: pl.DataFrame,
    *,
    regime_column: str = "regime",
) -> np.ndarray:
    """Attach the regime label that was current at each bundle row.

    ``labels`` is the output of :func:`scalping.regime.rules.classify` (or any
    frame with ``ts`` and ``regime_column``). The join is backward as-of: a row
    takes the newest label whose timestamp is at or before it, so a label that
    starts after the row can never be attached. Rows that precede the first label
    become :data:`~scalping.regime.strategy.UNKNOWN_REGIME`, which the selection
    step resolves to the global weights.

    The rule engine is causal per row, so computing the labels on the full frame
    and only reading the rows that reach the ensemble is safe. Mutating later bars
    cannot move an earlier label.
    """
    missing = [name for name in ("ts", regime_column) if name not in labels.columns]
    if missing:
        raise ValueError(f"regime labels are missing columns: {missing}")
    if labels["ts"].n_unique() != labels.height:
        raise ValueError("regime labels contain duplicate timestamps")

    frame = labels.select(pl.col("ts"), pl.col(regime_column))
    dtype = frame.schema["ts"]
    if not isinstance(dtype, pl.Datetime):
        raise TypeError(f"regime labels 'ts' must be a datetime, got {dtype}")
    if dtype.time_zone is None:
        frame = frame.with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
    else:
        frame = frame.with_columns(pl.col("ts").dt.convert_time_zone("UTC"))

    text = pl.col(regime_column).cast(pl.String)
    absent = text.is_null() | text.is_in(list(_MISSING_LABELS))
    frame = frame.select(
        pl.col("ts").alias("_label_ts"),
        pl.when(absent).then(pl.lit(UNKNOWN_REGIME)).otherwise(text).alias("_regime"),
    ).sort("_label_ts")

    ts = np.asarray(bundle.ts, dtype="datetime64[us]")
    if ts.size == 0 or frame.is_empty():
        return np.full(ts.size, UNKNOWN_REGIME, dtype=object)
    rows = (
        pl.DataFrame({"ts": ts, "_row": np.arange(ts.size)})
        .with_columns(pl.col("ts").dt.replace_time_zone("UTC"))
        .sort("ts")
    )

    joined = rows.join_asof(
        frame,
        left_on="ts",
        right_on="_label_ts",
        strategy="backward",
    ).sort("_row")
    values = joined["_regime"].fill_null(UNKNOWN_REGIME).cast(pl.String).to_numpy()
    return np.asarray(values, dtype=object)


def _mean_log_loss(matrix: np.ndarray, y_true: np.ndarray, classes: Sequence[int]) -> float:
    """Mean multiclass log loss of one member on one set of rows."""
    position = {int(label): index for index, label in enumerate(classes)}
    try:
        truth = np.asarray([position[int(value)] for value in np.asarray(y_true)], dtype=np.int64)
    except KeyError as error:  # pragma: no cover - OOFResult already rejects this
        raise ValueError(f"label {error} is outside the class list {tuple(classes)}") from error
    chosen = np.clip(
        np.asarray(matrix, dtype=np.float64)[np.arange(truth.size), truth], _MIN_LOG_LOSS, 1.0
    )
    return float(-np.log(chosen).mean())


def _inverse_weights(loss: Mapping[str, float], members: Sequence[str]) -> dict[str, float]:
    """Weights proportional to ``1 / mean log loss``, normalised to sum to one.

    This is a deterministic heuristic, not a fitted optimiser: no hyperparameter
    is tuned and no ordering of the rows changes the result. A member that is
    worse on a bucket gets a smaller weight; all weights stay positive, so the
    combination never silently drops a member.
    """
    values = np.asarray(
        [1.0 / max(float(loss[name]), _MIN_LOG_LOSS) for name in members],
        dtype=np.float64,
    )
    total = float(values.sum())
    if not np.isfinite(total) or total <= 0.0:  # pragma: no cover - losses are finite
        raise ValueError("cannot derive inverse-loss weights from these losses")
    return {name: float(value) for name, value in zip(members, values / total, strict=True)}


def fit_regime_weights(
    oof: OOFResult,
    regimes: Sequence[str] | np.ndarray,
    config: RegimeSelectionConfig | None = None,
) -> RegimeWeightTable:
    """Per-regime weights from out-of-fold predictions only.

    ``regimes`` is one label per OOF row, aligned with ``oof.y_true`` (use
    :func:`align_regime_labels` and index with ``oof.row_index``). A bucket with
    fewer than ``config.min_rows_per_regime`` rows keeps the global weights and
    is marked ``source="global"``; the global weights themselves always use every
    OOF row.

    Nothing here reads ``bundle.split``: an :class:`OOFResult` already contains
    only training-split rows scored out of fold. Passing a label vector of the
    wrong length, an empty label or a label spelling of "missing" is an error,
    not a silent bucket.
    """
    cfg = RegimeSelectionConfig() if config is None else config
    labels = _regime_vector(regimes, rows=int(np.asarray(oof.y_true).size))
    members = tuple(oof.members)
    classes = tuple(int(value) for value in oof.classes)

    global_loss = {
        name: _mean_log_loss(oof.probabilities[name], oof.y_true, classes) for name in members
    }
    global_weights = _inverse_weights(global_loss, members)

    per_regime: dict[str, RegimeWeights] = {}
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        rows = int(mask.sum())
        if rows < cfg.min_rows_per_regime:
            per_regime[label] = RegimeWeights(
                regime=label,
                rows=rows,
                source="global",
                weights=global_weights,
                loss=global_loss,
            )
            continue
        bucket_loss = {
            name: _mean_log_loss(
                np.asarray(oof.probabilities[name])[mask],
                np.asarray(oof.y_true)[mask],
                classes,
            )
            for name in members
        }
        per_regime[label] = RegimeWeights(
            regime=label,
            rows=rows,
            source="regime",
            weights=_inverse_weights(bucket_loss, members),
            loss=bucket_loss,
        )

    return RegimeWeightTable(
        members=members,
        classes=classes,
        rows=int(np.asarray(oof.y_true).size),
        min_rows_per_regime=cfg.min_rows_per_regime,
        global_weights=global_weights,
        global_loss=global_loss,
        per_regime=per_regime,
    )


class RegimeWeightedEnsemble:
    """Weighted probability average with one weight vector per row's regime.

    Each row is combined with the weights of its own regime; an unseen regime
    uses the table's global weights. The member set is fixed, so a regime changes
    how much each member counts, never which members exist.
    """

    def __init__(
        self,
        table: RegimeWeightTable,
        *,
        probability_floor: float = 0.0,
        renormalise: bool = True,
    ) -> None:
        if not 0.0 <= probability_floor < 1.0:
            raise ValueError("probability_floor must be in [0, 1)")
        if probability_floor > 0.0 and not renormalise:
            raise ValueError("probability_floor requires renormalise=True")
        self.table = table
        self.probability_floor = float(probability_floor)
        self.renormalise = bool(renormalise)

    @property
    def members(self) -> tuple[str, ...]:
        return tuple(self.table.members)

    def _prepare(
        self, probabilities: Mapping[str, np.ndarray]
    ) -> tuple[list[np.ndarray], int, int]:
        _check_members(self.table.members, probabilities)
        matrices: list[np.ndarray] = []
        rows: int | None = None
        columns: int | None = None
        for name in self.table.members:
            matrix = _probability_matrix(
                name,
                probabilities[name],
                rows=rows,
                columns=columns,
                renormalise=self.renormalise,
            )
            if rows is None:
                rows, columns = matrix.shape
            if self.probability_floor > 0.0:
                matrix = np.maximum(matrix, self.probability_floor)
                matrix = matrix / matrix.sum(axis=1, keepdims=True)
            matrices.append(matrix)
        if rows is None or columns is None:  # pragma: no cover - members are non-empty
            raise RuntimeError("no member probabilities were prepared")
        if columns != len(self.table.classes):
            raise ValueError(
                f"regime ensemble expects {len(self.table.classes)} class columns, got {columns}"
            )
        return matrices, rows, columns

    def predict_proba(
        self,
        probabilities: Mapping[str, np.ndarray],
        regimes: Sequence[str] | np.ndarray,
    ) -> np.ndarray:
        """Combine each row with the weights of that row's regime."""
        matrices, rows, _ = self._prepare(probabilities)
        labels = _regime_vector(regimes, rows=rows)
        entries = [self.table.for_regime(str(label)) for label in labels.tolist()]
        weights = np.asarray(
            [[entry.weights[name] for name in self.table.members] for entry in entries],
            dtype=np.float64,
        )
        stacked = np.stack(matrices, axis=0)
        return np.einsum("nm,mnc->nc", weights, stacked)


def _metrics_row(
    name: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    classes: tuple[int, ...],
) -> dict[str, Any]:
    matrix = _probability_matrix(name, probabilities, rows=y_true.size, columns=len(classes))
    labels = np.asarray(classes)[np.argmax(matrix, axis=1)]
    metrics = classification_metrics(y_true, labels, matrix, classes=classes)
    return {
        "name": name,
        "rows": metrics.rows,
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "log_loss": metrics.log_loss,
        "brier": metrics.brier,
    }


def _weighted(
    probabilities: Mapping[str, np.ndarray],
    weights: Mapping[str, float],
    members: tuple[str, ...],
    classes: tuple[int, ...],
) -> np.ndarray:
    """One fixed-weight average with the shared row/column guards."""
    matrices: list[np.ndarray] = []
    rows: int | None = None
    for name in members:
        matrix = _probability_matrix(
            name,
            probabilities[name],
            rows=rows,
            columns=len(classes),
        )
        if rows is None:
            rows = matrix.shape[0]
        matrices.append(matrix)
    if rows is None:  # pragma: no cover - members are non-empty
        raise RuntimeError("no member probabilities were provided")
    stacked = np.stack(matrices, axis=0)
    vector = np.asarray([float(weights[name]) for name in members], dtype=np.float64)
    return np.tensordot(vector, stacked, axes=(0, 0))


def regime_selection_report(
    oof: OOFResult,
    regimes: Sequence[str] | np.ndarray,
    table: RegimeWeightTable,
) -> dict[str, Any]:
    """Descriptive report of regime-dependent selection on the OOF rows.

    Every score is computed on the same OOF rows the weights came from, so the
    comparison is in-sample for the weights and must not be read as validation.
    The report states that explicitly under ``"descriptive_only"`` and contains
    no promotion flag: a better row than the global ensemble or the best member
    is a question for a later, independent evaluation, not a verdict.

    Contains per-regime rows (bucket size, share, weight source, weights, member
    metrics, regime-ensemble and global-ensemble metrics), coverage and fallback
    usage, and an overall comparison of the regime ensemble, the global
    ensemble and the best single member on the same rows.
    """
    labels = _regime_vector(regimes, rows=int(np.asarray(oof.y_true).size))
    members = tuple(oof.members)
    classes = tuple(int(value) for value in oof.classes)
    y_true = np.asarray(oof.y_true)
    probabilities = {name: np.asarray(oof.probabilities[name]) for name in members}

    clash = sorted(set(members) & set(_COMPARISON_NAMES))
    if clash:
        raise ValueError(f"member names {clash} are reserved for the comparison rows")

    ensemble = RegimeWeightedEnsemble(table)
    combined = ensemble.predict_proba(probabilities, labels)
    global_combined = _weighted(probabilities, table.global_weights, members, classes)

    member_rows = {
        name: _metrics_row(name, y_true, probabilities[name], classes) for name in members
    }
    best = min(member_rows.values(), key=lambda row: (-row["accuracy"], row["name"]))
    best_member = str(best["name"])

    regime_rows: list[dict[str, Any]] = []
    for label in sorted(set(labels.tolist())):
        mask = labels == label
        rows = int(mask.sum())
        entry = table.for_regime(label)
        bucket_truth = y_true[mask]
        member_metrics = sorted(
            (
                _metrics_row(name, bucket_truth, probabilities[name][mask], classes)
                for name in members
            ),
            key=lambda row: (-row["accuracy"], row["name"]),
        )
        regime_rows.append(
            {
                "regime": label,
                "rows": rows,
                "share": rows / int(y_true.size),
                "source": entry.source,
                "weights": {name: float(value) for name, value in entry.weights.items()},
                "members": member_metrics,
                "ensemble": _metrics_row(label, bucket_truth, combined[mask], classes),
                "global_ensemble": _metrics_row(
                    label, bucket_truth, global_combined[mask], classes
                ),
            }
        )

    fallback_rows = sum(int(row["rows"]) for row in regime_rows if row["source"] == "global")
    fallback_regimes = [str(row["regime"]) for row in regime_rows if row["source"] == "global"]
    weighted_regimes = [str(row["regime"]) for row in regime_rows if row["source"] == "regime"]
    unknown_rows = int((labels == UNKNOWN_REGIME).sum())

    comparison = sorted(
        [
            _metrics_row("regime_ensemble", y_true, combined, classes),
            _metrics_row("global_ensemble", y_true, global_combined, classes),
            *member_rows.values(),
        ],
        key=lambda row: (-row["accuracy"], row["name"]),
    )

    return {
        "rows": int(y_true.size),
        "classes": list(classes),
        "min_rows_per_regime": int(table.min_rows_per_regime),
        "global_weights": {name: float(value) for name, value in table.global_weights.items()},
        "regimes": regime_rows,
        "coverage": {
            "regime_weighted_rows": int(y_true.size) - fallback_rows,
            "fallback_rows": fallback_rows,
            "fallback_share": fallback_rows / int(y_true.size),
            "unknown_rows": unknown_rows,
            "regime_weighted_regimes": weighted_regimes,
            "fallback_regimes": fallback_regimes,
        },
        "fallback": {
            "rows": fallback_rows,
            "share": fallback_rows / int(y_true.size),
            "regimes": fallback_regimes,
        },
        "comparison": comparison,
        "best_member": best_member,
        "descriptive_only": True,
    }
