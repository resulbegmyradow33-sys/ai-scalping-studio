"""Feature-group ablation under one fixed protocol.

The question is narrow: for **this** dataset, model and spec, how much does the
validation score depend on each registered feature group? Three fixed run
families answer it, and nothing else is varied:

- ``all``: every selected group (the reference);
- ``without:<group>``: leave one group out;
- ``only:<group>``: that group alone.

The dataset is built once, so every run shares the same chronological split,
purge/embargo, seed, hyperparameters and flat-row policy. There is no
hyperparameter search and no test-split scoring: the validation split is the
only scored split, and the test rows stay reserved for the one-shot decision in
:func:`scalping.ml.train.train_classifier`. A run count of ``1 + 2 * groups``
is the whole cost, so the protocol is bounded by the feature-group registry.

What this is not:

- not a causal attribution. Dropping a group removes its columns, not their
  information: correlated groups can compensate, so a small delta does not mean
  the group is irrelevant, and a large delta does not prove it is the cause.
- not a trading result. A group that helps validation log loss can still lose
  money after costs; :mod:`scalping.ml.evaluate` is the next step, not this
  module.

Results carry no timestamp so identical inputs produce an identical payload.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Final

import numpy as np

from scalping.datasets.builder import build_dataset
from scalping.features.build import FeatureFrame
from scalping.features.registry import FEATURE_GROUPS, columns_for
from scalping.ml.models import build_model
from scalping.ml.train import TrainSpec, _baseline, _metrics

__all__ = [
    "ABLATION_METRICS",
    "PROTOCOL",
    "AblationResult",
    "AblationRun",
    "run_ablation",
]

#: Version label of the fixed protocol; bump only if the run families change.
PROTOCOL: Final[str] = "feature-group-ablation-v1"

#: Metrics compared across runs, in report order. ``delta = run - all``.
ABLATION_METRICS: Final[tuple[str, ...]] = (
    "accuracy",
    "balanced_accuracy",
    "log_loss",
    "brier",
    "ece",
)

#: Run families in output order.
_RUN_KINDS: Final[tuple[str, ...]] = ("all", "leave_one_out", "single_group")


@dataclass(frozen=True, slots=True)
class AblationRun:
    """One fixed-protocol fit: which groups were kept and how it scored."""

    name: str
    kind: str
    groups: tuple[str, ...]
    columns: tuple[str, ...]
    metrics: dict[str, float | int]
    delta: dict[str, float] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "groups": list(self.groups),
            "columns": list(self.columns),
            "features": len(self.columns),
            "metrics": dict(self.metrics),
            "delta_vs_all": None if self.delta is None else dict(self.delta),
        }


@dataclass(frozen=True, slots=True)
class AblationResult:
    """Every run plus the single dataset identity they were scored on."""

    symbol: str
    timeframe: str
    model: str
    dataset_id: str
    spec: TrainSpec
    columns: tuple[str, ...]
    split_counts: dict[str, int]
    baseline: dict[str, Any]
    runs: tuple[AblationRun, ...]
    protocol: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "model": self.model,
            "dataset_id": self.dataset_id,
            "spec": asdict(self.spec),
            "columns": list(self.columns),
            "split_counts": dict(self.split_counts),
            "baseline": self.baseline,
            "protocol": dict(self.protocol),
            "runs": [run.to_dict() for run in self.runs],
        }


def _resolve_groups(groups: tuple[str, ...]) -> tuple[str, ...]:
    """Reject an empty, unknown or duplicated selection before any fit."""
    if not groups:
        raise ValueError("ablation needs at least one feature group")
    unknown = [name for name in groups if name not in FEATURE_GROUPS]
    if unknown:
        raise ValueError(
            f"unknown feature groups: {', '.join(unknown)}; available: {', '.join(FEATURE_GROUPS)}"
        )
    resolved: list[str] = []
    for name in groups:
        if name not in resolved:
            resolved.append(name)
    if len(resolved) < 2:
        raise ValueError(
            "ablation needs at least two feature groups; leave-one-out over a single "
            "group would score an empty matrix"
        )
    return tuple(resolved)


def _run_columns(subset: tuple[str, ...]) -> tuple[str, ...]:
    """Columns for one run, refusing a group that expands to nothing."""
    columns = columns_for(subset)
    if not columns:
        raise ValueError(f"feature group selection {subset} has no columns")
    return columns


def _plan(groups: tuple[str, ...]) -> list[tuple[str, str, tuple[str, ...]]]:
    """The fixed run list: all, then leave-one-out, then single-group."""
    plans: list[tuple[str, str, tuple[str, ...]]] = [("all", "all", groups)]
    for group in groups:
        plans.append(
            (f"without:{group}", "leave_one_out", tuple(name for name in groups if name != group))
        )
    for group in groups:
        plans.append((f"only:{group}", "single_group", (group,)))
    return plans


def run_ablation(
    features: FeatureFrame,
    *,
    symbol: str,
    timeframe: str,
    spec: TrainSpec,
) -> AblationResult:
    """Run the fixed all / leave-one-out / single-group protocol.

    Every run fits a fresh ``build_model(spec.model, seed=spec.seed)`` on the
    same training rows and scores the same validation rows. Test rows are only
    counted, never predicted on. ``spec.evaluate_test`` is ignored on purpose:
    an ablation must not spend the locked split.
    """
    groups = _resolve_groups(tuple(spec.groups))
    # Fail on an unknown model before paying for the dataset build.
    build_model(spec.model, seed=spec.seed)

    bundle = build_dataset(
        features,
        symbol=symbol,
        timeframe=timeframe,
        groups=groups,
        horizon=spec.horizon,
        train_fraction=spec.train_fraction,
        validation_fraction=spec.validation_fraction,
        purge_bars=max(spec.purge_bars, spec.horizon),
        embargo_bars=spec.embargo_bars,
    )

    # Same flat-row policy as `ml train` and `ml cv`, so the row counts match.
    flat = bundle.y_direction == 0
    keep = np.ones(bundle.split.shape[0], dtype=bool) if spec.include_flat else ~flat
    train_mask = (bundle.split == 0) & keep
    validation_mask = (bundle.split == 1) & keep
    test_mask = (bundle.split == 2) & keep
    if int(train_mask.sum()) < 20:
        raise ValueError("not enough training rows; adjust fractions or horizon")
    if int(validation_mask.sum()) < 5:
        raise ValueError("not enough validation rows; adjust fractions or embargo")

    y_train = bundle.y_direction[train_mask]
    y_validation = bundle.y_direction[validation_mask]
    classes = tuple(int(value) for value in np.unique(y_train))
    if len(classes) < 2:
        raise ValueError(f"training split has a single class {list(classes)}")
    missing = np.setdiff1d(y_validation, np.asarray(classes))
    if missing.size:
        raise ValueError(f"validation labels {missing.tolist()} are absent from the training split")

    position = {name: index for index, name in enumerate(bundle.manifest.columns)}
    scored: list[tuple[str, str, tuple[str, ...], tuple[str, ...], dict[str, float | int]]] = []
    for name, kind, subset in _plan(groups):
        columns = _run_columns(subset)
        absent = [column for column in columns if column not in position]
        if absent:  # pragma: no cover - the registry and builder share one source
            raise ValueError(f"run {name}: columns missing from the dataset: {absent}")
        indices = [position[column] for column in columns]

        model = build_model(spec.model, seed=spec.seed)
        model.fit(bundle.X[train_mask][:, indices], y_train)
        probabilities = np.asarray(
            model.predict_proba(bundle.X[validation_mask][:, indices]), dtype=np.float64
        )
        if probabilities.shape != (y_validation.size, len(classes)):
            raise ValueError(
                f"run {name}: predict_proba returned shape {probabilities.shape}, "
                f"expected {(y_validation.size, len(classes))}"
            )
        totals = probabilities.sum(axis=1, keepdims=True)
        if not np.isfinite(totals).all() or np.any(totals <= 0.0):
            raise ValueError(f"run {name}: predict_proba rows do not sum to a positive value")
        # Float32 estimators (LightGBM) can sum to 1 +/- 1e-7; re-normalise so
        # log loss does not warn on rounding. Same treatment as `ml cv`.
        probabilities = probabilities / totals

        report = _metrics(y_validation, probabilities, classes)
        scored.append(
            (
                name,
                kind,
                subset,
                columns,
                {
                    "rows": int(report["rows"]),
                    "accuracy": float(report["accuracy"]),
                    "balanced_accuracy": float(report["balanced_accuracy"]),
                    "log_loss": float(report["log_loss"]),
                    "brier": float(report["brier"]),
                    "ece": float(report["calibration"]["ece"]),
                },
            )
        )

    reference = scored[0][4]
    runs = tuple(
        AblationRun(
            name=name,
            kind=kind,
            groups=subset,
            columns=columns,
            metrics=metrics,
            delta=None
            if name == "all"
            else {key: float(metrics[key]) - float(reference[key]) for key in ABLATION_METRICS},
        )
        for name, kind, subset, columns, metrics in scored
    )

    return AblationResult(
        symbol=symbol,
        timeframe=timeframe,
        model=spec.model,
        dataset_id=bundle.manifest.dataset_id,
        spec=spec,
        columns=bundle.manifest.columns,
        split_counts={
            "train": int(train_mask.sum()),
            "validation": int(validation_mask.sum()),
            "test": int(test_mask.sum()),
            "purged": int((bundle.split == -1).sum()),
            "flat_dropped": int(flat.sum()),
        },
        baseline=_baseline(y_train, y_validation, classes),
        runs=runs,
        protocol={
            "protocol": PROTOCOL,
            "run_kinds": list(_RUN_KINDS),
            "groups": len(groups),
            "runs": len(runs),
            "scored_split": "validation",
            "test_scored": False,
            "test_rows_reserved": int(test_mask.sum()),
            "evaluate_test_ignored": bool(spec.evaluate_test),
            "reference": "all",
            "delta": "run - all",
            "hyperparameter_search": False,
            "seed": spec.seed,
            "purge_bars": max(spec.purge_bars, spec.horizon),
            "embargo_bars": spec.embargo_bars,
        },
    )
