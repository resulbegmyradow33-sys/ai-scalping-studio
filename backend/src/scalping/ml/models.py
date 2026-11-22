"""Classical model factory.

Only the lightweight, CPU-friendly baselines are enabled by default. LightGBM
and XGBoost are optional: the factory reports a clear error when they are not
installed instead of silently substituting a different model.
"""

from __future__ import annotations

from typing import Any, Final

from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression

__all__ = ["MODEL_NAMES", "build_model"]

MODEL_NAMES: Final[tuple[str, ...]] = (
    "logistic",
    "random_forest",
    "extra_trees",
    "hist_gradient",
)


def build_model(name: str, *, seed: int = 0) -> Any:
    """Return an unfitted estimator for ``name``.

    ``seed`` is applied everywhere the estimator exposes ``random_state``.
    Class weights are balanced for the models that support them; the direction
    target is roughly symmetric but sessions and regimes are not.
    """
    key = name.strip().lower()
    if key == "logistic":
        return LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=seed,
        )
    if key == "random_forest":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    if key == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    if key == "hist_gradient":
        return HistGradientBoostingClassifier(
            max_iter=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_leaf=20,
            random_state=seed,
        )
    raise ValueError(
        f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}"
    )
