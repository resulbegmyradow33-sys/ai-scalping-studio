"""Classical model factory.

Only the lightweight, CPU-friendly baselines are enabled by default. LightGBM
and XGBoost are optional: the factory reports a clear error when they are not
installed instead of silently substituting a different model.
"""

from __future__ import annotations

from typing import Any, Final

import numpy as np
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
    "lightgbm",
    "xgboost",
)


class _XGBLabelClassifier:
    """XGBoost wrapper that accepts the dataset's ``-1/+1`` labels.

    XGBoost 3.x requires classes encoded as ``0..n-1``. The wrapper keeps the
    original class order so ``train_classifier`` and the evaluator can map
    probability columns back without leaking the encoding into the dataset.
    """

    def __init__(self, **params: Any) -> None:
        self._params = params
        self._model: Any = None
        self.classes_: np.ndarray = np.array([], dtype=np.int64)
        self.n_features_in_: int = 0

    def fit(self, X: np.ndarray, y: np.ndarray) -> _XGBLabelClassifier:
        try:
            from xgboost import XGBClassifier
        except (ImportError, OSError) as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "xgboost could not load its native library; on macOS install OpenMP "
                "with `brew install libomp`"
            ) from exc
        self.classes_ = np.unique(y)
        encoded = np.searchsorted(self.classes_, y)
        self._model = XGBClassifier(**self._params).fit(X, encoded)
        self.n_features_in_ = int(X.shape[1])
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


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
    if key == "lightgbm":
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as exc:  # pragma: no cover - platform dependent
            raise RuntimeError(
                "lightgbm could not load its native library; on macOS install OpenMP "
                "with `brew install libomp`"
            ) from exc
        return LGBMClassifier(
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        )
    if key == "xgboost":
        return _XGBLabelClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method="hist",
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        )
    raise ValueError(
        f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}"
    )
