"""Classical model factory.

Two task types live here and they are not interchangeable:

* ``direction`` — a probabilistic classifier over the dataset's ``-1/+1``
  labels (``0`` too when flat rows are kept). It exposes ``predict_proba`` and
  is the only kind that may become a promotion candidate.
* ``anomaly`` — an unsupervised outlier scorer fitted without labels. It
  exposes ``score_samples`` through :mod:`scalping.ml.anomaly`; it has no
  direction probability, no ``predict_proba`` and no promotion path.
  :func:`build_model` rejects anomaly names so a classifier training call can
  never silently treat one as a direction model.

Only the lightweight, CPU-friendly baselines are enabled by default. LightGBM
and XGBoost are optional: the factory reports a clear error when they are not
installed instead of silently substituting a different model.
"""

from __future__ import annotations

from typing import Any, Final, Literal

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression, SGDClassifier

__all__ = [
    "ALL_MODEL_NAMES",
    "ANOMALY_MODEL_NAMES",
    "MODEL_NAMES",
    "MODEL_TASKS",
    "TaskType",
    "build_anomaly_model",
    "build_model",
    "feature_importances",
    "is_promotion_candidate",
    "model_task",
]

TaskType = Literal["direction", "anomaly"]

#: Classifiers ``POST /api/ml/train`` and ``ml train`` accept, in registry order.
MODEL_NAMES: Final[tuple[str, ...]] = (
    "logistic",
    "random_forest",
    "extra_trees",
    "hist_gradient",
    "sgd",
    "lightgbm",
    "xgboost",
)

#: Unsupervised detectors with their own scoring path, never a classifier.
ANOMALY_MODEL_NAMES: Final[tuple[str, ...]] = ("isolation_forest",)

#: Every name :func:`model_task` resolves, across both task types.
ALL_MODEL_NAMES: Final[tuple[str, ...]] = MODEL_NAMES + ANOMALY_MODEL_NAMES

#: Task type per registered name; the direction names are exactly ``MODEL_NAMES``.
MODEL_TASKS: Final[dict[str, TaskType]] = {
    **{name: "direction" for name in MODEL_NAMES},
    "isolation_forest": "anomaly",
}


def model_task(name: str) -> TaskType:
    """Return ``"direction"`` or ``"anomaly"`` for a registered model name.

    Raises ``ValueError`` for an unknown name, listing both registries so a
    caller cannot mistake an anomaly detector for a missing classifier.
    """
    key = name.strip().lower()
    try:
        return MODEL_TASKS[key]
    except KeyError:
        raise ValueError(
            f"unknown model {key!r}; available: {', '.join(ALL_MODEL_NAMES)}"
        ) from None


def is_promotion_candidate(name: str) -> bool:
    """Whether a model may become a promotion candidate.

    Only direction classifiers qualify. An anomaly detector is a diagnostic
    that describes how unusual a feature row is; its score is neither a
    direction probability nor a trading signal, so no run of it can be
    promoted or championed.
    """
    return model_task(name) == "direction"


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

    @property
    def feature_importances_(self) -> np.ndarray:
        """Delegate the fitted booster's importances; raises until fitted."""
        if self._model is None:
            raise NotFittedError("this XGBoost wrapper is not fitted")
        return np.asarray(self._model.feature_importances_)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("model is not fitted")
        return self.classes_[np.argmax(self.predict_proba(X), axis=1)]


def build_model(name: str, *, seed: int = 0) -> Any:
    """Return an unfitted direction estimator for ``name``.

    ``seed`` is applied everywhere the estimator exposes ``random_state``.
    Class weights are balanced for the models that support them; the direction
    target is roughly symmetric but sessions and regimes are not.

    An anomaly name raises ``ValueError``: use :func:`build_anomaly_model` and
    :func:`scalping.ml.anomaly.detect_anomalies` instead.
    """
    key = name.strip().lower()
    if key in ANOMALY_MODEL_NAMES:
        raise ValueError(
            f"{key!r} is an anomaly detector, not a direction classifier; "
            "use build_anomaly_model() and scalping.ml.anomaly"
        )
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
    if key == "sgd":
        # Linear-in-time, bounded-memory baseline. ``shuffle=False`` makes the
        # update order the caller's row order (chronological here), so repeated
        # fits and the chunked ``partial_fit`` path are reproducible. The
        # default learning-rate schedule is scale-aware; features are not
        # standardised, so very different feature magnitudes still matter.
        return SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            max_iter=1000,
            tol=1e-3,
            shuffle=False,
            class_weight="balanced",
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
    raise ValueError(f"unknown model {name!r}; available: {', '.join(MODEL_NAMES)}")


def build_anomaly_model(name: str = "isolation_forest", *, seed: int = 0) -> IsolationForest:
    """Return an unfitted anomaly detector.

    ``contamination="auto"`` only sets the offset used by ``decision_function``;
    the operational threshold is chosen from the training score quantile in
    :mod:`scalping.ml.anomaly`, so the reported anomaly rate follows the
    caller's quantile rather than a guessed contamination.
    """
    key = name.strip().lower()
    if key != "isolation_forest":
        raise ValueError(
            f"unknown anomaly model {name!r}; available: {', '.join(ANOMALY_MODEL_NAMES)}"
        )
    return IsolationForest(
        n_estimators=200,
        max_samples="auto",
        contamination="auto",
        random_state=seed,
        n_jobs=1,
    )


def feature_importances(estimator: Any) -> np.ndarray | None:
    """Return the estimator's fitted feature importances, or ``None``.

    Tree ensembles (random forest, extra trees), LightGBM and the XGBoost wrapper
    expose ``feature_importances_``; scikit-learn's histogram gradient boosting
    and the linear models (logistic, SGD) do not, so this returns ``None``
    instead of inventing a number. For histogram gradient boosting use
    permutation importance if the question really needs it.

    The returned vector is a copy, one-dimensional, finite and non-negative,
    aligned with the feature columns the model was fitted on. An unfitted
    estimator also returns ``None``. Importances describe where the fitted model
    spent its splits; they are not causal effects and not out-of-sample
    evidence.
    """
    try:
        raw = getattr(estimator, "feature_importances_", None)
    except (AttributeError, NotFittedError):
        return None
    if raw is None:
        return None
    values = np.asarray(raw, dtype=np.float64).reshape(-1)
    if not np.isfinite(values).all():
        raise ValueError("estimator returned non-finite feature importances")
    if (values < 0.0).any():
        raise ValueError("estimator returned negative feature importances")
    expected = getattr(estimator, "n_features_in_", None)
    if expected is not None and values.size != int(expected):
        raise ValueError(
            f"feature importances have {values.size} entries, expected {int(expected)}"
        )
    return values.copy()
