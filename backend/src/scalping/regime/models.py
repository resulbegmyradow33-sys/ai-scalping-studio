"""Statistical regime models: K-Means, Gaussian mixture, HMM and a supervised baseline.

All four share one contract: build the causal feature frame, fit on the first
``train_fraction`` of the rows only, then label every row. Standardisation is
fitted on the training slice too, so a mutated future cannot move the scaler or
the cluster centres that produced an earlier label.

Unsupervised cluster ids are canonicalised by the mean ``ret_1`` of their
training members: ``state_0`` is the lowest-return state and the highest id the
highest-return one. Without that step, K-Means/GMM/HMM ids would depend on
initialisation and two runs of the same data could swap labels.

The supervised baseline learns the rule engine's summary label
(:func:`scalping.regime.rules.classify`) from the same features. It exists to
measure how much of the rule's structure a model can reproduce, not to replace
the rule.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import numpy as np
import polars as pl
from hmmlearn.hmm import GaussianHMM  # type: ignore[import-untyped]
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore[import-untyped]
from sklearn.mixture import GaussianMixture  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]

from scalping.regime.features import build_regime_features, feature_matrix
from scalping.regime.rules import classify

__all__ = ["MODEL_NAMES", "ModelConfig", "RegimeFit", "RegimeModelError", "fit_predict"]

ModelName = Literal["kmeans", "gmm", "hmm", "supervised"]
MODEL_NAMES: Final[tuple[ModelName, ...]] = ("kmeans", "gmm", "hmm", "supervised")


class RegimeModelError(ValueError):
    """The frame cannot support the requested model."""


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """Training choices shared by every model."""

    n_states: int = 4
    train_fraction: float = 0.7
    seed: int = 7
    min_rows: int = 120

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.min_rows < 20:
            raise ValueError("min_rows must be >= 20")


@dataclass(frozen=True, slots=True)
class RegimeFit:
    """Aligned labels plus the metadata a report needs."""

    model: str
    states: tuple[str, ...]
    labels: pl.DataFrame
    train_rows: int
    rows: int

    def counts(self) -> dict[str, int]:
        """Bars per state, in state order."""
        observed = dict(self.labels.group_by("state").len().iter_rows())
        return {state: int(observed.get(state, 0)) for state in self.states}


def _canonical(
    raw: np.ndarray,
    train_raw: np.ndarray,
    ret_1_train: np.ndarray,
    n_states: int,
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Map arbitrary cluster ids to ``state_0..state_{k-1}`` by mean training return."""
    means = np.full(n_states, np.inf)
    for cluster in range(n_states):
        mask = train_raw == cluster
        if mask.any():
            means[cluster] = float(ret_1_train[mask].mean())
    order = np.argsort(means, kind="stable")
    rank = {int(cluster): position for position, cluster in enumerate(order)}
    mapped = np.array([rank[int(cluster)] for cluster in raw], dtype=np.int64)
    return mapped, tuple(f"state_{position}" for position in range(n_states))


def _fit_unsupervised(
    model: ModelName,
    X: np.ndarray,
    split: int,
    config: ModelConfig,
) -> tuple[np.ndarray, tuple[str, ...]]:
    scaler = StandardScaler().fit(X[:split])
    X_train = scaler.transform(X[:split])
    X_all = scaler.transform(X)

    if model == "kmeans":
        estimator = KMeans(
            n_clusters=config.n_states, n_init=10, random_state=config.seed
        ).fit(X_train)
        train_raw = estimator.predict(X_train)
        raw = estimator.predict(X_all)
    elif model == "gmm":
        mixture = GaussianMixture(
            n_components=config.n_states,
            covariance_type="full",
            reg_covar=1e-6,
            random_state=config.seed,
        ).fit(X_train)
        train_raw = mixture.predict(X_train)
        raw = mixture.predict(X_all)
    else:
        hidden = GaussianHMM(
            n_components=config.n_states,
            covariance_type="diag",
            n_iter=100,
            random_state=config.seed,
        ).fit(X_train)
        train_raw = hidden.predict(X_train)
        raw = hidden.predict(X_all)

    return _canonical(raw, train_raw, X[:split, 0], config.n_states)


def _fit_supervised(
    X: np.ndarray, rule_labels: pl.Series, split: int, config: ModelConfig
) -> tuple[np.ndarray, tuple[str, ...]]:
    y = rule_labels.to_numpy().astype(str)
    train_y = y[:split]
    classes = tuple(sorted(set(train_y.tolist())))
    if len(classes) < 2:
        raise RegimeModelError("the training slice contains a single rule regime")
    index = {name: position for position, name in enumerate(classes)}
    estimator = HistGradientBoostingClassifier(
        max_iter=100, random_state=config.seed
    ).fit(X[:split], train_y)
    raw = estimator.predict(X)
    return np.array([index[str(name)] for name in raw], dtype=np.int64), classes


def fit_predict(
    frame: pl.DataFrame,
    model: ModelName = "kmeans",
    config: ModelConfig | None = None,
) -> RegimeFit:
    """Fit ``model`` on the first ``train_fraction`` rows and label every row."""
    if model not in MODEL_NAMES:
        raise ValueError(f"unknown regime model {model!r}; choose from {MODEL_NAMES}")
    cfg = ModelConfig() if config is None else config
    features = build_regime_features(frame)
    if features.height < cfg.min_rows:
        raise RegimeModelError(
            f"need at least {cfg.min_rows} bars for a regime model, got {features.height}"
        )
    split = int(features.height * cfg.train_fraction)
    if split < 3 * cfg.n_states:
        raise RegimeModelError(
            f"training slice of {split} bars is too short for {cfg.n_states} states"
        )
    X = feature_matrix(features)

    if model == "supervised":
        rule = classify(frame)
        if rule.height != features.height:
            raise RegimeModelError("rule labels are not aligned with the feature frame")
        ids, states = _fit_supervised(X, rule["regime"], split, cfg)
        state_names = np.array([states[value] for value in ids])
    else:
        ids, states = _fit_unsupervised(model, X, split, cfg)
        state_names = np.array([states[value] for value in ids])

    labels = pl.DataFrame(
        {
            "ts": features["ts"],
            "state": pl.Series(state_names, dtype=pl.String),
            "state_id": pl.Series(ids, dtype=pl.Int64),
        }
    )
    return RegimeFit(
        model=model,
        states=states,
        labels=labels,
        train_rows=split,
        rows=features.height,
    )
