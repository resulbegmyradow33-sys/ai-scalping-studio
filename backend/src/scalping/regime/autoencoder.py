"""Autoencoder representation for regime research.

The autoencoder compresses the eight causal regime features into a small latent
vector; K-Means then labels the latent space. It is a research variant, not a
forecast: the labels describe the past path exactly like the rule/model labels.

Leakage contract matches :mod:`scalping.regime.models`: the scaler, the network
and the cluster centres are fit on the first ``train_fraction`` rows only; every
row is then encoded and labelled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl
import torch
from sklearn.cluster import KMeans  # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler  # type: ignore[import-untyped]
from torch import nn

from scalping.regime.features import build_regime_features, feature_matrix
from scalping.regime.models import RegimeFit, RegimeModelError

__all__ = ["AutoencoderConfig", "fit_autoencoder"]


@dataclass(frozen=True, slots=True)
class AutoencoderConfig:
    """Small CPU-friendly architecture and training budget."""

    n_states: int = 4
    latent: int = 4
    hidden: int = 16
    epochs: int = 40
    learning_rate: float = 1e-2
    train_fraction: float = 0.7
    min_rows: int = 120
    seed: int = 7

    def __post_init__(self) -> None:
        if self.n_states < 2:
            raise ValueError("n_states must be >= 2")
        if not 1 <= self.latent <= 16:
            raise ValueError("latent must be in [1, 16]")
        if self.hidden < 2:
            raise ValueError("hidden must be >= 2")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if not np.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if not 0.5 <= self.train_fraction < 1.0:
            raise ValueError("train_fraction must be in [0.5, 1)")
        if self.min_rows < 20:
            raise ValueError("min_rows must be >= 20")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")


class _Autoencoder(nn.Module):
    def __init__(self, n_features: int, hidden: int, latent: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_features),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        return self.decoder(latent), latent


def _canonical(
    raw: np.ndarray, train_raw: np.ndarray, ret_1_train: np.ndarray, n_states: int
) -> tuple[np.ndarray, tuple[str, ...]]:
    means = np.full(n_states, np.inf)
    for cluster in range(n_states):
        mask = train_raw == cluster
        if mask.any():
            means[cluster] = float(ret_1_train[mask].mean())
    order = np.argsort(means, kind="stable")
    rank = {int(cluster): position for position, cluster in enumerate(order)}
    mapped = np.array([rank[int(cluster)] for cluster in raw], dtype=np.int64)
    return mapped, tuple(f"state_{position}" for position in range(n_states))


def fit_autoencoder(
    frame: pl.DataFrame,
    config: AutoencoderConfig | None = None,
) -> RegimeFit:
    """Fit a train-only autoencoder, cluster its latent space and label every row."""
    cfg = AutoencoderConfig() if config is None else config
    features = build_regime_features(frame)
    if features.height < cfg.min_rows:
        raise RegimeModelError(
            f"need at least {cfg.min_rows} bars for an autoencoder, got {features.height}"
        )
    split = int(features.height * cfg.train_fraction)
    if split < 3 * cfg.n_states:
        raise RegimeModelError(
            f"training slice of {split} bars is too short for {cfg.n_states} states"
        )
    X = feature_matrix(features)
    scaler = StandardScaler().fit(X[:split])
    scaled = scaler.transform(X).astype(np.float32)
    train = torch.from_numpy(np.ascontiguousarray(scaled[:split]))

    torch.manual_seed(cfg.seed)
    torch.set_num_threads(1)
    network = _Autoencoder(X.shape[1], cfg.hidden, cfg.latent)
    optimiser = torch.optim.Adam(network.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()
    network.train()
    for _ in range(cfg.epochs):
        optimiser.zero_grad(set_to_none=True)
        rebuilt, _ = network(train)
        loss = loss_fn(rebuilt, train)
        loss.backward()
        optimiser.step()

    network.eval()
    with torch.no_grad():
        all_tensor = torch.from_numpy(np.ascontiguousarray(scaled))
        train_latent = network.encoder(train).numpy()
        latent = network.encoder(all_tensor).numpy()

    estimator = KMeans(
        n_clusters=cfg.n_states, n_init=10, random_state=cfg.seed
    ).fit(train_latent)
    train_raw = estimator.predict(train_latent)
    raw = estimator.predict(latent)
    ids, states = _canonical(raw, train_raw, X[:split, 0], cfg.n_states)
    names = np.array([states[value] for value in ids])
    labels = pl.DataFrame(
        {
            "ts": features["ts"],
            "state": pl.Series(names, dtype=pl.String),
            "state_id": pl.Series(ids, dtype=pl.Int64),
        }
    )
    return RegimeFit(
        model="autoencoder",
        states=states,
        labels=labels,
        train_rows=split,
        rows=features.height,
    )
