"""Backtest runner: drive the simulator with a policy and summarise the run.

The engine owns no accounting. It instantiates exactly one
:class:`~scalping.env.simulator.Simulator`, calls ``step`` until the episode
terminates or truncates, records the equity the simulator reports after every
step and collects the ``Trade`` objects the simulator closed. Metrics are
computed from that output alone.

A run is identified by a hash of everything that changes its arithmetic:

* ``run_id(settings, seed, data_hash, policy_name=...)`` over the canonical
  JSON of the contract, risk and cost blocks plus the feature hashes.
* ``data_hash`` is the sha256 of ``FeatureFrame.X.tobytes()``, not of the row
  count, so any change in the feature matrix changes the id.

``BacktestResult.config_hash`` covers the configuration and data only;
``BacktestResult.run_id`` additionally includes the policy name, so two
policies over the same data get distinct ids.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from scalping.backtest.metrics import Metrics, compute
from scalping.config import Settings
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame
from scalping.types import Trade

__all__ = ["BacktestResult", "data_hash", "run_backtest", "run_id"]

#: Fixed simulator seed. The loop has no randomness; the seed only labels the run.
_SEED: int = 0


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """Outcome of one policy run over one feature frame."""

    run_id: str
    metrics: Metrics
    equity: np.ndarray
    trades: list[Trade]
    config_hash: str


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


def data_hash(features: FeatureFrame) -> str:
    """Full sha256 of the feature matrix bytes."""
    return hashlib.sha256(features.X.tobytes()).hexdigest()


def run_id(
    cfg: Settings,
    seed: int,
    data_hash: str,
    *,
    policy_name: str = "",
) -> str:
    """First 12 hex chars of the run's sha256; identical inputs give identical ids."""
    payload = {
        "contract": asdict(cfg.contract),
        "risk": asdict(cfg.risk),
        "costs": asdict(cfg.costs),
        "seed": int(seed),
        "data_hash": str(data_hash),
        "policy": str(policy_name),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:12]


def _policy_name(policy: Callable[[np.ndarray], int]) -> str:
    return getattr(policy, "__qualname__", None) or type(policy).__qualname__


def run_backtest(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
) -> BacktestResult:
    """Run ``policy`` over ``features`` and return equity, trades and metrics."""
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")

    name = _policy_name(policy)
    sim = Simulator(features, settings, strategy=name)
    obs = sim.reset(_SEED)

    equity: list[float] = []
    trades: list[Trade] = []
    while True:
        result = sim.step(int(policy(obs)))
        obs = result.obs
        equity.append(float(result.info["equity"]))
        trade = result.info["trade"]
        if trade is not None:
            trades.append(trade)
        if result.terminated or result.truncated:
            break

    values = np.asarray(equity, dtype=np.float64)
    digest = data_hash(features)
    config_hash = run_id(settings, _SEED, digest)
    return BacktestResult(
        run_id=run_id(settings, _SEED, digest, policy_name=name),
        metrics=compute(
            trades,
            values,
            float(settings.contract.initial_balance),
        ),
        equity=values,
        trades=trades,
        config_hash=config_hash,
    )
