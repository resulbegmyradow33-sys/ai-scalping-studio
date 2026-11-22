"""Backtest runner: drive the simulator with a policy and summarise the run.

The engine owns no accounting. It instantiates exactly one
:class:`~scalping.env.simulator.Simulator`, calls ``step`` until the episode
terminates or truncates, records the equity the simulator reports after every
step and collects the ``Trade`` objects the simulator closed. Metrics are
computed from that output alone.

A run is identified by a hash of everything that changes its arithmetic:

* ``run_id(settings, seed, data_hash, policy_name=...)`` over the canonical
  JSON of the contract, risk and cost blocks plus the data and policy hashes.
* ``data_hash`` covers the raw frame (shapes plus ``X``, ``ts``, ``ohlc``,
  ``spread`` and ``atr`` bytes), so a change in prices that leaves the
  normalised matrix untouched still changes the id.
* the policy identity is an explicit ``policy_name`` when given, otherwise the
  ``__qualname__`` plus a hash of the function's bytecode and constants. Two
  identical lambdas therefore share an id; two lambdas with different bodies do
  not, even when they are defined on the same line.

``BacktestResult.config_hash`` covers configuration and data only;
``BacktestResult.run_id`` additionally includes the policy identity.
"""

from __future__ import annotations

import hashlib
import json
import types
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import polars as pl

from scalping.backtest.metrics import Metrics, compute
from scalping.config import Settings
from scalping.env.simulator import WINDOW, ArmSelector, Simulator
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
    """Full sha256 of the raw frame: shapes plus every price and feature array."""
    digest = hashlib.sha256()
    for array in (features.X, features.ts, features.ohlc, features.spread, features.atr):
        values = np.ascontiguousarray(array)
        digest.update(f"{values.dtype}{values.shape}".encode())
        digest.update(values.tobytes())
    return digest.hexdigest()


def run_id(
    cfg: Settings,
    seed: int,
    data_hash: str,
    *,
    policy_name: str = "",
) -> str:
    """First 12 hex chars of the run's sha256; identical inputs give identical ids.

    ``policy_name`` is the policy identity (see :func:`run_backtest`); an empty
    string identifies a configuration-and-data fingerprint with no policy.
    """
    payload = {
        "contract": asdict(cfg.contract),
        "risk": asdict(cfg.risk),
        "costs": asdict(cfg.costs),
        "seed": int(seed),
        "data_hash": str(data_hash),
        "policy": str(policy_name),
    }
    return hashlib.sha256(_canonical(payload)).hexdigest()[:12]


def _const_fingerprint(value: object) -> str:
    if isinstance(value, types.CodeType):
        return f"<code {_code_fingerprint(value)}>"
    return repr(value)


def _code_fingerprint(code: types.CodeType) -> str:
    """Hash a code object without its filename, qualname or line numbers."""
    payload = "|".join(
        (
            code.co_code.hex(),
            ",".join(_const_fingerprint(value) for value in code.co_consts),
            ",".join(code.co_names),
            ",".join(code.co_varnames),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_label(policy: Callable[[np.ndarray], int], policy_name: str | None) -> str:
    if policy_name:
        return policy_name
    return getattr(policy, "__qualname__", None) or type(policy).__qualname__


def _policy_identity(policy: Callable[[np.ndarray], int], policy_name: str | None) -> str:
    """Explicit name first, then qualname plus bytecode/constants, else qualname."""
    if policy_name:
        return policy_name
    label = _policy_label(policy, None)
    code = getattr(policy, "__code__", None)
    if code is None:
        return label
    return f"{label}#{_code_fingerprint(code)[:12]}"


def run_backtest(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
    *,
    policy_name: str | None = None,
    max_steps: int | None = None,
    strategy_selector: ArmSelector | Callable[[np.ndarray], int] | None = None,
    labels: pl.DataFrame | None = None,
    window: int = WINDOW,
) -> BacktestResult:
    """Run ``policy`` over ``features`` and return equity, trades and metrics.

    ``policy_name`` overrides the derived policy identity in the run id. Without
    it, a lambda's qualname and bytecode hash form the identity, so two
    identical lambdas share an id and a different body gets a different one.
    ``max_steps`` overrides the simulator's default step cap, so a caller can
    evaluate a window longer than :data:`~scalping.env.simulator.MAX_STEPS`
    without truncating its tail. ``strategy_selector`` and ``labels`` give the
    simulator the +6 strategy context, so a policy trained with a frozen bandit
    receives the observation width it expects.
    """
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")

    label = _policy_label(policy, policy_name)
    identity = _policy_identity(policy, policy_name)
    sim = Simulator(
        features,
        settings,
        strategy=label,
        max_steps=max_steps,
        strategy_selector=strategy_selector,
        labels=labels,
        window=window,
    )
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
        run_id=run_id(settings, _SEED, digest, policy_name=identity),
        metrics=compute(
            trades,
            values,
            float(settings.contract.initial_balance),
            ts=features.ts,
        ),
        equity=values,
        trades=trades,
        config_hash=config_hash,
    )
