"""Evaluate a seed ensemble and its members on the same simulator slice.

Each member is a callable returning action probabilities; its simulated policy
takes the argmax. The ensemble averages the distributions (``PolicyEnsemble``)
and takes the argmax of the average. Every arm runs over the same bars with the
same settings, so the table is comparable.

This is a comparison harness, not a promotion: metrics are net of the
simulator's spread/commission/slippage and still need the usual out-of-sample
and cost-stress gates.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from scalping.backtest.engine import run_backtest
from scalping.config import Settings
from scalping.features.build import FeatureFrame
from scalping.rl.ensemble import PolicyEnsemble

__all__ = ["evaluate_seed_ensemble"]

Policy = Callable[[np.ndarray], Sequence[float]]


def _row(name: str, result, best_profit: float) -> dict[str, Any]:
    metrics = result.metrics
    return {
        "name": name,
        "trades": int(metrics.trades),
        "net_profit": float(metrics.net_profit),
        "expectancy_r": float(metrics.expectancy_r),
        "profit_factor": float(metrics.profit_factor),
        "max_drawdown": float(metrics.max_drawdown),
        "win_rate": float(metrics.win_rate),
        "delta_net_profit": float(metrics.net_profit) - best_profit,
    }


def evaluate_seed_ensemble(
    policies: Mapping[str, Policy],
    features: FeatureFrame,
    settings: Settings,
    *,
    weights: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Run each member and the averaged ensemble; return one row per arm."""
    if not policies:
        raise ValueError("need at least one member policy")
    if not all(callable(policy) for policy in policies.values()):
        raise TypeError("every member policy must be callable")

    ensemble = PolicyEnsemble(list(policies.values()), weights=weights)
    results: dict[str, Any] = {}
    for name, policy in policies.items():

        def member_action(obs: np.ndarray, _policy: Policy = policy) -> int:
            return int(np.argmax(_policy(obs)))

        results[name] = run_backtest(
            member_action,
            features,
            settings,
            policy_name=f"seed:{name}",
            max_steps=len(features.ts) + 1,
        )
    results["ensemble"] = run_backtest(
        ensemble.act,
        features,
        settings,
        policy_name="seed:ensemble",
        max_steps=len(features.ts) + 1,
    )

    best_profit = max(
        float(result.metrics.net_profit)
        for name, result in results.items()
        if name != "ensemble"
    )
    return [_row(name, result, best_profit) for name, result in results.items()]
