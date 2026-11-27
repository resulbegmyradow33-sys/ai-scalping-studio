"""Decompose the realized PPO reward into its terms.

The simulator returns one clipped reward per step. This module replays a policy
and reconstructs the four components from the public step output and the
documented formula:

```
equity      = Δequity / initial * 100
drawdown    = -50 * (new excess depth beyond 3%)
entry       = -entry_penalty on an opened step
risk        = +0.02 on the same opened step
total       = clip(equity + drawdown + entry + risk, -5, 5)
```

It then reports counterfactual totals with one component removed. This is an
**offline ablation of the realized signal**, not a retraining experiment: the
trajectory stays fixed, so it answers "how much of this run's reward came from
X", not "what policy would X produce".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from scalping.config import Settings
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame

__all__ = ["reward_breakdown"]

_DD_LIMIT = 0.03
_DD_PENALTY = 50.0
_RISK_BONUS = 0.02
_CLIP = 5.0


def reward_breakdown(
    policy: Callable[[np.ndarray], int],
    features: FeatureFrame,
    settings: Settings,
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Replay ``policy`` once and split every reward into its terms."""
    if not callable(policy):
        raise TypeError(f"policy must be callable, got {type(policy).__name__}")
    sim = Simulator(features, settings)
    observation = sim.reset(seed)
    initial = float(settings.contract.initial_balance)
    entry_penalty = float(sim.entry_penalty)

    equity_before = initial
    peak = initial
    previous_excess = 0.0
    totals = {"equity": 0.0, "drawdown": 0.0, "entry": 0.0, "risk": 0.0}
    parts: dict[str, list[float]] = {"equity": [], "drawdown": [], "entry": [], "risk": []}
    reconstructed: list[float] = []
    returned: list[float] = []
    opened_steps = 0
    steps = 0

    while True:
        result = sim.step(int(policy(observation)))
        observation = result.obs
        equity = float(result.info["equity"])
        opened = result.info["opened"] is not None
        opened_steps += int(opened)
        steps += 1

        equity_term = (equity - equity_before) / initial * 100.0
        peak = max(peak, equity)
        drawdown = (peak - equity) / peak if peak > 0 else 0.0
        excess = max(0.0, drawdown - _DD_LIMIT)
        drawdown_term = -_DD_PENALTY * max(0.0, excess - previous_excess)
        previous_excess = max(previous_excess, excess)
        entry_term = -entry_penalty if opened else 0.0
        risk_term = _RISK_BONUS if opened else 0.0

        for name, value in (
            ("equity", equity_term),
            ("drawdown", drawdown_term),
            ("entry", entry_term),
            ("risk", risk_term),
        ):
            totals[name] += value
            parts[name].append(value)
        reconstructed.append(
            float(np.clip(equity_term + drawdown_term + entry_term + risk_term, -_CLIP, _CLIP))
        )
        returned.append(float(result.reward))
        equity_before = equity
        if result.terminated or result.truncated:
            break

    mismatch = max(
        (abs(left - right) for left, right in zip(reconstructed, returned)), default=0.0
    )
    def without(keep: tuple[str, ...]) -> list[float]:
        return [
            _clip(sum(parts[name][index] for name in keep))
            for index in range(len(reconstructed))
        ]

    variants = {
        "all": reconstructed,
        "no_equity": without(("drawdown", "entry", "risk")),
        "no_drawdown": without(("equity", "entry", "risk")),
        "no_entry": without(("equity", "drawdown", "risk")),
        "no_risk": without(("equity", "drawdown", "entry")),
        "equity_only": without(("equity",)),
    }
    return {
        "steps": steps,
        "opened_steps": opened_steps,
        "totals": totals,
        "mean_reward": float(np.mean(returned)) if returned else 0.0,
        "reconstruction_max_error": mismatch,
        "variants": {
            name: {
                "mean": float(np.mean(values)) if values else 0.0,
                "total": float(np.sum(values)),
            }
            for name, values in variants.items()
        },
    }


def _clip(value: float) -> float:
    return float(np.clip(value, -_CLIP, _CLIP))
