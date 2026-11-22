"""Gymnasium wrapper around the deterministic :class:`Simulator`.

The wrapper is composition, not inheritance: ``env.sim`` is the accounting loop
and ``env.sim.broker`` stays reachable for tests and diagnostics. The spaces are
sized from ``sim.obs_dim``, so changing the feature count — or adding the
strategy context — never leaves a stale observation shape behind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
import polars as pl
from gymnasium import spaces

from scalping.config import Settings
from scalping.env.simulator import ArmSelector, Simulator, StepResult
from scalping.features.build import FeatureFrame

__all__ = ["TradingEnv"]


class TradingEnv(gym.Env):
    """Single-position scalping environment over a :class:`FeatureFrame`.

    ``strategy_selector``, when given, is called once per decision bar with the
    bandit's context row for that bar and returns an arm index (a position in
    :data:`~scalping.strategy.signals.ALL_STRATEGIES`). A bandit-like object
    with ``select(context)`` — :class:`~scalping.strategy.bandit.LinUCB` — is
    accepted in place of the callable. The observation gains that arm's one-hot
    and its signal strength, so the reset observation and every step describe
    the arm of the bar they cover. An entry is labelled with the arm chosen for
    its bar; without a selector the default label is ``"manual"``.

    ``labels`` is the optional :func:`~scalping.strategy.labeling.label_events`
    frame the context matrix's success columns are computed from. Training
    should pass ``label_events(df, settings)`` so the context is priced with the
    episode's account instead of the module default.

    ``start`` is the first decision bar of every episode, forwarded to the
    simulator. Training uses it to give each vectorised env a different slice of
    the same frame; the default keeps the historical behaviour.

    ``random_start`` and ``seed`` are forwarded to the simulator: with
    ``random_start=True`` every reset samples a new start from the usable range
    with a construction-seeded RNG, so long training runs cover the whole frame
    instead of replaying one window.
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        strategy_selector: Callable[[np.ndarray], int] | ArmSelector | None = None,
        labels: pl.DataFrame | None = None,
        start: int | None = None,
        random_start: bool = False,
        seed: int = 0,
        entry_penalty: float | None = None,
        gate_signals: bool = False,
        window: int = 60,
        flow: np.ndarray | None = None,
        flow_thr: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        self.sim = Simulator(
            features,
            settings,
            start=0 if start is None else start,
            strategy_selector=strategy_selector,
            labels=labels,
            random_start=random_start,
            seed=seed,
            entry_penalty=entry_penalty,
            gate_signals=gate_signals,
            window=window,
            flow=flow,
            flow_thr=flow_thr,
        )
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.sim.obs_dim,), dtype=np.float32)
        self._obs = np.zeros(self.sim.obs_dim, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        """Length of one observation vector (window, position and strategy context)."""
        return self.sim.obs_dim

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._obs = self.sim.reset(seed=seed)
        return self._obs.copy(), {}

    def step(self, action: int) -> StepResult:
        result = self.sim.step(int(action))
        # Keep a private copy so a caller mutating the returned obs cannot
        # corrupt the wrapper's own state.
        self._obs = result.obs.copy()
        return result
