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
from gymnasium import spaces

from scalping.config import Settings
from scalping.env.simulator import ArmSelector, Simulator, StepResult
from scalping.features.build import FeatureFrame

__all__ = ["TradingEnv"]


class TradingEnv(gym.Env):
    """Single-position scalping environment over a :class:`FeatureFrame`.

    ``strategy_selector``, when given, is called once per step with the bandit's
    context row for the decision bar and returns an arm index (a position in
    :data:`~scalping.strategy.signals.ALL_STRATEGIES`). A bandit-like object
    with ``select(context)`` — :class:`~scalping.strategy.bandit.LinUCB` — is
    accepted in place of the callable. The observation gains that arm's one-hot
    and its signal strength, and an entry made by the step is labelled with the
    arm's name; without a selector the default label is ``"manual"``. The reset
    observation carries an all-zero strategy block: no decision has been made.
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        strategy_selector: Callable[[np.ndarray], int] | ArmSelector | None = None,
    ) -> None:
        super().__init__()
        self.sim = Simulator(features, settings, strategy_selector=strategy_selector)
        self.strategy_selector = strategy_selector
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
