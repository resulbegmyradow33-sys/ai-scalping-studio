"""Gymnasium wrapper around the deterministic :class:`Simulator`.

The wrapper is composition, not inheritance: ``env.sim`` is the accounting loop
and ``env.sim.broker`` stays reachable for tests and diagnostics. The spaces are
sized from ``sim.obs_dim``, so changing the feature count never leaves a stale
observation shape behind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scalping.config import Settings
from scalping.env.simulator import Simulator, StepResult
from scalping.features.build import FeatureFrame

__all__ = ["TradingEnv"]


class TradingEnv(gym.Env):
    """Single-position scalping environment over a :class:`FeatureFrame`.

    ``strategy_selector``, when given, is called with the current observation
    before every step and labels any entry made by that step (the default
    label is ``"manual"``).
    """

    def __init__(
        self,
        features: FeatureFrame,
        settings: Settings,
        strategy_selector: Callable[[np.ndarray], str] | None = None,
    ) -> None:
        super().__init__()
        self.sim = Simulator(features, settings)
        self.strategy_selector = strategy_selector
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-np.inf, np.inf, (self.sim.obs_dim,), dtype=np.float32)
        self._obs = np.zeros(self.sim.obs_dim, dtype=np.float32)

    @property
    def obs_dim(self) -> int:
        """Length of one observation vector (window plus position features)."""
        return self.sim.obs_dim

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._obs = self.sim.reset(seed=seed)
        return self._obs, {}

    def step(self, action: int) -> StepResult:
        if self.strategy_selector is not None:
            self.sim.strategy = str(self.strategy_selector(self._obs))
        result = self.sim.step(int(action))
        # Keep a private copy so a caller mutating the returned obs cannot
        # corrupt the observation the strategy selector reads next step.
        self._obs = result.obs.copy()
        return result
