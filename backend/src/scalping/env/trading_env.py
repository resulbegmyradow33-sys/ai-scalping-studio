"""Gymnasium wrapper around the deterministic :class:`Simulator`.

The wrapper is composition, not inheritance: ``env.sim`` is the accounting loop
and ``env.sim.broker`` stays reachable for tests and diagnostics. The spaces are
sized from ``sim.obs_dim``, so changing the feature count never leaves a stale
observation shape behind.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Self

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from scalping.config import Settings
from scalping.env.simulator import Simulator, StepResult
from scalping.features.build import FeatureFrame

__all__ = ["ResetResult", "TradingEnv"]


class ResetResult(tuple):
    """``(obs, info)`` that also converts to the observation array.

    Gymnasium's contract is ``reset() -> (obs, info)`` while the project's
    determinism check compares two resets with ``np.allclose`` and the training
    loops want the bare vector. Iterating/unpacking yields ``(obs, info)``;
    ``np.asarray(result)`` yields ``obs``.
    """

    def __new__(cls, obs: np.ndarray, info: dict[str, Any]) -> Self:
        return super().__new__(cls, (obs, info))

    def __array__(self, dtype: np.dtype | None = None, copy: bool | None = None) -> np.ndarray:
        return np.asarray(self[0], dtype=dtype)


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
    ) -> ResetResult:
        super().reset(seed=seed)
        self._obs = self.sim.reset(seed=seed)
        return ResetResult(self._obs, {})

    def step(self, action: int) -> StepResult:
        if self.strategy_selector is not None:
            self.sim.strategy = str(self.strategy_selector(self._obs))
        result = self.sim.step(int(action))
        self._obs = result.obs
        return result
