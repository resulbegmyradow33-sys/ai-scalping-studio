"""Inference side of a trained run: load ``model.zip`` and act deterministically.

A run is more than its weights: the policy was trained on observations
normalised by ``VecNormalize``, so :func:`load_policy` also loads
``vecnormalize.pkl`` and applies the stored statistics before ``predict``. A run
without that file is rejected — predicting on unnormalised observations would
silently produce different actions than the backtest of the same policy.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from scalping.rl.registry import CONFIG_NAME, MODEL_NAME, VECNORMALIZE_NAME
from scalping.rl.runtime import pin_threads

__all__ = ["PolicyWrapper", "load_policy"]


class _IdentityEnv(gymnasium.Env):
    """A one-step env that only carries the run's observation space.

    ``VecNormalize.load`` needs a vectorised env to bind to even though inference
    only uses its running statistics; this stub never steps.
    """

    def __init__(self, obs_dim: int) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.observation_space = spaces.Box(-np.inf, np.inf, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(1)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(self.obs_dim, dtype=np.float32), {}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        return np.zeros(self.obs_dim, dtype=np.float32), 0.0, True, False, {}


class PolicyWrapper:
    """A trained policy with a single-observation ``act`` entry point."""

    def __init__(
        self,
        model: PPO,
        config: Mapping[str, Any],
        vec_normalize: VecNormalize,
    ) -> None:
        shape = model.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError(f"expected a 1-D observation space, got {model.observation_space}")
        if not isinstance(model.action_space, spaces.Discrete):
            raise TypeError(f"expected a Discrete action space, got {model.action_space}")
        self._model = model
        self._vec_normalize = vec_normalize
        self.config = dict(config)
        self.obs_dim = int(shape[0])
        self.n_actions = int(model.action_space.n)
        declared = config.get("obs_dim")
        if declared is not None and int(declared) != self.obs_dim:
            raise ValueError(
                f"config.json declares obs_dim={declared}, model expects {self.obs_dim}"
            )

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply the run's observation statistics, preserving a 1-D input shape."""
        array = np.asarray(obs, dtype=np.float32)
        single = array.ndim == 1
        if single:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.obs_dim:
            raise ValueError(
                f"expected an observation of width {self.obs_dim}, got shape {array.shape}"
            )
        normalized = np.asarray(self._vec_normalize.normalize_obs(array), dtype=np.float32)
        return normalized[0] if single else normalized

    def act(self, obs: np.ndarray) -> int:
        """Return the greedy action for one observation, always in ``0..n-1``.

        The observation is normalised with the run's stored statistics first;
        ``deterministic=True`` makes the action the argmax of the action
        distribution, so the same observation always maps to the same action.
        """
        array = self.normalize_obs(obs)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        action, _ = self._model.predict(array, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    def __call__(self, obs: np.ndarray) -> int:
        """Alias for :meth:`act`, for callables such as the backtest policy."""
        return self.act(obs)


def _load_vec_normalize(path: Path, obs_dim: int) -> VecNormalize:
    try:
        venv = DummyVecEnv([lambda: _IdentityEnv(obs_dim)])
        return VecNormalize.load(str(path), venv)
    except (
        OSError,
        AttributeError,
        AssertionError,
        EOFError,
        ValueError,
        pickle.UnpicklingError,
    ) as exc:
        raise ValueError(f"{path} is not a readable VecNormalize archive: {exc}") from exc


def load_policy(run_dir: str | Path) -> PolicyWrapper:
    """Load a run directory into a deterministic, obs-normalising policy."""
    pin_threads()
    directory = Path(run_dir)
    model_path = directory / MODEL_NAME
    config_path = directory / CONFIG_NAME
    vec_path = directory / VECNORMALIZE_NAME
    if not model_path.is_file():
        raise FileNotFoundError(f"no {MODEL_NAME} in {directory}")
    if not config_path.is_file():
        raise FileNotFoundError(f"no {CONFIG_NAME} in {directory}")
    if not vec_path.is_file():
        raise FileNotFoundError(f"no {VECNORMALIZE_NAME} in {directory}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    run_id = config.get("run_id")
    if run_id is not None and directory.name != str(run_id):
        raise ValueError(f"{config_path} belongs to run {run_id}, not to {directory.name}")

    declared = config.get("obs_dim")
    if declared is None:
        raise ValueError(f"{config_path} is missing obs_dim")
    obs_dim = int(declared)
    model = PPO.load(str(model_path), device="cpu")
    vec_normalize = _load_vec_normalize(vec_path, obs_dim)
    return PolicyWrapper(model, config, vec_normalize)
