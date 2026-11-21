"""Inference side of a trained run: load ``model.zip`` and act deterministically.

``load_policy`` is the only reader the evaluation and API layers need. It
re-reads ``config.json`` so a model copied out of its run directory, or a model
whose observation width no longer matches its config, fails with a clear error
instead of silently predicting on the wrong shape.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from scalping.rl.registry import MODEL_NAME

__all__ = ["CONFIG_NAME", "PolicyWrapper", "load_policy"]

#: File name of the resolved config inside a run directory.
CONFIG_NAME: str = "config.json"


class PolicyWrapper:
    """A trained policy with a single-observation ``act`` entry point."""

    def __init__(self, model: PPO, config: Mapping[str, Any]) -> None:
        shape = model.observation_space.shape
        if shape is None or len(shape) != 1:
            raise ValueError(f"expected a 1-D observation space, got {model.observation_space}")
        if not isinstance(model.action_space, spaces.Discrete):
            raise TypeError(f"expected a Discrete action space, got {model.action_space}")
        self._model = model
        self.config = dict(config)
        self.obs_dim = int(shape[0])
        self.n_actions = int(model.action_space.n)
        declared = config.get("obs_dim")
        if declared is not None and int(declared) != self.obs_dim:
            raise ValueError(
                f"config.json declares obs_dim={declared}, model expects {self.obs_dim}"
            )

    def act(self, obs: np.ndarray) -> int:
        """Return the greedy action for one observation, always in ``0..3``.

        The policy runs in evaluation mode and ``deterministic=True`` makes the
        action the argmax of the action distribution, so the same observation
        always maps to the same action.
        """
        array = np.asarray(obs, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != self.obs_dim:
            raise ValueError(
                f"expected an observation of width {self.obs_dim}, got shape {array.shape}"
            )
        action, _ = self._model.predict(array, deterministic=True)
        return int(np.asarray(action).reshape(-1)[0])

    def __call__(self, obs: np.ndarray) -> int:
        """Alias for :meth:`act`, for callables such as the backtest policy."""
        return self.act(obs)


def load_policy(run_dir: str | Path) -> PolicyWrapper:
    """Load ``runs/<run_id>/model.zip`` plus its config into a wrapper."""
    directory = Path(run_dir)
    model_path = directory / MODEL_NAME
    config_path = directory / CONFIG_NAME
    if not model_path.is_file():
        raise FileNotFoundError(f"no {MODEL_NAME} in {directory}")
    if not config_path.is_file():
        raise FileNotFoundError(f"no {CONFIG_NAME} in {directory}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise TypeError(f"{config_path} must contain a JSON object")
    run_id = config.get("run_id")
    if run_id is not None and directory.name != str(run_id):
        raise ValueError(f"{config_path} belongs to run {run_id}, not to {directory.name}")

    model = PPO.load(str(model_path), device="cpu")
    return PolicyWrapper(model, config)
