"""Callbacks that keep a run directory current while PPO learns.

``MetricsCallback`` writes ``metrics.json``; ``AtomicCheckpointCallback`` writes
``model.zip``. Both fire on the same step boundaries (multiples of
``checkpoint_every`` in environment steps), so the metrics file always describes
the checkpoint next to it. Both writes go through the registry's temp-file
replace, so an interrupt never leaves a half-written artefact.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from scalping.rl.registry import MODEL_NAME, save_model, write_json

__all__ = ["METRICS_NAME", "AtomicCheckpointCallback", "MetricsCallback"]

#: File name of the metrics snapshot inside a run directory.
METRICS_NAME: str = "metrics.json"

#: Number of actions of :class:`~scalping.env.trading_env.TradingEnv`.
N_ACTIONS: int = 4

#: Logger keys copied from stable-baselines3, in metrics order.
_LOSS_KEYS: tuple[str, ...] = (
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
)


def _finite(value: float | None) -> float:
    """``value`` as a finite float; a missing or non-finite number becomes 0."""
    if value is None:
        return 0.0
    number = float(value)
    return number if math.isfinite(number) else 0.0


class _StepTrigger:
    """Fire once each time the environment-step counter crosses a boundary."""

    def __init__(self, every: int) -> None:
        if isinstance(every, bool) or not isinstance(every, int) or every <= 0:
            raise ValueError(f"checkpoint_every must be a positive int, got {every!r}")
        self.every = every
        self.next = every

    def due(self, timesteps: int) -> bool:
        """Advance past every boundary at or below ``timesteps``."""
        if timesteps < self.next:
            return False
        while timesteps >= self.next:
            self.next += self.every
        return True


class MetricsCallback(BaseCallback):
    """Accumulate rollout statistics and write ``metrics.json`` every checkpoint.

    ``reward_mean`` is the mean reward of every environment step so far, not the
    mean of finished episodes: a 4 000-bar episode never ends inside a 1 024-step
    smoke run, so an episode-based mean would be missing exactly when the test
    needs it. ``ep_len_mean`` is the mean length of finished episodes, or the
    mean age of the episodes still running when none has finished yet.
    ``action_dist`` is the empirical action frequency, normalised to sum to 1.
    """

    def __init__(self, run_dir: str | Path, checkpoint_every: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self._trigger = _StepTrigger(checkpoint_every)
        self._reward_sum = 0.0
        self._reward_count = 0
        self._action_counts = np.zeros(N_ACTIONS, dtype=np.int64)
        self._episode_lengths: list[int] = []
        self._current_lengths = np.zeros(0, dtype=np.int64)
        self._metrics: dict[str, Any] = {}

    def _init_callback(self) -> None:
        # stable-baselines3 calls this again on every ``learn`` call; the run is
        # trained in chunks, so keeping the counters makes the metrics cumulative
        # over the whole run instead of restarting at every chunk.
        n_envs = self.training_env.num_envs
        if self._current_lengths.size != n_envs:
            self._current_lengths = np.zeros(n_envs, dtype=np.int64)

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            values = np.asarray(rewards, dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            self._reward_sum += float(values.sum())
            self._reward_count += int(values.size)

        actions = self.locals.get("actions")
        if actions is not None:
            arms = np.asarray(actions, dtype=np.int64).reshape(-1)
            arms = arms[(arms >= 0) & (arms < N_ACTIONS)]
            if arms.size:
                self._action_counts += np.bincount(arms, minlength=N_ACTIONS)[:N_ACTIONS]

        dones = self.locals.get("dones")
        if dones is not None:
            self._current_lengths += 1
            for index, done in enumerate(np.asarray(dones, dtype=bool).reshape(-1)):
                if done:
                    self._episode_lengths.append(int(self._current_lengths[index]))
                    self._current_lengths[index] = 0

        if self._trigger.due(self.num_timesteps):
            self.write()
        return True

    def payload(self) -> dict[str, Any]:
        """Build the metrics snapshot without writing it."""
        logger = self.model.logger.name_to_value if self.model.logger is not None else {}
        losses = [_finite(logger.get(key, 0.0)) for key in _LOSS_KEYS]

        total = int(self._action_counts.sum())
        if total:
            distribution = [float(count / total) for count in self._action_counts]
        else:
            distribution = [1.0 / N_ACTIONS] * N_ACTIONS

        if self._episode_lengths:
            ep_len_mean = float(np.mean(self._episode_lengths))
        elif self._current_lengths.size:
            ep_len_mean = float(self._current_lengths.mean())
        else:
            ep_len_mean = 0.0

        reward_mean = self._reward_sum / self._reward_count if self._reward_count else 0.0
        return {
            "step": int(self.num_timesteps),
            "reward_mean": _finite(reward_mean),
            "ep_len_mean": _finite(ep_len_mean),
            "entropy_loss": losses[0],
            "policy_loss": losses[1],
            "value_loss": losses[2],
            "action_dist": distribution,
        }

    def write(self) -> dict[str, Any]:
        """Write and return the current snapshot."""
        self._metrics = self.payload()
        write_json(self.run_dir / METRICS_NAME, self._metrics)
        return self._metrics

    @property
    def metrics(self) -> dict[str, Any]:
        """The last snapshot written, empty before the first checkpoint."""
        return dict(self._metrics)


class AtomicCheckpointCallback(BaseCallback):
    """Save ``model.zip`` every checkpoint without breaking the previous file."""

    def __init__(self, run_dir: str | Path, checkpoint_every: int, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self._trigger = _StepTrigger(checkpoint_every)

    def _on_step(self) -> bool:
        if self._trigger.due(self.num_timesteps):
            save_model(self.model, self.run_dir / MODEL_NAME)
        return True
