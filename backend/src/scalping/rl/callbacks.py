"""Callbacks that keep a run directory current while PPO learns.

``MetricsCallback`` writes ``metrics.json`` (latest snapshot) and appends
``metrics.jsonl`` (one line per checkpoint). ``CheckpointCallback`` saves
``model.zip`` and ``vecnormalize.pkl`` and then refreshes the manifest.
``ControlCallback`` implements pause/stop between steps and at rollout ends.

The callbacks fire on the same environment-step boundaries, in list order, so a
manifest written by the checkpoint callback always follows the metrics file for
the same step and the files it names.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from gymnasium import spaces
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecNormalize

from scalping.rl.registry import (
    METRICS_HISTORY_NAME,
    METRICS_NAME,
    MODEL_NAME,
    VECNORMALIZE_NAME,
    ManifestWriter,
    append_jsonl,
    save_model,
    save_vecnormalize,
    write_json,
)

__all__ = [
    "METRICS_HISTORY_NAME",
    "METRICS_NAME",
    "CheckpointCallback",
    "ControlCallback",
    "MetricsCallback",
    "TrainingHalted",
]

#: Logger keys copied from stable-baselines3, in metrics order.
_LOSS_KEYS: tuple[str, ...] = (
    "train/entropy_loss",
    "train/policy_gradient_loss",
    "train/value_loss",
)

#: Seconds between stop-event checks while a run is paused.
_PAUSE_POLL_SECONDS: Final[float] = 0.05


def _finite(value: Any) -> float | None:
    """``value`` as a finite float, or ``None`` when it is missing/non-finite."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


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
    """Accumulate rollout statistics and write the metrics files every checkpoint.

    ``reward_mean`` is the mean reward of every environment step so far, not the
    mean of finished episodes: a 4 000-bar episode never ends inside a 1 024-step
    smoke run, so an episode-based mean would be missing exactly when the test
    needs it. The reward the vectorised env reports is the one ``VecNormalize``
    returns (discounted-return normalised and clipped), which is the signal PPO
    actually learns from. ``reward_mean_interval`` covers only the steps since
    the previous checkpoint, which is what a reward curve should draw.
    ``ep_len_mean`` is the mean length of finished episodes, or the mean age of
    the episodes still running when none has finished yet. ``action_dist`` is the
    empirical action frequency, normalised to sum to 1.

    Losses are snapshotted at each rollout end because stable-baselines3 clears
    its logger on every dump and replaces the logger on every ``learn`` call. A
    checkpoint taken before the first update has no losses and writes ``null``;
    a zero would claim the loss was measured and happened to be zero.
    """

    def __init__(
        self,
        run_dir: str | Path,
        checkpoint_every: int,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self.status = "running"
        self._trigger = _StepTrigger(checkpoint_every)
        self._n_actions = 0
        self._action_counts = np.zeros(0, dtype=np.int64)
        self._reward_sum = 0.0
        self._reward_count = 0
        self._interval_reward_sum = 0.0
        self._interval_reward_count = 0
        self._last_interval_mean = 0.0
        self._episode_lengths: list[int] = []
        self._current_lengths = np.zeros(0, dtype=np.int64)
        self._losses: dict[str, float] = {}
        self._trade_count = 0
        self._last_history_step = -1
        self._metrics: dict[str, Any] = {}

    def _init_callback(self) -> None:
        # stable-baselines3 calls this again on every ``learn`` call; the run is
        # trained in chunks, so keeping the counters makes the metrics cumulative
        # over the whole run instead of restarting at every chunk.
        n_envs = self.training_env.num_envs
        if self._current_lengths.size != n_envs:
            self._current_lengths = np.zeros(n_envs, dtype=np.int64)
        action_space = self.training_env.action_space
        if not isinstance(action_space, spaces.Discrete):
            raise TypeError(f"expected a Discrete action space, got {action_space}")
        if self._n_actions != int(action_space.n):
            self._n_actions = int(action_space.n)
            self._action_counts = np.zeros(self._n_actions, dtype=np.int64)

    def reset_history(self) -> None:
        """Drop the previous invocation's snapshot and JSONL history."""
        (self.run_dir / METRICS_HISTORY_NAME).unlink(missing_ok=True)
        (self.run_dir / METRICS_NAME).unlink(missing_ok=True)
        self._last_history_step = -1

    def _on_step(self) -> bool:
        rewards = self.locals.get("rewards")
        if rewards is not None:
            values = np.asarray(rewards, dtype=np.float64).reshape(-1)
            values = values[np.isfinite(values)]
            total = float(values.sum())
            self._reward_sum += total
            self._interval_reward_sum += total
            self._reward_count += int(values.size)
            self._interval_reward_count += int(values.size)

        actions = self.locals.get("actions")
        if actions is not None and self._n_actions:
            arms = np.asarray(actions, dtype=np.int64).reshape(-1)
            arms = arms[(arms >= 0) & (arms < self._n_actions)]
            if arms.size:
                self._action_counts += np.bincount(arms, minlength=self._n_actions)

        dones = self.locals.get("dones")
        if dones is not None:
            self._current_lengths += 1
            for index, done in enumerate(np.asarray(dones, dtype=bool).reshape(-1)):
                if done:
                    self._episode_lengths.append(int(self._current_lengths[index]))
                    self._current_lengths[index] = 0

        infos = self.locals.get("infos")
        if infos:
            self._trade_count += sum(1 for info in infos if info.get("trade") is not None)

        if self._trigger.due(self.num_timesteps):
            self.write()
        return True

    def _on_rollout_end(self) -> None:
        snapshot = {
            key: value
            for key in _LOSS_KEYS
            if (value := _finite(self.model.logger.name_to_value.get(key))) is not None
        }
        if snapshot:
            self._losses.update(snapshot)

    def _loss(self, key: str) -> float | None:
        value = self._losses.get(key)
        if value is None:
            value = _finite(self.model.logger.name_to_value.get(key))
        return value

    def payload(self) -> dict[str, Any]:
        """Build the metrics snapshot without writing it."""
        if self._interval_reward_count:
            interval_mean = self._interval_reward_sum / self._interval_reward_count
        else:
            interval_mean = self._last_interval_mean

        total = int(self._action_counts.sum())
        if total:
            distribution: list[float] | None = [
                float(count / total) for count in self._action_counts
            ]
        else:
            # No action has been sampled yet; a uniform guess would claim a
            # measurement that does not exist.
            distribution = None

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
            "reward_mean_normalized": _finite(reward_mean),
            "reward_mean_interval": _finite(interval_mean),
            "ep_len_mean": _finite(ep_len_mean),
            "trades": int(self._trade_count),
            "entropy_loss": self._loss("train/entropy_loss"),
            "policy_loss": self._loss("train/policy_gradient_loss"),
            "value_loss": self._loss("train/value_loss"),
            "action_dist": distribution,
        }

    def write(self) -> dict[str, Any]:
        """Write the snapshot, append history if the step advanced, return it."""
        payload = self.payload()
        write_json(self.run_dir / METRICS_NAME, payload)
        if payload["step"] != self._last_history_step:
            append_jsonl(self.run_dir / METRICS_HISTORY_NAME, payload)
            self._last_history_step = int(payload["step"])
        self._metrics = payload
        if self._interval_reward_count:
            self._last_interval_mean = float(payload["reward_mean_interval"] or 0.0)
        self._interval_reward_sum = 0.0
        self._interval_reward_count = 0
        return payload

    @property
    def metrics(self) -> dict[str, Any]:
        """The last snapshot written, empty before the first checkpoint."""
        return dict(self._metrics)


class CheckpointCallback(BaseCallback):
    """Save the model and its normaliser, then refresh the manifest."""

    def __init__(
        self,
        run_dir: str | Path,
        checkpoint_every: int,
        vec_env: VecNormalize,
        metrics: MetricsCallback,
        manifest: ManifestWriter,
        extra: Callable[[], Mapping[str, Any]] | None = None,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.run_dir = Path(run_dir)
        self._trigger = _StepTrigger(checkpoint_every)
        self.vec_env = vec_env
        self.metrics_callback = metrics
        self.manifest = manifest
        self.extra = extra

    def _on_step(self) -> bool:
        if self._trigger.due(self.num_timesteps):
            self.save()
        return True

    def save(self) -> None:
        """Persist the model and normaliser, then publish the manifest."""
        save_model(self.model, self.run_dir / MODEL_NAME)
        save_vecnormalize(self.vec_env, self.run_dir / VECNORMALIZE_NAME)
        payload = self.metrics_callback.metrics
        reward_mean = payload.get("reward_mean")
        self.manifest.write(
            status=self.metrics_callback.status,
            steps=int(self.model.num_timesteps),
            reward_mean=float(reward_mean) if reward_mean is not None else None,
            extra=self.extra() if self.extra is not None else None,
        )


class TrainingHalted(RuntimeError):
    """Raised inside ``learn`` when a stop request must end the run now."""

    def __init__(self, status: str = "interrupted") -> None:
        self.status = status
        super().__init__(f"training {status}")


class ControlCallback(BaseCallback):
    """Turn ``stop_event``/``pause_event`` into a pause or a clean halt.

    Pause blocks the training thread and keeps the run loadable: the first pause
    writes a checkpoint and a ``paused`` manifest, a resume writes ``running``
    and continues the same model, and a stop while paused ends as ``interrupted``.
    """

    def __init__(
        self,
        stop_event: threading.Event | None,
        pause_event: threading.Event | None,
        checkpoint: Callable[[str], None],
        poll_seconds: float = _PAUSE_POLL_SECONDS,
    ) -> None:
        super().__init__(verbose=0)
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.checkpoint = checkpoint
        self.poll_seconds = poll_seconds

    def _requested(self, event: threading.Event | None) -> bool:
        return event is not None and event.is_set()

    def _on_step(self) -> bool:
        self.react()
        return True

    def _on_rollout_end(self) -> None:
        self.react()

    def react(self) -> None:
        """Halt on stop, block while paused, otherwise return immediately."""
        if self._requested(self.stop_event):
            raise TrainingHalted("interrupted")
        if self._requested(self.pause_event):
            self.pause_here()

    def pause_here(self) -> None:
        self.checkpoint("paused")
        while self._requested(self.pause_event):
            if self._requested(self.stop_event):
                raise TrainingHalted("interrupted")
            time.sleep(self.poll_seconds)
        self.checkpoint("running")
