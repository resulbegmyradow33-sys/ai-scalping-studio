"""Masked-action helpers for the discrete HOLD/BUY/SELL/CLOSE action space.

:meth:`~scalping.env.simulator.Simulator.action_masks` describes which actions
the accounting loop can execute on the current bar. Enforcing that mask inside
PPO needs a mask-aware policy: ``stable-baselines3``'s ``PPO`` has no mask hook,
and its categorical distribution keeps sampling masked actions. The supported
route is ``sb3-contrib``'s ``MaskablePPO`` (with ``MaskableEvalCallback`` for
evaluation), which reads ``env.action_masks()`` on the unwrapped env and masks
the logits before sampling. That dependency is deliberately not added in this
slice; training and the existing evaluation path are unchanged.

:func:`masked_action` is the interim evaluation rule: an action the mask forbids
becomes HOLD, so a rollout records the action the environment would actually
execute instead of an order it would reject. It is not a training fix — a policy
that proposes a masked action still wastes that step, and a masked action's
probability mass is not redistributed.
"""

from __future__ import annotations

from typing import Final

import numpy as np

from scalping.env.simulator import N_ACTIONS

__all__ = ["HOLD", "masked_action"]

#: Action index of HOLD; the shared order is 0 HOLD, 1 BUY, 2 SELL, 3 CLOSE.
HOLD: Final[int] = 0


def masked_action(action: int, mask: np.ndarray) -> int:
    """Return ``action`` when ``mask`` allows it, otherwise HOLD.

    ``mask`` is a boolean vector of length :data:`~scalping.env.simulator.N_ACTIONS`,
    normally the result of :meth:`Simulator.action_masks`. Every simulator mask
    allows HOLD, so a forbidden action degrades to waiting one bar rather than to
    a rejected order. A malformed mask or an action outside the space raises:
    silently mapping either to HOLD would hide a caller bug in an evaluation
    report.
    """
    array = np.asarray(mask)
    if array.dtype != np.bool_:
        raise TypeError(f"mask must be boolean, got {array.dtype}")
    if array.shape != (N_ACTIONS,):
        raise ValueError(f"mask must have shape ({N_ACTIONS},), got {array.shape}")
    if not array[HOLD]:
        raise ValueError("mask must allow HOLD: every simulator mask does")
    if isinstance(action, bool) or not isinstance(action, (int, np.integer)):
        raise TypeError(f"action must be an int, got {type(action).__name__}")
    index = int(action)
    if not 0 <= index < N_ACTIONS:
        raise ValueError(f"action must be in 0..{N_ACTIONS - 1}, got {index}")
    return index if bool(array[index]) else HOLD
