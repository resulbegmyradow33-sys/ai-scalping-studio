"""The validated v1 strategy: momentum gated by extreme tick-volume flow.

The rule is deliberately small and explicit, because it is the artifact that
survived five years of walk-forward testing while the PPO policy did not:

* take a ``momentum`` signal (EMA/displacement family) in its direction,
* only when ``|flow|`` exceeds the causal trailing quantile,
* exits come from the environment's ATR stop/target and the bar timeout.

``FlowMomentumRule.build`` turns a stored frame plus the flow side table into
one action per bar (0 hold, 1 buy, 2 sell). The environment still owns position
management, so an entry that arrives while a position is open is rejected with
``position_open``; that is exactly the trade population the labels priced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import polars as pl

__all__ = ["FlowMomentumRule"]

_HOLD = 0
_BUY = 1
_SELL = 2


@dataclass
class FlowMomentumRule:
    """One action per bar, built once and replayed in order."""

    actions: np.ndarray
    signal: np.ndarray | None = None
    flow: np.ndarray | None = None
    threshold: np.ndarray | None = None
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        actions = np.asarray(self.actions, dtype=np.int8)
        if actions.ndim != 1:
            raise ValueError("actions must be one-dimensional")
        self.actions = actions
        for name in ("signal", "flow", "threshold"):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float64)
                if array.shape != actions.shape:
                    raise ValueError(f"{name} must have shape {actions.shape}, got {array.shape}")
                setattr(self, name, array)

    def reset(self, index: int = 0) -> None:
        """Point the cursor at ``index``; explicit ``act(index)`` ignores it."""
        self._index = max(0, int(index))

    def act(self, index: int | None = None) -> int:
        if index is None:
            index = self._index
            self._index += 1
        if index < 0 or index >= len(self.actions):
            return _HOLD
        return int(self.actions[index])

    __call__ = act

    def blocked_reason(self, index: int) -> str | None:
        """Why this bar produced no entry, for the panel's decision message.

        ``None`` means "nothing to explain": either the bar had no momentum
        signal at all, or the action was an entry. ``no_flow`` is the rule's
        own gate, and it is the only reason the rule ever suppresses a signal.
        """
        if self.signal is None or self.flow is None or self.threshold is None:
            return None
        if index < 0 or index >= len(self.actions):
            return None
        if self.signal[index] == 0:
            return None
        extreme = (
            np.isfinite(self.flow[index])
            and np.isfinite(self.threshold[index])
            and abs(float(self.flow[index])) > float(self.threshold[index])
        )
        return None if extreme else "no_flow"

    @classmethod
    def build(
        cls,
        frame: pl.DataFrame,
        flow: np.ndarray,
        threshold: np.ndarray,
        signal: np.ndarray,
    ) -> FlowMomentumRule:
        height = frame.height
        for name, array in (("flow", flow), ("threshold", threshold), ("signal", signal)):
            if array.shape != (height,):
                raise ValueError(f"{name} must have shape ({height},), got {array.shape}")
        extreme = np.isfinite(flow) & np.isfinite(threshold) & (np.abs(flow) > threshold)
        actions = np.zeros(height, dtype=np.int8)
        actions[extreme & (signal > 0)] = _BUY
        actions[extreme & (signal < 0)] = _SELL
        return cls(
            actions=actions,
            signal=signal,
            flow=flow,
            threshold=threshold,
        )
