"""Deterministic order rejections: price gap and zero-volume bars."""

from __future__ import annotations

import numpy as np
import pytest

from scalping.backtest.engine import data_hash, run_id
from scalping.config import Settings
from scalping.env.rejections import RejectionConfig
from scalping.env.simulator import Simulator
from scalping.features.build import FeatureFrame

BUY, HOLD = 1, 0


def frame(
    n: int = 300,
    *,
    jump_at: int | None = None,
    jump: float = 0.0,
    zero_volume_at: int | None = None,
) -> FeatureFrame:
    ts = np.datetime64("2026-01-05T00:00", "us") + np.arange(n).astype("timedelta64[m]")
    price = np.full(n, 1000.0)
    if jump_at is not None:
        price[jump_at:] += jump
    ohlc = np.column_stack([price, price + 0.001, price - 0.001, price])
    volume = np.full(n, 10, dtype=np.int64)
    if zero_volume_at is not None:
        volume[zero_volume_at] = 0
    return FeatureFrame(
        X=np.zeros((n, 3), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(n, 0.20),
        atr=np.full(n, 5.0),
        valid_from=60,
        columns=["a", "b", "c"],
    )


def sim(frame_: FeatureFrame, *, delay: int = 1, rejections=None) -> Simulator:
    instance = Simulator(
        frame_, Settings(), execution_delay=delay, rejections=rejections
    )
    instance.reset(seed=1)
    return instance


def test_price_gap_after_delay_rejects_the_entry() -> None:
    instance = sim(frame(jump_at=120, jump=10.0), rejections=RejectionConfig(max_gap_atr=0.5))
    instance.step(BUY)

    result = instance.step(HOLD)

    assert result.info["blocked"] == "rejected_price_gap"
    assert result.info["position"] is None
    assert instance.pending_action is None


def test_small_gap_still_fills() -> None:
    instance = sim(frame(jump_at=120, jump=1.0), rejections=RejectionConfig(max_gap_atr=0.5))
    instance.step(BUY)

    result = instance.step(HOLD)

    assert result.info["blocked"] is None
    assert result.info["opened"] is not None


def test_zero_volume_rejects_an_immediate_entry() -> None:
    instance = sim(
        frame(zero_volume_at=119),
        delay=0,
        rejections=RejectionConfig(reject_zero_volume=True),
    )

    result = instance.step(BUY)

    assert result.info["blocked"] == "rejected_no_liquidity"
    assert result.info["position"] is None


def test_zero_volume_rejects_a_delayed_fill() -> None:
    instance = sim(frame(zero_volume_at=120), rejections=RejectionConfig(reject_zero_volume=True))
    instance.step(BUY)

    result = instance.step(HOLD)

    assert result.info["blocked"] == "rejected_no_liquidity"
    assert result.info["position"] is None
    assert instance.pending_action is None


def test_rejections_are_part_of_the_run_identity() -> None:
    digest = data_hash(frame())

    assert run_id(Settings(), 0, digest) != run_id(
        Settings(), 0, digest, rejections=RejectionConfig(max_gap_atr=0.5)
    )


def test_config_guard_rails() -> None:
    with pytest.raises(ValueError, match="at least one rule"):
        RejectionConfig()
    with pytest.raises(ValueError, match="max_gap_atr"):
        RejectionConfig(max_gap_atr=-1.0)
    with pytest.raises(TypeError, match="reject_zero_volume"):
        RejectionConfig(reject_zero_volume=1)  # type: ignore[arg-type]
    assert RejectionConfig(max_gap_atr=1.0).to_dict() == {
        "max_gap_atr": 1.0,
        "reject_zero_volume": False,
    }
