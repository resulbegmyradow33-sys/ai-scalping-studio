"""Autoencoder regime research: contract, determinism and leakage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal

from scalping.regime import AutoencoderConfig, RegimeModelError, fit_autoencoder

START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(closes: object, *, wick: float = 0.05) -> pl.DataFrame:
    close = np.asarray(closes, dtype=float)
    open_ = np.concatenate([[close[0]], close[:-1]])
    return pl.DataFrame(
        {
            "ts": [START + timedelta(minutes=i) for i in range(close.size)],
            "open": open_,
            "high": np.maximum(open_, close) + wick,
            "low": np.minimum(open_, close) - wick,
            "close": close,
            "bid_close": close,
            "ask_close": close + 0.2,
            "spread": np.full(close.size, 0.2),
            "volume": np.full(close.size, 10, dtype=np.int64),
        }
    )


def three_regimes(n: int = 250) -> np.ndarray:
    up = 100.0 * np.exp(np.linspace(0.0, 0.08, n))
    side = up[-1] + 0.3 * np.sin(np.arange(n) / 7.0)
    down = side[-1] * np.exp(np.linspace(0.0, -0.08, n))
    return np.concatenate([up, side, down])


def config(**overrides) -> AutoencoderConfig:
    base = {
        "n_states": 3,
        "latent": 3,
        "hidden": 8,
        "epochs": 12,
        "learning_rate": 1e-2,
        "seed": 7,
    }
    return AutoencoderConfig(**{**base, **overrides})  # type: ignore[arg-type]


def test_contract_and_determinism() -> None:
    frame = bars(three_regimes())

    first = fit_autoencoder(frame, config())
    second = fit_autoencoder(frame, config())

    assert first.model == "autoencoder"
    assert first.rows == frame.height
    assert first.labels.height == frame.height
    assert len(first.states) == 3
    assert first.labels["state"].n_unique() >= 2
    assert set(first.labels["state"].to_list()).issubset(set(first.states))
    assert sum(first.counts().values()) == first.rows
    assert_frame_equal(first.labels, second.labels)


def test_training_never_reads_past_the_split() -> None:
    path = three_regimes()
    base = fit_autoencoder(bars(path), config())

    mutated = path.copy()
    mutated[600:] *= 1.5
    changed = fit_autoencoder(bars(mutated), config())

    assert base.train_rows == 525
    assert_frame_equal(base.labels.head(600), changed.labels.head(600))


def test_short_frame_and_bad_config_are_rejected() -> None:
    with pytest.raises(RegimeModelError, match="at least"):
        fit_autoencoder(bars(np.linspace(100.0, 101.0, 40)), config())

    with pytest.raises(ValueError, match="n_states"):
        AutoencoderConfig(n_states=1)
    with pytest.raises(ValueError, match="latent"):
        AutoencoderConfig(latent=0)
    with pytest.raises(ValueError, match="epochs"):
        AutoencoderConfig(epochs=0)
    with pytest.raises(ValueError, match="train_fraction"):
        AutoencoderConfig(train_fraction=0.4)
