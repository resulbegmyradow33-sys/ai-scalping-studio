"""Deterministic synthetic M1 bars for offline development and tests.

The generator is a pure function of ``(seed, start, end)``: it never reads the
clock, and every random draw is keyed to a UTC calendar day, so the value of a
bar does not depend on how many bars were requested around it. The price level
is pinned by a slowly mean-reverting daily walk anchored at
:data:`ANCHOR_DATE`; each day's minute path is a Brownian bridge onto the next
day's open, which keeps the series continuous and every fetch of the same
timestamp identical.

Per day, minute log-returns are GBM increments whose volatility is scaled by a
three-state Markov regime (calm ``1x``, trend ``1.6x``, spike ``3x``). OHLC is
built from the previous close, the new close, and wicks proportional to the
bar's body, so ``low <= min(open, close) <= max(open, close) <= high`` holds by
construction. Spread is session-dependent and always positive; OHLC is the bid
series, so ``close`` is the bid quote and ``ask_close = close + spread``, the
same convention as :class:`~scalping.data.dukascopy.DukascopyProvider`.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from math import isfinite
from typing import Final

import numpy as np
import polars as pl

from scalping.data.base import BAR_SCHEMA, DataProvider

__all__ = ["PROVIDER_NAME", "SyntheticProvider"]

PROVIDER_NAME: Final[str] = "synthetic"
MINUTES_PER_DAY: Final[int] = 1440

#: Daily opens are anchored here so a bar's price does not depend on ``start``.
ANCHOR_DATE: Final[date] = date(2020, 1, 1)

_MINUTE_SIGMA: Final[float] = 0.00009
_DAY_SIGMA: Final[float] = 0.004
#: Pull of the daily walk back to ``start_price``; keeps a long horizon usable.
_DAY_REVERSION: Final[float] = 0.001
_REGIME_VOL: Final[tuple[float, ...]] = (1.0, 1.6, 3.0)
#: Rows are calm/trend/spike; the diagonal sets each regime's mean dwell time.
_REGIME_TRANSITION: Final[np.ndarray] = np.array(
    [
        [0.96, 0.03, 0.01],
        [0.04, 0.94, 0.02],
        [0.05, 0.05, 0.90],
    ]
)
_VOLUME_MEAN: Final[float] = 40.0
_MIN_WICK: Final[float] = 0.01
#: UTC hour -> base spread (price units): Asia 0.28, London 0.18, NY 0.22.
#: Hours 21:00-23:59 fall back to the Asian book.
_SESSION_SPREAD: Final[np.ndarray] = np.array([0.28] * 8 + [0.18] * 5 + [0.22] * 8 + [0.28] * 3)


def _as_utc(value: datetime) -> datetime:
    """Naive datetimes are assumed to already be UTC, matching ``conform_bars``."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _next_state_probabilities(state: int) -> np.ndarray:
    row = _REGIME_TRANSITION[state].copy()
    row[state] = 0.0
    return row / row.sum()


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=BAR_SCHEMA)


class SyntheticProvider(DataProvider):
    """Generate deterministic M1 bars for ``symbol`` between two instants.

    ``seed`` selects the random path, ``start_price`` is the price the anchored
    daily walk starts from in 2020, and ``vol_regimes=False`` disables the
    volatility multiplier so the series is a single calm GBM.

    ``fetch`` covers ``[start, end]`` inclusive, aligned to whole UTC minutes:
    a timestamp with seconds is rounded up to the next minute, and Saturdays
    and Sundays are dropped. Both bounds are required; the generator has no
    natural start or end to fall back on. Bars are continuous (each open equals the prior
    close) and quoted from the bid side, with ``ask_close = close + spread``.
    """

    name: str = PROVIDER_NAME

    def __init__(
        self, seed: int = 7, start_price: float = 2350.0, vol_regimes: bool = True
    ) -> None:
        if not isfinite(start_price) or start_price <= 0:
            raise ValueError("start_price must be a finite positive number")
        self.seed = int(seed)
        self.start_price = float(start_price)
        self.vol_regimes = bool(vol_regimes)

    def fetch(
        self,
        symbol: str,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> pl.DataFrame:
        if not symbol:
            raise ValueError("symbol must be non-empty")
        if start is None or end is None:
            raise ValueError("SyntheticProvider needs an explicit start and end")
        start, end = _as_utc(start), _as_utc(end)
        if end < start:
            return _empty()

        first, last = start.date(), end.date()
        trading, log_opens = self._daily_opens(first, last)

        frames: list[pl.DataFrame] = []
        for ordinal in range(first.toordinal(), last.toordinal() + 1):
            if (ordinal - 1) % 7 >= 5:  # 0001-01-01 was a Monday; 5/6 are Sat/Sun.
                continue
            low_minute, high_minute = self._minute_bounds(ordinal, start, end)
            if low_minute >= high_minute:
                continue
            index = int(np.searchsorted(trading, ordinal))
            frames.append(
                self._day_bars(
                    date.fromordinal(ordinal),
                    log_opens[index],
                    log_opens[index + 1],
                    low_minute,
                    high_minute,
                )
            )
        if not frames:
            return _empty()
        return pl.concat(frames).cast(BAR_SCHEMA)

    def _daily_opens(self, first: date, last: date) -> tuple[np.ndarray, np.ndarray]:
        """Return weekday ordinals and the log open of each, plus the next open."""
        walk_start = min(ANCHOR_DATE, first)
        ordinals = np.arange(walk_start.toordinal(), last.toordinal() + 4)
        trading = ordinals[(ordinals - 1) % 7 < 5]
        rng = np.random.default_rng([self.seed, 0])
        steps = rng.standard_normal(trading.size) * _DAY_SIGMA
        log_opens = np.empty(trading.size + 1)
        anchored = np.log(self.start_price)
        level = anchored
        log_opens[0] = level
        for index, step in enumerate(steps, start=1):
            level += -_DAY_REVERSION * (level - anchored) + step
            log_opens[index] = level
        return trading, log_opens

    @staticmethod
    def _minute_bounds(ordinal: int, start: datetime, end: datetime) -> tuple[int, int]:
        low = 0
        if ordinal == start.toordinal():
            low = start.hour * 60 + start.minute + (1 if start.second or start.microsecond else 0)
        high = MINUTES_PER_DAY
        if ordinal == end.toordinal():
            high = min(end.hour * 60 + end.minute + 1, MINUTES_PER_DAY)
        return low, high

    def _regime_multipliers(self, rng: np.random.Generator) -> np.ndarray:
        if not self.vol_regimes:
            return np.ones(MINUTES_PER_DAY)
        out = np.empty(MINUTES_PER_DAY)
        state = int(rng.integers(0, len(_REGIME_VOL)))
        position = 0
        while position < MINUTES_PER_DAY:
            stay = _REGIME_TRANSITION[state, state]
            stop = min(position + int(rng.geometric(1.0 - stay)), MINUTES_PER_DAY)
            out[position:stop] = _REGIME_VOL[state]
            position = stop
            state = int(rng.choice(len(_REGIME_VOL), p=_next_state_probabilities(state)))
        return out

    def _day_bars(
        self, day: date, log_open: float, log_next_open: float, low_minute: int, high_minute: int
    ) -> pl.DataFrame:
        ordinal = day.toordinal()
        noise_seed, regime_seed, volume_seed, spread_seed, wick_seed = np.random.SeedSequence(
            [self.seed, ordinal]
        ).spawn(5)
        noise_rng = np.random.default_rng(noise_seed)

        raw = (
            noise_rng.standard_normal(MINUTES_PER_DAY)
            * self._regime_multipliers(np.random.default_rng(regime_seed))
            * _MINUTE_SIGMA
        )
        path = np.cumsum(raw)
        # Brownian bridge: end the day exactly on the next anchored open.
        correction = (
            np.arange(1, MINUTES_PER_DAY + 1)
            / MINUTES_PER_DAY
            * (path[-1] - (log_next_open - log_open))
        )
        log_close = log_open + path - correction
        log_bar_open = np.concatenate(([log_open], log_close[:-1]))

        bar_open = np.exp(log_bar_open)
        close = np.exp(log_close)
        body = np.abs(close - bar_open)
        wick_rng = np.random.default_rng(wick_seed)
        upper = body * (0.25 + 0.75 * wick_rng.random(MINUTES_PER_DAY)) + _MIN_WICK
        lower = body * (0.25 + 0.75 * wick_rng.random(MINUTES_PER_DAY)) + _MIN_WICK
        body_high = np.maximum(bar_open, close)
        body_low = np.minimum(bar_open, close)
        high = body_high + upper
        low = body_low - lower

        hours = np.arange(MINUTES_PER_DAY) // 60
        spread = _SESSION_SPREAD[hours] * (
            0.9 + 0.2 * np.random.default_rng(spread_seed).random(MINUTES_PER_DAY)
        )
        bid_close = close
        ask_close = close + spread
        volume = (
            np.random.default_rng(volume_seed)
            .poisson(_VOLUME_MEAN, MINUTES_PER_DAY)
            .astype(np.int64)
        )

        minutes = np.arange(low_minute, high_minute, dtype="timedelta64[m]")
        ts = (np.datetime64(day.isoformat(), "m") + minutes).astype("datetime64[us]")
        window = slice(low_minute, high_minute)
        return pl.DataFrame(
            {
                "ts": pl.Series("ts", ts).dt.replace_time_zone("UTC"),
                "open": bar_open[window],
                "high": high[window],
                "low": low[window],
                "close": close[window],
                "bid_close": bid_close[window],
                "ask_close": ask_close[window],
                "spread": spread[window],
                "volume": volume[window],
            }
        ).with_columns(pl.lit(PROVIDER_NAME, dtype=pl.String).alias("provider"))
