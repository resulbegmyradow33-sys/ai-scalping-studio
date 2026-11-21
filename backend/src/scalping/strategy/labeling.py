"""Offline counterfactual labels and the bandit's market context.

Labeling
--------
Every bar on which a family fires gets **one** hypothetical trade for that
family. The trade opens at the **signal bar's** closing quote — the ask for a
long, the bid for a short — with the broker's ATR slippage, and it exits at the
first stop, target or ``max_bars_in_trade`` bars. Entry timing is the
environment's, not the design document's earlier "next bar open":
:class:`~scalping.env.simulator.Simulator` acts on the decision bar's close, and
the bar schema carries no ask open, so a label that entered one bar later would
price a trade the environment never takes. The exit, PnL, costs and
``r_multiple`` all come from the same
:class:`~scalping.env.broker.PaperBroker` the simulator uses, so a label's R is
the R that fill sequence produces.

A label whose hypothetical trade needs a bar past the end of the frame is
dropped, not closed early: an unresolved trade has no honest R. Every label
gets its own broker and the same starting equity, so no label can inherit state
from the one before it.

Timing detail: the entry bar's own high/low are never tested against the stop
(they printed before the quote the entry was filled at). Exits are evaluated
from the next bar onward, and ``bars`` counts those evaluation bars, so
``exit_index == signal_index + bars``.

Quotes
------
A broker CSV without ask quotes leaves ``ask_close`` and ``spread`` null. The
frame is run through :func:`~scalping.features.prepare.prepare_bars`, the same
spread model :func:`~scalping.features.build.build_features` uses, before any
bar is built, so OHLC-only sources produce labels instead of a ``TypeError``.
A frame missing a column outright still raises ``ValueError``.

Context
-------
:func:`context_matrix` is the bandit's feature vector. It is built only from
causal series (the feature builder's matrix, the realised labels' outcomes
observed at or before the row) and contains no frame-wide statistic, so a row
computed from a truncated frame is bit-identical to the same row computed from
the full frame.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.data.base import conform_bars
from scalping.env.broker import PaperBroker
from scalping.env.risk import RiskEngine
from scalping.features.build import FEATURE_COLUMNS, FeatureFrame
from scalping.features.indicators import atr
from scalping.features.prepare import prepare_bars
from scalping.strategy.signals import ALL_STRATEGIES, Strategy, strategy_signals
from scalping.types import Bar, Side

__all__ = [
    "CONTEXT_COLUMNS",
    "CONTEXT_DIM",
    "LABEL_COLUMNS",
    "context_matrix",
    "label_events",
    "strength_matrix",
]

#: Column order of :func:`label_events`. ``ts`` is the signal (decision) bar.
LABEL_COLUMNS: tuple[str, ...] = (
    "ts",
    "strategy",
    "side",
    "entry",
    "exit",
    "bars",
    "r_multiple",
    "reason",
)

_LABEL_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "ts": pl.Datetime("us", "UTC"),
    "strategy": pl.String,
    "side": pl.Int8,
    "entry": pl.Float64,
    "exit": pl.Float64,
    "bars": pl.Int32,
    "r_multiple": pl.Float64,
    "reason": pl.String,
}

#: ATR window used for the entry slippage and the 1x/2x ATR stop and target.
_ATR_WINDOW: int = 14

#: The three volatility-regime states, in ascending order.
_REGIME_STATES: tuple[float, ...] = (-1.0, 0.0, 1.0)

#: One-hot session flags, matching ``features.build``.
_SESSION_COLUMNS: tuple[str, ...] = ("asia", "london", "ny")

#: Feature columns copied straight into the context vector.
_ATR_PCT_COLUMN: str = "atr_norm"
_RSI_COLUMN: str = "rsi_14"
_TREND_COLUMN: str = "trend"
_SPREAD_ATR_COLUMN: str = "spread_norm"

#: Trailing bars over which a family's realised success is averaged.
_SUCCESS_WINDOW: int = 240

#: Success shown when no exit of that family resolved in the window.
_SUCCESS_NEUTRAL: float = 0.5

#: Trailing bars used to standardise ``atr_pct``.
_ATR_Z_WINDOW: int = 240

#: Bandit context width: regime(3) + session(3) + 4 scalars + success(5).
CONTEXT_COLUMNS: tuple[str, ...] = (
    *(f"regime_{int(state)}" for state in _REGIME_STATES),
    *_SESSION_COLUMNS,
    "atr_pct",
    "rsi",
    "trend",
    "spread_atr",
    *(f"success_{strategy.value}" for strategy in ALL_STRATEGIES),
)

CONTEXT_DIM: int = len(CONTEXT_COLUMNS)


@dataclass(frozen=True, slots=True)
class _Label:
    """One resolved hypothetical trade plus the indices used to place it."""

    index: int
    ts: datetime
    strategy: Strategy
    side: Side
    entry: float
    exit: float
    bars: int
    r_multiple: float
    reason: str
    exit_index: int


def _empty_labels() -> pl.DataFrame:
    return pl.DataFrame(schema=_LABEL_SCHEMA)


def _bar(row: int, frame: pl.DataFrame) -> Bar:
    """Build the broker's bar for row ``row`` from a conformed frame."""
    return Bar(
        ts=frame["ts"][row],
        open=float(frame["open"][row]),
        high=float(frame["high"][row]),
        low=float(frame["low"][row]),
        close=float(frame["close"][row]),
        bid_close=float(frame["bid_close"][row]),
        ask_close=float(frame["ask_close"][row]),
        spread=float(frame["spread"][row]),
        volume=int(frame["volume"][row]),
    )


def _labels(df: pl.DataFrame, settings: Settings) -> list[_Label]:
    """Resolve one hypothetical trade per signal, in ``(ts, strategy)`` order."""
    frame = conform_bars(df).sort("ts")
    if frame.is_empty():
        return []

    atr14 = atr(frame, _ATR_WINDOW).to_numpy()
    # An OHLC-only source (null ask/spread) gets the feature layer's spread
    # model before any bar is built, so a null quote cannot reach float().
    frame = prepare_bars(frame, settings.costs, atr14)

    open_ = frame["open"].to_numpy()
    high = frame["high"].to_numpy()
    low = frame["low"].to_numpy()
    close = frame["close"].to_numpy()
    bid = frame["bid_close"].to_numpy()
    ask = frame["ask_close"].to_numpy()
    spread = frame["spread"].to_numpy()
    volume = frame["volume"].to_numpy()
    usable = (
        np.isfinite(open_)
        & np.isfinite(high)
        & np.isfinite(low)
        & np.isfinite(close)
        & np.isfinite(bid)
        & np.isfinite(ask)
        & np.isfinite(spread)
        & np.isfinite(volume)
    )
    stamps = frame["ts"].to_list()
    bars = [_bar(index, frame) for index in range(frame.height)]

    signals = strategy_signals(frame)
    signal_columns = [signals[f"sig_{strategy.value}"].to_numpy() for strategy in ALL_STRATEGIES]

    risk = RiskEngine(settings.risk, settings.contract)
    equity = float(settings.contract.initial_balance)
    costs = settings.costs

    out: list[_Label] = []
    for index in range(frame.height):
        if not usable[index]:
            continue
        atr_value = float(atr14[index])
        if not isfinite(atr_value) or atr_value <= 0.0:
            continue
        if index + 1 >= frame.height:
            # No bar left to price the exit after the entry bar.
            continue
        for arm, strategy in enumerate(ALL_STRATEGIES):
            signal = int(signal_columns[arm][index])
            if signal == 0:
                continue
            side = Side.LONG if signal > 0 else Side.SHORT
            quote = ask[index] if side is Side.LONG else bid[index]
            slippage = costs.slippage_atr_frac * atr_value
            entry = quote + int(side) * slippage
            if not isfinite(entry) or entry <= 0.0:
                continue
            sl, tp = risk.sl_tp(entry, side, atr_value)
            sizing = risk.size(equity, entry, sl)
            if sizing.lot <= 0.0:
                continue

            broker = PaperBroker(settings.contract, costs, atr_value)
            broker.open(
                side,
                sizing.lot,
                stamps[index],
                bid[index],
                ask[index],
                sl,
                tp,
                strategy.value,
            )
            trade = None
            exit_index = -1
            for current in range(index + 1, frame.height):
                if not usable[current]:
                    # A missing bar breaks the path; the label stays unresolved.
                    break
                trade = broker.update(bars[current])
                if trade is not None:
                    exit_index = current
                    break
                if risk.must_force_close(current - index):
                    trade = broker.close(bars[current], "timeout")
                    exit_index = current
                    break
            if trade is None or exit_index < 0:
                # Ran out of bars (or into an unusable one): no honest R.
                continue
            out.append(
                _Label(
                    index=index,
                    ts=stamps[index],
                    strategy=strategy,
                    side=side,
                    entry=trade.entry,
                    exit=trade.exit,
                    bars=exit_index - index,
                    r_multiple=trade.r_multiple,
                    reason=trade.reason,
                    exit_index=exit_index,
                )
            )
    return out


def label_events(df: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    """Label every strategy signal with the R of its hypothetical trade.

    ``df`` is a frame of M1 bars (``ts`` plus OHLC and the ``bid_close`` /
    ``ask_close`` quote, as in :data:`~scalping.data.base.BAR_SCHEMA`). One row
    per ``(signal bar, family)`` is returned, sorted by ``ts`` then by the
    frozen :data:`~scalping.strategy.signals.ALL_STRATEGIES` order, with:

    ``ts``
        The signal (decision) bar.
    ``strategy``
        The family name.
    ``side``
        ``1`` for a long, ``-1`` for a short.
    ``entry`` / ``exit``
        Broker fill prices: the signal bar's ask (long) or bid (short) plus ATR
        slippage on entry, then the trigger level, gap open or liquidating
        quote on exit.
    ``bars``
        Bars between the entry bar (the signal bar) and the exit bar; at most
        ``settings.risk.max_bars_in_trade``.
    ``r_multiple``
        ``pnl / (lot * contract.size * |entry - sl|)``, straight from the
        broker.
    ``reason``
        ``"sl"``, ``"tp"`` or ``"timeout"``.

    Trades that would still be open at the last bar are dropped, so no label is
    ever priced against a bar that does not exist. A frame with no signals (or
    no usable ones) returns an empty frame with this schema, never ``None``.
    """
    labels = _labels(df, settings)
    if not labels:
        return _empty_labels()
    return pl.DataFrame(
        {
            "ts": [label.ts for label in labels],
            "strategy": [label.strategy.value for label in labels],
            "side": np.array([int(label.side) for label in labels], dtype=np.int8),
            "entry": np.array([label.entry for label in labels], dtype=np.float64),
            "exit": np.array([label.exit for label in labels], dtype=np.float64),
            "bars": np.array([label.bars for label in labels], dtype=np.int32),
            "r_multiple": np.array([label.r_multiple for label in labels], dtype=np.float64),
            "reason": [label.reason for label in labels],
        },
        schema=_LABEL_SCHEMA,
    )


def _feature_index(features: FeatureFrame) -> dict[str, int]:
    columns = features.columns or list(FEATURE_COLUMNS)
    return {name: position for position, name in enumerate(columns)}


def _feature_column(
    features: FeatureFrame, index: dict[str, int], name: str, fill: float
) -> np.ndarray:
    """One feature column as float64, with non-finite warm-up rows at ``fill``."""
    values = features.X[:, index[name]].astype(np.float64)
    return np.where(np.isfinite(values), values, fill)


def _bars_frame(features: FeatureFrame) -> pl.DataFrame:
    """Rebuild the broker's bars from a feature frame.

    ``FeatureFrame`` exposes what the broker needs: the bid OHLC, the quoted
    spread and the timestamps. Volume was not kept, so it is zero; the broker
    never reads it.
    """
    close = features.ohlc[:, 3]
    spread = np.where(
        np.isfinite(features.spread) & (features.spread >= 0.0), features.spread, 0.0
    )
    return pl.DataFrame(
        {
            "ts": pl.Series("ts", features.ts).cast(pl.Datetime("us")),
            "open": features.ohlc[:, 0],
            "high": features.ohlc[:, 1],
            "low": features.ohlc[:, 2],
            "close": close,
            "bid_close": close,
            "ask_close": close + spread,
            "spread": spread,
            "volume": np.zeros(close.size, dtype=np.int64),
        }
    )


def _success_matrix(
    outcomes: list[tuple[Strategy, int, float]], height: int
) -> np.ndarray:
    """Causal 240-bar success rate per family, neutral when nothing resolved.

    ``success[t, arm]`` is the fraction of ``arm``'s labels that exited with a
    positive R in bars ``t - 239 .. t``. A label enters the series at its exit
    bar, the first bar on which its outcome is observable, so row ``t`` never
    uses a trade that is still open. A window with no exit of that family reads
    :data:`_SUCCESS_NEUTRAL` (``0.5``): "no evidence" is not "all failures",
    which would teach the bandit to avoid an arm it has simply not seen.
    """
    success = np.zeros((height, len(ALL_STRATEGIES)), dtype=np.float64)
    for arm, strategy in enumerate(ALL_STRATEGIES):
        counts = np.zeros(height + 1, dtype=np.int64)
        wins = np.zeros(height + 1, dtype=np.int64)
        for label_strategy, exit_index, r_multiple in outcomes:
            if label_strategy is not strategy:
                continue
            if not 0 <= exit_index < height:  # pragma: no cover - filtered upstream
                continue
            counts[exit_index + 1] += 1
            wins[exit_index + 1] += int(r_multiple > 0.0)
        # prefix[m] = number of resolved labels with exit_index < m.
        prefix_counts = np.cumsum(counts)
        prefix_wins = np.cumsum(wins)
        rows = np.arange(height)
        starts = np.maximum(0, rows - _SUCCESS_WINDOW + 1)
        resolved = prefix_counts[rows + 1] - prefix_counts[starts]
        won = prefix_wins[rows + 1] - prefix_wins[starts]
        rates = np.full(height, _SUCCESS_NEUTRAL, dtype=np.float64)
        np.divide(won, resolved, out=rates, where=resolved > 0)
        success[:, arm] = rates
    return success


def _utc(stamp: datetime) -> datetime:
    """Timezone-aware UTC stamp; a naive value is read as UTC like ``conform_bars``."""
    return stamp.replace(tzinfo=UTC) if stamp.tzinfo is None else stamp.astimezone(UTC)


def _feature_stamps(features: FeatureFrame) -> list[datetime]:
    return [
        np.datetime64(value, "us").astype(datetime).replace(tzinfo=UTC)
        for value in features.ts
    ]


def _outcomes_from_table(
    table: pl.DataFrame, features: FeatureFrame
) -> list[tuple[Strategy, int, float]]:
    """Turn a :func:`label_events` frame into ``(strategy, exit index, R)``.

    Labels for a timestamp outside ``features`` are ignored (a caller may pass
    the labels of a wider frame), and so are labels whose exit bar lies past the
    frame: neither can be observed inside it.
    """
    required = {"ts", "strategy", "bars", "r_multiple"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise ValueError(f"labels missing columns: {missing}")
    height = int(features.X.shape[0])
    positions = {_utc(stamp): index for index, stamp in enumerate(_feature_stamps(features))}
    outcomes: list[tuple[Strategy, int, float]] = []
    for row in table.iter_rows(named=True):
        stamp = row["ts"]
        if stamp is None:
            continue
        index = positions.get(_utc(stamp))
        if index is None:
            continue
        exit_index = index + int(row["bars"])
        if exit_index >= height:
            continue
        outcomes.append((Strategy(row["strategy"]), exit_index, float(row["r_multiple"])))
    return outcomes


def _atr_z_score(features: FeatureFrame, index: dict[str, int]) -> np.ndarray:
    """ATR/close standardised against the previous 240 bars.

    The mean and standard deviation are shifted one bar, so the current bar does
    not contribute to the window it is compared against (the same convention the
    volatility-regime feature uses). A zero-variance or too-short history reads
    ``0.0``.
    """
    values = pl.Series(
        "atr_pct", features.X[:, index[_ATR_PCT_COLUMN]].astype(np.float64)
    ).fill_nan(None)
    work = pl.DataFrame({"value": values}).with_columns(
        pl.col("value")
        .rolling_mean(window_size=_ATR_Z_WINDOW, min_samples=1)
        .shift(1)
        .alias("mean"),
        pl.col("value")
        .rolling_std(window_size=_ATR_Z_WINDOW, min_samples=2)
        .shift(1)
        .alias("std"),
    )
    return (
        work.select(
            pl.when(pl.col("std") > 0.0)
            .then((pl.col("value") - pl.col("mean")) / pl.col("std"))
            .otherwise(0.0)
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias("z")
        )["z"]
        .to_numpy()
    )


def strength_matrix(features: FeatureFrame) -> np.ndarray:
    """The arms' ``(T, len(ALL_STRATEGIES))`` signal strength, one row per feature row.

    Column ``k`` is ``strength_<ALL_STRATEGIES[k]>`` from
    :func:`~scalping.strategy.signals.strategy_signals`, so it follows the same
    frozen arm order the bandit indexes. The environment appends the chosen
    arm's value to its observation; ``0.0`` is the strength of an arm that did
    not fire, matching the signal frame's own fill.

    Every input series is causal, so row ``t`` depends on bars at or before
    ``t`` only. The rows are checked against the feature frame's ``ts`` before
    they are returned: an unsorted frame would otherwise pair a row's strength
    with another bar's features while still looking causal. An empty frame
    returns a ``(0, 5)`` array.
    """
    height = int(features.X.shape[0])
    if height == 0:
        return np.zeros((0, len(ALL_STRATEGIES)), dtype=np.float32)
    signals = strategy_signals(_bars_frame(features))
    stamps = signals["ts"].to_numpy()
    if stamps.shape != (height,) or not np.array_equal(
        stamps, features.ts.astype("datetime64[us]")
    ):
        raise ValueError("strength rows are not aligned with the feature frame")
    matrix = np.column_stack(
        [
            signals[f"strength_{strategy.value}"].to_numpy().astype(np.float32, copy=True)
            for strategy in ALL_STRATEGIES
        ]
    )
    if matrix.shape != (height, len(ALL_STRATEGIES)):  # pragma: no cover - guard only
        raise RuntimeError(f"strength matrix shape drifted: {matrix.shape}")
    if not np.isfinite(matrix).all():  # pragma: no cover - signals fill this
        raise ValueError("strength matrix contains a non-finite value")
    return matrix


def context_matrix(features: FeatureFrame, labels: pl.DataFrame | None = None) -> np.ndarray:
    """The bandit's ``(T, CONTEXT_DIM)`` context, one row per feature row.

    Column order is :data:`CONTEXT_COLUMNS`: a one-hot of the volatility regime
    (down/flat/up), a one-hot of the session (Asia/London/New York; the hours
    after 21:00 UTC set none), ATR/close as a causal 240-bar z-score
    (the current bar is not part of the window; zero variance reads ``0.0``),
    RSI/100, the trend state, spread/ATR, and per family the share of its labels
    that finished positive among the exits of the trailing 240 bars.

    ``labels`` is an optional :func:`label_events` frame. Passing it avoids
    recomputing every hypothetical trade here, which matters once the caller
    already labeled the same frame. When it is ``None`` the labels are rebuilt
    from the feature bars with default :class:`~scalping.config.Settings`
    (10 000 initial balance, 0.5% risk, 120-bar limit); a ``FeatureFrame``
    carries no account, so the success indicator has no caller settings to use.
    Pass ``labels=label_events(df, settings)`` to make the indicator match a
    specific account. Rows with no exit of that family in the window read the
    neutral ``0.5``, never ``0.0``.

    Everything is causal. The scalar columns are copied from the feature
    builder's backward-looking matrix, and the success columns only consume a
    label once its exit bar has printed. Warm-up rows carry neutral values
    instead of NaN: ``0`` for the continuous columns and ``0.5`` for RSI. An
    empty frame returns a ``(0, CONTEXT_DIM)`` array.
    """
    height = int(features.X.shape[0])
    if height == 0:
        return np.zeros((0, CONTEXT_DIM), dtype=np.float64)

    index = _feature_index(features)
    regime = np.rint(_feature_column(features, index, "vol_regime", 0.0)).astype(np.int64)
    regime = np.clip(regime, -1, 1)
    scalars = np.column_stack(
        [
            _atr_z_score(features, index),
            np.clip(_feature_column(features, index, _RSI_COLUMN, 0.5), 0.0, 1.0),
            np.clip(_feature_column(features, index, _TREND_COLUMN, 0.0), -1.0, 1.0),
            _feature_column(features, index, _SPREAD_ATR_COLUMN, 0.0),
        ]
    )
    sessions = np.column_stack(
        [_feature_column(features, index, name, 0.0) for name in _SESSION_COLUMNS]
    )

    if labels is None:
        outcomes = [
            (label.strategy, label.exit_index, label.r_multiple)
            for label in _labels(_bars_frame(features), Settings())
        ]
    else:
        outcomes = _outcomes_from_table(labels, features)
    success = _success_matrix(outcomes, height)

    matrix = np.column_stack(
        [
            *((regime == int(state)).astype(np.float64) for state in _REGIME_STATES),
            sessions,
            scalars,
            success,
        ]
    )
    if matrix.shape != (height, CONTEXT_DIM):  # pragma: no cover - guard only
        raise RuntimeError(f"context matrix width drifted: {matrix.shape}")
    if not np.isfinite(matrix).all():  # pragma: no cover - guarded by construction
        raise ValueError("context matrix contains a non-finite value")
    return matrix
