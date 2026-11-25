"""Strategy performance by regime.

The report answers one question: *inside this regime, how did this strategy
actually trade?* It never predicts the regime and never re-prices a trade. The
regime is attached from the label that was current at the trade's entry bar, and
the trade statistics come from the simulator's closed ``Trade`` rows, costs
included.

A trade whose entry precedes the first label is reported under ``unknown``
instead of being dropped: a table that silently loses trades is worse than one
that admits missing coverage.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

import polars as pl
from polars.datatypes import DataType, DataTypeClass

from scalping.types import Trade

__all__ = ["UNKNOWN_REGIME", "regime_metrics", "strategy_regime_table", "trades_with_regime"]

UNKNOWN_REGIME: Final[str] = "unknown"

_TABLE_SCHEMA: Final[dict[str, DataType | DataTypeClass]] = {
    "strategy": pl.String,
    "regime": pl.String,
    "trades": pl.Int64,
    "wins": pl.Int64,
    "win_rate": pl.Float64,
    "expectancy_r": pl.Float64,
    "total_r": pl.Float64,
    "total_pnl": pl.Float64,
    "profit_factor": pl.Float64,
}

_TRADE_SCHEMA: Final[dict[str, DataType | DataTypeClass]] = {
    "entry_ts": pl.Datetime("us", "UTC"),
    "exit_ts": pl.Datetime("us", "UTC"),
    "strategy": pl.String,
    "side": pl.String,
    "r_multiple": pl.Float64,
    "pnl": pl.Float64,
    "regime": pl.String,
}


def trades_with_regime(
    trades: Sequence[Trade],
    labels: pl.DataFrame,
    *,
    regime_column: str = "regime",
) -> pl.DataFrame:
    """Attach the regime that was current at each trade's entry.

    ``labels`` needs ``ts`` and ``regime_column``. The join is backward as-of on
    the entry timestamp, so a label that starts after the entry can never be
    attached, and an entry exactly on a label timestamp uses that label.
    """
    missing = [name for name in ("ts", regime_column) if name not in labels.columns]
    if missing:
        raise ValueError(f"regime labels are missing columns: {missing}")
    if labels["ts"].n_unique() != labels.height:
        raise ValueError("regime labels contain duplicate timestamps")

    if not trades:
        return pl.DataFrame(schema=_TRADE_SCHEMA)

    rows = pl.DataFrame(
        {
            "entry_ts": [trade.entry_ts for trade in trades],
            "exit_ts": [trade.exit_ts for trade in trades],
            "strategy": [trade.strategy for trade in trades],
            "side": [trade.side.name for trade in trades],
            "r_multiple": [float(trade.r_multiple) for trade in trades],
            "pnl": [float(trade.pnl) for trade in trades],
        }
    ).sort("entry_ts")

    regime_frame = labels.select(
        pl.col("ts").alias("_regime_ts"),
        pl.col(regime_column).alias("regime"),
    ).sort("_regime_ts")

    joined = rows.join_asof(
        regime_frame,
        left_on="entry_ts",
        right_on="_regime_ts",
        strategy="backward",
    )
    return joined.drop("_regime_ts").with_columns(
        pl.col("regime").fill_null(UNKNOWN_REGIME).cast(pl.String)
    )


def strategy_regime_table(
    trades: Sequence[Trade],
    labels: pl.DataFrame,
    *,
    regime_column: str = "regime",
) -> pl.DataFrame:
    """Win rate, expectancy, total R and profit factor per strategy × regime.

    ``profit_factor`` is ``null`` when a bucket has wins but no losses: a finite
    value would invent a denominator. An empty bucket is not emitted at all.
    """
    per_trade = trades_with_regime(trades, labels, regime_column=regime_column)
    if per_trade.is_empty():
        return pl.DataFrame(schema=_TABLE_SCHEMA)

    wins = pl.when(pl.col("r_multiple") > 0).then(1).otherwise(0)
    gross_win = pl.when(pl.col("pnl") > 0).then(pl.col("pnl")).otherwise(0.0)
    gross_loss = pl.when(pl.col("pnl") < 0).then(-pl.col("pnl")).otherwise(0.0)
    table = (
        per_trade.group_by(["strategy", "regime"])
        .agg(
            pl.len().alias("trades"),
            wins.sum().alias("wins"),
            pl.col("r_multiple").mean().alias("expectancy_r"),
            pl.col("r_multiple").sum().alias("total_r"),
            pl.col("pnl").sum().alias("total_pnl"),
            gross_win.sum().alias("_gross_win"),
            gross_loss.sum().alias("_gross_loss"),
        )
        .with_columns(
            (pl.col("wins") / pl.col("trades")).alias("win_rate"),
            pl.when(pl.col("_gross_loss") > 0)
            .then(pl.col("_gross_win") / pl.col("_gross_loss"))
            .when(pl.col("_gross_win") > 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(0.0)
            .alias("profit_factor"),
            pl.col("trades").cast(pl.Int64),
            pl.col("wins").cast(pl.Int64),
        )
        .drop("_gross_win", "_gross_loss")
        .sort(["strategy", "regime"])
    )
    return table.select(
        "strategy",
        "regime",
        "trades",
        "wins",
        "win_rate",
        "expectancy_r",
        "total_r",
        "total_pnl",
        "profit_factor",
    )

_REGIME_SCHEMA: Final[dict[str, DataType | DataTypeClass]] = {
    "regime": pl.String,
    "trades": pl.Int64,
    "wins": pl.Int64,
    "win_rate": pl.Float64,
    "expectancy_r": pl.Float64,
    "total_r": pl.Float64,
    "total_pnl": pl.Float64,
    "profit_factor": pl.Float64,
    "avg_hold_minutes": pl.Float64,
}


def regime_metrics(
    trades: Sequence[Trade],
    labels: pl.DataFrame,
    *,
    regime_column: str = "regime",
) -> pl.DataFrame:
    """One-run metrics per regime: win rate, expectancy, PnL, PF and hold time.

    ``strategy_regime_table`` answers "which strategy suits this regime"; this
    function describes **one run's trades** by the regime current at entry.
    ``profit_factor`` is ``null`` for a bucket with wins but no losses; a trade
    before the first label lands in ``unknown``.
    """
    per_trade = trades_with_regime(trades, labels, regime_column=regime_column)
    if per_trade.is_empty():
        return pl.DataFrame(schema=_REGIME_SCHEMA)
    wins = pl.when(pl.col("r_multiple") > 0).then(1).otherwise(0)
    gross_win = pl.when(pl.col("pnl") > 0).then(pl.col("pnl")).otherwise(0.0)
    gross_loss = pl.when(pl.col("pnl") < 0).then(-pl.col("pnl")).otherwise(0.0)
    hold = (
        (pl.col("exit_ts") - pl.col("entry_ts")).dt.total_seconds() / 60.0
    ).alias("hold_minutes")
    table = (
        per_trade.with_columns(hold)
        .group_by("regime")
        .agg(
            pl.len().alias("trades"),
            wins.sum().alias("wins"),
            pl.col("r_multiple").mean().alias("expectancy_r"),
            pl.col("r_multiple").sum().alias("total_r"),
            pl.col("pnl").sum().alias("total_pnl"),
            pl.col("hold_minutes").mean().alias("avg_hold_minutes"),
            gross_win.sum().alias("_gross_win"),
            gross_loss.sum().alias("_gross_loss"),
        )
        .with_columns(
            (pl.col("wins") / pl.col("trades")).alias("win_rate"),
            pl.when(pl.col("_gross_loss") > 0)
            .then(pl.col("_gross_win") / pl.col("_gross_loss"))
            .when(pl.col("_gross_win") > 0)
            .then(pl.lit(None, dtype=pl.Float64))
            .otherwise(0.0)
            .alias("profit_factor"),
            pl.col("trades").cast(pl.Int64),
            pl.col("wins").cast(pl.Int64),
        )
        .drop("_gross_win", "_gross_loss")
        .sort("regime")
    )
    return table.select(
        "regime",
        "trades",
        "wins",
        "win_rate",
        "expectancy_r",
        "total_r",
        "total_pnl",
        "profit_factor",
        "avg_hold_minutes",
    )

