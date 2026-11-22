"""Regime evaluation: transitions, dwell time and (later) strategy tables.

The two functions here are descriptive. They answer "how sticky is each state"
and "where does the sequence go next", which is how a regime report avoids the
mistake of reading one label as a forecast.
"""

from __future__ import annotations

from typing import Final

import polars as pl

__all__ = ["dwell_stats", "transition_matrix"]

_REQUIRED: Final[tuple[str, ...]] = ("ts", "state")


def _require(labels: pl.DataFrame) -> None:
    missing = [name for name in _REQUIRED if name not in labels.columns]
    if missing:
        raise ValueError(f"labels are missing columns: {missing}")


def transition_matrix(labels: pl.DataFrame) -> pl.DataFrame:
    """Row-normalised state-to-state transitions, including unseen pairs.

    Every observed state gets a full row, so a state that never changed has a
    diagonal of 1 and zero elsewhere instead of a missing cell.
    """
    _require(labels)
    states = labels["state"].unique().sort().to_list()
    if not states:
        return pl.DataFrame(
            schema={"from": pl.String, "to": pl.String, "count": pl.Int64, "probability": pl.Float64}
        )

    pairs = labels.select(
        pl.col("state").alias("from"),
        pl.col("state").shift(-1).alias("to"),
    ).drop_nulls("to")
    counts = pairs.group_by(["from", "to"]).agg(pl.len().alias("count"))
    grid = pl.DataFrame(
        {
            "from": [state for state in states for _ in states],
            "to": [state for _ in states for state in states],
        }
    )
    grid = grid.join(counts, on=["from", "to"], how="left").with_columns(
        pl.col("count").fill_null(0).cast(pl.Int64)
    )
    totals = grid.group_by("from").agg(pl.col("count").sum().alias("total"))
    return (
        grid.join(totals, on="from")
        .with_columns(
            pl.when(pl.col("total") > 0)
            .then(pl.col("count") / pl.col("total"))
            .otherwise(0.0)
            .alias("probability")
        )
        .drop("total")
        .sort(["from", "to"])
    )


def dwell_stats(labels: pl.DataFrame) -> pl.DataFrame:
    """Bars, share and run length per state.

    A high ``mean_run`` means the label is persistent; a low one means the
    classifier flickers and should not drive a slow strategy.
    """
    _require(labels)
    if labels.is_empty():
        return pl.DataFrame(
            schema={
                "state": pl.String,
                "runs": pl.Int64,
                "bars": pl.Int64,
                "share": pl.Float64,
                "mean_run": pl.Float64,
                "max_run": pl.Int64,
            }
        )
    runs = (
        labels.with_columns(
            (pl.col("state") != pl.col("state").shift(1))
            .fill_null(True)
            .cum_sum()
            .alias("_run")
        )
        .group_by(["_run", "state"])
        .agg(pl.len().alias("length"))
    )
    return (
        runs.group_by("state")
        .agg(
            pl.len().alias("runs"),
            pl.col("length").sum().alias("bars"),
            pl.col("length").mean().alias("mean_run"),
            pl.col("length").max().alias("max_run"),
        )
        .with_columns(
            (pl.col("bars") / labels.height).alias("share"),
        )
        .with_columns(
            pl.col("runs").cast(pl.Int64),
            pl.col("bars").cast(pl.Int64),
            pl.col("max_run").cast(pl.Int64),
        )
        .sort("state")
    )
