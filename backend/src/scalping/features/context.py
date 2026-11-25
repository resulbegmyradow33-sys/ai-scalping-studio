"""Session flags and higher timeframe context.

Sessions are UTC buckets: Asia ``00:00-07:59``, London ``08:00-12:59``, New York
``13:00-20:59``. They are mutually exclusive; ``ny_open_30`` is the first half
hour of the New York session and sits inside ``ny``. Hours after 21:00 UTC are
labelled by no session.

Higher timeframe context
------------------------
``htf_context`` only joins a bucket once it **closed**. Each aggregated bar is
moved forward by one bucket width (its availability time) and then joined with
``join_asof(strategy="backward")``, so a row at ``10:07`` sees the 5m bucket
that ended at ``10:00``, never the one still forming. Aggregating a truncated
frame therefore cannot change any joined value.

The default ``m5``/``m15`` columns are joined on every build. ``m30``/``h4``
frames are optional (``build_features(..., groups=("htf",))``) and add the
compact :data:`MTF_CONTEXT_COLUMNS` set under the same closed-bucket rule.
"""

from __future__ import annotations

from datetime import timedelta

import polars as pl

from scalping.features.indicators import atr, rsi
from scalping.features.liquidity import fvg
from scalping.features.structure import market_structure

__all__ = ["HTF_COLUMNS", "MTF_CONTEXT_COLUMNS", "htf_context", "session_flags"]

#: Feature suffixes computed on each default higher timeframe (M5/M15).
_HTF_SUFFIXES: tuple[str, ...] = (
    "ret_1",
    "rsi_14",
    "atr_norm",
    "ema_gap_21",
    "pos_range_20",
    "bos",
    "fvg_dist",
)

#: Compact suffixes computed on the optional M30/H4 context frames. Trend state,
#: one-bucket return and (H4 only) volatility scale, not a second copy of the
#: whole M5 ladder.
_CONTEXT_SUFFIXES: dict[str, tuple[str, ...]] = {
    "m30": ("trend", "ret"),
    "h4": ("trend", "ret", "atr_frac"),
}

_ASIA_HOURS: tuple[int, int] = (0, 8)
_LONDON_HOURS: tuple[int, int] = (8, 13)
_NY_HOURS: tuple[int, int] = (13, 21)

#: Every column ``htf_context`` produces by default, in order.
HTF_COLUMNS: tuple[str, ...] = tuple(
    f"{timeframe}_{suffix}" for timeframe in ("m5", "m15") for suffix in _HTF_SUFFIXES
)

#: Extra closed-bucket columns the optional ``htf`` group adds to the matrix.
MTF_CONTEXT_COLUMNS: tuple[str, ...] = tuple(
    f"{timeframe}_{suffix}"
    for timeframe, suffixes in _CONTEXT_SUFFIXES.items()
    for suffix in suffixes
)


def session_flags(ts: pl.Series) -> pl.DataFrame:
    """Boolean ``asia``/``london``/``ny``/``ny_open_30`` flags for UTC ``ts``."""
    hour = ts.dt.hour()
    minute = ts.dt.minute()
    asia = (hour >= _ASIA_HOURS[0]) & (hour < _ASIA_HOURS[1])
    london = (hour >= _LONDON_HOURS[0]) & (hour < _LONDON_HOURS[1])
    ny = (hour >= _NY_HOURS[0]) & (hour < _NY_HOURS[1])
    return pl.DataFrame(
        {
            "asia": asia.fill_null(False),
            "london": london.fill_null(False),
            "ny": ny.fill_null(False),
            # The first half hour of the NY session: 13:00-13:29 UTC.
            "ny_open_30": (ny & (hour == _NY_HOURS[0]) & (minute < 30)).fill_null(False),
        }
    )


def _htf_features(frame: pl.DataFrame, prefix: str) -> pl.DataFrame:
    """Causal features for one aggregated timeframe."""
    ordered = frame.sort("ts")
    close = pl.col("close")
    low20 = pl.col("low").rolling_min(20, min_samples=1)
    width = pl.col("high").rolling_max(20, min_samples=1) - low20
    rsi_14 = rsi(ordered, 14, min_samples=2)
    structure = market_structure(ordered, 2)["bos"]
    gaps = fvg(ordered)
    return ordered.select(
        pl.col("ts"),
        (close / close.shift(1) - 1.0).alias(f"{prefix}_ret_1"),
        (rsi_14 / 100.0).alias(f"{prefix}_rsi_14"),
        (atr(ordered, 14) / close).alias(f"{prefix}_atr_norm"),
        (close / close.ewm_mean(span=21, adjust=False) - 1.0).alias(f"{prefix}_ema_gap_21"),
        pl.when(width > 0.0)
        .then((close - low20) / width)
        .otherwise(0.5)
        .alias(f"{prefix}_pos_range_20"),
        structure.alias(f"{prefix}_bos"),
        gaps["fvg_dist"].alias(f"{prefix}_fvg_dist"),
    )


def _context_features(frame: pl.DataFrame, prefix: str, suffixes: tuple[str, ...]) -> pl.DataFrame:
    """Causal trend/return/volatility context for one optional timeframe."""
    ordered = frame.sort("ts")
    close = pl.col("close")
    structure = market_structure(ordered, 2)
    context = ordered.with_columns(
        structure["trend"].cast(pl.Float64).alias(f"{prefix}_trend"),
        (close / close.shift(1) - 1.0).alias(f"{prefix}_ret"),
        # ATR(14) relative to price: the same units as the M1 ``atr_norm``, so
        # an H4 volatility reading can be compared across instruments.
        (atr(ordered, 14) / close).alias(f"{prefix}_atr_frac"),
    )
    return context.select("ts", *[f"{prefix}_{suffix}" for suffix in suffixes])


def _bucket_width(frame: pl.DataFrame) -> pl.Expr:
    """Availability offset: one bucket width, taken from the bucket spacing."""
    step = frame["ts"].diff().drop_nulls().min()
    if not isinstance(step, timedelta) or step <= timedelta(0):
        raise ValueError("a higher timeframe frame needs at least two distinct buckets")
    return pl.duration(microseconds=round(step.total_seconds() * 1_000_000))


def htf_context(
    m1: pl.DataFrame,
    m5: pl.DataFrame,
    m15: pl.DataFrame,
    *,
    m30: pl.DataFrame | None = None,
    h4: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Join the last closed bucket onto every M1 ``ts``.

    Returns one row per M1 row, in input order, with ``ts`` plus the columns in
    :data:`HTF_COLUMNS`; passing ``m30``/``h4`` appends
    :data:`MTF_CONTEXT_COLUMNS`. Rows before the first closed bucket carry
    nulls; the builder forward fills them.

    The rule is the same for every timeframe: a bucket stamped ``s`` only
    becomes visible at ``s + width``, where ``width`` is the frame's own
    spacing, and ``join_asof(backward)`` includes the exact boundary. The
    bucket containing the decision bar is never used.
    """
    left = m1.select("ts").with_row_index("_row")
    joined = left
    sources: list[tuple[str, pl.DataFrame | None, tuple[str, ...], bool]] = [
        ("m5", m5, _HTF_SUFFIXES, True),
        ("m15", m15, _HTF_SUFFIXES, True),
        ("m30", m30, _CONTEXT_SUFFIXES["m30"], False),
        ("h4", h4, _CONTEXT_SUFFIXES["h4"], False),
    ]
    for prefix, frame, suffixes, required in sources:
        if frame is None:
            if not required:
                continue
            raise ValueError(f"missing {prefix} frame for the default higher-timeframe context")
        if frame.height < 2:
            joined = joined.with_columns(
                [pl.lit(None, dtype=pl.Float64).alias(f"{prefix}_{suffix}") for suffix in suffixes]
            )
            continue
        ready = (
            (
                _htf_features(frame, prefix)
                if required
                else _context_features(frame, prefix, suffixes)
            )
            .sort("ts")
            .with_columns((pl.col("ts") + _bucket_width(frame)).alias("ts"))
        )
        joined = joined.sort("ts").join_asof(ready, on="ts", strategy="backward")
    return joined.sort("_row").drop("_row")
