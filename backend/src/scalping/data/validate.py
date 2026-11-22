"""Structural validation for stored bars and ticks.

The store enforces a schema and UTC, but it cannot know whether an imported
file has duplicate stamps, broken OHLC relationships, negative spreads or a
weekend-sized hole. This module reports those problems without changing or
filling the data: a validator that silently repairs prices is worse than no
validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Final, Literal, cast

import polars as pl

from scalping.data.base import BAR_SCHEMA

__all__ = [
    "Issue",
    "ValidationReport",
    "validate_bars",
    "validate_ticks",
]

Severity = Literal["error", "warning"]

_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "bid_close", "ask_close")


@dataclass(frozen=True, slots=True)
class Issue:
    """One class of problem, with a bounded sample of offending rows."""

    code: str
    severity: Severity
    count: int
    detail: str
    sample: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "count": self.count,
            "detail": self.detail,
            "sample": list(self.sample),
        }


@dataclass(frozen=True, slots=True)
class ValidationReport:
    """Result of validating one frame."""

    symbol: str
    timeframe: str
    rows: int
    ok: bool
    issues: tuple[Issue, ...] = field(default_factory=tuple)
    first_ts: str | None = None
    last_ts: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "ok": self.ok,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "issues": [issue.to_dict() for issue in self.issues],
        }


def _sample_ts(frame: pl.DataFrame, mask: pl.Expr, limit: int = 5) -> tuple[str, ...]:
    try:
        values = frame.filter(mask).get_column("ts").head(limit).to_list()
    except (pl.exceptions.PolarsError, KeyError):
        return ()
    return tuple(str(value) for value in values)


def _stamp(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _gap_report(frame: pl.DataFrame, threshold_minutes: int) -> Issue | None:
    if frame.height < 3:
        return None
    stamps = frame.get_column("ts").sort()
    diffs = stamps.diff().drop_nulls().dt.total_minutes()
    gaps = diffs.filter(diffs > threshold_minutes)
    count = int(gaps.len())
    if count == 0:
        return None
    # The timestamp after each gap: diff is attached to the later bar.
    later = stamps.slice(1).filter(diffs > threshold_minutes)
    sample = tuple(str(value) for value in later.head(5).to_list())
    largest = int(cast(int, gaps.max() or 0))
    return Issue(
        code="time_gaps",
        severity="warning",
        count=count,
        detail=f"{count} gaps > {threshold_minutes} min; largest {largest} min",
        sample=sample,
    )


def validate_bars(
    df: pl.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "M1",
    gap_threshold_minutes: int = 10,
) -> ValidationReport:
    """Validate a bar frame without mutating it.

    Errors are structural and always fail the report (missing columns, broken
    OHLC, negative values, duplicates, non-UTC or unsorted stamps). Time gaps
    are warnings: weekends and holidays are legitimate, but a gap inside a
    trading day is worth seeing.
    """
    issues: list[Issue] = []
    rows = df.height
    if rows == 0:
        return ValidationReport(symbol=symbol, timeframe=timeframe, rows=0, ok=True)

    missing = [name for name in BAR_SCHEMA if name not in df.columns]
    if missing:
        issues.append(
            Issue(
                code="missing_columns",
                severity="error",
                count=len(missing),
                detail=", ".join(missing),
            )
        )
        return ValidationReport(
            symbol=symbol, timeframe=timeframe, rows=rows, ok=False, issues=tuple(issues)
        )

    if df["ts"].dtype != pl.Datetime("us", "UTC"):
        issues.append(
            Issue(
                code="timestamp_dtype",
                severity="error",
                count=rows,
                detail=f"expected Datetime('us','UTC'), got {df['ts'].dtype}",
            )
        )
    if not df["ts"].is_sorted():
        issues.append(
            Issue(
                code="unsorted_timestamps",
                severity="error",
                count=rows,
                detail="timestamps are not sorted ascending",
            )
        )

    duplicates = rows - df["ts"].n_unique()
    if duplicates:
        issues.append(
            Issue(
                code="duplicate_timestamps",
                severity="error",
                count=duplicates,
                detail=f"{duplicates} duplicate ts values",
            )
        )

    nulls = sum(int(cast(int, value)) for value in df.null_count().row(0))
    if nulls:
        issues.append(
            Issue(
                code="null_values",
                severity="error",
                count=nulls,
                detail=f"{nulls} null cells",
            )
        )

    finite_mask = pl.all_horizontal(
        [pl.col(name).is_finite() for name in _PRICE_COLUMNS if name in df.columns]
    )
    non_finite = rows - int(df.filter(finite_mask).height)
    if non_finite:
        issues.append(
            Issue(
                code="non_finite_prices",
                severity="error",
                count=non_finite,
                detail=f"{non_finite} rows with non-finite prices",
                sample=_sample_ts(df, ~finite_mask),
            )
        )

    if {"open", "high", "low", "close"} <= set(df.columns):
        broken = df.filter(
            (pl.col("low") > pl.col("high"))
            | (pl.col("low") > pl.min_horizontal(pl.col("open"), pl.col("close")))
            | (pl.col("high") < pl.max_horizontal(pl.col("open"), pl.col("close")))
        )
        if broken.height:
            issues.append(
                Issue(
                    code="ohlc_inconsistent",
                    severity="error",
                    count=broken.height,
                    detail="low/high do not bracket open/close",
                    sample=_sample_ts(df, ~(
                        (pl.col("low") <= pl.col("high"))
                        & (pl.col("low") <= pl.min_horizontal(pl.col("open"), pl.col("close")))
                        & (pl.col("high") >= pl.max_horizontal(pl.col("open"), pl.col("close")))
                    )),
                )
            )

    if {"bid_close", "ask_close"} <= set(df.columns):
        crossed = df.filter(pl.col("ask_close") < pl.col("bid_close"))
        if crossed.height:
            issues.append(
                Issue(
                    code="crossed_quotes",
                    severity="error",
                    count=crossed.height,
                    detail="ask_close below bid_close",
                    sample=_sample_ts(df, pl.col("ask_close") < pl.col("bid_close")),
                )
            )

    for name in ("spread", "volume"):
        if name not in df.columns:
            continue
        negative = df.filter(pl.col(name) < 0)
        if negative.height:
            issues.append(
                Issue(
                    code=f"negative_{name}",
                    severity="error",
                    count=negative.height,
                    detail=f"{negative.height} negative {name} rows",
                    sample=_sample_ts(df, pl.col(name) < 0),
                )
            )

    gap = _gap_report(df, gap_threshold_minutes)
    if gap is not None:
        issues.append(gap)

    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=rows,
        ok=ok,
        issues=tuple(issues),
        first_ts=_stamp(df["ts"].min()),
        last_ts=_stamp(df["ts"].max()),
    )


def validate_ticks(df: pl.DataFrame, *, symbol: str = "") -> ValidationReport:
    """Validate a tick frame with ``ts, bid, ask, bid_volume, ask_volume``."""
    issues: list[Issue] = []
    rows = df.height
    if rows == 0:
        return ValidationReport(symbol=symbol, timeframe="tick", rows=0, ok=True)
    required = ("ts", "bid", "ask", "bid_volume", "ask_volume")
    missing = [name for name in required if name not in df.columns]
    if missing:
        return ValidationReport(
            symbol=symbol,
            timeframe="tick",
            rows=rows,
            ok=False,
            issues=(
                Issue(
                    code="missing_columns",
                    severity="error",
                    count=len(missing),
                    detail=", ".join(missing),
                ),
            ),
        )

    if not df["ts"].is_sorted():
        issues.append(
            Issue(
                code="unsorted_timestamps",
                severity="error",
                count=rows,
                detail="ticks are not sorted ascending",
            )
        )
    crossed = df.filter(pl.col("ask") < pl.col("bid")).height
    if crossed:
        issues.append(
            Issue(
                code="crossed_quotes",
                severity="error",
                count=crossed,
                detail="ask below bid",
            )
        )
    for name in ("bid", "ask", "bid_volume", "ask_volume"):
        bad = df.filter(~pl.col(name).is_finite() | (pl.col(name) < 0)).height
        if bad:
            issues.append(
                Issue(
                    code=f"invalid_{name}",
                    severity="error",
                    count=bad,
                    detail=f"{bad} non-finite or negative rows",
                )
            )
    nulls = int(df.null_count().sum_horizontal().item())
    if nulls:
        issues.append(
            Issue(code="null_values", severity="error", count=nulls, detail=f"{nulls} null cells")
        )
    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        symbol=symbol,
        timeframe="tick",
        rows=rows,
        ok=ok,
        issues=tuple(issues),
        first_ts=_stamp(df["ts"].min()),
        last_ts=_stamp(df["ts"].max()),
    )
