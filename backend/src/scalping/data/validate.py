"""Structural validation for stored bars and ticks.

The store enforces a schema and UTC, but it cannot know whether an imported
file has duplicate stamps, broken OHLC relationships, negative spreads or a
weekend-sized hole. This module reports those problems without changing or
filling the data: a validator that silently repairs prices is worse than no
validator.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Final, Literal, cast

import polars as pl

from scalping.data.base import BAR_SCHEMA
from scalping.data.registry import SessionSpec, session_for
from scalping.features.availability import capabilities_from_frame

__all__ = [
    "Issue",
    "ValidationReport",
    "validate_bars",
    "validate_ticks",
]

Severity = Literal["error", "warning"]

_PRICE_COLUMNS: Final[tuple[str, ...]] = ("open", "high", "low", "close", "bid_close", "ask_close")
_QUOTE_COLUMNS: Final[tuple[str, ...]] = ("bid_close", "ask_close", "spread")
_BAR_MINUTES: Final[dict[str, int]] = {
    "M1": 1,
    "M5": 5,
    "M15": 15,
    "M30": 30,
    "H1": 60,
    "H4": 240,
    "D1": 1440,
    "W1": 10080,
}


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
    capabilities: dict[str, bool] | None = None
    gap_breakdown: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "rows": self.rows,
            "ok": self.ok,
            "first_ts": self.first_ts,
            "last_ts": self.last_ts,
            "capabilities": self.capabilities,
            "gap_breakdown": self.gap_breakdown,
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


def _gap_parts(
    frame: pl.DataFrame,
    threshold_minutes: int,
    session: SessionSpec | None,
    bar_minutes: int | None,
) -> tuple[Issue | None, dict[str, Any] | None]:
    """Return the legacy time-gap warning plus a session/missing-minute breakdown."""
    if frame.height < 2:
        return None, None
    stamps = frame.get_column("ts").sort()
    diffs = stamps.diff().drop_nulls().dt.total_minutes()
    gap_mask = diffs > threshold_minutes
    count = int(gap_mask.sum())
    if count == 0:
        return None, None
    # The timestamp after each gap: diff is attached to the later bar.
    later = stamps.slice(1).filter(gap_mask)
    sample = tuple(str(value) for value in later.head(5).to_list())
    gap_minutes = [int(value) for value in diffs.filter(gap_mask).to_list()]
    largest = max(gap_minutes) if gap_minutes else 0

    in_session = 0
    session_closed = 0
    missing_minutes: int | None = None
    if session is not None:
        earlier = stamps.slice(0, stamps.len() - 1).filter(gap_mask).to_list()
        later_moments = later.to_list()
        # Classify by the gap midpoint: a weekend reopen has an open later
        # stamp but the interval itself is session-closed.
        midpoints = [
            (start + (end - start) / 2).astimezone(UTC)
            if (start + (end - start) / 2).tzinfo is not None
            else (start + (end - start) / 2).replace(tzinfo=UTC)
            for start, end in zip(earlier, later_moments, strict=True)
        ]
        in_session = sum(1 for moment in midpoints if session.is_open(moment))
        session_closed = count - in_session
        if bar_minutes is not None and bar_minutes > 0:
            missing_minutes = sum(
                max(0, round(minutes / bar_minutes) - 1)
                for minutes, moment in zip(gap_minutes, midpoints, strict=True)
                if session.is_open(moment)
            )
    breakdown = {
        "gaps": count,
        "in_session": in_session if session is not None else None,
        "session_closed": session_closed if session is not None else None,
        "largest_minutes": largest,
        "missing_minutes": missing_minutes,
        "bar_minutes": bar_minutes,
    }
    detail = f"{count} gaps > {threshold_minutes} min; largest {largest} min"
    if session is not None:
        detail += f"; {in_session} in-session"
        if missing_minutes is not None:
            detail += f", {missing_minutes} missing bar(s)"
    return (
        Issue(
            code="time_gaps",
            severity="warning",
            count=count,
            detail=detail,
            sample=sample,
        ),
        breakdown,
    )


def validate_bars(
    df: pl.DataFrame,
    *,
    symbol: str = "",
    timeframe: str = "M1",
    gap_threshold_minutes: int = 10,
    include_outliers: bool = True,
    session: SessionSpec | None = None,
    bar_minutes: int | None = None,
) -> ValidationReport:
    """Validate a bar frame without mutating it.

    Errors are structural and always fail the report (missing columns, broken
    OHLC, negative values, duplicates, non-UTC or unsorted stamps). Time gaps
    are warnings: weekends and holidays are legitimate, but a gap inside a
    trading day is worth seeing. When ``session`` is omitted it is derived from
    a registered ``symbol``; ``gap_breakdown`` then separates in-session from
    session-closed gaps and, with a known bar width, estimates missing bars.

    A bid-only export is not a structural error: null ``bid_close``/
    ``ask_close``/``spread`` cells become a ``missing_quotes`` warning and the
    report's ``capabilities`` say ``real_quotes=false``. Null cells in OHLC or
    volume remain errors.
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

    null_counts = df.null_count().row(0)
    total_nulls = sum(int(cast(int, value)) for value in null_counts)
    quote_nulls = sum(
        int(cast(int, null_counts[df.columns.index(name)]))
        for name in _QUOTE_COLUMNS
        if name in df.columns
    )
    structural_nulls = total_nulls - quote_nulls
    if structural_nulls:
        issues.append(
            Issue(
                code="null_values",
                severity="error",
                count=structural_nulls,
                detail=f"{structural_nulls} null non-quote cells",
            )
        )
    if quote_nulls:
        quote_null_rows = int(
            df.select(
                pl.any_horizontal([pl.col(name).is_null() for name in _QUOTE_COLUMNS if name in df.columns])
                .sum()
            ).item()
        )
        issues.append(
            Issue(
                code="missing_quotes",
                severity="warning",
                count=quote_null_rows,
                detail=(
                    f"{quote_null_rows} rows have a null bid/ask/spread cell; "
                    "real_quotes=false and the spread model must fill it"
                ),
                sample=_sample_ts(
                    df,
                    pl.any_horizontal([pl.col(name).is_null() for name in _QUOTE_COLUMNS if name in df.columns]),
                ),
            )
        )

    non_finite_mask = pl.any_horizontal(
        [
            pl.col(name).is_not_null() & ~pl.col(name).is_finite()
            for name in _PRICE_COLUMNS
            if name in df.columns
        ]
    )
    non_finite = int(df.filter(non_finite_mask).height)
    if non_finite:
        issues.append(
            Issue(
                code="non_finite_prices",
                severity="error",
                count=non_finite,
                detail=f"{non_finite} rows with non-finite prices",
                sample=_sample_ts(df, non_finite_mask),
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

    if session is None and symbol:
        try:
            session = session_for(symbol)
        except ValueError:
            session = None
    width = _BAR_MINUTES.get(timeframe.upper()) if bar_minutes is None else bar_minutes
    gap, gap_breakdown = _gap_parts(df, gap_threshold_minutes, session, width)
    if gap is not None:
        issues.append(gap)

    if include_outliers:
        # Imported here: the outlier module is retrospective QA and imports
        # ``Issue`` from this module, so a top-level import would be circular.
        from scalping.data.quality import detect_bar_outliers

        issues.extend(
            detect_bar_outliers(df, gap_threshold_minutes=gap_threshold_minutes)
        )

    capabilities = capabilities_from_frame(df)
    ok = not any(issue.severity == "error" for issue in issues)
    return ValidationReport(
        symbol=symbol,
        timeframe=timeframe,
        rows=rows,
        ok=ok,
        issues=tuple(issues),
        first_ts=_stamp(df["ts"].min()),
        last_ts=_stamp(df["ts"].max()),
        capabilities={
            "bar_volume": capabilities.bar_volume,
            "real_quotes": capabilities.real_quotes,
            "modeled_spread": capabilities.modeled_spread,
            "spread": capabilities.spread,
            "tick_flow": capabilities.tick_flow,
            "m1_source": capabilities.m1_source,
        },
        gap_breakdown=gap_breakdown,
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
