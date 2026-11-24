"""Measure one pipeline stage, repeatedly, without hidden state.

The harness exists because "it feels fast" is not a measurement. A stage is an
arbitrary zero-argument callable; :func:`measure_stage` runs it ``warmup`` times
(discarded), then ``repeats`` times, and records:

* wall time per repeat (``perf_counter`` by default);
* process CPU user and system time per repeat (``getrusage(RUSAGE_SELF)``);
* high-water RSS at the start and end of each repeat (``ru_maxrss`` converted
  to MiB), so the delta is the growth of the high-water mark, not the area under
  a residency curve;
* optional bytes written under ``disk_path`` during each repeat;
* ``rows`` and ``bytes`` when the stage returns a mapping with those keys.

Aggregation is deliberately plain: median, minimum and maximum over the
recorded repeats. No trimming, no percentile interpolation and no cross-stage
normalisation. The raw per-repeat samples are kept in the result, so a reader
can re-derive every aggregate and see a failed repeat instead of a smooth
average that hides it.

A stage that raises is *recorded*, not swallowed silently: ``ok=False`` and the
exception text are part of the result. ``strict=True`` re-raises as
:class:`StageError` after the measurement exists, for callers that prefer a
hard failure.

Nothing here reads a data store, opens a socket or keeps module-level mutable
state. The clock, CPU and RSS probes are injectable only so tests can pin the
aggregation maths; production callers use the defaults.
"""

from __future__ import annotations

import math
import os
import resource
import statistics
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from scalping.orchestrator.resources import rss_mb_from_ru_maxrss

__all__ = [
    "RunMeasurement",
    "StageError",
    "StageMeasurement",
    "measure_stage",
]

_MIB: Final[float] = 1024.0 * 1024.0


def _cpu_times() -> tuple[float, float]:
    """This process's cumulative ``(user, system)`` CPU seconds."""
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return float(usage.ru_utime), float(usage.ru_stime)


def _rss_mb() -> float:
    """This process's high-water RSS in MiB; platform conversion is centralised."""
    usage = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return rss_mb_from_ru_maxrss(usage)


def _dir_size_mb(path: Path) -> float:
    """Bytes under ``path`` in MiB, ignoring races and unreadable entries."""
    if not path.exists():
        return 0.0
    total = 0
    stack: list[str] = [str(path)]
    while stack:
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False):
                            total += entry.stat(follow_symlinks=False).st_size
                    except OSError:
                        continue
        except OSError:
            continue
    return float(total) / _MIB


def _finite(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def _median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


@dataclass(frozen=True, slots=True)
class RunMeasurement:
    """One measured repeat, before aggregation."""

    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    cpu_total_seconds: float
    rss_start_mb: float | None
    rss_end_mb: float | None
    disk_start_mb: float | None
    disk_end_mb: float | None
    ok: bool
    error: str = ""

    @property
    def rss_delta_mb(self) -> float | None:
        if self.rss_start_mb is None or self.rss_end_mb is None:
            return None
        return max(self.rss_end_mb - self.rss_start_mb, 0.0)

    @property
    def disk_delta_mb(self) -> float | None:
        if self.disk_start_mb is None or self.disk_end_mb is None:
            return None
        return max(self.disk_end_mb - self.disk_start_mb, 0.0)

    def to_dict(self) -> dict[str, float | bool | str | None]:
        return {
            "wall_seconds": round(_finite(self.wall_seconds), 6),
            "cpu_user_seconds": round(_finite(self.cpu_user_seconds), 6),
            "cpu_system_seconds": round(_finite(self.cpu_system_seconds), 6),
            "cpu_total_seconds": round(_finite(self.cpu_total_seconds), 6),
            "rss_start_mb": _round_optional(self.rss_start_mb),
            "rss_end_mb": _round_optional(self.rss_end_mb),
            "rss_delta_mb": _round_optional(self.rss_delta_mb),
            "disk_start_mb": _round_optional(self.disk_start_mb),
            "disk_end_mb": _round_optional(self.disk_end_mb),
            "disk_delta_mb": _round_optional(self.disk_delta_mb),
            "ok": self.ok,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class StageMeasurement:
    """Aggregated cost of one stage; JSON-safe through :meth:`to_dict`.

    ``cpu_*_median`` values are medians of the per-repeat CPU readings, and
    ``cpu_total_seconds_median`` is the median of each repeat's user+system
    sum, not the sum of the two medians. ``rss_delta_mb`` is the largest
    per-repeat growth of the high-water mark. ``disk_delta_mb`` is the largest
    per-repeat growth of ``disk_path``. All values are local to this process on
    this machine at this time.
    """

    name: str
    repeats: int
    warmup: int
    measured_repeats: int
    wall_seconds_median: float
    wall_seconds_min: float
    wall_seconds_max: float
    cpu_user_seconds_median: float
    cpu_system_seconds_median: float
    cpu_total_seconds_median: float
    rss_start_mb: float | None
    rss_peak_mb: float | None
    rss_delta_mb: float | None
    disk_delta_mb: float | None
    rows: int | None
    bytes_processed: int | None
    ok: bool
    error: str = ""
    probe_error: str = ""
    runs: tuple[RunMeasurement, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "repeats": self.repeats,
            "warmup": self.warmup,
            "measured_repeats": self.measured_repeats,
            "wall_seconds": {
                "median": round(_finite(self.wall_seconds_median), 6),
                "min": round(_finite(self.wall_seconds_min), 6),
                "max": round(_finite(self.wall_seconds_max), 6),
            },
            "cpu_seconds": {
                "user_median": round(_finite(self.cpu_user_seconds_median), 6),
                "system_median": round(_finite(self.cpu_system_seconds_median), 6),
                "total_median": round(_finite(self.cpu_total_seconds_median), 6),
            },
            "rss_mb": {
                "start": _round_optional(self.rss_start_mb),
                "peak": _round_optional(self.rss_peak_mb),
                "delta": _round_optional(self.rss_delta_mb),
            },
            "disk_delta_mb": _round_optional(self.disk_delta_mb),
            "rows": self.rows,
            "bytes": self.bytes_processed,
            "ok": self.ok,
            "error": self.error,
            "probe_error": self.probe_error,
            "runs": [run.to_dict() for run in self.runs],
        }


class StageError(RuntimeError):
    """A stage failed and the caller asked for a hard failure."""

    def __init__(self, measurement: StageMeasurement) -> None:
        super().__init__(f"stage {measurement.name!r} failed: {measurement.error}")
        self.measurement = measurement


def _round_optional(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(_finite(float(value)), digits)


def _as_int(value: object) -> int | None:
    """Accept a non-negative int (or integral float); reject bool and garbage."""
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value if value >= 0 else None
    if isinstance(value, float) and math.isfinite(value) and value.is_integer() and value >= 0:
        return int(value)
    return None


def _processed(result: object) -> tuple[int | None, int | None]:
    """Read ``rows`` and ``bytes``/``bytes_processed`` from a stage's mapping."""
    if not isinstance(result, Mapping):
        return None, None
    rows = _as_int(result.get("rows"))
    raw_bytes = result.get("bytes", result.get("bytes_processed"))
    return rows, _as_int(raw_bytes)


def _empty_measurement(
    name: str,
    repeats: int,
    warmup: int,
    error: str,
    probe_error: str,
    runs: tuple[RunMeasurement, ...],
) -> StageMeasurement:
    return StageMeasurement(
        name=name,
        repeats=repeats,
        warmup=warmup,
        measured_repeats=len(runs),
        wall_seconds_median=0.0,
        wall_seconds_min=0.0,
        wall_seconds_max=0.0,
        cpu_user_seconds_median=0.0,
        cpu_system_seconds_median=0.0,
        cpu_total_seconds_median=0.0,
        rss_start_mb=None,
        rss_peak_mb=None,
        rss_delta_mb=None,
        disk_delta_mb=None,
        rows=None,
        bytes_processed=None,
        ok=False,
        error=error,
        probe_error=probe_error,
        runs=runs,
    )


def measure_stage(
    name: str,
    fn: Callable[[], object],
    *,
    repeats: int = 1,
    warmup: int = 0,
    disk_path: str | Path | None = None,
    strict: bool = False,
    clock: Callable[[], float] = time.perf_counter,
    cpu_times: Callable[[], tuple[float, float]] = _cpu_times,
    rss_mb: Callable[[], float] = _rss_mb,
) -> StageMeasurement:
    """Run ``fn`` and return its aggregated wall/CPU/RSS/disk cost.

    ``warmup`` calls happen before any timestamp is recorded; their return
    values are ignored. A failure during a measured repeat stops the repeat
    loop, keeps the failed repeat in ``runs`` and returns ``ok=False``. A
    failure during warmup returns a zeroed, failed measurement with no runs.
    ``strict=True`` raises :class:`StageError` instead of returning that failed
    measurement. The exception from ``fn`` is never re-raised unchanged, so the
    error text is always available in the report.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be a non-empty string")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
        raise ValueError(f"repeats must be an int >= 1, got {repeats!r}")
    if isinstance(warmup, bool) or not isinstance(warmup, int) or warmup < 0:
        raise ValueError(f"warmup must be an int >= 0, got {warmup!r}")
    if not callable(fn):
        raise TypeError(f"fn must be callable, got {type(fn).__name__}")
    root = None if disk_path is None else Path(disk_path)

    probe_errors: list[str] = []

    def rss_reading() -> float | None:
        try:
            return float(rss_mb())
        except Exception as exc:  # noqa: BLE001 - a probe failure is data, not a crash
            probe_errors.append(f"{type(exc).__name__}: {exc}")
            return None

    for _ in range(warmup):
        try:
            fn()
        except Exception as exc:
            measurement = _empty_measurement(
                name,
                repeats,
                warmup,
                f"warmup failed: {type(exc).__name__}: {exc}",
                "; ".join(probe_errors),
                (),
            )
            if strict:
                raise StageError(measurement) from exc
            return measurement

    runs: list[RunMeasurement] = []
    error = ""
    rows: int | None = None
    bytes_processed: int | None = None

    for _ in range(repeats):
        rss_start = rss_reading()
        disk_start = None if root is None else _dir_size_mb(root)
        user_start, system_start = cpu_times()
        started = clock()
        try:
            result = fn()
        except Exception as exc:  # noqa: BLE001 - a failed stage still has a cost
            wall = max(_finite(clock() - started), 0.0)
            user_end, system_end = cpu_times()
            rss_end = rss_reading()
            disk_end = None if root is None else _dir_size_mb(root)
            error = f"{type(exc).__name__}: {exc}"
            rows = None
            bytes_processed = None
            runs.append(
                RunMeasurement(
                    wall_seconds=wall,
                    cpu_user_seconds=max(user_end - user_start, 0.0),
                    cpu_system_seconds=max(system_end - system_start, 0.0),
                    cpu_total_seconds=max(user_end - user_start, 0.0)
                    + max(system_end - system_start, 0.0),
                    rss_start_mb=rss_start,
                    rss_end_mb=rss_end,
                    disk_start_mb=disk_start,
                    disk_end_mb=disk_end,
                    ok=False,
                    error=error,
                )
            )
            break

        wall = max(_finite(clock() - started), 0.0)
        user_end, system_end = cpu_times()
        rss_end = rss_reading()
        disk_end = None if root is None else _dir_size_mb(root)
        user = max(user_end - user_start, 0.0)
        system = max(system_end - system_start, 0.0)
        runs.append(
            RunMeasurement(
                wall_seconds=wall,
                cpu_user_seconds=user,
                cpu_system_seconds=system,
                cpu_total_seconds=user + system,
                rss_start_mb=rss_start,
                rss_end_mb=rss_end,
                disk_start_mb=disk_start,
                disk_end_mb=disk_end,
                ok=True,
            )
        )
        rows, bytes_processed = _processed(result)

    if not runs:  # pragma: no cover - repeats >= 1 and warmup failures return early
        measurement = _empty_measurement(
            name, repeats, warmup, error or "no repeats", "; ".join(probe_errors), ()
        )
        if strict:
            raise StageError(measurement)
        return measurement

    walls = [run.wall_seconds for run in runs]
    users = [run.cpu_user_seconds for run in runs]
    systems = [run.cpu_system_seconds for run in runs]
    totals = [run.cpu_total_seconds for run in runs]
    deltas = [run.rss_delta_mb for run in runs if run.rss_delta_mb is not None]
    disk_deltas = [run.disk_delta_mb for run in runs if run.disk_delta_mb is not None]
    starts = [run.rss_start_mb for run in runs if run.rss_start_mb is not None]
    ends = [run.rss_end_mb for run in runs if run.rss_end_mb is not None]

    ok = error == ""
    measurement = StageMeasurement(
        name=name,
        repeats=repeats,
        warmup=warmup,
        measured_repeats=len(runs),
        wall_seconds_median=_median(walls),
        wall_seconds_min=min(walls),
        wall_seconds_max=max(walls),
        cpu_user_seconds_median=_median(users),
        cpu_system_seconds_median=_median(systems),
        cpu_total_seconds_median=_median(totals),
        rss_start_mb=starts[0] if starts else None,
        rss_peak_mb=max(ends) if ends else None,
        rss_delta_mb=max(deltas) if deltas else None,
        disk_delta_mb=max(disk_deltas) if disk_deltas else None,
        rows=rows,
        bytes_processed=bytes_processed,
        ok=ok,
        error=error,
        probe_error="; ".join(probe_errors),
        runs=tuple(runs),
    )
    if strict and not ok:
        raise StageError(measurement)
    return measurement
