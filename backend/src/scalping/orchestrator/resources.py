"""Measured resource guard for the Mac mini M2 8 GB target.

Audit P3 item 3 asks for a measured RAM/CPU/disk guard and P6 item 2 for
enforced resource bounds. Slice one supplies the *measurement*, not the
scheduling policy: :class:`MeasuredResourceGuard` implements the orchestrator's
``ResourceGuard`` hook (``can_start(kind) -> (bool, str)``) using numbers read
from the operating system, so a refusal names the binding limit and the value
that breached it.

Design rules, in order of importance:

* **Stdlib only.** ``resource``, ``shutil``, ``os`` and ``time``. No psutil, no
  subprocess, no network. A probe that shells out would itself consume the
  resources it is trying to police.
* **Fail closed.** A probe that raises, returns garbage or produces a
  non-finite value is a refusal with ``probe_failed: ...`` as the reason. The
  guard never raises into the orchestrator; a broken probe must not crash the
  loop that keeps paper trading alive.
* **One measurement per queue drain.** ``can_start`` caches the snapshot for
  ``cache_seconds`` (1 s by default) so draining ten jobs does not run ten
  probes. The cache is monotonic-clock based; a backwards clock forces a fresh
  probe.
* **Measurements, not promises.** These numbers describe *this* machine at
  *this* instant. They cannot predict a later allocation, and on macOS they
  cannot see memory held by other processes (see :class:`SystemProbe`).

Profiles are immutable module constants: :data:`M2_8GB_PROFILE` is the safe
default for the target machine and :data:`EXPLORATORY_PROFILE` is permissive but
still bounded. ``0`` disables an individual limit; a default-constructed
:class:`ResourceLimits` therefore allows everything and must be opted into
explicitly, never treated as promotion evidence.
"""

from __future__ import annotations

import math
import os
import resource
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final, Protocol, Self, runtime_checkable

__all__ = [
    "EXPLORATORY_PROFILE",
    "M2_8GB_PROFILE",
    "MeasuredResourceGuard",
    "Measurement",
    "Probe",
    "ResourceLimits",
    "ResourceProbeError",
    "ResourceSample",
    "ResourceSnapshot",
    "SystemProbe",
    "rss_mb_from_ru_maxrss",
]

#: Bytes per MiB. ``ru_maxrss`` is a byte count on macOS and a KiB count on
#: Linux; every other RAM number in this module is MiB.
_MIB: Final[float] = 1024.0 * 1024.0

#: Total RAM assumed when ``os.sysconf`` cannot report physical pages. 8192 MiB
#: is the target machine class; on a larger host the fallback under-states
#: capacity, so a limit derived from it refuses earlier rather than later.
ASSUMED_TOTAL_RAM_MB: Final[float] = 8192.0

#: Platforms whose ``ru_maxrss`` is bytes rather than KiB. macOS and the BSDs.
_RU_MAXRSS_BYTES_PLATFORMS: Final[tuple[str, ...]] = ("darwin", "freebsd", "openbsd", "netbsd")


class ResourceProbeError(RuntimeError):
    """A probe could not produce a usable snapshot.

    Raised by probes and by :meth:`MeasuredResourceGuard.snapshot`; the guard's
    ``can_start`` converts it into a refusal instead of propagating it.
    """


@dataclass(frozen=True, slots=True)
class ResourceLimits:
    """Ceilings and floors for one machine profile; ``0`` disables a limit.

    A measurement exactly *at* a ceiling (or at the disk floor) is already a
    refusal: the guard refuses at the boundary rather than one tick past it, so
    a limit is a hard edge and not a warning level.

    ``max_rss_mb`` bounds *system* RAM in use (``total - available``).
    ``max_process_rss_mb`` bounds *this* process's high-water RSS; on the M2 the
    orchestrator is the process that loads data and runs handlers, so the second
    is the more precise control. Both are reported in MiB.
    """

    max_rss_mb: float = 0.0
    max_load_1m: float = 0.0
    min_free_disk_mb: float = 0.0
    max_process_rss_mb: float = 0.0

    def __post_init__(self) -> None:
        for name in ("max_rss_mb", "max_load_1m", "min_free_disk_mb", "max_process_rss_mb"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
            if float(value) < 0.0:
                raise ValueError(f"{name} must be >= 0 (0 disables the limit)")

    @property
    def enabled(self) -> bool:
        """True when at least one limit is active."""
        return any(
            float(getattr(self, name)) > 0.0
            for name in ("max_rss_mb", "max_load_1m", "min_free_disk_mb", "max_process_rss_mb")
        )


#: Safe default for the M2 8 GB target: a heavy job may grow the orchestrator to
#: 4.5 GiB, total system use may not pass 6 GiB, load must stay below 6 on 8
#: cores, and the data volume keeps at least 2 GiB free.
M2_8GB_PROFILE: Final[ResourceLimits] = ResourceLimits(
    max_rss_mb=6000.0,
    max_load_1m=6.0,
    min_free_disk_mb=2048.0,
    max_process_rss_mb=4500.0,
)

#: Owner-supervised exploration only. Every limit is still finite and below the
#: machine's capacity, so it is permissive rather than unbounded: 7.3 GiB system
#: use, 6.8 GiB process, load 7.5, and a 512 MiB disk floor.
EXPLORATORY_PROFILE: Final[ResourceLimits] = ResourceLimits(
    max_rss_mb=7500.0,
    max_load_1m=7.5,
    min_free_disk_mb=512.0,
    max_process_rss_mb=7000.0,
)


@dataclass(frozen=True, slots=True)
class ResourceSnapshot:
    """One probe reading. All memory values are MiB, ``timestamp`` is the
    probe clock (epoch seconds by default)."""

    timestamp: float
    total_ram_mb: float
    available_ram_mb: float
    load1: float
    free_disk_mb: float
    process_rss_mb: float

    def __post_init__(self) -> None:
        for name in (
            "timestamp",
            "total_ram_mb",
            "available_ram_mb",
            "load1",
            "free_disk_mb",
            "process_rss_mb",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite")
        if self.total_ram_mb < 0.0 or self.available_ram_mb < 0.0:
            raise ValueError("RAM values must be >= 0")
        if self.load1 < 0.0 or self.free_disk_mb < 0.0 or self.process_rss_mb < 0.0:
            raise ValueError("load, disk and RSS values must be >= 0")
        if self.available_ram_mb > self.total_ram_mb:
            raise ValueError("available_ram_mb cannot exceed total_ram_mb")

    @property
    def used_ram_mb(self) -> float:
        """System RAM in use, floored at zero so a rounding skew cannot go negative."""
        return max(self.total_ram_mb - self.available_ram_mb, 0.0)

    def to_dict(self) -> dict[str, float]:
        return {
            "timestamp": float(self.timestamp),
            "total_ram_mb": round(float(self.total_ram_mb), 3),
            "available_ram_mb": round(float(self.available_ram_mb), 3),
            "used_ram_mb": round(self.used_ram_mb, 3),
            "load1": round(float(self.load1), 3),
            "free_disk_mb": round(float(self.free_disk_mb), 3),
            "process_rss_mb": round(float(self.process_rss_mb), 3),
        }


@runtime_checkable
class Probe(Protocol):
    """Source of :class:`ResourceSnapshot` values."""

    def snapshot(self) -> ResourceSnapshot: ...


def rss_mb_from_ru_maxrss(ru_maxrss: float, *, platform: str | None = None) -> float:
    """Convert ``resource.getrusage().ru_maxrss`` to MiB.

    macOS and the BSDs report **bytes**, Linux reports **KiB**. ``platform``
    defaults to ``sys.platform`` and exists so the branch is testable on one
    machine. A negative reading is clamped to zero.
    """
    value = float(ru_maxrss)
    if not math.isfinite(value):
        raise ValueError("ru_maxrss must be finite")
    if value < 0.0:
        return 0.0
    name = sys.platform if platform is None else platform
    if name.startswith(_RU_MAXRSS_BYTES_PLATFORMS):
        return value / _MIB
    return value / 1024.0


@dataclass(slots=True)
class SystemProbe:
    """Stdlib probe: ``resource``, ``shutil``, ``os`` and ``time`` only.

    ``disk_path`` is the volume whose free space matters — pass the data or
    artefact root, not whichever directory the process happens to be in. The
    default is the filesystem root.

    Platform notes, deliberately written down because they are limitations and
    not implementation details:

    * **Total RAM** comes from ``SC_PHYS_PAGES * SC_PAGE_SIZE``. When
      ``os.sysconf`` does not expose those (or reports nonsense) the probe falls
      back to :data:`ASSUMED_TOTAL_RAM_MB` = 8192 MiB.
    * **Available RAM** comes from ``SC_AVPHYS_PAGES`` where the platform
      exposes it (Linux). macOS raises ``ValueError`` for that name, so the
      fallback is ``total - own RSS``. That number ignores every other process
      and every byte the kernel accounts elsewhere; it is an *optimistic upper
      bound* on what is actually available. The guard's other limits exist
      partly because this one is weak.
    * **Load** is ``os.getloadavg()[0]``, the 1-minute average over runnable and
      uninterruptible tasks. On a machine without load average the probe fails
      and the guard fails closed, which is the intended behaviour.
    * **Process RSS** is the high-water mark from ``RUSAGE_SELF``. It never
      decreases, so it measures "the largest this process has ever been", not
      current residency. That is what a memory ceiling wants, but it cannot show
      a leak being freed.
    * No subprocess and no network call happens here, on any path.
    """

    disk_path: Path = field(default_factory=lambda: Path(os.path.abspath(os.sep)))
    clock: Callable[[], float] = time.time

    def snapshot(self) -> ResourceSnapshot:
        try:
            process_rss_mb = self.process_rss_mb()
            total_ram_mb = self.total_ram_mb()
            available_ram_mb = self.available_ram_mb(total_ram_mb, process_rss_mb)
            load1 = float(os.getloadavg()[0])
            free_disk_mb = self.free_disk_mb()
            timestamp = float(self.clock())
        except ResourceProbeError:
            raise
        except Exception as exc:  # pragma: no cover - depends on the host platform
            raise ResourceProbeError(f"{type(exc).__name__}: {exc}") from exc
        return ResourceSnapshot(
            timestamp=timestamp,
            total_ram_mb=total_ram_mb,
            available_ram_mb=available_ram_mb,
            load1=load1,
            free_disk_mb=free_disk_mb,
            process_rss_mb=process_rss_mb,
        )

    @staticmethod
    def total_ram_mb() -> float:
        """Physical RAM in MiB, or :data:`ASSUMED_TOTAL_RAM_MB` when unknown."""
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return ASSUMED_TOTAL_RAM_MB
        if pages <= 0 or page_size <= 0:
            return ASSUMED_TOTAL_RAM_MB
        return float(pages * page_size) / _MIB

    @staticmethod
    def available_ram_mb(total_ram_mb: float, process_rss_mb: float) -> float:
        """Available RAM in MiB; see the class docstring for the macOS fallback."""
        try:
            pages = os.sysconf("SC_AVPHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
        except (ValueError, OSError, AttributeError):
            return max(total_ram_mb - process_rss_mb, 0.0)
        if pages < 0 or page_size <= 0:
            return max(total_ram_mb - process_rss_mb, 0.0)
        return min(float(pages * page_size) / _MIB, total_ram_mb)

    def free_disk_mb(self) -> float:
        """Free space on ``disk_path`` in MiB."""
        return float(shutil.disk_usage(self.disk_path).free) / _MIB

    @staticmethod
    def process_rss_mb() -> float:
        """This process's high-water RSS in MiB, via ``getrusage``."""
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss_mb_from_ru_maxrss(usage)


@dataclass(frozen=True, slots=True)
class ResourceSample:
    """What one ``measure`` block cost, JSON-safe for the API layer later.

    ``rss_delta_mb`` is the growth of the high-water mark, not the area under the
    curve; a job that allocates and frees 1 GiB in the middle still shows it, a
    job that stays flat shows zero.
    """

    kind: str
    started_at: float
    elapsed_seconds: float
    cpu_seconds: float
    rss_start_mb: float | None
    rss_peak_mb: float | None
    rss_delta_mb: float | None
    ok: bool
    error: str = ""
    probe_error: str = ""

    def to_dict(self) -> dict[str, float | str | bool | None]:
        return {
            "kind": self.kind,
            "started_at": round(float(self.started_at), 3),
            "elapsed_seconds": round(float(self.elapsed_seconds), 6),
            "cpu_seconds": round(float(self.cpu_seconds), 6),
            "rss_start_mb": _round_optional(self.rss_start_mb),
            "rss_peak_mb": _round_optional(self.rss_peak_mb),
            "rss_delta_mb": _round_optional(self.rss_delta_mb),
            "ok": self.ok,
            "error": self.error,
            "probe_error": self.probe_error,
        }


def _round_optional(value: float | None, digits: int = 3) -> float | None:
    return None if value is None else round(float(value), digits)


class Measurement:
    """Context manager returned by :meth:`MeasuredResourceGuard.measure`.

    ``sample`` is ``None`` until the block exits, then holds the
    :class:`ResourceSample` even when the block raised — a failed job still has
    a measured cost, which is exactly when the number is interesting. An
    exception from the block propagates unchanged; the measurement never masks
    it.
    """

    def __init__(self, guard: MeasuredResourceGuard, kind: str) -> None:
        self._guard = guard
        self.kind = kind
        self.sample: ResourceSample | None = None
        self._started_at = 0.0
        self._wall_started = 0.0
        self._cpu_started = 0.0
        self._rss_start: float | None = None
        self._start_error = ""

    def __enter__(self) -> Self:
        self._started_at = float(self._guard.wall_clock())
        self._wall_started = float(self._guard.clock())
        self._cpu_started = float(self._guard.cpu_clock())
        self._rss_start, self._start_error = self._guard._probe_process_rss()
        return self

    def __exit__(self, exc_type: object, exc: BaseException | None, tb: object) -> None:
        elapsed = max(float(self._guard.clock()) - self._wall_started, 0.0)
        cpu = max(float(self._guard.cpu_clock()) - self._cpu_started, 0.0)
        rss_peak, peak_error = self._guard._probe_process_rss()
        probe_error = "; ".join(part for part in (self._start_error, peak_error) if part)
        delta: float | None = None
        if self._rss_start is not None and rss_peak is not None:
            delta = max(rss_peak - self._rss_start, 0.0)
        self.sample = ResourceSample(
            kind=self.kind,
            started_at=self._started_at,
            elapsed_seconds=elapsed,
            cpu_seconds=cpu,
            rss_start_mb=self._rss_start,
            rss_peak_mb=rss_peak,
            rss_delta_mb=delta,
            ok=exc_type is None,
            error="" if exc is None else f"{type(exc).__name__}: {exc}",
            probe_error=probe_error,
        )
        self._guard._record_sample(self.sample)


class MeasuredResourceGuard:
    """The ``ResourceGuard`` veto backed by real probe readings.

    ``can_start(kind)`` measures at most once per ``cache_seconds`` and refuses
    on the first breached limit in this fixed order: ``max_rss_mb``,
    ``max_process_rss_mb``, ``max_load_1m``, ``min_free_disk_mb``. The order is
    part of the contract so an audit log is reproducible; it is not a severity
    ranking.

    ``kind`` is accepted for the protocol and not used in the decision: the
    machine is shared, so a heavy training job and a light paper step face the
    same limits. Per-kind budgets belong in the caller's limits, not in a hidden
    table here.

    Failures are refusals, never exceptions: a probe that raises, returns the
    wrong type or produces a non-finite value yields
    ``(False, "probe_failed: TypeError: ...")``. The failing probe is cached for
    the same short window so a broken probe cannot be hammered by a queue drain.
    """

    def __init__(
        self,
        limits: ResourceLimits | None = None,
        *,
        probe: Probe | None = None,
        clock: Callable[[], float] = time.monotonic,
        cpu_clock: Callable[[], float] = time.process_time,
        wall_clock: Callable[[], float] = time.time,
        cache_seconds: float = 1.0,
        sample_limit: int = 64,
    ) -> None:
        if limits is not None and not isinstance(limits, ResourceLimits):
            raise TypeError("limits must be a ResourceLimits or None")
        if isinstance(cache_seconds, bool) or not isinstance(cache_seconds, (int, float)):
            raise TypeError("cache_seconds must be a number")
        if not math.isfinite(float(cache_seconds)):
            raise ValueError("cache_seconds must be finite")
        if float(cache_seconds) < 0.0:
            raise ValueError("cache_seconds must be >= 0 (0 disables caching)")
        if isinstance(sample_limit, bool) or not isinstance(sample_limit, int):
            raise TypeError("sample_limit must be an int")
        if sample_limit < 0:
            raise ValueError("sample_limit must be >= 0 (0 disables the ring)")

        self.limits: ResourceLimits = ResourceLimits() if limits is None else limits
        self.probe: Probe = SystemProbe() if probe is None else probe
        self.clock = clock
        self.cpu_clock = cpu_clock
        self.wall_clock = wall_clock
        self.cache_seconds: float = float(cache_seconds)
        self.sample_limit: int = sample_limit
        self._lock = threading.Lock()
        self._snapshot: ResourceSnapshot | None = None
        self._measured_at: float | None = None
        self._probe_error: str | None = None
        self._samples: deque[ResourceSample] = deque(maxlen=max(sample_limit, 1))

    # ------------------------------------------------------------------ public

    def can_start(self, kind: str) -> tuple[bool, str]:
        """Allow or refuse starting ``kind``; never raises.

        A refusal's reason names the limit and the measured value, for example
        ``"max_load_1m=6 exceeded: load1=7.42"``, so the orchestrator's
        ``resource_refused`` event explains itself without a second lookup.
        """
        try:
            snapshot, error = self._current()
        except Exception as exc:  # noqa: BLE001 - a probe bug must not kill the loop
            return False, f"probe_failed: {type(exc).__name__}: {exc}"
        if snapshot is None:
            return False, f"probe_failed: {error or 'no snapshot'}"
        return self._evaluate(snapshot)

    def snapshot(self, *, force: bool = False) -> ResourceSnapshot:
        """Current reading, from cache unless ``force``; raises on probe failure."""
        with self._lock:
            now = float(self.clock())
            if force or not self._is_fresh(now):
                self._measure_locked(now)
            if self._snapshot is None:
                raise ResourceProbeError(self._probe_error or "no snapshot")
            return self._snapshot

    @property
    def last_snapshot(self) -> ResourceSnapshot | None:
        """Most recent successful reading, without probing."""
        with self._lock:
            return self._snapshot

    @property
    def last_error(self) -> str | None:
        """Reason of the most recent probe failure, or ``None``."""
        with self._lock:
            return self._probe_error

    def measure(self, kind: str) -> Measurement:
        """Time ``kind`` and record wall, CPU and high-water RSS growth."""
        return Measurement(self, kind)

    @property
    def samples(self) -> tuple[ResourceSample, ...]:
        """Recent samples, oldest first, bounded by ``sample_limit``."""
        with self._lock:
            return tuple(self._samples)

    # --------------------------------------------------------------- internals

    def _is_fresh(self, now: float) -> bool:
        if self._measured_at is None:
            return False
        age = now - self._measured_at
        # A backwards clock is not a reason to trust an old reading.
        return 0.0 <= age < self.cache_seconds

    def _current(self) -> tuple[ResourceSnapshot | None, str | None]:
        with self._lock:
            now = float(self.clock())
            if not self._is_fresh(now):
                self._measure_locked(now)
            return self._snapshot, self._probe_error

    def _measure_locked(self, now: float) -> None:
        self._measured_at = now
        try:
            snapshot = self.probe.snapshot()
            if not isinstance(snapshot, ResourceSnapshot):
                raise ResourceProbeError(
                    f"probe returned {type(snapshot).__name__}, expected ResourceSnapshot"
                )
        except Exception as exc:  # noqa: BLE001 - any probe failure becomes a refusal
            self._snapshot = None
            self._probe_error = f"{type(exc).__name__}: {exc}"
            return
        self._snapshot = snapshot
        self._probe_error = None

    def _evaluate(self, snapshot: ResourceSnapshot) -> tuple[bool, str]:
        limits = self.limits
        used = snapshot.used_ram_mb
        if limits.max_rss_mb and used >= limits.max_rss_mb:
            return False, (f"max_rss_mb={_fmt(limits.max_rss_mb)} exceeded: used_ram_mb={used:.1f}")
        if limits.max_process_rss_mb and snapshot.process_rss_mb >= limits.max_process_rss_mb:
            return False, (
                f"max_process_rss_mb={_fmt(limits.max_process_rss_mb)} exceeded: "
                f"process_rss_mb={snapshot.process_rss_mb:.1f}"
            )
        if limits.max_load_1m and snapshot.load1 >= limits.max_load_1m:
            return False, (
                f"max_load_1m={_fmt(limits.max_load_1m)} exceeded: load1={snapshot.load1:.2f}"
            )
        if limits.min_free_disk_mb and snapshot.free_disk_mb <= limits.min_free_disk_mb:
            return False, (
                f"min_free_disk_mb={_fmt(limits.min_free_disk_mb)} breached: "
                f"free_disk_mb={snapshot.free_disk_mb:.1f}"
            )
        return True, ""

    def _probe_process_rss(self) -> tuple[float | None, str]:
        """Own-process RSS for ``measure``; a probe failure is reported, not raised."""
        try:
            return float(self.probe.snapshot().process_rss_mb), ""
        except Exception as exc:  # noqa: BLE001 - a failed measurement is data, not a crash
            return None, f"{type(exc).__name__}: {exc}"

    def _record_sample(self, sample: ResourceSample) -> None:
        if self.sample_limit <= 0:
            return
        with self._lock:
            self._samples.append(sample)


def _fmt(value: float) -> str:
    """Format a limit without a trailing ``.0`` so the reason reads naturally."""
    return f"{float(value):g}"
