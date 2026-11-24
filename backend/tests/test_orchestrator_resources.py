"""Measured resource guard: limits, caching, fail-closed probes and samples."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from scalping.orchestrator import (
    JobStore,
    Orchestrator,
    ResourceGuard,
)
from scalping.orchestrator.resources import (
    ASSUMED_TOTAL_RAM_MB,
    EXPLORATORY_PROFILE,
    M2_8GB_PROFILE,
    MeasuredResourceGuard,
    Probe,
    ResourceLimits,
    ResourceProbeError,
    ResourceSnapshot,
    SystemProbe,
    rss_mb_from_ru_maxrss,
)

T0 = 1_000.0


class _Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self, now: float = T0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class _FakeProbe:
    """Returns queued snapshots (or raises queued exceptions) and counts calls."""

    def __init__(self, *items: ResourceSnapshot | Exception) -> None:
        self.items = list(items)
        self.calls = 0

    def snapshot(self) -> ResourceSnapshot:
        self.calls += 1
        index = min(self.calls - 1, len(self.items) - 1)
        item = self.items[index]
        if isinstance(item, Exception):
            raise item
        return item


def _snapshot(
    *,
    total_ram_mb: float = 8192.0,
    available_ram_mb: float = 6000.0,
    load1: float = 1.0,
    free_disk_mb: float = 100_000.0,
    process_rss_mb: float = 500.0,
    timestamp: float = T0,
) -> ResourceSnapshot:
    return ResourceSnapshot(
        timestamp=timestamp,
        total_ram_mb=total_ram_mb,
        available_ram_mb=available_ram_mb,
        load1=load1,
        free_disk_mb=free_disk_mb,
        process_rss_mb=process_rss_mb,
    )


def _guard(*items: ResourceSnapshot | Exception, **kwargs: object) -> MeasuredResourceGuard:
    """A guard over a fake probe; cache off unless the test asks for it."""
    kwargs.setdefault("cache_seconds", 0.0)
    return MeasuredResourceGuard(
        M2_8GB_PROFILE,
        probe=_FakeProbe(*items),
        clock=_Clock(),
        **kwargs,  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------- limits


def test_default_limits_disable_everything() -> None:
    limits = ResourceLimits()
    assert not limits.enabled
    guard = MeasuredResourceGuard(
        limits,
        probe=_FakeProbe(
            _snapshot(available_ram_mb=1.0, load1=99.0, free_disk_mb=1.0, process_rss_mb=99_000.0)
        ),
        cache_seconds=0.0,
    )
    assert guard.can_start("train") == (True, "")


@pytest.mark.parametrize(
    ("limits", "snapshot"),
    [
        (
            ResourceLimits(max_rss_mb=6000.0),
            _snapshot(total_ram_mb=8192.0, available_ram_mb=2192.0),
        ),
        (
            ResourceLimits(max_process_rss_mb=4500.0),
            _snapshot(process_rss_mb=4500.0),
        ),
        (ResourceLimits(max_load_1m=6.0), _snapshot(load1=6.0)),
        (ResourceLimits(min_free_disk_mb=2048.0), _snapshot(free_disk_mb=2048.0)),
    ],
)
def test_each_limit_refuses_at_the_boundary(
    limits: ResourceLimits, snapshot: ResourceSnapshot
) -> None:
    guard = MeasuredResourceGuard(limits, probe=_FakeProbe(snapshot), cache_seconds=0.0)
    allowed, reason = guard.can_start("train")
    assert allowed is False
    assert reason


@pytest.mark.parametrize(
    ("limits", "snapshot"),
    [
        (
            ResourceLimits(max_rss_mb=6000.0),
            _snapshot(total_ram_mb=8192.0, available_ram_mb=2192.4),
        ),
        (ResourceLimits(max_process_rss_mb=4500.0), _snapshot(process_rss_mb=4499.9)),
        (ResourceLimits(max_load_1m=6.0), _snapshot(load1=5.99)),
        (ResourceLimits(min_free_disk_mb=2048.0), _snapshot(free_disk_mb=2048.1)),
    ],
)
def test_each_limit_allows_just_inside_the_boundary(
    limits: ResourceLimits, snapshot: ResourceSnapshot
) -> None:
    guard = MeasuredResourceGuard(limits, probe=_FakeProbe(snapshot), cache_seconds=0.0)
    assert guard.can_start("train") == (True, "")


def test_reason_names_the_limit_and_the_measurement() -> None:
    cases = [
        (
            ResourceLimits(max_rss_mb=6000.0),
            _snapshot(total_ram_mb=8192.0, available_ram_mb=1792.0),
            "max_rss_mb=6000",
            "used_ram_mb=6400.0",
        ),
        (
            ResourceLimits(max_process_rss_mb=4500.0),
            _snapshot(process_rss_mb=4602.5),
            "max_process_rss_mb=4500",
            "process_rss_mb=4602.5",
        ),
        (
            ResourceLimits(max_load_1m=6.0),
            _snapshot(load1=7.42),
            "max_load_1m=6",
            "load1=7.42",
        ),
        (
            ResourceLimits(min_free_disk_mb=2048.0),
            _snapshot(free_disk_mb=1024.0),
            "min_free_disk_mb=2048",
            "free_disk_mb=1024.0",
        ),
    ]
    for limits, snapshot, limit_text, value_text in cases:
        guard = MeasuredResourceGuard(limits, probe=_FakeProbe(snapshot), cache_seconds=0.0)
        allowed, reason = guard.can_start("train")
        assert allowed is False
        assert limit_text in reason
        assert value_text in reason


def test_first_breached_limit_in_the_documented_order_wins() -> None:
    guard = MeasuredResourceGuard(
        M2_8GB_PROFILE,
        probe=_FakeProbe(_snapshot(total_ram_mb=8192.0, available_ram_mb=1000.0, load1=7.9)),
        cache_seconds=0.0,
    )
    _, reason = guard.can_start("train")
    assert reason.startswith("max_rss_mb=")


def test_disabled_limit_is_not_evaluated() -> None:
    limits = ResourceLimits(max_rss_mb=0.0, max_load_1m=6.0, min_free_disk_mb=0.0)
    guard = MeasuredResourceGuard(
        limits,
        probe=_FakeProbe(_snapshot(available_ram_mb=1.0, load1=1.0, free_disk_mb=1.0)),
        cache_seconds=0.0,
    )
    assert guard.can_start("train") == (True, "")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_rss_mb": -1.0},
        {"max_load_1m": -0.1},
        {"min_free_disk_mb": -5.0},
        {"max_process_rss_mb": -2.0},
        {"max_load_1m": float("nan")},
        {"max_rss_mb": float("inf")},
        {"max_rss_mb": True},
        {"max_load_1m": "6"},
    ],
)
def test_invalid_limits_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises((TypeError, ValueError)):
        ResourceLimits(**kwargs)  # type: ignore[arg-type]


# -------------------------------------------------------------- probe failures


def test_probe_failure_fails_closed_with_the_reason() -> None:
    guard = MeasuredResourceGuard(
        M2_8GB_PROFILE, probe=_FakeProbe(RuntimeError("boom")), cache_seconds=0.0
    )
    allowed, reason = guard.can_start("train")
    assert allowed is False
    assert reason == "probe_failed: RuntimeError: boom"


def test_probe_returning_the_wrong_type_fails_closed() -> None:
    class _BadProbe:
        def snapshot(self) -> ResourceSnapshot:  # type: ignore[return-value]
            return {"load1": 0.0}  # type: ignore[return-value]

    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=_BadProbe(), cache_seconds=0.0)
    allowed, reason = guard.can_start("train")
    assert allowed is False
    assert reason.startswith("probe_failed: ")
    assert "expected ResourceSnapshot" in reason


def test_non_finite_snapshot_fails_closed() -> None:
    class _NanProbe:
        def snapshot(self) -> ResourceSnapshot:
            return ResourceSnapshot(
                timestamp=T0,
                total_ram_mb=8192.0,
                available_ram_mb=6000.0,
                load1=float("nan"),
                free_disk_mb=100_000.0,
                process_rss_mb=500.0,
            )

    guard = MeasuredResourceGuard(
        ResourceLimits(max_load_1m=6.0), probe=_NanProbe(), cache_seconds=0.0
    )
    allowed, reason = guard.can_start("train")
    assert allowed is False
    assert "probe_failed: ValueError" in reason


def test_failed_probe_is_cached_for_the_window() -> None:
    clock = _Clock()
    probe = _FakeProbe(RuntimeError("broken"), _snapshot())
    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=probe, clock=clock, cache_seconds=1.0)
    assert guard.can_start("train")[0] is False
    clock.advance(0.4)
    assert guard.can_start("train")[0] is False
    assert probe.calls == 1
    clock.advance(0.7)
    assert guard.can_start("train") == (True, "")
    assert probe.calls == 2


def test_snapshot_raises_a_typed_error_on_probe_failure() -> None:
    guard = MeasuredResourceGuard(
        M2_8GB_PROFILE, probe=_FakeProbe(RuntimeError("no")), cache_seconds=0.0
    )
    with pytest.raises(ResourceProbeError, match="RuntimeError: no"):
        guard.snapshot()
    assert guard.last_error == "RuntimeError: no"
    assert guard.last_snapshot is None


# --------------------------------------------------------------------- caching


def test_cache_reuses_one_measurement_across_a_queue_drain() -> None:
    clock = _Clock()
    probe = _FakeProbe(_snapshot(), _snapshot())
    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=probe, clock=clock, cache_seconds=1.0)
    for _ in range(5):
        assert guard.can_start("train") == (True, "")
        clock.advance(0.1)
    assert probe.calls == 1
    clock.advance(0.6)
    assert guard.can_start("train") == (True, "")
    assert probe.calls == 2


def test_zero_cache_measures_every_call() -> None:
    probe = _FakeProbe(_snapshot(), _snapshot(), _snapshot())
    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=probe, clock=_Clock(), cache_seconds=0.0)
    for _ in range(3):
        guard.can_start("train")
    assert probe.calls == 3


def test_backwards_clock_forces_a_fresh_measurement() -> None:
    clock = _Clock()
    probe = _FakeProbe(_snapshot(), _snapshot())
    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=probe, clock=clock, cache_seconds=1.0)
    guard.can_start("train")
    clock.now -= 5.0
    guard.can_start("train")
    assert probe.calls == 2


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf")])
def test_invalid_cache_seconds_are_rejected(value: float) -> None:
    with pytest.raises(ValueError):
        MeasuredResourceGuard(M2_8GB_PROFILE, cache_seconds=value)


def test_invalid_sample_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        MeasuredResourceGuard(M2_8GB_PROFILE, sample_limit=-1)
    with pytest.raises(TypeError):
        MeasuredResourceGuard(M2_8GB_PROFILE, sample_limit=1.5)  # type: ignore[arg-type]


# --------------------------------------------------------------- RSS conversion


def test_ru_maxrss_is_bytes_on_macos() -> None:
    assert rss_mb_from_ru_maxrss(16 * 1024 * 1024, platform="darwin") == pytest.approx(16.0)


def test_ru_maxrss_is_kib_on_linux() -> None:
    assert rss_mb_from_ru_maxrss(16 * 1024, platform="linux") == pytest.approx(16.0)


def test_ru_maxrss_clamps_negative_and_rejects_non_finite() -> None:
    assert rss_mb_from_ru_maxrss(-1.0, platform="darwin") == 0.0
    with pytest.raises(ValueError):
        rss_mb_from_ru_maxrss(float("nan"), platform="darwin")


def test_system_probe_reads_this_machine() -> None:
    probe = SystemProbe()
    snapshot = probe.snapshot()
    assert snapshot.total_ram_mb > 0.0
    assert snapshot.available_ram_mb >= 0.0
    assert snapshot.load1 >= 0.0
    assert snapshot.free_disk_mb > 0.0
    assert snapshot.process_rss_mb > 0.0


def test_system_probe_disk_path_is_configurable(tmp_path: Path) -> None:
    probe = SystemProbe(disk_path=tmp_path)
    assert probe.free_disk_mb() > 0.0


def test_assumed_total_ram_is_the_target_machine_class() -> None:
    assert ASSUMED_TOTAL_RAM_MB == 8192.0


# ------------------------------------------------------------------ measuring


def test_measure_records_elapsed_cpu_and_finite_numbers() -> None:
    clock = _Clock()
    cpu = _Clock(0.0)
    wall = _Clock(1_700_000_000.0)
    probe = _FakeProbe(_snapshot(process_rss_mb=100.0), _snapshot(process_rss_mb=350.0))
    guard = MeasuredResourceGuard(
        M2_8GB_PROFILE,
        probe=probe,
        clock=clock,
        cpu_clock=cpu,
        wall_clock=wall,
        cache_seconds=0.0,
        sample_limit=4,
    )

    with guard.measure("train") as measurement:
        clock.advance(2.5)
        cpu.advance(1.75)

    sample = measurement.sample
    assert sample is not None
    assert sample.kind == "train"
    assert sample.ok is True
    assert sample.error == ""
    assert sample.started_at == pytest.approx(1_700_000_000.0)
    assert sample.elapsed_seconds == pytest.approx(2.5)
    assert sample.cpu_seconds == pytest.approx(1.75)
    assert sample.rss_start_mb == pytest.approx(100.0)
    assert sample.rss_peak_mb == pytest.approx(350.0)
    assert sample.rss_delta_mb == pytest.approx(250.0)
    for value in (sample.elapsed_seconds, sample.cpu_seconds, sample.rss_delta_mb):
        assert value is not None and math.isfinite(value)
    assert guard.samples[-1] is sample


def test_measure_records_a_failed_block_without_masking_it() -> None:
    guard = _guard(_snapshot(), _snapshot(process_rss_mb=900.0))
    with pytest.raises(ValueError, match="job blew up"), guard.measure("train") as measurement:
        raise ValueError("job blew up")

    sample = measurement.sample
    assert sample is not None
    assert sample.ok is False
    assert sample.error == "ValueError: job blew up"
    assert sample.rss_peak_mb == pytest.approx(900.0)


def test_measure_survives_a_probe_failure() -> None:
    guard = _guard(RuntimeError("no rss"))
    with guard.measure("train") as measurement:
        pass

    sample = measurement.sample
    assert sample is not None
    assert sample.rss_start_mb is None
    assert sample.rss_peak_mb is None
    assert sample.rss_delta_mb is None
    assert "RuntimeError: no rss" in sample.probe_error
    assert sample.ok is True


def test_samples_ring_is_bounded_and_keeps_the_newest() -> None:
    guard = _guard(_snapshot(), sample_limit=2)
    for _ in range(3):
        with guard.measure("train") as measurement:
            pass
    assert len(guard.samples) == 2
    assert guard.samples[-1] is measurement.sample


def test_samples_ring_can_be_disabled() -> None:
    guard = _guard(_snapshot(), sample_limit=0)
    with guard.measure("train"):
        pass
    assert guard.samples == ()


def test_sample_is_json_safe() -> None:
    guard = _guard(_snapshot(), _snapshot(process_rss_mb=700.0))
    with guard.measure("train") as measurement:
        pass
    assert measurement.sample is not None
    payload = measurement.sample.to_dict()
    assert json.loads(json.dumps(payload))["kind"] == "train"
    assert payload["rss_delta_mb"] == pytest.approx(200.0)


def test_snapshot_is_json_safe() -> None:
    payload = _snapshot().to_dict()
    assert json.loads(json.dumps(payload))["used_ram_mb"] == pytest.approx(2192.0)


# ------------------------------------------------------------------- profiles


def test_default_profile_is_the_documented_safe_profile() -> None:
    assert M2_8GB_PROFILE == ResourceLimits(
        max_rss_mb=6000.0,
        max_load_1m=6.0,
        min_free_disk_mb=2048.0,
        max_process_rss_mb=4500.0,
    )


def test_exploratory_profile_is_permissive_but_bounded() -> None:
    assert EXPLORATORY_PROFILE.enabled
    assert 0.0 < EXPLORATORY_PROFILE.max_rss_mb < ASSUMED_TOTAL_RAM_MB
    assert 0.0 < EXPLORATORY_PROFILE.max_process_rss_mb < ASSUMED_TOTAL_RAM_MB
    assert 0.0 < EXPLORATORY_PROFILE.max_load_1m < 8.0
    assert EXPLORATORY_PROFILE.min_free_disk_mb > 0.0
    for name in ("max_rss_mb", "max_process_rss_mb", "max_load_1m"):
        assert float(getattr(EXPLORATORY_PROFILE, name)) >= float(getattr(M2_8GB_PROFILE, name)), (
            f"{name} should be at least as permissive as the default profile"
        )
    assert EXPLORATORY_PROFILE.min_free_disk_mb <= M2_8GB_PROFILE.min_free_disk_mb


def test_profiles_are_immutable() -> None:
    with pytest.raises(AttributeError):
        M2_8GB_PROFILE.max_rss_mb = 999_999.0  # type: ignore[misc]


# ------------------------------------------------------------ protocol wiring


def test_guard_and_probe_satisfy_the_protocols() -> None:
    guard = MeasuredResourceGuard(M2_8GB_PROFILE, probe=_FakeProbe(_snapshot()))
    assert isinstance(guard, ResourceGuard)
    assert isinstance(SystemProbe(), Probe)


def test_the_orchestrator_defers_on_a_measured_refusal(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "orchestrator.sqlite3")
    guard = MeasuredResourceGuard(
        ResourceLimits(max_load_1m=6.0),
        probe=_FakeProbe(_snapshot(load1=7.42)),
        cache_seconds=0.0,
    )
    orchestrator = Orchestrator(
        store,
        handlers={"train": lambda job, context: {"ok": True}},
        heavy_kinds=("train",),
        guard=guard,
        guard_retry_seconds=60.0,
    )
    job, _ = store.enqueue("train", dedupe_key="train:measured-guard")

    outcome = orchestrator.run_once()

    assert outcome.status == "resource_refused"
    assert outcome.job is not None
    assert outcome.job.attempts == 0
    assert "max_load_1m=6" in (outcome.job.error or "")
    events = [event for event in store.events(job.id) if event.event == "resource_refused"]
    assert len(events) == 1
    assert "load1=7.42" in events[0].detail


# ------------------------------------------------------------------- stdlib-ness


def test_resources_module_uses_no_subprocess_or_network_imports() -> None:
    source = Path("src/scalping/orchestrator/resources.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    forbidden = {"subprocess", "socket", "urllib", "http", "requests", "httpx", "psutil"}
    assert imported & forbidden == set()
    assert "__import__" not in source
    assert "os.system" not in source
    assert "os.popen" not in source
