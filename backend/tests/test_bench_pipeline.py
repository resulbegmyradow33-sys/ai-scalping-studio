"""Tests for the M2 measurement harness.

The assertions pin the parts that must not drift: plain median/min/max
aggregation, finite RSS maths, JSON-safe output, failure isolation, the claims
block and the synthetic default that never opens the real store. No test needs
market data, a network or the 17-year store.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from scalping.bench import (
    Stage,
    StageError,
    dataset_build_stage,
    discovery_smoke_stage,
    evidence_build_stage,
    features_build_stage,
    measure_stage,
    paper_advance_stage,
    run_suite,
)
from scalping.data.store import ParquetStore


class _Sequence:
    """Callable returning the next queued value; fails loudly when exhausted."""

    def __init__(self, values: list[Any]) -> None:
        self._values = iter(values)

    def __call__(self) -> Any:
        try:
            return next(self._values)
        except StopIteration as exc:  # pragma: no cover - a test bug, not a product path
            raise AssertionError("sequence exhausted") from exc


def _boom() -> None:
    raise RuntimeError("boom")


def _noop() -> dict[str, int]:
    return {"rows": 3}


def test_aggregation_is_plain_median_min_max_with_injected_timings() -> None:
    # Three repeats: wall 2.0, 3.0, 7.0.
    clock = _Sequence([0.0, 2.0, 10.0, 13.0, 20.0, 27.0])
    cpu = _Sequence(
        [
            (0.0, 0.0),
            (1.0, 0.5),
            (10.0, 0.0),
            (11.0, 0.5),
            (20.0, 0.0),
            (22.0, 1.0),
        ]
    )
    rss = _Sequence([100.0, 140.0, 140.0, 150.0, 150.0, 150.0])

    measurement = measure_stage(
        "fake",
        _noop,
        repeats=3,
        clock=clock,
        cpu_times=cpu,
        rss_mb=rss,
    )

    assert measurement.ok is True
    assert measurement.measured_repeats == 3
    assert measurement.wall_seconds_min == pytest.approx(2.0)
    assert measurement.wall_seconds_median == pytest.approx(3.0)
    assert measurement.wall_seconds_max == pytest.approx(7.0)
    # Per-repeat user deltas are 1, 1, 2; system deltas 0.5, 0.5, 1.0.
    assert measurement.cpu_user_seconds_median == pytest.approx(1.0)
    assert measurement.cpu_system_seconds_median == pytest.approx(0.5)
    assert measurement.cpu_total_seconds_median == pytest.approx(1.5)
    assert measurement.rows == 3
    assert measurement.bytes_processed is None
    assert [run.wall_seconds for run in measurement.runs] == [2.0, 3.0, 7.0]


def test_one_sample_median_equals_min_and_max() -> None:
    clock = _Sequence([5.0, 7.5])
    measurement = measure_stage("single", _noop, repeats=1, clock=clock)

    assert measurement.measured_repeats == 1
    assert measurement.wall_seconds_median == pytest.approx(2.5)
    assert measurement.wall_seconds_min == measurement.wall_seconds_median
    assert measurement.wall_seconds_max == measurement.wall_seconds_median


def test_peak_rss_delta_is_the_largest_growth_and_finite() -> None:
    rss = _Sequence([100.0, 140.0, 140.0, 150.0])
    measurement = measure_stage("rss", _noop, repeats=2, rss_mb=rss)

    assert measurement.rss_start_mb == pytest.approx(100.0)
    assert measurement.rss_peak_mb == pytest.approx(150.0)
    assert measurement.rss_delta_mb == pytest.approx(40.0)
    assert measurement.rss_delta_mb is not None
    assert math.isfinite(measurement.rss_delta_mb)

    # The production probe is also finite on this host; it may be zero.
    live = measure_stage("live-rss", _noop, repeats=1)
    assert live.rss_peak_mb is not None and math.isfinite(live.rss_peak_mb)
    assert live.rss_delta_mb is not None and live.rss_delta_mb >= 0.0


def test_rows_bytes_and_optional_disk_delta(tmp_path: Path) -> None:
    def write_result() -> dict[str, int]:
        (tmp_path / "payload.bin").write_bytes(b"x" * 4096)
        return {"rows": 11, "bytes": 4096}

    measurement = measure_stage("disk", write_result, repeats=1, disk_path=tmp_path)

    assert measurement.rows == 11
    assert measurement.bytes_processed == 4096
    assert measurement.disk_delta_mb is not None
    assert measurement.disk_delta_mb > 0.0


def test_measurement_and_report_are_json_safe(tmp_path: Path) -> None:
    measurement = measure_stage("json", _noop, repeats=1)
    encoded = json.dumps(measurement.to_dict(), allow_nan=False)
    assert '"wall_seconds"' in encoded

    out = tmp_path / "report.json"
    report = run_suite([Stage("json", _noop)], out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == json.loads(json.dumps(report, allow_nan=False))
    assert loaded["schema"] == "scalping.bench.suite/1"


def test_failed_stage_is_recorded_and_does_not_abort_the_suite(tmp_path: Path) -> None:
    report = run_suite(
        [
            Stage("before", _noop),
            Stage("broken", _boom),
            Stage("after", _noop),
        ],
        tmp_path / "report.json",
    )
    stages = {stage["name"]: stage for stage in report["stages"]}

    assert stages["before"]["ok"] is True
    assert stages["broken"]["ok"] is False
    assert "RuntimeError: boom" in stages["broken"]["error"]
    assert stages["after"]["ok"] is True
    assert report["summary"]["failed"] == 1
    assert report["summary"]["ok"] == 2


def test_strict_suite_raises_instead_of_writing_a_report(tmp_path: Path) -> None:
    out = tmp_path / "strict.json"

    with pytest.raises(StageError, match="broken"):
        run_suite([Stage("broken", _boom)], out, strict=True)

    assert not out.exists()


def test_suite_claims_block_is_explicit(tmp_path: Path) -> None:
    report = run_suite([Stage("claim", _noop)], tmp_path / "report.json")
    claims = report["claims"]

    assert claims["kind"] == "local_measurement"
    assert claims["validation"] is False
    assert claims["profitability"] is False
    assert claims["m2_acceptance"] is False
    assert "not validate" in claims["text"]
    assert report["environment"]["python"]
    assert report["environment"]["cpu_count"] is None or report["environment"]["cpu_count"] >= 1


def test_synthetic_default_never_reads_the_named_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sentinel = tmp_path / "must-not-be-read"
    seen: list[Path] = []
    original = ParquetStore.read

    def recording_read(self: ParquetStore, symbol: str, timeframe: str = "M1") -> Any:
        seen.append(Path(self.root).resolve())
        return original(self, symbol, timeframe)

    monkeypatch.setattr(ParquetStore, "read", recording_read)

    stages = [
        features_build_stage(sentinel),
        dataset_build_stage(sentinel),
        discovery_smoke_stage(sentinel, max_trials=1),
        evidence_build_stage(sentinel),
        paper_advance_stage(sentinel),
    ]
    report = run_suite(stages, tmp_path / "report.json")

    assert [stage["ok"] for stage in report["stages"]] == [True] * len(stages)
    assert sentinel.resolve() not in seen
    assert seen, "discovery/paper should read their own temporary synthetic stores"


def test_real_store_flag_is_the_only_path_that_touches_the_store(tmp_path: Path) -> None:
    empty = tmp_path / "empty-store"
    empty.mkdir()

    with pytest.raises(FileNotFoundError):
        features_build_stage(empty, real_store=True, rows=10)


def test_no_network_socket_is_opened(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> socket.socket:
        raise AssertionError("the bench harness must not open a socket")

    monkeypatch.setattr(socket, "socket", forbidden)
    report = run_suite([features_build_stage(tmp_path / "unused")], tmp_path / "report.json")
    assert report["stages"][0]["ok"] is True


def test_cli_writes_the_same_json_it_prints(tmp_path: Path) -> None:
    backend = Path(__file__).resolve().parents[1]
    out = tmp_path / "cli-report.json"
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/bench_pipeline.py",
            "--out",
            str(out),
            "--max-trials",
            "1",
        ],
        cwd=backend,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    printed = json.loads(proc.stdout)
    written = json.loads(out.read_text(encoding="utf-8"))
    assert printed == written
    assert [stage["name"] for stage in printed["stages"]] == [
        "features_build",
        "dataset_build",
        "discovery_smoke",
        "evidence_build",
        "paper_advance",
    ]
    assert printed["claims"]["m2_acceptance"] is False
