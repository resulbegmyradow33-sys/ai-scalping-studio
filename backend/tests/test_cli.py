"""End-to-end tests for ``python -m scalping.cli``.

Every command writes exactly one JSON line to stdout and exits 0; data errors
write one JSON line to stderr and exit 2. The tests run the real module as a
subprocess so ``argparse``, stdout/stderr discipline and exit codes are all
covered, not just the handler functions.
"""

from __future__ import annotations

import http.client
import json
import select
import socket
import subprocess
import sys
import time
from dataclasses import fields
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scalping import cli
from scalping.backtest.metrics import Metrics
from scalping.data.store import ParquetStore
from scalping.features.build import FEATURE_COLUMNS, build_features

FIXTURES = Path(__file__).parent / "fixtures"


def run(*args: str, cwd: str = ".") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )


def _read_stdout_line(proc: subprocess.Popen[str], timeout: float = 30.0) -> str:
    """Read the child's first stdout line, failing instead of blocking forever."""
    assert proc.stdout is not None
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        ready, _, _ = select.select([proc.stdout], [], [], 0.1)
        if ready:
            line = proc.stdout.readline()
            if line:
                return line
        elif proc.poll() is not None:
            # EOF (or a line buffered just before an early exit).
            line = proc.stdout.readline()
            if line:
                return line
            break
    raise AssertionError(f"serve printed no JSON line (exit code {proc.poll()})")


def _stop(proc: subprocess.Popen[str], timeout: float = 15.0) -> None:
    """Terminate the child and guarantee the test cannot hang on it."""
    if proc.poll() is None:
        proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)


def test_synth_backtest_roundtrip(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-01",
        "--end",
        "2026-01-08",
        "--root",
        str(tmp_path),
        cwd=".",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(r.stdout)["rows"] > 0

    out = tmp_path / "bt.json"
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "hold",
        "--root",
        str(tmp_path),
        "--out",
        str(out),
        cwd=".",
    )
    assert r.returncode == 0, r.stderr
    assert json.loads(out.read_text())["metrics"]["trades"] == 0


def test_backtest_without_data_exits_2(tmp_path):
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "hold",
        "--root",
        str(tmp_path),
        "--out",
        str(tmp_path / "x.json"),
        cwd=".",
    )
    assert r.returncode == 2


def test_synth_prints_one_json_line_and_respects_root(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-06",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1 and r.stdout.endswith("\n")
    payload = json.loads(r.stdout)
    assert set(payload) == {"rows", "symbol", "first_ts", "last_ts"}
    assert payload["symbol"] == "XAUUSD"
    assert payload["rows"] == 2_880  # Monday and Tuesday, 1440 M1 bars each
    assert payload["first_ts"].startswith("2026-01-05T00:00:00")
    assert payload["last_ts"].startswith("2026-01-06T23:59:00")
    assert (tmp_path / "XAUUSD" / "M1").is_dir()


def test_backtest_report_metrics_keys_match_metrics(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-06",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr

    out = tmp_path / "runs" / "latest.json"
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "hold",
        "--root",
        str(tmp_path),
        "--out",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1
    summary = json.loads(r.stdout)
    assert summary["out"] == str(out)

    report = json.loads(out.read_text())
    assert set(report) == {"run_id", "config_hash", "metrics", "trades"}
    assert set(report["metrics"]) == {field.name for field in fields(Metrics)}
    assert report["metrics"]["trades"] == 0
    assert report["trades"] == []
    assert report["run_id"] != report["config_hash"]


def test_rule_policy_is_a_named_baseline(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-05",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr

    out = tmp_path / "rule.json"
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "rule",
        "--root",
        str(tmp_path),
        "--out",
        str(out),
    )
    assert r.returncode == 0, r.stderr
    report = json.loads(out.read_text())
    assert isinstance(report["trades"], list)
    assert set(report["metrics"]) == {field.name for field in fields(Metrics)}
    assert len(report["trades"]) > 0, "the rule baseline must actually trade on this data"
    assert {trade["strategy"] for trade in report["trades"]} == {"rule:ret_5-baseline"}

    # Independently re-derive the feature at the first trade's entry bar: the
    # baseline may only buy when that bar's ret_5 was positive. Look the column
    # up by name and the bar up by timestamp, so this never mirrors the CLI's
    # flattened-observation index.
    stored = ParquetStore(tmp_path).read("XAUUSD", "M1")
    features = build_features(stored)
    entry = datetime.fromisoformat(report["trades"][0]["entry_ts"])
    stamp = np.datetime64(entry.astimezone(UTC).replace(tzinfo=None), "us")
    bar = int(np.searchsorted(features.ts, stamp, side="left"))
    assert features.ts[bar] == stamp, "entry_ts must be a decision bar in the frame"
    assert features.X[bar, FEATURE_COLUMNS.index("ret_5")] > 0.0


def test_csv_ingest_prints_a_summary(tmp_path):
    source = FIXTURES / "xauusd_gmt3.csv"
    r = run(
        "data",
        "csv",
        "--path",
        str(source),
        "--tz-offset",
        "3",
        "--symbol",
        "XAUUSD",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1
    payload = json.loads(r.stdout)
    assert payload["rows"] == 5
    assert payload["symbol"] == "XAUUSD"
    assert payload["first_ts"] == "2026-01-05T07:00:00+00:00"
    assert payload["last_ts"] == "2026-01-05T07:04:00+00:00"


def test_backtest_with_too_few_bars_exits_2(tmp_path):
    source = FIXTURES / "xauusd_gmt3.csv"  # five bars, far below the 120-bar minimum
    r = run(
        "data",
        "csv",
        "--path",
        str(source),
        "--tz-offset",
        "3",
        "--symbol",
        "XAUUSD",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr

    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "hold",
        "--root",
        str(tmp_path),
        "--out",
        str(tmp_path / "short.json"),
    )
    assert r.returncode == 2
    assert r.stdout == ""
    assert "120" in json.loads(r.stderr)["error"]
    assert not (tmp_path / "short.json").exists()


def test_csv_missing_file_exits_2_with_json_error(tmp_path):
    r = run(
        "data",
        "csv",
        "--path",
        str(tmp_path / "missing.csv"),
        "--symbol",
        "XAUUSD",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 2
    assert r.stdout == ""
    assert r.stderr.count("\n") == 1
    error = json.loads(r.stderr)
    assert set(error) == {"error"}
    assert "missing.csv" in error["error"]


def test_non_utf8_csv_exits_2_with_json_error(tmp_path):
    path = tmp_path / "latin1.csv"
    path.write_bytes(b"2026.01.05 10:00,2350.10,2351.00,2349.80,2350.60,\xff\n")

    r = run(
        "data",
        "csv",
        "--path",
        str(path),
        "--tz-offset",
        "3",
        "--symbol",
        "XAUUSD",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 2
    assert r.stdout == ""
    assert r.stderr.count("\n") == 1
    error = json.loads(r.stderr)
    assert set(error) == {"error"}
    assert "latin1.csv" in error["error"]


def test_unknown_policy_exits_2_with_json_error(tmp_path):
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "nope",
        "--root",
        str(tmp_path),
        "--out",
        str(tmp_path / "x.json"),
    )

    assert r.returncode == 2
    assert r.stdout == ""
    assert "invalid choice" in json.loads(r.stderr)["error"]
    assert not (tmp_path / "x.json").exists()


def test_start_after_end_exits_2_with_json_error(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-02-01",
        "--end",
        "2026-01-01",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 2
    assert r.stdout == ""
    error = json.loads(r.stderr)
    assert "after" in error["error"]
    assert not (tmp_path / "XAUUSD").exists()


def test_serve_prints_one_json_line_and_keeps_stdout_silent(tmp_path):
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "scalping.cli",
            "serve",
            "--port",
            "0",  # ephemeral: the CLI binds first and reports the real port
            "--root",
            str(tmp_path),
        ],
        cwd=".",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        line = _read_stdout_line(proc)
        payload = json.loads(line)
        assert payload["status"] == "serving"
        assert payload["port"] > 0

        connection = http.client.HTTPConnection("127.0.0.1", payload["port"], timeout=10)
        try:
            connection.request("GET", "/api/health")
            response = connection.getresponse()
            assert response.status == 200
            assert json.loads(response.read())["status"] == "ok"
        finally:
            connection.close()
    finally:
        # Terminate before draining the pipes: waiting on fill-then-read order
        # cannot deadlock because both streams are read only after exit.
        _stop(proc)

    assert proc.stdout is not None and proc.stderr is not None
    rest = proc.stdout.read()
    stderr = proc.stderr.read()
    assert rest == "", f"serve wrote extra stdout after its JSON line: {rest!r}"
    assert "Application startup complete" in stderr  # uvicorn logs, never stdout
    assert "GET /api/health" not in stderr  # access_log=False: no access entries


def test_serve_bind_failure_exits_1_with_json_error(tmp_path):
    holder = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    holder.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    holder.bind(("127.0.0.1", 0))
    holder.listen(1)
    port = holder.getsockname()[1]
    try:
        r = run("serve", "--port", str(port), "--root", str(tmp_path))
    finally:
        holder.close()

    assert r.returncode == 1
    assert r.stdout == ""
    assert r.stderr.count("\n") == 1
    error = json.loads(r.stderr)
    assert set(error) == {"error", "type"}
    assert "bind" in error["error"].lower()
    assert str(port) in error["error"]


def test_main_maps_system_exit_to_one_json_line(monkeypatch, capsys):
    """A ``sys.exit()`` escaping uvicorn must still obey the JSON contract."""

    def explode(_args):
        raise SystemExit(1)

    class StubParser:
        def parse_args(self, _argv):
            return SimpleNamespace(handler=explode)

    monkeypatch.setattr(cli, "_build_parser", lambda: StubParser())
    code = cli.main([])

    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    assert captured.err.count("\n") == 1
    error = json.loads(captured.err)
    assert set(error) == {"error", "type"}
    assert error["type"] == "SystemExit"
