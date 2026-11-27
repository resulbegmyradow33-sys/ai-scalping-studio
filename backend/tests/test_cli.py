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
import polars as pl

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


def test_cli_import_does_not_load_the_api():
    """A data command must not pay for FastAPI just to serialise a report."""
    code = (
        "import sys\n"
        "import scalping.cli\n"
        "print('scalping.api' in sys.modules)\n"
        "print('fastapi' in sys.modules)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["False", "False"]


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
    assert set(payload) == {"rows", "symbol", "first_ts", "last_ts", "provenance"}
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


def test_backtest_without_out_exits_2(tmp_path):
    r = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "hold",
        "--root",
        str(tmp_path),
    )

    assert r.returncode == 2
    assert r.stdout == ""
    assert "--out" in json.loads(r.stderr)["error"]


def test_backtest_montecarlo_reports_the_distribution(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr

    out = tmp_path / "mc.json"
    r = run(
        "backtest",
        "montecarlo",
        "--symbol",
        "XAUUSD",
        "--policy",
        "rule",
        "--root",
        str(tmp_path),
        "--n-paths",
        "32",
        "--seed",
        "3",
        "--mode",
        "bootstrap",
        "--block-size",
        "4",
        "--ruin-fraction",
        "0.5",
        "--confidence",
        "0.05,0.5,0.95",
        "--out",
        str(out),
    )

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1
    payload = json.loads(r.stdout)
    assert payload["paths"] == 32
    assert payload["trades"] > 0
    assert payload["mode"] == "bootstrap"
    assert set(payload["final_pnl"]) == {"p05", "p50", "p95"}
    assert 0.0 <= payload["probability_of_ruin"] <= 1.0
    assert payload["ruin_threshold"] == 5_000.0
    assert payload["worst_path"]["ending_balance"] <= payload["ending_balance"]["p50"]
    assert payload["out"] == str(out)

    full = json.loads(out.read_text())
    assert len(full["outcomes"]) == 32
    assert full["confidence_levels"] == [0.05, 0.5, 0.95]


def test_backtest_sensitivity_reports_rows_and_breakeven(tmp_path):
    r = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr

    out = tmp_path / "sens.json"
    r = run(
        "backtest",
        "sensitivity",
        "--symbol",
        "XAUUSD",
        "--policy",
        "rule",
        "--root",
        str(tmp_path),
        "--commission",
        "0,7,28",
        "--out",
        str(out),
    )

    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1
    payload = json.loads(r.stdout)
    assert payload["base"]["parameter"] == "base"
    assert [axis["parameter"] for axis in payload["axes"]] == ["commission_per_lot"]
    rows = payload["axes"][0]["rows"]
    assert [row["value"] for row in rows] == [0.0, 7.0, 28.0]
    assert all(row["trades"] >= 0 for row in rows)
    assert "breakeven_multiplier" in payload["axes"][0]
    assert out.is_file()

    written = json.loads(out.read_text())
    assert written == {key: value for key, value in payload.items() if key != "out"}


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


def test_cli_json_import_and_derive(tmp_path):
    path = tmp_path / "bars.json"
    rows = [
        {
            "timestamp": f"2026-01-05T00:0{i}:00Z",
            "open": 100.0 + i,
            "high": 101.0 + i,
            "low": 99.0 + i,
            "close": 100.5 + i,
            "volume": 10,
        }
        for i in range(5)
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    root = tmp_path / "store"
    imported = run("data", "json", "--path", str(path), "--root", str(root))
    assert imported.returncode == 0, imported.stderr
    payload = json.loads(imported.stdout)
    assert payload["rows"] == 5
    assert payload["provenance"]["source"] == "json:bars.json"

    derived = run(
        "data",
        "derive",
        "--root",
        str(root),
        "--timeframes",
        "M5,M30,H1,H4",
    )
    assert derived.returncode == 0, derived.stderr
    written = json.loads(derived.stdout)["written"]
    assert set(written) == {"M5", "M30", "H1", "H4"}
    assert ParquetStore(root).read("XAUUSD", "M5").height == 1


def test_cli_dataset_build_writes_a_manifest(tmp_path):
    root = tmp_path / "store"
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    out = tmp_path / "dataset.json"
    built = run(
        "dataset",
        "build",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert built.returncode == 0, built.stderr
    payload = json.loads(built.stdout)
    assert payload["columns"] == 19  # price (14) + momentum (5)
    assert payload["test"] > 0
    manifest = json.loads(out.read_text())
    assert manifest["dataset_id"] == payload["dataset_id"]
    assert manifest["purge_bars"] == 5


def _synth_two_days(root: Path) -> None:
    """Write a clean Monday–Tuesday M1 slice used by the validate tests."""
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
        str(root),
    )
    assert r.returncode == 0, r.stderr


def test_data_validate_prints_one_json_line_and_exits_0(tmp_path):
    _synth_two_days(tmp_path)
    r = run(
        "data",
        "validate",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    assert r.stderr == ""
    assert r.stdout.count("\n") == 1
    report = json.loads(r.stdout)
    assert report["ok"] is True
    assert report["symbol"] == "XAUUSD"
    assert report["rows"] == 2_880


def test_data_validate_exits_2_when_a_frame_is_broken(tmp_path):
    _synth_two_days(tmp_path)
    frame = ParquetStore(tmp_path).read("XAUUSD", "M1")
    broken = frame.with_columns((pl.col("bid_close") - 1.0).alias("ask_close"))
    broken.write_parquet(tmp_path / "XAUUSD" / "M1" / "2026" / "01.parquet")

    r = run("data", "validate", "--root", str(tmp_path))
    assert r.returncode == 2
    assert r.stderr == ""
    report = json.loads(r.stdout)
    assert report["ok"] is False
    assert "crossed_quotes" in {issue["code"] for issue in report["issues"]}


def test_data_validate_without_a_slice_exits_2_with_json_error(tmp_path):
    r = run(
        "data",
        "validate",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "H1",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 2
    assert r.stdout == ""
    error = json.loads(r.stderr)
    assert "XAUUSD" in error["error"] and "H1" in error["error"]


def test_cli_ml_train_writes_a_model_and_metrics(tmp_path):
    root = tmp_path / "store"
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    out = tmp_path / "ml"
    trained = run(
        "ml",
        "train",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--model",
        "logistic",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert trained.returncode == 0, trained.stderr
    payload = json.loads(trained.stdout)
    assert payload["model"] == "logistic"
    assert payload["validation"] > 0
    assert Path(payload["model_path"]).is_file()
    assert Path(payload["metrics_path"]).is_file()
    assert payload["validation_metrics"]["rows"] == payload["validation"]


def test_cli_ml_evaluate_trades_a_saved_model(tmp_path):
    root = tmp_path / "store"
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    out = tmp_path / "ml"
    trained = run(
        "ml",
        "train",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--model",
        "logistic",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert trained.returncode == 0, trained.stderr
    run_dir = Path(json.loads(trained.stdout)["model_path"]).parent
    evaluated = run(
        "ml",
        "evaluate",
        "--run",
        str(run_dir),
        "--root",
        str(root),
        "--split",
        "validation",
        "--stress",
        "2",
    )
    assert evaluated.returncode == 0, evaluated.stderr
    payload = json.loads(evaluated.stdout)
    assert payload["split"] == "validation"
    assert {"trades", "net_profit", "profit_factor"} <= set(payload["metrics"])
    assert payload["hold"]["trades"] == 0
    assert "x2" in payload["stress"]


def test_cli_ml_train_reports_fitted_temperature(tmp_path):
    root = tmp_path / "store"
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-07",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    out = tmp_path / "ml"
    trained = run(
        "ml",
        "train",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--model",
        "logistic",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--calibrate",
        "temperature",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert trained.returncode == 0, trained.stderr
    payload = json.loads(trained.stdout)
    calibration = payload["calibration"]
    assert calibration["method"] == "temperature"
    assert calibration["fitted_on"] == "validation"
    assert calibration["rows"] == payload["validation"]
    assert calibration["temperature"] > 0.0
    assert calibration["after"]["log_loss"] <= calibration["before"]["log_loss"]

    stored = json.loads(Path(payload["metrics_path"]).read_text(encoding="utf-8"))
    assert stored["spec"]["calibrate"] == "temperature"
    assert stored["calibration"]["temperature"] == calibration["temperature"]


def test_cli_ml_cv_prints_json_and_writes_report(tmp_path):
    root = tmp_path / "store"
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-06",
        "--root",
        str(root),
    )
    assert synth.returncode == 0, synth.stderr
    out = tmp_path / "cv.json"
    completed = run(
        "ml",
        "cv",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--model",
        "logistic",
        "--groups",
        "price,momentum",
        "--horizon",
        "5",
        "--folds",
        "4",
        "--purge",
        "5",
        "--embargo",
        "2",
        "--min-train-rows",
        "50",
        "--root",
        str(root),
        "--out",
        str(out),
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stderr == ""
    assert completed.stdout.count("\n") == 1
    payload = json.loads(completed.stdout)
    assert payload["model"] == "logistic"
    assert payload["spec"]["horizon"] == 5
    assert payload["spec"]["include_flat"] is False
    assert payload["spec"]["seed"] == 0
    assert payload["columns"]
    assert len(payload["folds"]) == 4
    assert payload["split_counts"]["train"] > 0
    assert set(payload["aggregate"]) == {
        "accuracy",
        "balanced_accuracy",
        "log_loss",
        "brier",
        "ece",
        "mce",
    }
    assert "out" in payload
    report = json.loads(out.read_text())
    assert "out" not in report
    assert report["dataset_id"] == payload["dataset_id"]
    assert report["spec"] == payload["spec"]


def test_cli_ml_cv_without_data_exits_2(tmp_path):
    completed = run(
        "ml",
        "cv",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--root",
        str(tmp_path / "empty"),
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    error = json.loads(completed.stderr)
    assert "XAUUSD" in error["error"] and "M1" in error["error"]


def test_cli_import_does_not_load_boosting_libraries():
    code = (
        "import sys\n"
        "import scalping.cli\n"
        "print('lightgbm' in sys.modules, 'xgboost' in sys.modules)\n"
    )
    import subprocess
    import sys as _sys

    completed = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == "False False"


def test_backtest_mask_actions_flag(tmp_path):
    synth = run(
        "data",
        "synth",
        "--symbol",
        "XAUUSD",
        "--start",
        "2026-01-05",
        "--end",
        "2026-01-08",
        "--root",
        str(tmp_path),
    )
    assert synth.returncode == 0, synth.stderr

    out = tmp_path / "masked.json"
    result = run(
        "backtest",
        "--symbol",
        "XAUUSD",
        "--policy",
        "rule",
        "--mask-actions",
        "--root",
        str(tmp_path),
        "--out",
        str(out),
    )

    assert result.returncode == 0, result.stderr
    assert out.is_file()
    assert "run_id" in json.loads(result.stdout)


def test_data_status_prints_coverage_providers_and_provenance(tmp_path):
    _synth_two_days(tmp_path)
    r = run(
        "data",
        "status",
        "--symbol",
        "XAUUSD",
        "--timeframe",
        "M1",
        "--root",
        str(tmp_path),
    )
    assert r.returncode == 0, r.stderr
    payload = json.loads(r.stdout)
    assert payload["rows"] == 2_880
    assert payload["providers"]
    assert payload["first_ts"] and payload["last_ts"]
    assert payload["provenance"] is None or isinstance(payload["provenance"], dict)
