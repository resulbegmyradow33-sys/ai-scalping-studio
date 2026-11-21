"""End-to-end tests for ``python -m scalping.cli``.

Every command writes exactly one JSON line to stdout and exits 0; data errors
write one JSON line to stderr and exit 2. The tests run the real module as a
subprocess so ``argparse``, stdout/stderr discipline and exit codes are all
covered, not just the handler functions.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields
from pathlib import Path

from scalping.backtest.metrics import Metrics

FIXTURES = Path(__file__).parent / "fixtures"


def run(*args: str, cwd: str = ".") -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "scalping.cli", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


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
    assert all(trade["strategy"] == "rule:ret_5-baseline" for trade in report["trades"])


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
