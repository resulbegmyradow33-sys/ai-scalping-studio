"""Market-data file triggers: one ``dataset.build`` per new symbol/timeframe file.

The source is a plain ``scan``/``seen`` pair, so every test drives it directly
or through ``scan_once``. The store is a real temporary SQLite file; nothing is
mocked except the deliberate crash between the two phases.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scalping.orchestrator import (
    EVENT_DEFERRED,
    EVENT_REFUSED,
    KIND_DATASET_BUILD,
    DataFileTriggerSource,
    JobStore,
    Orchestrator,
    Trigger,
    TriggerSource,
    scan_once,
)

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "orchestrator.sqlite3")


def _orchestrator(store: JobStore) -> Orchestrator:
    return Orchestrator(store)


def _data_root(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    root.mkdir()
    return root


def _write(root: Path, relative: str, content: bytes) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def _events(store: JobStore, name: str) -> list[dict[str, object]]:
    details: list[dict[str, object]] = []
    for event in store.events():
        if event.event == name:
            assert event.detail is not None
            parsed = json.loads(event.detail)
            assert isinstance(parsed, dict)
            details.append(parsed)
    return details


class _CrashBeforeSeen:
    """Delegate ``scan`` but fail the acknowledgement, like a killed process."""

    def __init__(self, source: DataFileTriggerSource) -> None:
        self._source = source
        self.name = source.name

    def scan(self) -> list[Trigger]:
        return self._source.scan()

    def seen(self, triggers: Sequence[Trigger]) -> None:
        raise RuntimeError("simulated crash before acknowledgement")


def test_new_file_emits_one_dataset_build(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    path = _write(root, "XAUUSD/M1/2026/01.parquet", b"PAR1-bars")
    source = DataFileTriggerSource(root, store=store)
    assert isinstance(source, TriggerSource)

    outcome = scan_once(_orchestrator(store), [source], now=T0)

    assert len(outcome.queued) == 1
    job = outcome.queued[0]
    assert job.kind == KIND_DATASET_BUILD
    assert set(job.payload) == {"symbol", "timeframe", "notes"}
    assert job.payload["symbol"] == "XAUUSD"
    assert job.payload["timeframe"] == "M1"
    assert str(job.payload["notes"]).startswith(
        f"auto-triggered by data file {path.relative_to(root).as_posix()} "
    )

    again = scan_once(_orchestrator(store), [source], now=T0)
    assert again.queued == ()
    assert again.duplicates == ()
    assert len(store.list_jobs()) == 1


def test_payload_is_accepted_by_the_dataset_handler(tmp_path: Path) -> None:
    """The source and the handler must agree on the contract, not just exist."""
    from scalping.orchestrator.handlers import KIND_DATASET_BUILD as HANDLER_KIND
    from scalping.orchestrator.handlers import DatasetRequest

    assert KIND_DATASET_BUILD == HANDLER_KIND, "trigger and handler disagree on the kind"

    store = _store(tmp_path)
    root = _data_root(tmp_path)
    _write(root, "XAUUSD/H1/bars.csv", b"ts,open,high,low,close\n")
    source = DataFileTriggerSource(root, store=store)
    outcome = scan_once(_orchestrator(store), [source], now=T0)

    request = DatasetRequest.from_payload(outcome.queued[0].payload)
    assert request.symbol == "XAUUSD"
    assert request.timeframe == "H1"
    assert request.horizon == 1


def test_cursor_survives_a_store_restart(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    _write(root, "XAUUSD/M1/2026/01.parquet", b"bars")

    first_store = _store(tmp_path)
    first = DataFileTriggerSource(root, store=first_store)
    assert len(scan_once(_orchestrator(first_store), [first], now=T0).queued) == 1
    first_store.close()

    second_store = JobStore(tmp_path / "orchestrator.sqlite3")
    second = DataFileTriggerSource(root, store=second_store)
    outcome = scan_once(_orchestrator(second_store), [second], now=T0)

    assert outcome.queued == ()
    assert outcome.duplicates == ()
    assert len(second_store.list_jobs()) == 1


def test_changed_file_emits_once(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    path = _write(root, "XAUUSD/M1/2026/01.parquet", b"AAAA")
    source = DataFileTriggerSource(root, store=store)
    orchestrator = _orchestrator(store)

    first = scan_once(orchestrator, [source], now=T0)
    assert len(first.queued) == 1
    first_digest = first.queued[0].payload["notes"].split("sha256 ")[1].split(",")[0]

    stamp = path.stat().st_mtime_ns
    path.write_bytes(b"BBBB")  # same size, new bytes
    os.utime(path, ns=(stamp + 2_000_000_000, stamp + 2_000_000_000))
    changed = scan_once(orchestrator, [source], now=T0)

    assert len(changed.queued) == 1
    assert changed.queued[0].dedupe_key != first.queued[0].dedupe_key
    second_digest = changed.queued[0].payload["notes"].split("sha256 ")[1].split(",")[0]
    assert second_digest != first_digest
    assert scan_once(orchestrator, [source], now=T0).queued == ()
    assert len(store.list_jobs()) == 2


def test_identical_bytes_under_another_path_emit_nothing(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    _write(root, "XAUUSD/M1/a.parquet", b"same bytes")
    source = DataFileTriggerSource(root, store=store)
    orchestrator = _orchestrator(store)
    assert len(scan_once(orchestrator, [source], now=T0).queued) == 1

    _write(root, "XAUUSD/M1/b.parquet", b"same bytes")
    copied = scan_once(orchestrator, [source], now=T0)

    assert copied.queued == ()
    assert copied.duplicates == ()  # no trigger at all: the checksum was seen
    assert len(store.list_jobs()) == 1


def test_unknown_layout_is_refused_without_a_job(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    _write(root, "loose.parquet", b"no symbol")
    _write(root, "XAUUSD/bars.parquet", b"no timeframe")
    _write(root, "XAUUSD/YEARLY/bars.parquet", b"unknown timeframe")
    _write(root, "XAUUSD/M1/ok.parquet", b"valid")
    source = DataFileTriggerSource(root, store=store)
    orchestrator = _orchestrator(store)

    first = scan_once(orchestrator, [source], now=T0)

    assert [job.payload["symbol"] for job in first.queued] == ["XAUUSD"]
    assert len(store.list_jobs()) == 1
    refusals = {
        str(item["relative_path"]): str(item["reason"]) for item in _events(store, EVENT_REFUSED)
    }
    assert refusals == {
        "loose.parquet": "unknown_layout",
        "XAUUSD/bars.parquet": "unknown_layout",
        "XAUUSD/YEARLY/bars.parquet": "unknown_timeframe",
    }

    second = scan_once(orchestrator, [source], now=T0)
    assert second.queued == ()
    # The refusal is recorded once per (path, size, mtime), not every tick.
    assert len(_events(store, EVENT_REFUSED)) == 3


def test_missing_root_is_a_no_op(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = tmp_path / "not-created-yet"
    source = DataFileTriggerSource(root, store=store)
    orchestrator = _orchestrator(store)

    outcome = scan_once(orchestrator, [source], now=T0)

    assert outcome.queued == ()
    assert outcome.duplicates == ()
    assert store.list_jobs() == []
    assert source.scan() == []  # still a no-op outside scan_once


def test_crash_between_scan_and_seen_still_queues_one_job(tmp_path: Path) -> None:
    root = _data_root(tmp_path)
    _write(root, "XAUUSD/M1/2026/01.parquet", b"bars")

    first_store = _store(tmp_path)
    crashing = _CrashBeforeSeen(DataFileTriggerSource(root, store=first_store))
    with pytest.raises(RuntimeError, match="simulated crash"):
        scan_once(_orchestrator(first_store), [crashing], now=T0)
    assert len(first_store.list_jobs()) == 1
    first_store.close()

    # Restart: the cursor was never persisted, so the file is scanned again and
    # the identical dedupe key returns the existing job instead of a new one.
    second_store = JobStore(tmp_path / "orchestrator.sqlite3")
    source = DataFileTriggerSource(root, store=second_store)
    outcome = scan_once(_orchestrator(second_store), [source], now=T0)

    assert outcome.queued == ()
    assert len(outcome.duplicates) == 1
    assert len(second_store.list_jobs()) == 1


def test_max_emits_defers_without_losing_files(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    for symbol in ("AAA", "BBB", "CCC"):
        _write(root, f"{symbol}/M1/2026/01.parquet", symbol.encode())
    source = DataFileTriggerSource(root, store=store, max_emits=2)
    orchestrator = _orchestrator(store)

    first = scan_once(orchestrator, [source], now=T0)
    assert len(first.queued) == 2
    assert _events(store, EVENT_DEFERRED)[0]["deferred"] == 1

    second = scan_once(orchestrator, [source], now=T0)
    assert len(second.queued) == 1
    assert scan_once(orchestrator, [source], now=T0).queued == ()
    assert len(store.list_jobs()) == 3


def test_since_skips_old_files_until_they_change(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    path = _write(root, "XAUUSD/M1/2026/01.parquet", b"old")
    old = int(datetime(2020, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    os.utime(path, ns=(old, old))
    source = DataFileTriggerSource(root, store=store, since=T0)
    orchestrator = _orchestrator(store)

    assert scan_once(orchestrator, [source], now=T0).queued == ()

    new = int(datetime(2026, 1, 1, 1, tzinfo=UTC).timestamp() * 1_000_000_000)
    os.utime(path, ns=(new, new))
    assert len(scan_once(orchestrator, [source], now=T0).queued) == 1


def test_timeframe_and_symbol_are_normalised(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    _write(root, "xauusd/m1/bars.csv", b"ts\n")
    source = DataFileTriggerSource(root, store=store)

    outcome = scan_once(_orchestrator(store), [source], now=T0)

    assert outcome.queued[0].payload["symbol"] == "XAUUSD"
    assert outcome.queued[0].payload["timeframe"] == "M1"


def test_dot_prefixed_paths_are_ignored(tmp_path: Path) -> None:
    store = _store(tmp_path)
    root = _data_root(tmp_path)
    _write(root, ".feature-cache/XAUUSD/M1/2026/01.parquet", b"cache")
    _write(root, "XAUUSD/M1/.01.parquet.tmp", b"partial")
    source = DataFileTriggerSource(root, store=store)

    outcome = scan_once(_orchestrator(store), [source], now=T0)

    assert outcome.queued == ()
    assert _events(store, EVENT_REFUSED) == []


def test_constructor_refuses_a_file_as_the_data_root(tmp_path: Path) -> None:
    path = tmp_path / "bars.parquet"
    path.write_bytes(b"x")
    with pytest.raises(NotADirectoryError):
        DataFileTriggerSource(path, store=_store(tmp_path))


def test_constructor_refuses_a_kind_it_cannot_serve(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dataset.build"):
        DataFileTriggerSource(
            _data_root(tmp_path),
            store=_store(tmp_path),
            kind="discovery.run",
        )


# ------------------------------------------------------------------------ CLI


def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "orchestrator",
        "run",
        "--once",
        "--profile",
        "none",
        "--db",
        str(tmp_path / "orchestrator.sqlite3"),
        "--root",
        str(tmp_path / "data"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--archive-root",
        str(tmp_path / "archive"),
        "--max-jobs-per-tick",
        "0",
        *extra,
    ]


def test_cli_watch_data_dir_enqueues_a_dataset_build(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scalping import cli

    root = _data_root(tmp_path)
    _write(root, "XAUUSD/M1/2026/01.parquet", b"bars")
    args = _cli_args(tmp_path, "--watch-data-dir", str(root))

    assert cli.main(args) == 0
    first = json.loads(capsys.readouterr().out.strip())
    assert [job["kind"] for job in first["tick"]["scan"]["queued"]] == ["dataset.build"]
    assert first["tick"]["scan"]["queued"][0]["payload"]["symbol"] == "XAUUSD"

    assert cli.main(args) == 0
    second = json.loads(capsys.readouterr().out.strip())
    assert second["tick"]["scan"]["queued"] == []


def test_cli_rejects_watch_kind_dataset_build_with_a_file_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from scalping import cli

    docs = tmp_path / "documents"
    docs.mkdir()
    args = _cli_args(tmp_path, "--watch-dir", str(docs), "--watch-kind", "dataset.build")

    assert cli.main(args) == 2
    error = json.loads(capsys.readouterr().err.strip())["error"]
    assert "--watch-data-dir" in error


def test_cli_rejects_two_watch_roots(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from scalping import cli

    docs = tmp_path / "documents"
    docs.mkdir()
    root = _data_root(tmp_path)
    args = _cli_args(
        tmp_path,
        "--watch-dir",
        str(docs),
        "--watch-data-dir",
        str(root),
    )

    assert cli.main(args) == 2
    error = json.loads(capsys.readouterr().err.strip())["error"]
    assert "--watch-dir" in error and "--watch-data-dir" in error
