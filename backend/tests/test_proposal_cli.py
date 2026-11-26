"""CLI wiring for approved proposals as ``discovery.run`` queue work.

The bridge source already exists; these tests pin the operator-facing seam:

* ``--watch-proposals`` is repeatable and binds one source per explicit path, in
  flag order, with one shared ``--proposal-template`` JSON file;
* only owner-approved proposals are enqueued and the payload carries the
  template's gate, stress and seed;
* paths with ``?``, ``#``, ``%`` and spaces are probed correctly without
  creating a sidecar or a truncated file;
* a missing, non-file, unreadable, non-writable, empty, stub-schema or
  incompatible-schema proposal database is refused before the orchestrator
  database is created;
* the cursor and the job dedupe key survive a CLI restart;
* ``--enqueue-only`` queues exactly one job and executes none;
* the two watch flags are mutually dependent and conflict with ``--no-scan``;
* a broken or unreadable template is a CLI error, never a silently defaulted
  search policy.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from scalping import cli
from scalping.knowledge import (
    DocumentIngestor,
    KnowledgeIndex,
    ProposalStore,
    extract_hypotheses,
    propose_from_hypothesis,
)
from scalping.orchestrator import JobStore
from scalping.orchestrator.handlers import DiscoveryRequest
from scalping.orchestrator.proposal_handlers import (
    DiscoveryTemplate,
    ProposalQueueTriggerSource,
)

KIND_DISCOVERY_RUN = "discovery.run"
SYMBOL = "XAUUSD"
TIMEFRAME = "M15"
TEXT = """# Claims

When volatility expands, momentum signals become less reliable. A signed order-flow imbalance predicts short-term continuation. The more the spread widens, the lower the edge.

## Liquidity

Sweeps of session lows lead to reversals. A session breakout after a quiet Asian range tends to continue.
"""
TEMPLATE: dict[str, Any] = {
    "max_trials": 25,
    "seed": 7,
    "gate": {"min_trades": 10, "max_drawdown": 0.25},
    "stress": [1.5, 2.0],
    "population": 8,
    "generations": 3,
    "patience": 2,
    "horizon": 2,
}


# --------------------------------------------------------------------------- helpers


def _cli_args(
    tmp_path: Path,
    *extra: str,
    max_jobs_per_tick: str | None = "0",
    db: Path | None = None,
) -> list[str]:
    args = [
        "orchestrator",
        "run",
        "--once",
        "--profile",
        "none",
        "--db",
        str(db or tmp_path / "orchestrator.sqlite3"),
        "--root",
        str(tmp_path / "data"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--archive-root",
        str(tmp_path / "archive"),
    ]
    if max_jobs_per_tick is not None:
        args.extend(["--max-jobs-per-tick", max_jobs_per_tick])
    return [*args, *extra]


def _template_file(
    tmp_path: Path, document: dict[str, Any] | None = None, name: str = "policy"
) -> Path:
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(TEMPLATE if document is None else document), encoding="utf-8")
    return path


def _seed_proposals(
    tmp_path: Path, *, name: str, approve: int = 1, total: int = 2
) -> tuple[Path, list[str], list[str]]:
    """Create a proposal DB with approved and still-proposed records."""
    db = tmp_path / f"proposals-{name}.sqlite3"
    index = KnowledgeIndex(tmp_path / f"knowledge-{name}.sqlite3")
    store = ProposalStore(db)
    approved: list[str] = []
    proposed: list[str] = []
    try:
        source = tmp_path / f"notes-{name}.md"
        source.write_text(TEXT, encoding="utf-8")
        document = DocumentIngestor(tmp_path / f"knowledge-{name}").ingest(source)
        index.add_document(document)
        hypotheses = list(extract_hypotheses(index, document.doc_id).created)
        assert len(hypotheses) >= total
        for position, hypothesis in enumerate(hypotheses[:total]):
            record = store.create(
                propose_from_hypothesis(
                    hypothesis, symbol=SYMBOL, timeframe=TIMEFRAME, index=index
                ),
                index=index,
            )
            if position < approve:
                store.approve(record.proposal_id, "scope and data are clear", "owner")
                approved.append(record.proposal_id)
            else:
                proposed.append(record.proposal_id)
    finally:
        store.close()
        index.close()
    return db, approved, proposed


def _tick(capsys: pytest.CaptureFixture[str]) -> dict[str, Any]:
    payload = json.loads(capsys.readouterr().out.strip())
    assert "tick" in payload
    return payload["tick"]


def _error(capsys: pytest.CaptureFixture[str]) -> str:
    lines = [line for line in capsys.readouterr().err.splitlines() if line.strip()]
    assert len(lines) == 1
    payload = json.loads(lines[0])
    return str(payload["error"])


def _watch_args(*paths: Path, template: Path, max_jobs: str = "0") -> list[str]:
    args: list[str] = []
    for path in paths:
        args.extend(["--watch-proposals", str(path)])
    args.extend(["--proposal-template", str(template)])
    return args


# ------------------------------------------------------------------------------- tests


def test_cli_binds_one_source_per_path_in_flag_order(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    first, _, _ = _seed_proposals(tmp_path, name="a")
    second, _, _ = _seed_proposals(tmp_path, name="b")
    template = _template_file(tmp_path)
    args = _cli_args(
        tmp_path,
        *_watch_args(second, first, template=template),
    )

    assert cli.main(args) == 0
    scan = _tick(capsys)["scan"]
    assert scan is not None
    assert scan["sources"] == [
        f"proposals:{second.resolve()}",
        f"proposals:{first.resolve()}",
    ]
    assert [job["kind"] for job in scan["queued"]] == [KIND_DISCOVERY_RUN] * 2
    jobs = _jobs(tmp_path / "orchestrator.sqlite3")
    assert len(jobs) == 2


def test_cli_payload_carries_the_template_policy_and_validates(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="policy")
    template = _template_file(tmp_path)

    assert cli.main(_cli_args(tmp_path, *_watch_args(db, template=template))) == 0
    scan = _tick(capsys)["scan"]
    assert scan is not None and len(scan["queued"]) == 1
    payload = scan["queued"][0]["payload"]
    assert payload["symbol"] == SYMBOL
    assert payload["timeframe"] == TIMEFRAME
    assert payload["seed"] == 7
    assert payload["max_trials"] == 25
    assert payload["stress"] == [1.5, 2.0]
    assert payload["gate"]["min_trades"] == 10
    assert payload["gate"]["max_drawdown"] == 0.25
    assert payload["population"] == 8
    assert payload["generations"] == 3
    assert payload["patience"] == 2
    assert payload["horizon"] == 2
    # The same payload the production handler accepts, without running it.
    assert DiscoveryRequest.from_payload(payload).seed == 7


def test_cli_emits_only_owner_approved_proposals(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, approved, proposed = _seed_proposals(tmp_path, name="approval", approve=1, total=2)
    template = _template_file(tmp_path)

    assert cli.main(_cli_args(tmp_path, *_watch_args(db, template=template))) == 0
    scan = _tick(capsys)["scan"]
    assert scan is not None and len(scan["queued"]) == 1
    assert approved[0] in scan["queued"][0]["dedupe_key"]
    assert proposed[0] not in scan["queued"][0]["dedupe_key"]
    store = ProposalStore(db)
    try:
        assert next(
            record for record in store.list() if record.proposal_id == approved[0]
        ).status == ("approved")
        assert next(
            record for record in store.list() if record.proposal_id == proposed[0]
        ).status == ("proposed")
    finally:
        store.close()


def test_cli_scan_enqueues_without_executing_discovery_or_writing_an_archive(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="no-exec")
    template = _template_file(tmp_path)
    args = _cli_args(tmp_path, *_watch_args(db, template=template))

    assert cli.main(args) == 0
    tick = _tick(capsys)
    assert tick["error"] is None and tick["outcomes"] == []
    scan = tick["scan"]
    assert scan is not None and [job["kind"] for job in scan["queued"]] == [KIND_DISCOVERY_RUN]
    jobs = _jobs(tmp_path / "orchestrator.sqlite3")
    assert len(jobs) == 1 and jobs[0].status == "queued" and jobs[0].attempts == 0
    store = JobStore(tmp_path / "orchestrator.sqlite3")
    try:
        names = [event.event for event in store.events()]
    finally:
        store.close()
    assert "started" not in names
    assert "proposal.discovery_queued" in names
    assert not (tmp_path / "runs" / "discovery").exists()


def test_cli_restart_dedupes_and_a_lost_cursor_deduplicates_again(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="restart")
    template = _template_file(tmp_path)
    args = _cli_args(tmp_path, *_watch_args(db, template=template))
    orchestrator_db = tmp_path / "orchestrator.sqlite3"

    assert cli.main(args) == 0
    first = _tick(capsys)["scan"]
    assert first is not None and len(first["queued"]) == 1

    assert cli.main(args) == 0
    second = _tick(capsys)["scan"]
    assert second is not None and second["queued"] == [] and second["duplicates"] == []
    assert len(_jobs(orchestrator_db)) == 1

    jobs = JobStore(orchestrator_db)
    try:
        source = ProposalQueueTriggerSource(
            ProposalStore(db),
            store=jobs,
            template=DiscoveryTemplate.from_mapping(json.loads(template.read_text())),
        )
        jobs.clear_state(source.cursor_key)
    finally:
        jobs.close()
    assert cli.main(args) == 0
    third = _tick(capsys)["scan"]
    assert third is not None and third["queued"] == []
    assert [job["kind"] for job in third["duplicates"]] == [KIND_DISCOVERY_RUN]
    assert len(_jobs(orchestrator_db)) == 1


@pytest.mark.parametrize("case", ["missing", "directory", "empty", "garbage", "no_schema"])
def test_cli_refuses_an_invalid_proposal_db_before_creating_the_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    template = _template_file(tmp_path)
    path = tmp_path / "proposals.sqlite3"
    if case == "missing":
        pass
    elif case == "directory":
        path.mkdir()
    elif case == "empty":
        path.touch()
    elif case == "garbage":
        path.write_text("not a sqlite database", encoding="utf-8")
    else:
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE unrelated (id INTEGER PRIMARY KEY)")
    args = _cli_args(tmp_path, *_watch_args(path, template=template))

    assert cli.main(args) == 2
    message = _error(capsys)
    assert message
    db = tmp_path / "orchestrator.sqlite3"
    assert not db.exists()
    if case == "empty":
        assert path.stat().st_size == 0
    if case in {"empty", "garbage", "no_schema"}:
        probe = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tables = probe.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        except sqlite3.DatabaseError:
            tables = []  # a non-SQLite file certainly has no proposals table
        finally:
            probe.close()
        assert ("proposals",) not in tables


def test_cli_requires_the_flag_pair(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="pair")
    template = _template_file(tmp_path)
    watch_only = _cli_args(tmp_path, "--watch-proposals", str(db))
    template_only = _cli_args(tmp_path, "--proposal-template", str(template))

    assert cli.main(watch_only) == 2
    assert "template" in _error(capsys)
    assert cli.main(template_only) == 2
    assert "watch" in _error(capsys)
    assert not (tmp_path / "orchestrator.sqlite3").exists()


def test_cli_no_scan_conflicts_with_both_proposal_flags(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="conflict")
    template = _template_file(tmp_path)
    with_watch = _cli_args(
        tmp_path,
        *_watch_args(db, template=template),
        "--no-scan",
    )
    with_template = _cli_args(
        tmp_path,
        "--proposal-template",
        str(template),
        "--no-scan",
    )

    assert cli.main(with_watch) == 2
    assert "no-scan" in _error(capsys)
    assert cli.main(with_template) == 2
    assert "no-scan" in _error(capsys)
    assert not (tmp_path / "orchestrator.sqlite3").exists()


def test_cli_refuses_a_missing_or_invalid_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="bad-template")
    missing = tmp_path / "absent.json"
    cases = {
        "missing": missing,
        "malformed": _template_file(tmp_path, {"stress": [1.5]}, name="malformed"),
        "unknown": _template_file(tmp_path, {**TEMPLATE, "gateway": {}}, name="unknown"),
        "scalar": _template_file(tmp_path, 7, name="scalar"),  # type: ignore[arg-type]
    }
    # An incomplete policy is invalid too: gate and stress are required.
    cases["incomplete"] = _template_file(
        tmp_path,
        {key: value for key, value in TEMPLATE.items() if key != "stress"},
        name="incomplete",
    )
    for name, path in cases.items():
        args = _cli_args(tmp_path, *_watch_args(db, template=path))
        assert cli.main(args) == 2, name
        assert _error(capsys), name
        assert not (tmp_path / "orchestrator.sqlite3").exists(), name


def test_probe_encodes_special_characters_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    weird = tmp_path / "weird?#% space"
    weird.mkdir()
    db, _, _ = _seed_proposals(weird, name="special")
    before = set(weird.iterdir()) | set(tmp_path.iterdir())

    assert cli._validate_proposal_database(str(db)) == db
    after = set(weird.iterdir()) | set(tmp_path.iterdir())
    assert after == before
    # The old unencoded URI opened (and created) the path truncated at '?'.
    assert not (tmp_path / "weird").exists()

    template = _template_file(tmp_path)
    assert cli.main(_cli_args(tmp_path, *_watch_args(db, template=template))) == 0
    tick = _tick(capsys)
    scan = tick["scan"]
    assert scan is not None and scan["sources"] == [f"proposals:{db.resolve()}"]
    assert len(scan["queued"]) == 1
    assert not (tmp_path / "weird").exists()


@pytest.mark.parametrize("case", ["stub", "no_sources", "future_version"])
def test_cli_refuses_a_broken_proposal_schema_before_creating_the_queue(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    case: str,
) -> None:
    template = _template_file(tmp_path)
    path = tmp_path / "proposals.sqlite3"
    if case == "stub":
        with sqlite3.connect(path) as connection:
            connection.execute("CREATE TABLE proposals (id INTEGER PRIMARY KEY)")
    else:
        seeded = _seed_proposals(tmp_path, name="schema")[0]
        seeded.rename(path)
        with sqlite3.connect(path) as connection:
            if case == "no_sources":
                connection.execute("DROP TABLE proposal_sources")
            else:
                connection.execute("DELETE FROM schema_version")
                connection.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (99, 'future')"
                )
    args = _cli_args(tmp_path, *_watch_args(path, template=template))

    assert cli.main(args) == 2
    message = _error(capsys)
    assert message
    assert not (tmp_path / "orchestrator.sqlite3").exists()


def test_cli_refuses_a_non_writable_proposal_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="readonly")
    template = _template_file(tmp_path)
    os.chmod(db, 0o444)
    try:
        if os.access(db, os.W_OK):
            pytest.skip("elevated privileges can write a 0444 file")
        assert cli.main(_cli_args(tmp_path, *_watch_args(db, template=template))) == 2
        assert "writable" in _error(capsys)
        assert not (tmp_path / "orchestrator.sqlite3").exists()
    finally:
        os.chmod(db, 0o644)


def test_cli_refuses_a_wal_database_whose_directory_is_not_writable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="wal")
    connection = sqlite3.connect(db)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
    finally:
        connection.close()
    template = _template_file(tmp_path)
    queue = tmp_path / "queue"
    queue.mkdir()
    os.chmod(tmp_path, 0o555)
    try:
        if os.access(tmp_path, os.W_OK):
            pytest.skip("elevated privileges can write to a 0555 directory")
        assert (
            cli.main(
                _cli_args(
                    tmp_path, *_watch_args(db, template=template), db=queue / "orchestrator.sqlite3"
                )
            )
            == 2
        )
        assert "writable" in _error(capsys)
        assert not (queue / "orchestrator.sqlite3").exists()
    finally:
        os.chmod(tmp_path, 0o755)


def test_cli_enqueue_only_queues_without_claiming_with_the_default_budget(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, approved, _ = _seed_proposals(tmp_path, name="enqueue-only")
    template = _template_file(tmp_path)
    args = _cli_args(
        tmp_path,
        *_watch_args(db, template=template),
        "--enqueue-only",
        max_jobs_per_tick=None,
    )

    assert cli.main(args) == 0
    tick = _tick(capsys)
    assert tick["error"] is None and tick["outcomes"] == []
    scan = tick["scan"]
    assert scan is not None and len(scan["queued"]) == 1
    assert approved[0] in scan["queued"][0]["dedupe_key"]
    jobs = _jobs(tmp_path / "orchestrator.sqlite3")
    assert len(jobs) == 1 and jobs[0].status == "queued" and jobs[0].attempts == 0
    assert not (tmp_path / "runs" / "discovery").exists()


def test_cli_enqueue_only_requires_once_and_conflicts_with_no_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="enqueue-gate")
    template = _template_file(tmp_path)
    forever = [
        "orchestrator",
        "run",
        "--forever",
        "--profile",
        "none",
        *_watch_args(db, template=template),
        "--enqueue-only",
    ]
    no_scan = _cli_args(
        tmp_path, *_watch_args(db, template=template), "--enqueue-only", "--no-scan"
    )

    assert cli.main(forever) == 2
    assert "once" in _error(capsys)
    assert cli.main(no_scan) == 2
    assert "no-scan" in _error(capsys)
    assert not (tmp_path / "orchestrator.sqlite3").exists()


def test_cli_refuses_an_unreadable_template(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="unreadable-template")
    template = _template_file(tmp_path)
    os.chmod(template, 0)
    try:
        if os.access(template, os.R_OK):
            pytest.skip("elevated privileges can read a 0000 file")
        assert cli.main(_cli_args(tmp_path, *_watch_args(db, template=template))) == 2
        assert "template" in _error(capsys)
        assert not (tmp_path / "orchestrator.sqlite3").exists()
    finally:
        os.chmod(template, 0o600)


def test_cli_names_the_same_database_twice_and_creates_one_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    db, _, _ = _seed_proposals(tmp_path, name="twice")
    template = _template_file(tmp_path)

    assert cli.main(_cli_args(tmp_path, *_watch_args(db, db, template=template))) == 0
    scan = _tick(capsys)["scan"]
    assert scan is not None
    assert len(scan["sources"]) == 2
    assert len(scan["queued"]) == 1
    # The second source shares the first source's cursor and is a quiet no-op;
    # the duplicate is the same job, never a second one.
    assert scan["duplicates"] == []
    assert len(_jobs(tmp_path / "orchestrator.sqlite3")) == 1


def _jobs(db: Path) -> list:
    store = JobStore(db)
    try:
        return store.list_jobs(kind=KIND_DISCOVERY_RUN)
    finally:
        store.close()
