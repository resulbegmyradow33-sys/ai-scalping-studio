"""Autonomy status API: honest missing states, real counts and read-only GETs.

The API is a viewer. These tests pin the properties that make it safe to poll:
a missing store is unconfigured, an unreadable store is not "empty", and a GET
never creates, migrates or rewrites a database.
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from discovery_fakes import ToyCandidate
from scalping.api.app import create_app
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.protocols import TrialResult, canonical_digest
from scalping.orchestrator.core import Orchestrator
from scalping.orchestrator.store import JobStore
from scalping.paper.ledger import PaperLedger
from scalping.types import Side

T0 = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
DATASET_ID = "validation:XAUUSD:M5:abcd1234"
SCOPE = "XAUUSD:M5"


@dataclass(frozen=True, slots=True)
class _MetaCandidate:
    """A candidate whose payload carries the optional policy metadata."""

    payload: dict[str, Any]

    def canonical_dict(self) -> dict[str, Any]:
        return dict(self.payload)

    def canonical_hash(self) -> str:
        return canonical_digest(self.canonical_dict())

    def complexity(self) -> int:
        return 1


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """Configured paths plus pre-created data and runs directories.

    The directories exist before the first request so a GET that creates a file
    is visible as a new entry, not hidden by the directory itself appearing.
    """
    paths = {
        "data_root": tmp_path / "data",
        "runs_root": tmp_path / "runs",
        "paper": tmp_path / "paper.sqlite",
        "orchestrator": tmp_path / "runs" / "orchestrator.sqlite3",
        "archive": tmp_path / "runs" / "discovery" / "discovery.sqlite",
    }
    paths["data_root"].mkdir()
    paths["runs_root"].mkdir()
    return paths


@pytest.fixture
def client(paths: dict[str, Path]) -> TestClient:
    app = create_app(
        data_root=paths["data_root"],
        runs_root=paths["runs_root"],
        paper_ledger=paths["paper"],
        orchestrator_db=paths["orchestrator"],
        discovery_archive=paths["archive"],
    )
    return TestClient(app)


def _files_under(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _seed_paper(path: Path) -> None:
    """Two sessions, one open position and one settled trade."""
    with PaperLedger(path) as book:
        for session_id in ("open", "traded"):
            book.open_session(
                session_id,
                symbol="XAUUSD",
                timeframe="M1",
                strategy="seed",
            )
            signal_id = book.record_signal(session_id, T0, "buy")
            order_id = book.record_order(
                session_id,
                signal_id=signal_id,
                ts=T0,
                side=Side.LONG,
                lot=0.1,
                requested_price=2350.0,
            )
            book.record_fill(
                order_id,
                ts=T0,
                price=2350.1,
                lot=0.1,
                sl=2349.0,
                tp=2352.0,
            )
        book.record_trade(
            "traded",
            side=Side.LONG,
            entry_ts=T0,
            entry_price=2350.1,
            exit_ts=T0 + timedelta(minutes=5),
            exit_price=2352.1,
            lot=0.1,
            pnl=18.0,
            r_multiple=1.4,
            exit_reason="tp",
        )


def _seed_orchestrator(path: Path) -> None:
    """One failed job, one running job and one queued job, plus three controls."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with JobStore(path) as store:
        failed, _ = store.enqueue("discovery.run", dedupe_key="a")
        running, _ = store.enqueue("dataset.build", dedupe_key="b", max_retries=1)
        claimed = store.claim_next()
        assert claimed is not None and claimed.id == failed.id
        store.fail(claimed.id, "gate rejected")
        claimed = store.claim_next()
        assert claimed is not None and claimed.id == running.id
        store.enqueue("paper.advance", dedupe_key="c")
        Orchestrator(store).pause("maintenance")
        Orchestrator(store).safe_mode(True, "drill")
        Orchestrator(store).disable_research(True, "budget")


def _seed_archive(path: Path) -> str:
    """One promoted candidate and one rejected decision."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate = _MetaCandidate(
        {
            "format": "dsl-v1",
            "mode": "paper",
            "policy_id": "policy-42",
            "digest": "feedface",
        }
    )
    with DiscoveryArchive(path) as archive:
        record = archive.add_candidate(candidate, code_version="test-2026-11-24")
        archive.record_trial(
            record.candidate_id,
            TrialResult(metrics={"net_profit": 12.5}, accepted=True, rejections=()),
            dataset_id=DATASET_ID,
            code_version="test-2026-11-24",
            seed=3,
        )
        archive.promote(
            record.candidate_id,
            scope=SCOPE,
            reason="beats baseline after costs",
            metric="net_profit",
        )
        archive.record_decision(record.candidate_id, "rejected", "cost stress fails")
    return record.candidate_id


# ----------------------------------------------------------------- missing stores


def test_status_of_missing_stores_reports_unconfigured_without_zeros(
    client: TestClient, paths: dict[str, Path]
) -> None:
    response = client.get("/api/autonomy/status")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body == {
        "paper": {"configured": False, "readable": False},
        "orchestrator": {"configured": False, "readable": False},
        "champions": [],
        "champions_readable": False,
        "discoveries": {"configured": False, "readable": False},
        "data": {"configured": True, "readable": True, "symbols": []},
    }
    assert not paths["paper"].exists()
    assert not paths["orchestrator"].exists()
    assert not paths["archive"].exists()


def test_list_endpoints_of_missing_stores_omit_items_and_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    champions = client.get("/api/autonomy/champions")
    assert champions.status_code == 200
    assert champions.json() == {"configured": False, "readable": False}

    jobs = client.get("/api/autonomy/jobs")
    assert jobs.status_code == 200
    assert jobs.json() == {"configured": False, "readable": False, "limit": 50}

    events = client.get("/api/autonomy/events")
    assert events.status_code == 200
    assert events.json() == {"configured": False, "readable": False, "limit": 100}

    for key in ("paper", "orchestrator", "archive"):
        assert not paths[key].exists()


def test_get_leaves_precreated_directories_empty(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """A GET on absent stores creates neither the database nor its WAL sidecars."""
    for url in (
        "/api/autonomy/status",
        "/api/autonomy/champions",
        "/api/autonomy/jobs",
        "/api/autonomy/events",
    ):
        assert client.get(url).status_code == 200, url

    assert _files_under(paths["runs_root"]) == []
    assert _files_under(paths["data_root"]) == []
    for key in ("paper", "orchestrator", "archive"):
        assert not paths[key].exists()
        assert not Path(f"{paths[key]}-wal").exists()
        assert not Path(f"{paths[key]}-shm").exists()


def test_an_unreadable_store_is_not_reported_as_empty(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """Existing corrupt files are configured but not readable; no zero is invented."""
    for key in ("paper", "orchestrator", "archive"):
        paths[key].parent.mkdir(parents=True, exist_ok=True)
        paths[key].write_bytes(b"this is not a sqlite database")

    status = client.get("/api/autonomy/status").json()
    assert status["paper"] == {"configured": True, "readable": False}
    assert status["orchestrator"] == {"configured": True, "readable": False}
    assert status["champions"] == []
    assert status["champions_readable"] is False
    assert status["discoveries"] == {"configured": True, "readable": False}

    jobs = client.get("/api/autonomy/jobs").json()
    assert jobs == {"configured": True, "readable": False, "limit": 50}
    events = client.get("/api/autonomy/events").json()
    assert events == {"configured": True, "readable": False, "limit": 100}
    champions = client.get("/api/autonomy/champions").json()
    assert champions == {"configured": True, "readable": False}


@pytest.mark.skipif(os.geteuid() == 0, reason="root can read a mode-000 file")
def test_a_file_that_cannot_be_opened_is_configured_but_unreadable(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_orchestrator(paths["orchestrator"])
    paths["orchestrator"].chmod(0)
    try:
        status = client.get("/api/autonomy/status").json()
        jobs = client.get("/api/autonomy/jobs").json()
    finally:
        paths["orchestrator"].chmod(0o600)

    assert status["orchestrator"] == {"configured": True, "readable": False}
    assert jobs == {"configured": True, "readable": False, "limit": 50}


def test_a_stray_empty_file_is_configured_but_unreadable(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """No schema is not an empty queue; it is a store this reader cannot use."""
    paths["orchestrator"].parent.mkdir(parents=True, exist_ok=True)
    paths["orchestrator"].write_bytes(b"")
    body = client.get("/api/autonomy/status").json()
    assert body["orchestrator"] == {"configured": True, "readable": False}
    assert paths["orchestrator"].read_bytes() == b""


def test_status_lists_symbols_with_a_month_partition(
    client: TestClient, paths: dict[str, Path]
) -> None:
    partition = paths["data_root"] / "XAUUSD" / "M1" / "2026" / "01.parquet"
    partition.parent.mkdir(parents=True, exist_ok=True)
    partition.touch()
    (paths["data_root"] / "EMPTY" / "M1").mkdir(parents=True)
    deep = paths["data_root"] / "DEEP" / "M1" / "2026" / "01" / "02" / "x.parquet"
    deep.parent.mkdir(parents=True, exist_ok=True)
    deep.touch()
    (paths["data_root"] / "imports").mkdir(parents=True)

    body = client.get("/api/autonomy/status").json()
    assert body["data"] == {"configured": True, "readable": True, "symbols": ["XAUUSD"]}


# ------------------------------------------------------------------- seeded state


def test_status_reports_real_paper_and_orchestrator_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_paper(paths["paper"])
    _seed_orchestrator(paths["orchestrator"])

    body = client.get("/api/autonomy/status").json()
    assert body["paper"] == {
        "configured": True,
        "readable": True,
        "sessions": 2,
        "open_positions": 1,
        "closed_trades": 1,
    }
    orchestrator = body["orchestrator"]
    assert orchestrator["configured"] is True
    assert orchestrator["readable"] is True
    assert (orchestrator["queued"], orchestrator["running"], orchestrator["failed"]) == (1, 1, 1)
    assert orchestrator["paused"] is True
    assert orchestrator["safe_mode"] is True
    assert orchestrator["research_disabled"] is True
    assert orchestrator["last_event"]["event"] == "research_disabled"


def test_status_reports_champion_metadata_and_discovery_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    candidate_id = _seed_archive(paths["archive"])

    body = client.get("/api/autonomy/status").json()
    assert body["champions_readable"] is True
    assert body["champions"] == [
        {
            "scope": SCOPE,
            "candidate_id": candidate_id,
            "mode": "paper",
            "promoted_at": body["champions"][0]["promoted_at"],
            "policy_id": "policy-42",
            "policy_digest": "feedface",
        }
    ]
    assert body["discoveries"] == {
        "configured": True,
        "readable": True,
        "runs": 1,
        "decisions": 2,
    }


def test_champions_route_reads_the_same_archive_without_migrating(
    client: TestClient, paths: dict[str, Path]
) -> None:
    candidate_id = _seed_archive(paths["archive"])
    before = paths["archive"].read_bytes()
    response = client.get("/api/autonomy/champions")
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert [item["candidate_id"] for item in body["items"]] == [candidate_id]
    assert body["items"][0]["mode"] == "paper"
    assert paths["archive"].read_bytes() == before


def test_an_older_archive_without_champions_answers_readable_and_empty(
    client: TestClient, paths
) -> None:
    """A v1 archive has candidates/trials/decisions but no champion tables."""
    path = paths["archive"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE candidates (candidate_id TEXT PRIMARY KEY)")
        conn.execute("CREATE TABLE trials (id INTEGER PRIMARY KEY)")
        conn.execute("CREATE TABLE decisions (id INTEGER PRIMARY KEY)")
        conn.commit()
    finally:
        conn.close()
    before = path.read_bytes()

    body = client.get("/api/autonomy/champions").json()
    assert body == {"configured": True, "readable": True, "items": []}
    status = client.get("/api/autonomy/status").json()
    assert status["champions"] == []
    assert status["champions_readable"] is True
    assert status["discoveries"] == {
        "configured": True,
        "readable": True,
        "runs": 0,
        "decisions": 0,
    }
    # A viewer must not run the v2 migration that opening DiscoveryArchive would.
    assert path.read_bytes() == before


def test_champions_without_metadata_fall_back_to_the_candidate_digest(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """Toy payloads carry no mode or policy; the archive id is still a real digest."""
    path = paths["archive"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with DiscoveryArchive(path) as archive:
        record = archive.add_candidate(ToyCandidate((1, 2)), code_version="test-2026-11-24")
        archive.record_trial(
            record.candidate_id,
            TrialResult(metrics={"net_profit": 1.0}, accepted=True, rejections=()),
            dataset_id=DATASET_ID,
            code_version="test-2026-11-24",
            seed=1,
        )
        archive.promote(
            record.candidate_id,
            scope=SCOPE,
            reason="small but measured",
            metric="net_profit",
        )

    body = client.get("/api/autonomy/champions").json()
    assert body["readable"] is True
    item = body["items"][0]
    assert item["mode"] == "dsl"
    assert item.get("policy_id") is None
    assert item["policy_digest"] == item["candidate_id"]


def test_a_null_promotion_stamp_is_not_stringified(
    client: TestClient, paths: dict[str, Path]
) -> None:
    path = paths["archive"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute("CREATE TABLE champions (scope TEXT, candidate_id TEXT, promoted_at TEXT)")
        conn.execute("INSERT INTO champions VALUES ('XAUUSD:M5', 'abc', NULL)")
        conn.commit()
    finally:
        conn.close()

    item = client.get("/api/autonomy/champions").json()["items"][0]
    assert item.get("promoted_at") is None


def test_a_v1_queue_without_defer_count_is_unreadable_everywhere(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """`/status` and `/jobs` must agree that a v1 jobs table cannot be served."""
    path = paths["orchestrator"]
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            CREATE TABLE jobs (
                id INTEGER PRIMARY KEY, kind TEXT, status TEXT, attempts INTEGER,
                priority INTEGER, created_at TEXT, scheduled_at TEXT,
                started_at TEXT, finished_at TEXT
            )
            """
        )
        conn.execute(
            "INSERT INTO jobs VALUES (1, 'discovery.run', 'queued', 0, 0, "
            "'2026-01-01', '2026-01-01', NULL, NULL)"
        )
        conn.commit()
    finally:
        conn.close()

    status = client.get("/api/autonomy/status").json()
    assert status["orchestrator"] == {"configured": True, "readable": False}
    assert client.get("/api/autonomy/jobs").json() == {
        "configured": True,
        "readable": False,
        "limit": 50,
    }


# ------------------------------------------------------------------------- jobs


def test_jobs_are_newest_first_and_bounded(client: TestClient, paths: dict[str, Path]) -> None:
    _seed_orchestrator(paths["orchestrator"])
    body = client.get("/api/autonomy/jobs", params={"limit": 2}).json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert body["limit"] == 2
    assert len(body["items"]) == 2
    assert [item["status"] for item in body["items"]] == ["queued", "running"]
    assert [item["attempts"] for item in body["items"]] == [0, 1]
    assert all("defer_count" in item for item in body["items"])


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"limit": -1}])
def test_jobs_limit_is_bounded(client: TestClient, params: dict[str, int]) -> None:
    assert client.get("/api/autonomy/jobs", params=params).status_code == 422


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 501}, {"limit": -1}])
def test_events_limit_is_bounded(client: TestClient, params: dict[str, int]) -> None:
    assert client.get("/api/autonomy/events", params=params).status_code == 422


def test_events_are_newest_first_and_do_not_expose_detail(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_orchestrator(paths["orchestrator"])
    body = client.get("/api/autonomy/events", params={"limit": 3}).json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert [item["event"] for item in body["items"]] == [
        "research_disabled",
        "safe_mode_on",
        "paused",
    ]
    assert body["items"][0].get("job_id") is None
    assert all("detail" not in item for item in body["items"])


# ------------------------------------------------------------------- read-only


def test_get_is_served_while_a_writer_holds_a_transaction(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """A dashboard GET must not wait for the orchestrator's ``BEGIN IMMEDIATE``."""
    _seed_orchestrator(paths["orchestrator"])
    writer = JobStore(paths["orchestrator"])
    writer.connection.execute("BEGIN IMMEDIATE")
    try:
        status = client.get("/api/autonomy/status")
        jobs = client.get("/api/autonomy/jobs")
        events = client.get("/api/autonomy/events")
    finally:
        writer.connection.rollback()
        writer.close()

    assert status.status_code == 200, status.text
    assert status.json()["orchestrator"]["queued"] == 1
    assert jobs.status_code == 200 and len(jobs.json()["items"]) == 3
    assert events.status_code == 200 and events.json()["items"]


def test_get_leaves_seeded_databases_byte_identical_and_pins_the_tree(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_paper(paths["paper"])
    _seed_orchestrator(paths["orchestrator"])
    _seed_archive(paths["archive"])
    before = {key: paths[key].read_bytes() for key in ("paper", "orchestrator", "archive")}

    for url in (
        "/api/autonomy/status",
        "/api/autonomy/champions",
        "/api/autonomy/jobs",
        "/api/autonomy/events",
    ):
        assert client.get(url).status_code == 200, url

    for key, payload in before.items():
        assert paths[key].read_bytes() == payload, key

    # A read-only WAL connection creates ``-shm``/``-wal`` sidecars. Nothing else
    # may appear: a regular write connection would also create the database or
    # change its bytes, which the snapshot above rejects.
    expected = set()
    for key in ("paper", "orchestrator", "archive"):
        name = str(paths[key])
        expected.update({name, f"{name}-wal", f"{name}-shm"})
    actual = {str(path) for path in paths["runs_root"].parent.rglob("*") if path.is_file()}
    assert actual == expected


def test_every_autonomy_route_is_read_only(client: TestClient) -> None:
    routes = {
        path: set(methods)
        for path, methods in client.app.openapi()["paths"].items()
        if path.startswith("/api/autonomy")
    }
    assert set(routes) == {
        "/api/autonomy/status",
        "/api/autonomy/champions",
        "/api/autonomy/jobs",
        "/api/autonomy/events",
    }
    assert all(methods == {"get"} for methods in routes.values())
    assert client.post("/api/autonomy/status").status_code == 405
