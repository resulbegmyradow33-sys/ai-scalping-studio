"""Autonomy status API: honest missing states, real counts and read-only GETs.

The API is a viewer. These tests pin the two properties that make it safe to
poll: a missing store is reported as unconfigured rather than as zeros, and a
GET never creates, migrates or rewrites a database.
"""

from __future__ import annotations

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
    return {
        "data_root": tmp_path / "data",
        "runs_root": tmp_path / "runs",
        "paper": tmp_path / "paper.sqlite",
        "orchestrator": tmp_path / "runs" / "orchestrator.sqlite3",
        "archive": tmp_path / "runs" / "discovery" / "discovery.sqlite",
    }


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
        "paper": {"configured": False},
        "orchestrator": {"configured": False},
        "champions": [],
        "discoveries": {"configured": False},
        "data": {"configured": False},
    }
    assert not paths["paper"].exists()
    assert not paths["orchestrator"].exists()
    assert not paths["archive"].exists()


def test_list_endpoints_of_missing_stores_are_empty_pages(
    client: TestClient, paths: dict[str, Path]
) -> None:
    champions = client.get("/api/autonomy/champions")
    assert champions.status_code == 200
    assert champions.json() == {"configured": False, "items": []}

    jobs = client.get("/api/autonomy/jobs")
    assert jobs.status_code == 200
    assert jobs.json() == {"configured": False, "items": [], "limit": 50}

    events = client.get("/api/autonomy/events")
    assert events.status_code == 200
    assert events.json() == {"configured": False, "items": [], "limit": 100}

    for key in ("paper", "orchestrator", "archive"):
        assert not paths[key].exists()


def test_a_stray_empty_file_is_reported_as_an_existing_store(client: TestClient, paths) -> None:
    """The file exists, so it is configured; nothing is invented behind it."""
    paths["orchestrator"].parent.mkdir(parents=True)
    paths["orchestrator"].write_bytes(b"")
    body = client.get("/api/autonomy/status").json()
    assert body["orchestrator"] == {"configured": True}
    assert paths["orchestrator"].read_bytes() == b""


def test_status_lists_symbols_that_have_timeframe_partitions(
    client: TestClient, paths: dict[str, Path]
) -> None:
    partition = paths["data_root"] / "XAUUSD" / "M1" / "2026" / "01.parquet"
    partition.parent.mkdir(parents=True)
    partition.touch()
    (paths["data_root"] / "EMPTY" / "M1").mkdir(parents=True)
    (paths["data_root"] / "imports").mkdir(parents=True)

    body = client.get("/api/autonomy/status").json()
    assert body["data"] == {"configured": True, "symbols": ["XAUUSD"]}


# ------------------------------------------------------------------- seeded state


def test_status_reports_real_paper_and_orchestrator_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_paper(paths["paper"])
    _seed_orchestrator(paths["orchestrator"])

    body = client.get("/api/autonomy/status").json()
    assert body["paper"] == {
        "configured": True,
        "sessions": 2,
        "open_positions": 1,
        "closed_trades": 1,
    }
    orchestrator = body["orchestrator"]
    assert orchestrator["configured"] is True
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
    assert body["discoveries"] == {"configured": True, "runs": 1, "decisions": 2}


def test_champions_route_reads_the_same_archive_without_migrating(
    client: TestClient, paths: dict[str, Path]
) -> None:
    candidate_id = _seed_archive(paths["archive"])
    before = paths["archive"].read_bytes()
    response = client.get("/api/autonomy/champions")
    assert response.status_code == 200
    body = response.json()
    assert body["configured"] is True
    assert [item["candidate_id"] for item in body["items"]] == [candidate_id]
    assert body["items"][0]["mode"] == "paper"
    assert paths["archive"].read_bytes() == before


def test_an_older_archive_without_champions_answers_empty(client: TestClient, paths) -> None:
    """A v1 archive has candidates/trials/decisions but no champion tables."""
    import sqlite3

    path = paths["archive"]
    path.parent.mkdir(parents=True)
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
    assert body == {"configured": True, "items": []}
    status = client.get("/api/autonomy/status").json()
    assert status["champions"] == []
    assert status["discoveries"] == {"configured": True, "runs": 0, "decisions": 0}
    # A viewer must not run the v2 migration that opening DiscoveryArchive would.
    assert path.read_bytes() == before


def test_champions_without_metadata_fall_back_to_the_candidate_digest(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """Toy payloads carry no mode or policy; the archive id is still a real digest."""
    path = paths["archive"]
    path.parent.mkdir(parents=True)
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

    item = client.get("/api/autonomy/champions").json()["items"][0]
    assert item["mode"] == "dsl"
    assert item["policy_id"] is None
    assert item["policy_digest"] == item["candidate_id"]


# ------------------------------------------------------------------------- jobs


def test_jobs_are_newest_first_and_bounded(client: TestClient, paths: dict[str, Path]) -> None:
    _seed_orchestrator(paths["orchestrator"])
    body = client.get("/api/autonomy/jobs", params={"limit": 2}).json()
    assert body["configured"] is True
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


def test_events_are_newest_first(client: TestClient, paths: dict[str, Path]) -> None:
    _seed_orchestrator(paths["orchestrator"])
    body = client.get("/api/autonomy/events", params={"limit": 3}).json()
    assert body["configured"] is True
    assert [item["event"] for item in body["items"]] == [
        "research_disabled",
        "safe_mode_on",
        "paused",
    ]
    assert body["items"][0]["job_id"] is None


# ------------------------------------------------------------------- read-only


def test_get_never_creates_a_missing_store(client: TestClient, paths: dict[str, Path]) -> None:
    for url in (
        "/api/autonomy/status",
        "/api/autonomy/champions",
        "/api/autonomy/jobs",
        "/api/autonomy/events",
    ):
        assert client.get(url).status_code == 200, url
    for key in ("paper", "orchestrator", "archive"):
        assert not paths[key].exists()
    assert not paths["runs_root"].exists()


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


def test_get_leaves_seeded_databases_byte_identical(
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


def test_every_autonomy_route_is_read_only(client: TestClient) -> None:
    paths = {
        path: set(methods)
        for path, methods in client.app.openapi()["paths"].items()
        if path.startswith("/api/autonomy")
    }
    assert set(paths) == {
        "/api/autonomy/status",
        "/api/autonomy/champions",
        "/api/autonomy/jobs",
        "/api/autonomy/events",
    }
    assert all(methods == {"get"} for methods in paths.values())
    assert client.post("/api/autonomy/status").status_code == 405
