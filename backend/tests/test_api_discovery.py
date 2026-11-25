"""Read-only discovery evidence API.

The archive is the evidence log of an autonomous search. The API over it is a
review tool: it must report what is stored, distinguish "nothing archived" from
"archive unreadable", never invent a zero or a metric, and leave the database
exactly as it found it. These tests pin those properties plus the request
boundary (validated ids, bounded limits, no traversal).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from discovery_fakes import ToyCandidate
from scalping.api.app import create_app
from scalping.api.schemas import DISCOVERY_CANDIDATE_PATTERN
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.protocols import TrialResult, canonical_digest

SCOPE = "XAUUSD:M5"
SCOPE_M15 = "XAUUSD:M15"
SCOPE_EUR = "EURUSD:M5"
DATASET_M5 = "validation:XAUUSD:M5:aaaa1111"
DATASET_M15 = "validation:XAUUSD:M15:bbbb2222"
DATASET_EUR = "validation:EURUSD:M5:cccc3333"
CODE_VERSION = "test-2026-11-25"


@pytest.fixture
def paths(tmp_path: Path) -> dict[str, Path]:
    """Configured paths plus pre-created data/runs directories.

    The directories exist before the first request, so a GET that creates a file
    is visible as a new entry instead of being hidden by a parent appearing.
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


def _tree(root: Path) -> dict[str, str]:
    """Every file under ``root`` with its sha256, so a write cannot hide."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _seed_archive(path: Path) -> dict[str, str]:
    """A parent, a mutated child and an unrelated symbol; one champion.

    Returns the candidate ids by label. ``expected`` is also the expected
    evidence: two accepted M5 trials (12.5 and 20.0), one rejected M5 trial, one
    accepted M15 trial for the child and one accepted EURUSD trial.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with DiscoveryArchive(path) as archive:
        parent = archive.add_candidate(ToyCandidate((1, 2, 3)), code_version=CODE_VERSION)
        child = archive.add_candidate(
            ToyCandidate((2, 2, 3)),
            code_version=CODE_VERSION,
            parent_a=parent.candidate_id,
            mutation="mutation",
        )
        other = archive.add_candidate(ToyCandidate((4, 5)), code_version=CODE_VERSION)
        archive.record_trial(
            parent.candidate_id,
            TrialResult(metrics={"net_profit": 12.5, "trades": 2.0}, accepted=True, rejections=()),
            dataset_id=DATASET_M5,
            code_version=CODE_VERSION,
            seed=1,
        )
        archive.record_trial(
            parent.candidate_id,
            TrialResult(
                metrics={"net_profit": -1.0},
                accepted=False,
                rejections=("below_threshold",),
            ),
            dataset_id=DATASET_M5,
            code_version=CODE_VERSION,
            seed=2,
        )
        archive.record_trial(
            child.candidate_id,
            TrialResult(metrics={"net_profit": 20.0, "trades": 3.0}, accepted=True, rejections=()),
            dataset_id=DATASET_M5,
            code_version=CODE_VERSION,
            seed=3,
        )
        archive.record_trial(
            child.candidate_id,
            TrialResult(metrics={"net_profit": 3.0}, accepted=True, rejections=()),
            dataset_id=DATASET_M15,
            code_version=CODE_VERSION,
            seed=4,
        )
        archive.record_trial(
            other.candidate_id,
            TrialResult(metrics={"net_profit": 9.0}, accepted=True, rejections=()),
            dataset_id=DATASET_EUR,
            code_version=CODE_VERSION,
            seed=5,
        )
        archive.promote_exploratory(
            parent.candidate_id,
            scope=SCOPE,
            reason="beats baseline after costs",
            metric="net_profit",
        )
        archive.record_decision(
            parent.candidate_id,
            "rejected",
            "cost stress fails",
            detail={"net_profit": 12.5, "stress": "x2"},
        )
        archive.record_decision(
            child.candidate_id,
            "no_trade",
            "no tradable candidate",
            detail={"reason_code": "no_tradable_candidate"},
        )
    return {
        "parent": parent.candidate_id,
        "child": child.candidate_id,
        "other": other.candidate_id,
    }


def _mark_evidence_champion(path: Path) -> None:
    """Give the seeded champion the v4 governance fields a reader exposes.

    A real evidence promotion needs a full policy bundle; this test only needs a
    row that carries the columns, so the row is updated directly in the fixture
    archive after ``DiscoveryArchive`` has closed.
    """
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "UPDATE champions SET mode = 'evidence', policy_id = 'promotion-v1',"
            " policy_version = 1, policy_digest = ?, evidence_checksum = ?",
            ("a" * 64, "b" * 64),
        )
        conn.commit()
    finally:
        conn.close()


def _minimal_v1_archive(path: Path) -> None:
    """The v1 schema: four tables, no champion governance, no detail column."""
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_id = canonical_digest({"format": "toy-v1", "nodes": [1]})
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);
            INSERT INTO schema_version VALUES (1, '2026-01-01T00:00:00+00:00');
            CREATE TABLE candidates (
                candidate_id TEXT PRIMARY KEY,
                canonical_json TEXT NOT NULL,
                complexity INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                parent_a TEXT,
                parent_b TEXT,
                mutation TEXT,
                code_version TEXT NOT NULL,
                trials INTEGER NOT NULL
            );
            CREATE TABLE trials (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                dataset_id TEXT NOT NULL,
                code_version TEXT NOT NULL,
                seed INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                accepted INTEGER NOT NULL,
                rejections_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE TABLE lineage (
                candidate_id TEXT NOT NULL,
                parent_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                PRIMARY KEY (candidate_id, parent_id, relation)
            );
            CREATE TABLE decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT NOT NULL,
                champion_id TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            "INSERT INTO candidates VALUES (?, ?, 1, '2026-01-01T00:00:00+00:00',"
            " NULL, NULL, NULL, ?, 1)",
            (candidate_id, json.dumps({"format": "toy-v1", "nodes": [1]}), CODE_VERSION),
        )
        conn.execute(
            "INSERT INTO trials (candidate_id, dataset_id, code_version, seed,"
            " metrics_json, accepted, rejections_json, created_at)"
            " VALUES (?, ?, ?, 1, '{\"net_profit\": 2.0}', 1, '[]',"
            " '2026-01-01T00:00:00+00:00')",
            (candidate_id, DATASET_M5, CODE_VERSION),
        )
        conn.execute(
            "INSERT INTO decisions (candidate_id, decision, reason, champion_id,"
            " created_at) VALUES (?, 'no_change', 'baseline stands', NULL,"
            " '2026-01-01T00:00:00+00:00')",
            (candidate_id,),
        )
        conn.commit()
    finally:
        conn.close()


# ------------------------------------------------------------- missing stores


def test_missing_archive_reports_unconfigured_without_invented_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    summary = client.get("/api/discovery/summary")
    assert summary.status_code == 200
    assert summary.json() == {
        "configured": False,
        "readable": False,
        "champions_readable": False,
        "limit": 100,
    }

    candidates = client.get("/api/discovery/candidates")
    assert candidates.status_code == 200
    assert candidates.json() == {
        "configured": False,
        "readable": False,
        "metric": "net_profit",
        "limit": 100,
        "offset": 0,
    }

    trials = client.get("/api/discovery/trials")
    assert trials.status_code == 200
    assert trials.json() == {
        "configured": False,
        "readable": False,
        "limit": 100,
        "offset": 0,
    }

    unknown = canonical_digest({"format": "toy-v1", "nodes": [9]})
    missing = client.get(f"/api/discovery/candidates/{unknown}")
    assert missing.status_code == 404
    assert set(missing.json()) == {"detail"}

    assert not paths["archive"].exists()
    assert _tree(paths["runs_root"]) == {}


def test_corrupt_archive_is_configured_but_unreadable(
    client: TestClient, paths: dict[str, Path]
) -> None:
    paths["archive"].parent.mkdir(parents=True, exist_ok=True)
    paths["archive"].write_bytes(b"this is not a sqlite database")
    before = _tree(paths["archive"].parent)

    summary = client.get("/api/discovery/summary").json()
    assert summary["configured"] is True
    assert summary["readable"] is False
    assert "candidates" not in summary
    assert "trials" not in summary
    assert "champions" not in summary

    candidates = client.get("/api/discovery/candidates").json()
    assert candidates == {
        "configured": True,
        "readable": False,
        "metric": "net_profit",
        "limit": 100,
        "offset": 0,
    }
    trials = client.get("/api/discovery/trials").json()
    assert trials == {
        "configured": True,
        "readable": False,
        "limit": 100,
        "offset": 0,
    }

    unknown = canonical_digest({"format": "toy-v1", "nodes": [9]})
    detail = client.get(f"/api/discovery/candidates/{unknown}")
    assert detail.status_code == 409
    assert str(paths["archive"]) not in detail.text

    assert _tree(paths["archive"].parent) == before


def test_a_v1_archive_without_champions_is_readable_without_migration(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """v1 has candidates/trials/decisions but no champion tables or detail_json."""
    _minimal_v1_archive(paths["archive"])
    before = _tree(paths["archive"].parent)

    summary = client.get("/api/discovery/summary").json()
    assert summary["configured"] is True
    assert summary["readable"] is True
    assert summary["candidates"] == 1
    assert summary["trials"] == 1
    assert summary["decisions"] == 1
    assert summary["champions_readable"] is True
    assert summary["champions"] == []

    candidate_id = canonical_digest({"format": "toy-v1", "nodes": [1]})
    detail = client.get(f"/api/discovery/candidates/{candidate_id}").json()
    assert detail["readable"] is True
    assert detail["decisions"][0]["decision"] == "no_change"
    assert detail["decisions"][0].get("detail") is None

    # A viewer must not run the v2-v4 migrations opening DiscoveryArchive would.
    assert _tree(paths["archive"].parent) == before


# ----------------------------------------------------------------- seeded state


def test_summary_reports_real_totals_and_champion_governance(
    client: TestClient, paths: dict[str, Path]
) -> None:
    ids = _seed_archive(paths["archive"])
    _mark_evidence_champion(paths["archive"])

    body = client.get("/api/discovery/summary").json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert body["candidates"] == 3
    assert body["trials"] == 5
    assert body["accepted_trials"] == 4
    assert body["rejected_trials"] == 1
    assert body["datasets"] == 3
    # Three decisions: the promotion itself, the rejected trial and the no_trade.
    assert body["decisions"] == 3
    assert body["champions_readable"] is True
    champion = body["champions"][0]
    assert champion["scope"] == SCOPE
    assert champion["candidate_id"] == ids["parent"]
    assert champion["mode"] == "evidence"
    assert champion["policy_id"] == "promotion-v1"
    assert champion["policy_version"] == 1
    assert champion["policy_digest"] == "a" * 64
    assert champion["evidence_checksum"] == "b" * 64
    assert champion["trials"] == 2
    assert champion["accepted_trials"] == 1


def test_summary_scope_filters_trials_and_candidates(
    client: TestClient, paths: dict[str, Path]
) -> None:
    _seed_archive(paths["archive"])

    m5 = client.get("/api/discovery/summary", params={"scope": SCOPE}).json()
    assert m5["scope"] == SCOPE
    assert (m5["candidates"], m5["trials"], m5["accepted_trials"]) == (2, 3, 2)
    assert m5["datasets"] == 1
    # A and B both have a trial on M5; all three decisions concern them.
    assert m5["decisions"] == 3
    assert [item["scope"] for item in m5["champions"]] == [SCOPE]

    m15 = client.get("/api/discovery/summary", params={"scope": SCOPE_M15}).json()
    assert (m15["candidates"], m15["trials"], m15["accepted_trials"]) == (1, 1, 1)
    assert m15["champions"] == []

    unknown = client.get("/api/discovery/summary", params={"scope": "GBPUSD:H1"}).json()
    assert (unknown["candidates"], unknown["trials"], unknown["datasets"]) == (0, 0, 0)
    assert unknown["champions"] == []


def test_candidate_list_carries_spec_hash_complexity_and_best_metrics(
    client: TestClient, paths: dict[str, Path]
) -> None:
    ids = _seed_archive(paths["archive"])

    body = client.get("/api/discovery/candidates").json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert body["total"] == 3
    assert body["metric"] == "net_profit"
    items = {item["candidate_id"]: item for item in body["items"]}
    assert set(items) == {ids["parent"], ids["child"], ids["other"]}

    child = items[ids["child"]]
    assert child["complexity"] == 3
    assert child["trials"] == 2
    assert child["accepted_trials"] == 2
    assert child["best_value"] == 20.0
    assert child["best_metrics"] == {"net_profit": 20.0, "trades": 3.0}
    assert child["mutation"] == "mutation"
    assert child["parent_a"] == ids["parent"]

    parent = items[ids["parent"]]
    assert parent["trials"] == 2
    assert parent["accepted_trials"] == 1
    # The rejected -1.0 trial must not become the "best".
    assert parent["best_value"] == 12.5

    # A candidate whose accepted trials never publish the requested metric does
    # not get an invented score.
    missing_metric = client.get(
        "/api/discovery/candidates", params={"metric": "expectancy_r"}
    ).json()
    assert all(item.get("best_value") is None for item in missing_metric["items"])
    assert all(item.get("best_metrics") is None for item in missing_metric["items"])


def test_candidate_scope_narrows_the_evidence_counts(
    client: TestClient, paths: dict[str, Path]
) -> None:
    ids = _seed_archive(paths["archive"])
    body = client.get("/api/discovery/candidates", params={"scope": SCOPE}).json()
    assert body["total"] == 2
    items = {item["candidate_id"]: item for item in body["items"]}
    assert set(items) == {ids["parent"], ids["child"]}
    assert items[ids["child"]]["trials"] == 1  # the M15 trial is out of scope
    assert items[ids["child"]]["best_value"] == 20.0

    eur = client.get("/api/discovery/candidates", params={"scope": SCOPE_EUR}).json()
    assert eur["total"] == 1
    assert eur["items"][0]["candidate_id"] == ids["other"]


def test_candidate_detail_returns_trials_lineage_and_decisions(
    client: TestClient, paths: dict[str, Path]
) -> None:
    ids = _seed_archive(paths["archive"])
    response = client.get(f"/api/discovery/candidates/{ids['child']}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["configured"] is True
    assert body["readable"] is True
    assert body["spec"] == {"format": "toy-v1", "nodes": [2, 2, 3]}
    assert canonical_digest(body["spec"]) == body["candidate"]["candidate_id"]
    assert body["candidate"]["complexity"] == 3
    assert body["trial_total"] == 2

    # Newest first; the full evidence row is present.
    assert [trial["id"] for trial in body["trials"]] == [4, 3]
    assert body["trials"][0]["dataset_id"] == DATASET_M15
    assert body["trials"][0].get("dataset_checksum") is None
    assert body["trials"][0].get("settings_digest") is None
    assert body["trials"][1]["metrics"] == {"net_profit": 20.0, "trades": 3.0}
    assert body["trials"][1]["seed"] == 3
    assert body["trials"][1]["accepted"] is True
    assert body["trials"][1]["rejections"] == []

    assert body["lineage"] == [{"parent_id": ids["parent"], "relation": "mutation"}]
    assert body["decision_total"] == 1
    assert body["decisions"][0]["decision"] == "no_trade"
    assert body["decisions"][0]["detail"] == {"reason_code": "no_tradable_candidate"}


def test_candidate_detail_pages_and_reports_totals(
    client: TestClient, paths: dict[str, Path]
) -> None:
    ids = _seed_archive(paths["archive"])
    first = client.get(
        f"/api/discovery/candidates/{ids['parent']}", params={"limit": 1, "offset": 0}
    ).json()
    second = client.get(
        f"/api/discovery/candidates/{ids['parent']}", params={"limit": 1, "offset": 1}
    ).json()
    assert first["trial_total"] == 2
    assert second["trial_total"] == 2
    assert [trial["id"] for trial in first["trials"]] == [2]
    assert [trial["id"] for trial in second["trials"]] == [1]
    assert first["trials"][0]["accepted"] is False
    assert first["trials"][0]["rejections"] == ["below_threshold"]


def test_unknown_candidate_is_a_404(client: TestClient, paths: dict[str, Path]) -> None:
    _seed_archive(paths["archive"])
    unknown = canonical_digest({"format": "toy-v1", "nodes": [9]})
    response = client.get(f"/api/discovery/candidates/{unknown}")
    assert response.status_code == 404
    assert unknown in response.json()["detail"]


def test_trials_endpoint_filters_and_bounds(client: TestClient, paths: dict[str, Path]) -> None:
    ids = _seed_archive(paths["archive"])

    everything = client.get("/api/discovery/trials").json()
    assert everything["total"] == 5
    assert [trial["id"] for trial in everything["items"]] == [5, 4, 3, 2, 1]
    assert everything["items"][0]["dataset_id"] == DATASET_EUR

    by_candidate = client.get(
        "/api/discovery/trials", params={"candidate_id": ids["parent"]}
    ).json()
    assert by_candidate["total"] == 2
    assert {trial["candidate_id"] for trial in by_candidate["items"]} == {ids["parent"]}

    accepted = client.get("/api/discovery/trials", params={"accepted": False}).json()
    assert accepted["total"] == 1
    assert accepted["items"][0]["accepted"] is False

    by_dataset = client.get("/api/discovery/trials", params={"dataset_id": DATASET_M15}).json()
    assert by_dataset["total"] == 1
    assert by_dataset["items"][0]["candidate_id"] == ids["child"]

    by_scope = client.get("/api/discovery/trials", params={"scope": SCOPE}).json()
    assert by_scope["total"] == 3
    assert all("XAUUSD:M5" in trial["dataset_id"] for trial in by_scope["items"])

    page = client.get("/api/discovery/trials", params={"limit": 2, "offset": 2}).json()
    assert page["total"] == 5
    assert [trial["id"] for trial in page["items"]] == [3, 2]
    assert (page["limit"], page["offset"]) == (2, 2)


# ---------------------------------------------------------------- request edges


@pytest.mark.parametrize(
    ("url", "params"),
    [
        ("/api/discovery/summary", {"limit": 0}),
        ("/api/discovery/summary", {"limit": 501}),
        ("/api/discovery/summary", {"limit": -1}),
        ("/api/discovery/candidates", {"limit": 0}),
        ("/api/discovery/candidates", {"limit": 501}),
        ("/api/discovery/candidates", {"offset": -1}),
        ("/api/discovery/trials", {"limit": 0}),
        ("/api/discovery/trials", {"limit": 501}),
        ("/api/discovery/trials", {"offset": -1}),
        ("/api/discovery/summary", {"scope": ""}),
        ("/api/discovery/summary", {"scope": "../etc"}),
        ("/api/discovery/candidates", {"scope": "XAUUSD M5"}),
        ("/api/discovery/candidates", {"metric": "../net_profit"}),
        ("/api/discovery/trials", {"dataset_id": "../data"}),
        ("/api/discovery/trials", {"candidate_id": "not-a-hash"}),
        ("/api/discovery/trials", {"accepted": "maybe"}),
    ],
)
def test_request_bounds_and_validated_ids_are_422(
    client: TestClient, url: str, params: dict[str, Any]
) -> None:
    assert client.get(url, params=params).status_code == 422


def test_candidate_id_is_a_sha256_and_traversal_never_reaches_a_lookup(
    client: TestClient, paths: dict[str, Path]
) -> None:
    import re

    assert re.fullmatch(DISCOVERY_CANDIDATE_PATTERN, "a" * 64)
    assert not re.fullmatch(DISCOVERY_CANDIDATE_PATTERN, "A" * 64)
    assert not re.fullmatch(DISCOVERY_CANDIDATE_PATTERN, "../" + "a" * 61)

    _seed_archive(paths["archive"])
    for value in (
        "..",
        "../",
        "%2E%2E%2F%2E%2E%2Fetc%2Fpasswd",
        "a" * 63,
        "g" * 64,
        "A" * 64,
        canonical_digest({"format": "toy-v1", "nodes": [9]}) + "0",
    ):
        response = client.get(f"/api/discovery/candidates/{value}")
        assert response.status_code in {404, 422}, value
        assert "passwd" not in response.text


@pytest.mark.parametrize(
    ("url", "params"),
    [
        (f"/api/discovery/candidates/{'a' * 64}", {"limit": 0}),
        (f"/api/discovery/candidates/{'a' * 64}", {"limit": 501}),
        (f"/api/discovery/candidates/{'a' * 64}", {"offset": -1}),
        (f"/api/discovery/candidates/{'a' * 64}", {"metric": "bad metric"}),
    ],
)
def test_candidate_detail_query_bounds_are_422(
    client: TestClient, url: str, params: dict[str, Any]
) -> None:
    assert client.get(url, params=params).status_code == 422


def test_every_discovery_route_is_read_only(client: TestClient) -> None:
    routes = {
        path: set(methods)
        for path, methods in client.app.openapi()["paths"].items()  # type: ignore[attr-defined]
        if path.startswith("/api/discovery")
    }
    assert set(routes) == {
        "/api/discovery/summary",
        "/api/discovery/candidates",
        "/api/discovery/candidates/{candidate_id}",
        "/api/discovery/trials",
    }
    assert all(methods == {"get"} for methods in routes.values())
    assert client.post("/api/discovery/candidates").status_code == 405
    assert client.post("/api/discovery/summary").status_code == 405


# ------------------------------------------------------------------- read only


def test_reads_leave_the_archive_directory_byte_identical(
    client: TestClient, paths: dict[str, Path]
) -> None:
    """Every route, including a GET, must not create a DB, WAL or SHM file."""
    ids = _seed_archive(paths["archive"])
    _mark_evidence_champion(paths["archive"])
    before = _tree(paths["archive"].parent)
    assert set(before) == {"discovery.sqlite"}

    for url, params in (
        ("/api/discovery/summary", {}),
        ("/api/discovery/summary", {"scope": SCOPE}),
        ("/api/discovery/candidates", {}),
        ("/api/discovery/candidates", {"scope": SCOPE, "metric": "net_profit"}),
        (f"/api/discovery/candidates/{ids['parent']}", {}),
        ("/api/discovery/trials", {"scope": SCOPE}),
    ):
        assert client.get(url, params=params).status_code == 200, url

    assert _tree(paths["archive"].parent) == before
    assert not Path(f"{paths['archive']}-wal").exists()
    assert not Path(f"{paths['archive']}-shm").exists()

    # The rows themselves are untouched: the same summary reads the same numbers.
    again = client.get("/api/discovery/summary").json()
    assert (again["candidates"], again["trials"], again["decisions"]) == (3, 5, 3)
