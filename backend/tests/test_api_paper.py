"""Paper API tests: the session record, its history and the read/write split.

The API writes two things only — a session row and its close — and reads
everything else. Reads are expected to work while a writer holds a transaction,
which the concurrency test proves with a second connection holding
``BEGIN IMMEDIATE``.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.paper.ledger import PaperLedger
from scalping.types import Side

BODY = {"symbol": "XAUUSD", "timeframe": "M1", "strategy": "flow_momentum"}
T0 = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


@pytest.fixture
def client(tmp_path):
    app = create_app(data_root=tmp_path)
    return TestClient(app)


def _create(client, **overrides) -> str:
    response = client.post("/api/paper/sessions", json={**BODY, **overrides})
    assert response.status_code == 201, response.text
    return str(response.json()["session_id"])


def _seed(ledger_path: Path, session_id: str, *, close_position: bool, marks: int = 5) -> None:
    """Write a deterministic session directly, without the HTTP layer."""
    with PaperLedger(ledger_path) as book:
        book.open_session(session_id, symbol="XAUUSD", timeframe="M1", strategy="seed")
        signal_id = book.record_signal(session_id, T0, "buy")
        order_id = book.record_order(
            session_id,
            signal_id=signal_id,
            ts=T0,
            side=Side.LONG,
            lot=0.1,
            requested_price=2350.0,
        )
        book.record_fill(order_id, ts=T0, price=2350.1, lot=0.1, sl=2349.0, tp=2352.0)
        for index in range(marks):
            book.record_equity(
                session_id,
                T0 + timedelta(minutes=index),
                10_000.0 + index,
                10_000.0 + index,
            )
        if close_position:
            book.record_trade(
                session_id,
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


# ------------------------------------------------------------------ open and list


def test_open_session_returns_a_token_and_lists_it(client):
    session_id = _create(client)
    assert re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id)

    listed = client.get("/api/paper/sessions").json()
    assert listed["total"] == 1
    assert listed["limit"] == 100 and listed["offset"] == 0
    assert listed["items"][0]["session_id"] == session_id
    assert listed["items"][0]["status"] == "running"
    assert listed["items"][0]["timeframe"] == "M1"


def test_open_session_records_the_request_config(client):
    session_id = _create(client, config={"note": "smoke"})
    summary = client.get(f"/api/paper/sessions/{session_id}").json()
    assert summary["config"] == {"origin": "api", "extra": {"note": "smoke"}}
    assert summary["status"] == "running"


def test_open_session_ignores_any_path_in_the_body(client, tmp_path):
    """Extra fields are not request options: the ledger stays where the app put it."""
    evil = tmp_path / "evil.sqlite"
    response = client.post(
        "/api/paper/sessions",
        json={**BODY, "ledger": str(evil), "path": str(evil), "session_id": "chosen"},
    )
    assert response.status_code == 201, response.text
    assert response.json()["session_id"] != "chosen"
    assert not evil.exists()
    assert (tmp_path / "paper.sqlite").exists()


def test_the_factory_can_move_the_ledger(tmp_path):
    custom = tmp_path / "state" / "paper.db"
    app = create_app(data_root=tmp_path / "data", paper_ledger=custom)
    with TestClient(app) as client:
        assert client.post("/api/paper/sessions", json=BODY).status_code == 201
    assert custom.exists()
    assert app.state.paper_ledger_path == custom
    assert not (tmp_path / "data" / "paper.sqlite").exists()


def test_listing_an_absent_ledger_creates_nothing(client, tmp_path):
    response = client.get("/api/paper/sessions")
    assert response.status_code == 200
    assert response.json() == {"items": [], "total": 0, "limit": 100, "offset": 0}
    assert not (tmp_path / "paper.sqlite").exists()


# ------------------------------------------------------------------------ summary


def test_summary_reports_account_state_and_open_position(client, tmp_path):
    ledger = tmp_path / "paper.sqlite"
    _seed(ledger, "open-one", close_position=False, marks=3)

    summary = client.get("/api/paper/sessions/open-one").json()
    assert summary["balance"] == 10_002.0
    assert summary["equity"] == 10_002.0
    assert summary["trade_count"] == 0
    assert summary["last_bar_ts"] is not None
    position = summary["open_position"]
    assert position is not None
    assert position["side"] == "long" and position["status"] == "open"
    assert position["sl"] == 2349.0 and position["tp"] == 2352.0


def test_summary_counts_closed_trades(client, tmp_path):
    _seed(tmp_path / "paper.sqlite", "traded", close_position=True)
    summary = client.get("/api/paper/sessions/traded").json()
    assert summary["trade_count"] == 1
    assert summary["open_position"] is None


def test_summary_of_a_fresh_session_has_no_balance(client):
    session_id = _create(client)
    summary = client.get(f"/api/paper/sessions/{session_id}").json()
    assert summary["balance"] is None and summary["equity"] is None
    assert summary["last_bar_ts"] is None
    assert summary["trade_count"] == 0


# -------------------------------------------------------------------------- close


def test_close_is_idempotent_in_effect_and_refuses_the_retry(client):
    session_id = _create(client)
    first = client.post(f"/api/paper/sessions/{session_id}/close")
    assert first.status_code == 200
    closed = first.json()
    assert closed["status"] == "closed" and closed["ended_at"] is not None

    second = client.post(f"/api/paper/sessions/{session_id}/close")
    assert second.status_code == 409
    assert "already closed" in second.json()["detail"]

    after = client.get(f"/api/paper/sessions/{session_id}").json()
    assert after["status"] == "closed"
    assert after["ended_at"] == closed["ended_at"]

    # The ledger's own close cannot move the stamp either.
    with PaperLedger(Path(client.app.state.paper_ledger_path)) as book:
        stamp = datetime.fromisoformat(closed["ended_at"].removesuffix("Z") + "+00:00")
        assert book.close_session(session_id).ended_at == stamp


# --------------------------------------------------------------------- pagination


@pytest.mark.parametrize(
    "limit,offset",
    [(0, 0), (1001, 0), (100, -1)],
)
@pytest.mark.parametrize(
    "suffix",
    ["/positions", "/orders", "/history", "/equity"],
)
def test_pagination_bounds_are_enforced(client, suffix, limit, offset):
    session_id = _create(client)
    response = client.get(
        f"/api/paper/sessions/{session_id}{suffix}",
        params={"limit": limit, "offset": offset},
    )
    assert response.status_code == 422, response.text


def test_pagination_bounds_apply_to_the_session_list(client):
    assert client.get("/api/paper/sessions", params={"limit": 0}).status_code == 422
    assert client.get("/api/paper/sessions", params={"limit": 1001}).status_code == 422
    assert client.get("/api/paper/sessions", params={"offset": -1}).status_code == 422


def test_pages_slice_rows_and_report_the_total(client, tmp_path):
    _seed(tmp_path / "paper.sqlite", "paged", close_position=True, marks=5)

    history = client.get(
        "/api/paper/sessions/paged/history", params={"limit": 1, "offset": 0}
    ).json()
    assert history["total"] == 1 and len(history["items"]) == 1
    assert history["items"][0]["exit_reason"] == "tp"

    positions = client.get("/api/paper/sessions/paged/positions", params={"limit": 2}).json()
    assert positions["total"] == 1
    assert positions["items"][0]["status"] == "closed"

    orders = client.get("/api/paper/sessions/paged/orders", params={"limit": 1, "offset": 5}).json()
    assert orders["total"] == 1 and orders["items"] == []


def test_equity_returns_the_newest_window_in_order(client, tmp_path):
    _seed(tmp_path / "paper.sqlite", "curve", close_position=False, marks=5)

    tail = client.get("/api/paper/sessions/curve/equity", params={"limit": 2}).json()
    assert tail["total"] == 5
    assert [point["balance"] for point in tail["items"]] == [10_003.0, 10_004.0]

    older = client.get("/api/paper/sessions/curve/equity", params={"limit": 2, "offset": 2}).json()
    assert [point["balance"] for point in older["items"]] == [10_001.0, 10_002.0]


# ---------------------------------------------------------- concurrency and ids


def test_reads_work_while_a_writer_holds_a_transaction(client, tmp_path):
    """A GET must not wait for, or join, the runner's ``BEGIN IMMEDIATE``."""
    ledger = tmp_path / "paper.sqlite"
    _seed(ledger, "committed", close_position=False)
    writer = PaperLedger(ledger)
    writer.connection.execute("BEGIN IMMEDIATE")
    writer.connection.execute(
        """
        INSERT INTO sessions (
            session_id, symbol, timeframe, strategy, started_at, status, config_json
        ) VALUES ('uncommitted', 'XAUUSD', 'M1', 'seed', ?, 'running', '{}')
        """,
        (T0.isoformat(),),
    )
    try:
        listed = client.get("/api/paper/sessions")
        summary = client.get("/api/paper/sessions/committed")
        history = client.get("/api/paper/sessions/committed/history")
    finally:
        writer.connection.rollback()
        writer.close()

    assert listed.status_code == 200
    ids = [item["session_id"] for item in listed.json()["items"]]
    assert ids == ["committed"]
    assert summary.status_code == 200 and summary.json()["balance"] == 10_004.0
    assert history.status_code == 200


def test_unknown_session_is_404_everywhere(client):
    for suffix in ["", "/positions", "/orders", "/history", "/equity", "/close"]:
        method = client.post if suffix == "/close" else client.get
        response = method(f"/api/paper/sessions/unknown{suffix}")
        assert response.status_code == 404, (suffix, response.text)
        assert "unknown paper session" in response.json()["detail"]


@pytest.mark.parametrize("session_id", [".hidden", "a" * 65, "with space", "a.b"])
def test_invalid_session_ids_are_422(client, session_id):
    assert client.get(f"/api/paper/sessions/{session_id}").status_code == 422


@pytest.mark.parametrize("raw", ["..", "../..", "..%2Fetc", "a%2Fb", "%2e%2e%2f"])
def test_session_id_traversal_is_rejected(client, raw):
    response = client.get(f"/api/paper/sessions/{raw}")
    assert response.status_code in {404, 422}, response.text


def test_bad_bodies_are_422(client, tmp_path):
    assert client.post("/api/paper/sessions", json={**BODY, "symbol": "../x"}).status_code == 422
    assert client.post("/api/paper/sessions", json={**BODY, "timeframe": "Y1"}).status_code == 422
    assert client.post("/api/paper/sessions", json={**BODY, "strategy": ""}).status_code == 422
    assert not (tmp_path / "paper.sqlite").exists()


def test_a_stray_empty_file_is_read_as_an_empty_ledger(client, tmp_path):
    """A read treats a table-less file as "no ledger yet" and leaves it byte-identical."""
    stray = tmp_path / "paper.sqlite"
    stray.write_bytes(b"")
    assert client.get("/api/paper/sessions").json()["total"] == 0
    assert stray.read_bytes() == b""
