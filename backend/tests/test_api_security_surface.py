"""Security regression tests for the HTTP surface.

The API is a local-only research panel, so the threat model is "a request must
never reach outside what the app factory configured, must never write on a
``GET``, and must never echo the machine's file layout". Each test pins one of
those properties with a request a browser or a script can actually send:

* ids that name runs, ML runs, paper sessions or CSV imports are single opaque
  tokens; separators and absurd lengths are a 4xx from request parsing, never a
  disk probe or a 500;
* every ``GET`` leaves a seeded store byte-identical;
* error bodies carry the request's own value, not an absolute path;
* request values reach SQL only as parameters, never as text;
* the CORS allowlist and the absence of an auth layer are explicit, so binding
  this app to a public interface is a decision, not an accident.

The dependency set is the project's own: pytest, FastAPI's test client and the
ledger/job-store helpers already used by the API tests.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from scalping.api.app import ALLOWED_ORIGINS, create_app
from scalping.orchestrator.store import JobStore
from scalping.paper.ledger import PaperLedger
from scalping.types import Side

T0 = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
DECOY_MARKER = "DECOY-PATH-ESCAPED"


# --------------------------------------------------------------------- fixtures


@pytest.fixture
def roots(tmp_path: Path) -> dict[str, Path]:
    """Configured roots that exist before the first request.

    The directories are pre-created so a handler that writes anything is
    visible as a new file, not hidden by its parent appearing.
    """
    data = tmp_path / "data"
    runs = tmp_path / "runs"
    data.mkdir()
    runs.mkdir()
    return {"tmp": tmp_path, "data": data, "runs": runs}


@pytest.fixture
def client(roots: dict[str, Path]) -> TestClient:
    """The default ledger/queue/archive layout under the roots above."""
    app = create_app(data_root=roots["data"], runs_root=roots["runs"])
    # A 500 body is part of what these tests inspect.
    return TestClient(app, raise_server_exceptions=False)


def _files_under(root: Path) -> list[str]:
    return sorted(str(path.relative_to(root)) for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tables(path: Path) -> list[str]:
    with PaperLedger.open_reader(path) as book:
        rows = book.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
        ).fetchall()
    return [str(row[0]) for row in rows]


def _seed_paper(path: Path) -> None:
    """One session with a settled trade, an equity mark and a closed position."""
    with PaperLedger(path) as book:
        book.open_session("sec1", symbol="XAUUSD", timeframe="M1", strategy="seed")
        signal_id = book.record_signal("sec1", T0, "buy")
        order_id = book.record_order(
            "sec1",
            signal_id=signal_id,
            ts=T0,
            side=Side.LONG,
            lot=0.1,
            requested_price=2350.0,
        )
        book.record_fill(order_id, ts=T0, price=2350.1, lot=0.1, sl=2349.0, tp=2352.0)
        book.record_equity("sec1", T0, 10_000.0, 10_000.0)
        book.record_trade(
            "sec1",
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with JobStore(path) as store:
        store.enqueue("paper.advance", dedupe_key="sec")


def _plant_decoy(directory: Path, name: str = "manifest.json") -> Path:
    """A file outside the configured roots that a traversal would reach."""
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    target.write_text(f'{{"run_id": "outside", "marker": "{DECOY_MARKER}", "status": "completed"}}')
    return target


PAPER_READS = (
    "/api/paper/sessions",
    "/api/paper/sessions/sec1",
    "/api/paper/sessions/sec1/positions",
    "/api/paper/sessions/sec1/orders",
    "/api/paper/sessions/sec1/history",
    "/api/paper/sessions/sec1/equity",
)

AUTONOMY_READS = (
    "/api/autonomy/status",
    "/api/autonomy/champions",
    "/api/autonomy/jobs",
    "/api/autonomy/events",
)


# ---------------------------------------------------------------- path traversal


def test_train_run_ids_cannot_escape_the_runs_root(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Every train run-id route answers 4xx and never reads the decoy manifest."""
    _plant_decoy(roots["tmp"] / "outside")
    ids = ["../outside", "..%2Foutside", "%2e%2e%2foutside", "a%2Fb", "..%5Coutside", "a" * 5000]

    for raw in ids:
        detail = client.get(f"/api/runs/{raw}")
        assert detail.status_code in {404, 422}, (raw, detail.status_code, detail.text)
        assert DECOY_MARKER not in detail.text, raw
        assert str(roots["tmp"]) not in detail.text, raw

        control = client.post(f"/api/runs/{raw}/control", json={"action": "pause"})
        assert control.status_code in {404, 422}, (raw, control.status_code)

    bodies = [
        {"run_id": "../outside"},
        {"run_id": "../../../etc/passwd"},
        {"run_id": "/etc/passwd"},
        {"run_id": "..\\outside"},
        {"run_id": "a" * 5000},
        {"run_id": "a\u0000b"},
    ]
    for body in bodies:
        for endpoint in ("/api/backtest", "/api/evaluate"):
            response = client.post(endpoint, json={**body, "symbol": "XAUUSD"})
            assert response.status_code in {404, 422}, (endpoint, body, response.status_code)
            assert response.status_code != 500
            assert DECOY_MARKER not in response.text


def test_run_id_length_is_rejected_before_the_filesystem(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Regression: an over-long id used to reach ``os.stat`` and raise OSError.

    ``ENAMETOOLONG`` escaped as a 500 and put the absolute run path in the
    traceback. The boundary now answers 422 without touching the disk.
    """
    oversized = "a" * 5000
    detail = client.get(f"/api/runs/{oversized}")
    assert detail.status_code == 422, detail.text
    backtest = client.post("/api/backtest", json={"run_id": oversized, "symbol": "XAUUSD"})
    assert backtest.status_code == 422, backtest.text
    evaluate = client.post("/api/evaluate", json={"run_id": oversized, "symbol": "XAUUSD"})
    assert evaluate.status_code == 422, evaluate.text
    assert str(roots["runs"]) not in detail.text


def test_the_runner_guard_is_a_second_line_of_defence(roots: dict[str, Path]) -> None:
    """A direct caller (no HTTP schema) is refused by ``_run_dir`` itself."""
    from scalping.api.train_runner import TrainRunner, UnknownRunError
    from scalping.config import Settings
    from scalping.data.store import ParquetStore

    runner = TrainRunner(ParquetStore(roots["data"]), Settings(), roots["runs"])
    for bad in ("", ".", "..", "../outside", "a/b", "a\\b", "a" * 5000):
        with pytest.raises(UnknownRunError):
            runner._run_dir(bad)


def test_ml_and_paper_ids_cannot_escape_their_stores(
    client: TestClient, roots: dict[str, Path]
) -> None:
    _plant_decoy(roots["tmp"] / "outside", "meta.json")
    _plant_decoy(roots["tmp"] / "outside", "report.json")

    for raw in ("..%2Foutside", "%2e%2e%2foutside", "a%2Fb", "a" * 5000):
        ml = client.get(f"/api/ml/runs/{raw}")
        assert ml.status_code in {404, 422}, (raw, ml.status_code)
        assert DECOY_MARKER not in ml.text

        paper = client.get(f"/api/paper/sessions/{raw}")
        assert paper.status_code in {404, 422}, (raw, paper.status_code)

    for body in ({"run_id": "../outside"}, {"run_id": "a" * 5000}, {"run_id": "a\u0000b"}):
        response = client.post("/api/ml/evaluate", json=body)
        assert response.status_code in {404, 422}, (body, response.status_code)


def test_csv_import_error_names_no_absolute_path(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Regression: the 422 used to interpolate ``<data_root>/imports`` verbatim."""
    response = client.post(
        "/api/data/download",
        json={
            "provider": "csv",
            "symbol": "XAUUSD",
            "path": "missing.csv",
            "start": "2026-01-05T00:00:00Z",
            "end": "2026-01-07T00:00:00Z",
        },
    )

    assert response.status_code == 422, response.text
    assert "missing.csv" in response.json()["detail"]
    assert str(roots["data"]) not in response.text
    assert str(roots["tmp"]) not in response.text


# --------------------------------------------------------------------- read-only


def test_paper_reads_leave_a_seeded_ledger_byte_identical(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Every paper ``GET`` and the autonomy view read without writing."""
    ledger = roots["data"] / "paper.sqlite"
    _seed_paper(ledger)
    before_hash = _sha256(ledger)
    before_tables = _tables(ledger)
    before_files = set(_files_under(roots["data"]))

    for url in PAPER_READS + ("/api/paper/sessions/unknown", *AUTONOMY_READS):
        response = client.get(url)
        assert response.status_code in {200, 404}, (url, response.status_code)

    assert _sha256(ledger) == before_hash
    assert _tables(ledger) == before_tables
    # A read-only WAL connection may create its own sidecars; no other file is
    # allowed to appear, and the database itself is byte-identical above.
    after_files = set(_files_under(roots["data"]))
    assert after_files <= before_files | {"paper.sqlite-wal", "paper.sqlite-shm"}


def test_get_surface_creates_no_store_when_the_ledger_is_absent(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Pre-created directories stay empty: no ledger, queue or archive is made."""
    for url in PAPER_READS + AUTONOMY_READS:
        assert client.get(url).status_code in {200, 404}, url

    for key in ("data", "runs"):
        assert _files_under(roots[key]) == [], key
    for name in ("paper.sqlite", "orchestrator.sqlite3"):
        assert not (roots["data"] / name).exists()
    assert not (roots["runs"] / "discovery" / "discovery.sqlite").exists()


def test_autonomy_gets_leave_seeded_stores_byte_identical(
    client: TestClient, roots: dict[str, Path]
) -> None:
    orchestrator = roots["runs"] / "orchestrator.sqlite3"
    _seed_orchestrator(orchestrator)
    _seed_paper(roots["data"] / "paper.sqlite")
    before = {path: _sha256(path) for path in (orchestrator, roots["data"] / "paper.sqlite")}

    for url in AUTONOMY_READS:
        assert client.get(url).status_code == 200, url

    for path, digest in before.items():
        assert _sha256(path) == digest, path


# ------------------------------------------------------------ information leak


def test_error_bodies_do_not_expose_the_machine_layout(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """404/409/422 bodies carry the request's value, never an absolute path."""
    _seed_paper(roots["data"] / "paper.sqlite")
    needles = (str(roots["data"]), str(roots["runs"]), str(roots["tmp"]))

    responses = [
        client.get("/api/runs/nope"),
        client.post("/api/runs/nope/control", json={"action": "pause"}),
        client.post("/api/backtest", json={"run_id": "nope", "symbol": "XAUUSD"}),
        client.post("/api/evaluate", json={"run_id": "nope", "symbol": "XAUUSD"}),
        client.get("/api/ml/runs/nope"),
        client.post("/api/ml/evaluate", json={"run_id": "nope"}),
        client.get("/api/paper/sessions/nope"),
        client.get("/api/paper/sessions/nope/history"),
        client.get("/api/data/status", params={"symbol": "XAU/USD"}),
    ]
    for response in responses:
        assert response.status_code >= 400, response.url
        for needle in needles:
            assert needle not in response.text, (response.url, needle)
        assert "Traceback" not in response.text, response.url
        assert 'File "' not in response.text, response.url


def test_unhandled_errors_do_not_publish_a_traceback(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """A corrupt ledger is a local misconfiguration, not a stack-trace export."""
    (roots["data"] / "paper.sqlite").write_bytes(b"not a database at all")

    response = client.get("/api/paper/sessions")

    assert "Traceback" not in response.text
    assert 'File "' not in response.text
    assert str(roots["data"]) not in response.text
    # The autonomy view already degrades gracefully; this route may not, but it
    # must never answer 200 with invented state.
    assert response.status_code in {500, 503}


def test_loader_errors_do_not_embed_absolute_paths(
    client: TestClient, roots: dict[str, Path]
) -> None:
    """Regression: loader exceptions carried the path they failed to open.

    Both a partial ML run (meta written, metrics missing) and a train run with
    a manifest but no policy used to answer 409 with ``'/abs/...'`` inside the
    detail. The reason survives; the layout does not.
    """
    ml_dir = roots["runs"] / "ml" / "abc123"
    ml_dir.mkdir(parents=True)
    (ml_dir / "meta.json").write_text(
        '{"run_id": "abc123", "status": "completed", "symbol": "XAUUSD", "timeframe": "H1"}'
    )
    # A present but unreadable artefact makes ``load_result`` fail after the
    # route's "has model" check, which is where the path used to surface.
    (ml_dir / "model.joblib").write_bytes(b"\x00garbage")
    ml = client.post("/api/ml/evaluate", json={"run_id": "abc123"})

    train_dir = roots["runs"] / "abc123"
    train_dir.mkdir(parents=True)
    (train_dir / "manifest.json").write_text(
        '{"run_id": "abc123", "status": "completed", "created_at": "2026-01-01T00:00:00+00:00"'
        ', "steps": 1}'
    )
    train = client.post("/api/evaluate", json={"run_id": "abc123", "symbol": "XAUUSD"})

    for response in (ml, train):
        assert response.status_code == 409, response.text
        assert str(roots["tmp"]) not in response.text
        assert str(roots["runs"]) not in response.text
        assert "/metrics.json" not in response.text
        assert "/model.zip" not in response.text
    assert "cannot be loaded" in ml.json()["detail"]
    assert "cannot be loaded" in train.json()["detail"]


def test_public_error_text_scrubs_paths_and_keeps_the_reason(tmp_path: Path) -> None:
    from scalping.api.errors import public_error_text

    missing = FileNotFoundError(2, "No such file or directory", str(tmp_path / "runs" / "a" / "b"))
    scrubbed = public_error_text(missing, root=tmp_path / "runs")
    assert str(tmp_path) not in scrubbed
    assert "No such file or directory" in scrubbed
    assert "<root><path>" not in scrubbed

    assert public_error_text(ValueError("expected 6/8 fields")) == "expected 6/8 fields"
    relative = public_error_text(ValueError("cannot open data/imports/x.csv"), root=Path("data"))
    assert relative == "cannot open data/imports/x.csv"
    denied = public_error_text(PermissionError(13, "Permission denied", "/etc/shadow"))
    assert "/etc/shadow" not in denied
    assert "<path>" in denied
    assert denied.endswith("'<path>'")


# -------------------------------------------------------------------- injection


def test_sql_metacharacters_in_ids_are_rejected_and_stores_are_untouched(
    client: TestClient, roots: dict[str, Path]
) -> None:
    ledger = roots["data"] / "paper.sqlite"
    _seed_paper(ledger)
    before_hash = _sha256(ledger)
    before_tables = _tables(ledger)

    hostile = [
        "' OR 1=1--",
        "x'; DROP TABLE sessions;--",
        "1 UNION SELECT session_id FROM sessions",
        "sec1' AND '1'='1",
    ]
    for value in hostile:
        quoted = value.replace(" ", "%20").replace("'", "%27").replace(";", "%3B")
        paper = client.get(f"/api/paper/sessions/{quoted}")
        assert paper.status_code in {404, 422}, (value, paper.status_code)
        run = client.post("/api/backtest", json={"run_id": value, "symbol": "XAUUSD"})
        assert run.status_code in {404, 422}, (value, run.status_code)
        ml = client.post("/api/ml/evaluate", json={"run_id": value})
        assert ml.status_code in {404, 422}, (value, ml.status_code)

    assert _sha256(ledger) == before_hash
    assert _tables(ledger) == before_tables
    assert "sessions" in before_tables


@pytest.mark.parametrize(
    ("url", "params"),
    [
        ("/api/paper/sessions", {"limit": "abc"}),
        ("/api/paper/sessions", {"limit": "1.5"}),
        ("/api/paper/sessions", {"limit": "true"}),
        ("/api/paper/sessions", {"offset": "-1"}),
        ("/api/paper/sessions/sec1/history", {"limit": 0}),
        ("/api/paper/sessions/sec1/history", {"limit": 1001}),
        ("/api/autonomy/jobs", {"limit": 501}),
        ("/api/autonomy/events", {"limit": 0}),
        ("/api/regime", {"symbol": "XAUUSD", "limit": 5001}),
    ],
)
def test_pagination_parameters_are_bounded_integers(
    client: TestClient, url: str, params: dict[str, Any]
) -> None:
    """No request may ask for an unbounded page or a non-integer window."""
    response = client.get(url, params=params)
    assert response.status_code == 422, (url, params, response.text)


# --------------------------------------------------------------- auth / CORS


def test_cors_allows_only_the_local_panel_origins(client: TestClient) -> None:
    assert ALLOWED_ORIGINS
    assert "*" not in ALLOWED_ORIGINS
    assert all(
        origin.startswith(("http://localhost:", "http://127.0.0.1:")) for origin in ALLOWED_ORIGINS
    )

    evil = client.options(
        "/api/paper/sessions",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert evil.status_code == 400
    assert evil.headers.get("access-control-allow-origin") is None

    local = client.options(
        "/api/paper/sessions",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert local.status_code == 200
    assert local.headers.get("access-control-allow-origin") == "http://localhost:3000"
    assert local.headers.get("access-control-allow-credentials") is None

    read = client.get("/api/health", headers={"Origin": "http://evil.example"})
    assert read.headers.get("access-control-allow-origin") is None


def test_no_authentication_layer_is_declared(client: TestClient) -> None:
    """Documents the local-only assumption: there is no auth or CSRF token.

    If an auth dependency is added, this test should be updated deliberately —
    it exists so that "the API has no authentication" is a recorded property,
    and so binding the server to a non-loopback interface is a conscious act.
    """
    schema = client.app.openapi()
    assert schema.get("components", {}).get("securitySchemes", {}) == {}
    for path, operations in schema["paths"].items():
        for method, operation in operations.items():
            assert "security" not in operation, (path, method)

    # No credentials are required for a write either.
    response = client.post(
        "/api/paper/sessions",
        json={"symbol": "XAUUSD", "timeframe": "M1", "strategy": "manual"},
    )
    assert response.status_code == 201, response.text
