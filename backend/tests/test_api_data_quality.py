"""Data-quality report endpoint.

The endpoint is a view over the existing validator, so these tests check the
HTTP contract and the mapping rules, not the validator itself: clean vs dirty
counts, the status thresholds, provenance, the explicit unconfigured state and
the read-only guarantee.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from fastapi.testclient import TestClient

from scalping.api.app import create_app
from scalping.data.provenance import build_provenance, save_provenance

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
START = datetime(2026, 1, 5, tzinfo=UTC)


def bars(n: int = 60, step_minutes: int = 1) -> pl.DataFrame:
    """A structurally clean slice: valid OHLC, positive spread, zigzag closes."""
    stamps = [START + timedelta(minutes=step_minutes * i) for i in range(n)]
    close = [100.0 + 0.01 * (i % 2) for i in range(n)]
    open_ = [100.0] * n
    high = [max(o, c) + 0.5 for o, c in zip(open_, close, strict=True)]
    low = [min(o, c) - 0.5 for o, c in zip(open_, close, strict=True)]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "bid_close": [c - 0.05 for c in close],
            "ask_close": [c + 0.15 for c in close],
            "spread": [0.2] * n,
            "volume": [10 + (i % 5) for i in range(n)],
            "provider": ["test"] * n,
        }
    )


def _set(frame: pl.DataFrame, column: str, index: int, value: float | datetime) -> pl.DataFrame:
    """Replace one cell; the tests keep frames sorted where that matters."""
    dtype = frame.schema[column]
    return frame.with_columns(
        pl.when(pl.int_range(pl.len()) == index)
        .then(pl.lit(value, dtype=dtype))
        .otherwise(pl.col(column))
        .alias(column)
    )


def write_store(
    root: Path,
    frame: pl.DataFrame,
    *,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
) -> None:
    """Write parquet directly so a test can store rows the store would dedupe."""
    target = root / symbol / timeframe / "2026" / "01.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)


def write_provenance(
    root: Path,
    frame: pl.DataFrame,
    *,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    synthetic: bool = True,
    source: str = "provider:synthetic",
) -> None:
    save_provenance(
        root,
        build_provenance(
            frame, symbol=symbol, timeframe=timeframe, source=source, synthetic=synthetic
        ),
    )


def quality(client: TestClient, **params: object) -> dict[str, object]:
    response = client.get(
        "/api/data/quality",
        params={"symbol": SYMBOL, "timeframe": TIMEFRAME, **params},
    )
    assert response.status_code == 200, response.text
    body: dict[str, object] = response.json()
    return body


@pytest.fixture
def client(tmp_path: Path):
    with TestClient(create_app(data_root=tmp_path)) as test_client:
        yield test_client


def test_clean_store_is_ok_with_zero_counts_and_provenance(
    tmp_path: Path, client: TestClient
) -> None:
    frame = bars()
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["configured"] is True
    assert body["status"] == "ok"
    assert body["status_rule"] == "clean"
    assert body["rows"] == 60
    assert body["first_ts"] == START.isoformat()
    assert body["last_ts"] == (START + timedelta(minutes=59)).isoformat()
    assert body["duplicates"] == 0
    assert body["gaps"] == 0
    assert body["ohlc_violations"] == 0
    assert body["crossed_quotes"] == 0
    assert body["negative_spread"] == 0
    assert body["negative_volume"] == 0
    assert body["outliers"] == 0
    assert body["outliers_checked"] is True
    assert body["issues"] == []
    provenance = body["provenance"]
    assert isinstance(provenance, dict)
    assert provenance["synthetic"] is True
    assert provenance["source"] == "provider:synthetic"
    assert str(provenance["dataset_id"]).startswith(f"{SYMBOL}:{TIMEFRAME}:")
    assert body["thresholds"] == {
        "gap_threshold_minutes": 10,
        "outlier_return_z": 8.0,
        "outlier_volume_z": 8.0,
        "outlier_min_rows": 30,
    }


def test_configured_store_without_a_sidecar_warns(tmp_path: Path, client: TestClient) -> None:
    """A clean slice with no sidecar is a provenance gap, not a data error.

    ``docs/data-pipeline.md`` maps a missing sidecar onto ``warn`` with
    ``status_rule = provenance_missing`` and leaves ``provenance`` null instead
    of guessing a source or a synthetic flag.
    """
    write_store(tmp_path, bars())

    body = quality(client)

    assert body["configured"] is True
    assert body["status"] == "warn"
    assert body["status_rule"] == "provenance_missing"
    assert body["provenance"] is None
    assert body["rows"] == 60
    assert body["duplicates"] == 0
    assert body["gaps"] == 0


def test_missing_store_is_explicit_and_creates_nothing(tmp_path: Path) -> None:
    with TestClient(create_app(data_root=tmp_path)) as client:
        body = quality(client)

    assert body["configured"] is False
    assert body["status"] == "fail"
    assert body["status_rule"] == "store_empty"
    assert body["rows"] == 0
    assert body["first_ts"] is None
    assert body["last_ts"] is None
    # "not checked" must not masquerade as a clean zero.
    for key in (
        "duplicates",
        "gaps",
        "ohlc_violations",
        "crossed_quotes",
        "negative_spread",
        "negative_volume",
        "outliers",
    ):
        assert body[key] is None, key
    assert body["outliers_checked"] is False
    assert body["provenance"] is None
    assert body["issues"] == []
    assert list(tmp_path.rglob("*")) == []


@pytest.mark.parametrize(
    "params",
    [
        {"symbol": "NOTREAL", "timeframe": "M1"},
        {"symbol": "xauusd", "timeframe": "M1"},
        {"symbol": "../../etc", "timeframe": "M1"},
        {"symbol": "XAUUSD", "timeframe": "M2"},
        {"symbol": "XAUUSD", "timeframe": "../x"},
    ],
)
def test_unknown_or_unsafe_ids_are_422(client: TestClient, params: dict[str, str]) -> None:
    response = client.get("/api/data/quality", params=params)
    assert response.status_code == 422, response.text


def test_duplicate_timestamps_fail_the_report(tmp_path: Path, client: TestClient) -> None:
    frame = _set(bars(), "ts", 5, START + timedelta(minutes=4))
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["duplicates"] == 1
    assert body["status"] == "fail"
    assert body["status_rule"] == "duplicate_timestamps"
    assert any(
        issue["code"] == "duplicate_timestamps"
        for issue in body["issues"]  # type: ignore[union-attr]
    )


def test_gap_is_a_warning_with_its_count(tmp_path: Path, client: TestClient) -> None:
    frame = pl.concat([bars().head(20), bars().tail(30)])
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["rows"] == 50
    assert body["gaps"] == 1
    assert body["status"] == "warn"
    assert body["status_rule"] == "time_gaps"


def test_ohlc_violation_fails_the_report(tmp_path: Path, client: TestClient) -> None:
    frame = _set(bars(), "low", 10, 103.0)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["ohlc_violations"] == 1
    assert body["status"] == "fail"
    assert body["status_rule"] == "ohlc_inconsistent"


def test_crossed_quotes_fail_the_report(tmp_path: Path, client: TestClient) -> None:
    frame = _set(bars(), "ask_close", 10, 99.0)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["crossed_quotes"] == 1
    assert body["status"] == "fail"
    assert body["status_rule"] == "crossed_quotes"


@pytest.mark.filterwarnings("ignore:invalid value encountered in log1p:RuntimeWarning")
def test_negative_spread_and_volume_are_counted_separately(
    tmp_path: Path, client: TestClient
) -> None:
    frame = _set(_set(bars(), "spread", 10, -0.1), "volume", 12, -5)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["negative_spread"] == 1
    assert body["negative_volume"] == 1
    assert body["status"] == "fail"
    # The validator reports the negative spread before the volume, so it is
    # the rule that names the failure.
    assert body["status_rule"] == "negative_spread"


def test_price_outlier_warns_without_failing(tmp_path: Path, client: TestClient) -> None:
    frame = _set(_set(bars(), "close", 40, 105.0), "high", 40, 105.5)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["ohlc_violations"] == 0
    assert body["outliers_checked"] is True
    assert isinstance(body["outliers"], int) and body["outliers"] >= 1
    assert body["status"] == "warn"
    assert body["status_rule"] == "price_outliers"


def test_volume_outlier_warns_without_failing(tmp_path: Path, client: TestClient) -> None:
    frame = _set(bars(), "volume", 45, 1_000_000)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert isinstance(body["outliers"], int) and body["outliers"] >= 1
    assert body["status"] == "warn"
    assert body["status_rule"] == "volume_outliers"


def test_outlier_pass_can_be_skipped_for_large_slices(tmp_path: Path, client: TestClient) -> None:
    frame = _set(bars(), "volume", 45, 1_000_000)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client, include_outliers=False)

    assert body["outliers"] is None
    assert body["outliers_checked"] is False
    assert body["status"] == "ok"
    assert body["status_rule"] == "clean"


def test_gap_threshold_follows_the_bar_width(tmp_path: Path, client: TestClient) -> None:
    frame = bars(step_minutes=60)
    write_store(tmp_path, frame, timeframe="H1")
    write_provenance(tmp_path, frame, timeframe="H1")

    body = quality(client, timeframe="H1")

    assert body["thresholds"]["gap_threshold_minutes"] == 60  # type: ignore[index]
    assert body["gaps"] == 0
    assert body["status"] == "ok"


def test_short_slice_does_not_invent_an_outlier_zero(tmp_path: Path, client: TestClient) -> None:
    frame = bars(5)
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)

    body = quality(client)

    assert body["rows"] == 5
    assert body["outliers"] is None
    assert body["outliers_checked"] is False
    assert body["status"] == "ok"


def test_missing_provenance_warns(tmp_path: Path, client: TestClient) -> None:
    write_store(tmp_path, bars())

    body = quality(client)

    assert body["provenance"] is None
    assert body["status"] == "warn"
    assert body["status_rule"] == "provenance_missing"


def test_quality_report_does_not_write(tmp_path: Path, client: TestClient) -> None:
    frame = bars()
    write_store(tmp_path, frame)
    write_provenance(tmp_path, frame)
    before = _snapshot(tmp_path)

    first = quality(client)
    second = quality(client)
    after = _snapshot(tmp_path)

    assert first == second
    assert before == after


def _snapshot(root: Path) -> dict[str, tuple[int, str]]:
    return {
        str(path.relative_to(root)): (
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
