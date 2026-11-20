"""Dukascopy provider: tick decoding, backoff, hour cache, and M1 aggregation.

Every test drives the provider through ``httpx.MockTransport``; nothing here
can open a socket, so the suite is safe to run when the feed is rate-limited.
"""

import lzma
import struct
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from scalping.data import dukascopy
from scalping.data.base import BAR_SCHEMA, conform_bars
from scalping.data.dukascopy import DataUnavailable, DukascopyProvider, decode_ticks

HOUR = datetime(2026, 1, 5, 10, tzinfo=UTC)
#: Fixed clock after HOUR has fully elapsed so cache decisions are deterministic.
PAST = datetime(2026, 1, 6, tzinfo=UTC)


def test_decode_ticks_known_record():
    payload = struct.pack(">IIIff", 60_000, 235_1234, 235_1000, 1.5, 2.5)
    ticks = decode_ticks(payload)
    assert ticks[0]["ms"] == 60_000
    assert ticks[0]["ask"] == pytest.approx(2351.234)
    assert ticks[0]["bid"] == pytest.approx(2351.0)


def test_retries_on_429_then_succeeds(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, text="Too Many Requests")
        return httpx.Response(200, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = DukascopyProvider(tmp_path, client=client, base_delay=0.01, now=lambda: PAST)
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=UTC))
    assert calls["n"] == 3


def test_gives_up_after_max_retries(tmp_path):
    client = httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(429)))
    p = DukascopyProvider(
        tmp_path, client=client, max_retries=2, base_delay=0.01, now=lambda: PAST
    )
    with pytest.raises(DataUnavailable):
        p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=UTC))


def test_cache_prevents_second_request(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=b"")

    client = httpx.Client(transport=httpx.MockTransport(handler))
    p = DukascopyProvider(tmp_path, client=client, base_delay=0.01, now=lambda: PAST)
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=UTC))
    p._fetch_hour("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=UTC))
    assert calls["n"] == 1


# --- helpers ---------------------------------------------------------------


def _ticks(*records: tuple[int, int, int, float, float]) -> bytes:
    return b"".join(struct.pack(">IIIff", *record) for record in records)


def _provider(tmp_path, handler, **kwargs) -> DukascopyProvider:
    client = httpx.Client(transport=httpx.MockTransport(handler))
    kwargs.setdefault("base_delay", 0.01)
    kwargs.setdefault("now", lambda: PAST)
    return DukascopyProvider(tmp_path, client=client, **kwargs)


# --- decoding --------------------------------------------------------------


def test_decode_ticks_multiple_records_and_scaling():
    payload = _ticks(
        (0, 2_351_500, 2_351_000, 1.0, 2.0),
        (500, 2_352_250, 2_352_000, 0.5, 0.75),
    )
    ticks = decode_ticks(payload)
    assert [tick["ms"] for tick in ticks] == [0, 500]
    assert ticks[1]["ask"] == pytest.approx(2352.25)
    assert ticks[1]["bid"] == pytest.approx(2352.0)
    assert ticks[1]["ask_vol"] == pytest.approx(0.5)
    assert ticks[1]["bid_vol"] == pytest.approx(0.75)


def test_decode_ticks_empty_and_truncated():
    assert decode_ticks(b"") == []
    with pytest.raises(ValueError):
        decode_ticks(b"\x00" * 19)


# --- URL / cache layout ----------------------------------------------------


def test_url_uses_zero_based_month_and_cache_layout(tmp_path):
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=b"")

    p = _provider(tmp_path, handler)
    frame = p._fetch_hour("XAUUSD", HOUR)

    assert seen == ["https://datafeed.dukascopy.com/datafeed/XAUUSD/2026/00/05/10h_ticks.bi5"]
    assert (tmp_path / "XAUUSD" / "20260105" / "10.parquet").exists()
    assert frame.is_empty()
    assert frame.schema == BAR_SCHEMA


def test_empty_payload_caches_an_empty_frame(tmp_path):
    p = _provider(tmp_path, lambda request: httpx.Response(200, content=b""))
    p._fetch_hour("XAUUSD", HOUR)

    cached = DukascopyProvider(
        tmp_path, client=httpx.Client(transport=httpx.MockTransport(lambda r: httpx.Response(500)))
    )._fetch_hour("XAUUSD", HOUR)
    assert cached.is_empty()
    assert cached.columns == list(BAR_SCHEMA)


def test_cached_ticks_roundtrip_without_request(tmp_path):
    calls = {"n": 0}
    payload = _ticks((0, 2_351_500, 2_351_000, 1.0, 2.0))

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=payload)

    p = _provider(tmp_path, handler)
    first = p._fetch_hour("XAUUSD", HOUR)
    second = p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 1
    assert second.equals(first)
    assert second.schema == BAR_SCHEMA
    assert second["volume"].to_list() == [1]


# --- retry policy ----------------------------------------------------------


def test_retry_delay_schedule(tmp_path, monkeypatch):
    sleeps = []
    monkeypatch.setattr(dukascopy, "_sleep", sleeps.append)

    p = _provider(tmp_path, lambda request: httpx.Response(429), max_retries=5, base_delay=1.0)
    with pytest.raises(DataUnavailable) as excinfo:
        p._fetch_hour("XAUUSD", HOUR)

    assert sleeps == [1.0, 2.0, 4.0, 8.0, 16.0]
    assert excinfo.value.symbol == "XAUUSD"
    assert excinfo.value.hour == HOUR
    assert excinfo.value.status == 429


def test_503_is_retryable(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503) if calls["n"] == 1 else httpx.Response(200, content=b"")

    p = _provider(tmp_path, handler)
    p._fetch_hour("XAUUSD", HOUR)
    assert calls["n"] == 2


def test_404_returns_empty_without_retry(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    p = _provider(tmp_path, handler)
    frame = p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 1
    assert frame.is_empty()
    assert frame.schema == BAR_SCHEMA


def test_404_for_elapsed_hour_is_negatively_cached(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    path = tmp_path / "XAUUSD" / "20260105" / "10.parquet"
    p = _provider(tmp_path, handler)
    first = p._fetch_hour("XAUUSD", HOUR)
    second = p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 1
    assert first.is_empty() and second.is_empty()
    assert path.exists()
    assert second.schema == BAR_SCHEMA


def test_404_for_current_hour_is_not_cached(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    now = HOUR + timedelta(minutes=30)
    p = _provider(tmp_path, handler, now=lambda: now)
    p._fetch_hour("XAUUSD", HOUR)
    p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 2
    assert not (tmp_path / "XAUUSD" / "20260105" / "10.parquet").exists()


def test_404_for_future_hour_is_not_cached(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(404)

    now = HOUR - timedelta(hours=1)
    p = _provider(tmp_path, handler, now=lambda: now)
    p._fetch_hour("XAUUSD", HOUR)
    p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 2
    assert not (tmp_path / "XAUUSD" / "20260105" / "10.parquet").exists()


def test_in_progress_hour_is_returned_but_not_cached(tmp_path):
    calls = {"n": 0}
    payload = _ticks((0, 2_351_500, 2_351_000, 1.0, 2.0))

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=payload)

    now = HOUR + timedelta(minutes=30)
    p = _provider(tmp_path, handler, now=lambda: now)
    first = p._fetch_hour("XAUUSD", HOUR)
    second = p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 2
    assert first.height == 1
    assert second.equals(first)
    assert not (tmp_path / "XAUUSD" / "20260105" / "10.parquet").exists()


def test_hour_that_just_elapsed_is_cached(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, content=b"")

    now = HOUR + timedelta(hours=1)  # hour_end <= now
    p = _provider(tmp_path, handler, now=lambda: now)
    p._fetch_hour("XAUUSD", HOUR)
    p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 1
    assert (tmp_path / "XAUUSD" / "20260105" / "10.parquet").exists()


def test_unsupported_symbol_is_rejected(tmp_path):
    p = _provider(tmp_path, lambda request: httpx.Response(200, content=b""))

    with pytest.raises(ValueError, match="unsupported symbol"):
        p._fetch_hour("EURUSD", HOUR)
    with pytest.raises(ValueError, match="unsupported symbol"):
        p.fetch("EURUSD", HOUR, HOUR)
    with pytest.raises(ValueError, match="unsupported symbol"):
        DukascopyProvider(tmp_path, symbol="EURUSD")


def test_supported_symbols_have_a_point_divisor(tmp_path):
    assert dukascopy.POINT_DIVISORS["XAUUSD"] == 1000
    assert dukascopy.POINT_DIVISORS["XAGUSD"] == 1000

    p = _provider(tmp_path, lambda request: httpx.Response(200, content=b""), symbol="XAGUSD")
    assert p._fetch_hour("XAGUSD", HOUR).is_empty()


def test_hour_bars_use_the_validated_point_divisor(tmp_path, monkeypatch):
    monkeypatch.setitem(dukascopy.POINT_DIVISORS, "XAUUSD", 100)
    payload = _ticks((0, 2_351_500, 2_351_000, 1.0, 2.0))

    p = _provider(tmp_path, lambda request: httpx.Response(200, content=payload))
    frame = p._fetch_hour("XAUUSD", HOUR)

    assert frame["ask_close"].to_list() == [pytest.approx(23515.0)]
    assert frame["bid_close"].to_list() == [pytest.approx(23510.0)]


def test_context_manager_closes_injected_client(tmp_path):
    class RecordingClient:
        def __init__(self):
            self.closed = 0

        def close(self):
            self.closed += 1

    client = RecordingClient()
    with DukascopyProvider(tmp_path, client=client) as p:
        assert p.name == "dukascopy"

    assert client.closed == 1


def test_other_statuses_fail_immediately(tmp_path):
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(403)

    p = _provider(tmp_path, handler)
    with pytest.raises(DataUnavailable) as excinfo:
        p._fetch_hour("XAUUSD", HOUR)
    assert excinfo.value.status == 403
    assert calls["n"] == 1


def test_exhaustion_counts_match_schedule(tmp_path, monkeypatch):
    calls = {"n": 0}
    sleeps = []
    monkeypatch.setattr(dukascopy, "_sleep", sleeps.append)

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503)

    p = _provider(tmp_path, handler, max_retries=2)
    with pytest.raises(DataUnavailable):
        p._fetch_hour("XAUUSD", HOUR)

    assert calls["n"] == 3  # initial attempt + max_retries
    assert sleeps == [0.01, 0.02]


# --- LZMA ------------------------------------------------------------------


def test_lzma_payload_is_decompressed(tmp_path):
    payload = lzma.compress(
        _ticks((60_000, 2_351_500, 2_351_000, 1.0, 2.0)), format=lzma.FORMAT_ALONE
    )
    assert payload[:1] == b"\x5d"  # LZMA-alone properties byte

    p = _provider(tmp_path, lambda request: httpx.Response(200, content=payload))
    frame = p._fetch_hour("XAUUSD", HOUR)

    assert frame.height == 1
    assert frame["bid_close"].to_list() == [pytest.approx(2351.0)]
    assert frame["ask_close"].to_list() == [pytest.approx(2351.5)]


# --- M1 aggregation --------------------------------------------------------


def test_hour_aggregates_ticks_to_m1(tmp_path):
    payload = _ticks(
        (0, 2_351_500, 2_351_300, 1.0, 1.0),  # 10:00 open bid 2351.3
        (30_000, 2_351_700, 2_351_500, 1.0, 1.0),  # 10:00 high bid 2351.5
        (59_000, 2_351_300, 2_351_100, 1.0, 1.0),  # 10:00 low/last bid 2351.1
        (61_000, 2_351_450, 2_351_200, 1.0, 1.0),  # 10:01 single tick
    )
    p = _provider(tmp_path, lambda request: httpx.Response(200, content=payload))
    frame = p._fetch_hour("XAUUSD", HOUR)

    assert frame.schema == BAR_SCHEMA
    assert frame.height == 2
    assert frame["ts"].to_list() == [
        datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 10, 1, tzinfo=UTC),
    ]
    assert frame["open"].to_list() == [pytest.approx(2351.3), pytest.approx(2351.2)]
    assert frame["high"].to_list() == [pytest.approx(2351.5), pytest.approx(2351.2)]
    assert frame["low"].to_list() == [pytest.approx(2351.1), pytest.approx(2351.2)]
    assert frame["close"].to_list() == [pytest.approx(2351.1), pytest.approx(2351.2)]
    assert frame["bid_close"].to_list() == [pytest.approx(2351.1), pytest.approx(2351.2)]
    assert frame["ask_close"].to_list() == [pytest.approx(2351.3), pytest.approx(2351.45)]
    assert frame["spread"].to_list() == [pytest.approx(0.2), pytest.approx(0.25)]
    assert frame["volume"].to_list() == [3, 1]
    assert frame["provider"].to_list() == ["dukascopy", "dukascopy"]


def test_fetch_walks_hours_and_conforms(tmp_path):
    seen = []

    def handler(request):
        hour = request.url.path.rsplit("/", 1)[-1][:2]
        seen.append(hour)
        if hour == "10":
            return httpx.Response(200, content=_ticks((0, 1_000_500, 1_000_000, 1.0, 1.0)))
        return httpx.Response(200, content=_ticks((0, 2_000_500, 2_000_000, 1.0, 1.0)))

    p = _provider(tmp_path, handler)
    frame = p.fetch(
        "XAUUSD", datetime(2026, 1, 5, 10, 0, tzinfo=UTC), datetime(2026, 1, 5, 11, 59, tzinfo=UTC)
    )

    assert seen == ["10", "11"]
    assert frame.schema == BAR_SCHEMA
    assert frame.equals(conform_bars(frame))
    assert frame["ts"].to_list() == [
        datetime(2026, 1, 5, 10, 0, tzinfo=UTC),
        datetime(2026, 1, 5, 11, 0, tzinfo=UTC),
    ]
    assert frame["provider"].unique().to_list() == ["dukascopy"]


def test_fetch_filters_to_requested_bounds(tmp_path):
    def handler(request):
        return httpx.Response(200, content=_ticks((0, 1_000_500, 1_000_000, 1.0, 1.0)))

    p = _provider(tmp_path, handler)
    frame = p.fetch(
        "XAUUSD", datetime(2026, 1, 5, 10, 30, tzinfo=UTC), datetime(2026, 1, 5, 11, 59, tzinfo=UTC)
    )
    assert frame["ts"].to_list() == [datetime(2026, 1, 5, 11, 0, tzinfo=UTC)]


def test_fetch_requires_bounds(tmp_path):
    p = _provider(tmp_path, lambda request: httpx.Response(200, content=b""))
    with pytest.raises(ValueError):
        p.fetch("XAUUSD")
    with pytest.raises(ValueError):
        p.fetch("XAUUSD", datetime(2026, 1, 5, 10, tzinfo=UTC))
