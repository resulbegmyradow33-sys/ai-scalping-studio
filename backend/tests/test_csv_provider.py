import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scalping.data.base import BAR_SCHEMA
from scalping.data.csv_provider import CsvProvider, DataError

FIXTURES = Path(__file__).parent / "fixtures"
FIX = FIXTURES / "xauusd_gmt3.csv"


def test_converts_broker_offset_to_utc():
    df = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df["ts"].dt.hour()[0] == 7  # file says 10:00 GMT+3
    assert df["ts"][0] == datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
    assert df["ts"].dtype.time_zone == "UTC"


def test_missing_ask_columns_yields_null_spread():
    df = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df["ask_close"].null_count() == df.height
    assert df["spread"].null_count() == df.height
    assert df["provider"][0] == "csv_no_spread"
    assert df["bid_close"].to_list() == df["close"].to_list()


def test_duplicate_timestamp_raises_with_line_number():
    with pytest.raises(DataError, match="line 3") as info:
        CsvProvider(FIX.with_name("dupes.csv"), tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert (info.value.line, info.value.column) == (3, "ts")


def test_bad_float_raises_data_error():
    with pytest.raises(DataError, match="close") as info:
        CsvProvider(FIX.with_name("bad_float.csv"), tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert (info.value.line, info.value.column) == (2, "close")


def test_frame_conforms_to_bar_schema():
    df = CsvProvider(FIX, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df.schema == BAR_SCHEMA
    assert df.height == 5
    assert df["ts"].dt.second().to_list() == [0] * 5


def test_bounds_are_inclusive_and_none_is_unbounded():
    provider = CsvProvider(FIX, tz_offset_hours=3)
    low = datetime(2026, 1, 5, 7, 1, tzinfo=UTC)
    high = datetime(2026, 1, 5, 7, 3, tzinfo=UTC)
    assert [ts.minute for ts in provider.fetch("XAUUSD", low, high)["ts"]] == [1, 2, 3]
    assert provider.fetch("XAUUSD", None, high).height == 4
    assert provider.fetch("XAUUSD", low, None).height == 4


@pytest.mark.parametrize(("offset", "hour"), [(3, 7), (0, 10), (-5, 15)])
def test_offset_conversion_matches_broker_clock(offset, hour):
    df = CsvProvider(FIX, tz_offset_hours=offset).fetch("XAUUSD", None, None)
    assert df["ts"].dt.hour()[0] == hour


def test_auto_detects_offset_from_spring_forward_gap(tmp_path):
    path = tmp_path / "dst.csv"
    path.write_text(
        "2026.03.29 01:59,1,2,0.5,1.5,10\n"
        "2026.03.29 03:00,1,2,0.5,1.5,10\n"
        "2026.03.29 03:01,1,2,0.5,1.5,10\n"
    )
    df = CsvProvider(path).fetch("XAUUSD", None, None)
    # 01:59 local (+3) is 22:59 UTC the day before; 03:00 local is 00:00 UTC.
    assert df["ts"].dt.hour().to_list() == [22, 0, 0]


def test_auto_detect_warns_and_falls_back_without_dst_gap(caplog):
    with caplog.at_level(logging.WARNING, logger="scalping.data.csv_provider"):
        df = CsvProvider(FIX).fetch("XAUUSD", None, None)
    assert df["ts"].dt.hour()[0] == 7
    assert any("offset" in record.message.lower() for record in caplog.records)


def test_empty_file_returns_schema_correct_frame(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("")
    df = CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df.is_empty()
    assert df.schema == BAR_SCHEMA


def test_wrong_field_count_reports_line(tmp_path):
    path = tmp_path / "short.csv"
    path.write_text("2026.01.05 10:00,1,2,3,4\n")
    with pytest.raises(DataError, match="line 1") as info:
        CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert info.value.column is None


def test_ask_columns_are_used_when_present(tmp_path):
    path = tmp_path / "quoted.csv"
    path.write_text("2026.01.05 10:00,1.0,2.0,0.5,1.5,1.4,1.6,10\n")
    df = CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df["provider"][0] == "csv"
    assert df["ask_close"][0] == pytest.approx(1.6)
    assert df["spread"][0] == pytest.approx(0.2)


def test_bad_volume_reports_volume_column(tmp_path):
    path = tmp_path / "bad_volume.csv"
    path.write_text("2026.01.05 10:00,1,2,0.5,1.5,many\n")
    with pytest.raises(DataError, match="volume") as info:
        CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert (info.value.line, info.value.column) == (1, "volume")


def test_column_map_reorders_positional_columns(tmp_path):
    # This broker writes volume before close: datetime,open,high,low,volume,close.
    path = tmp_path / "swapped.csv"
    path.write_text("2026.01.05 10:00,1.0,2.0,0.5,10,1.5\n")
    df = CsvProvider(
        path,
        tz_offset_hours=3,
        column_map={"volume": "close", "close": "volume"},
    ).fetch("XAUUSD", None, None)
    assert df["close"][0] == pytest.approx(1.5)
    assert df["volume"][0] == 10


def test_zero_field_rows_are_skipped(tmp_path):
    path = tmp_path / "gappy.csv"
    path.write_text(
        "2026.01.05 10:00,1.0,2.0,0.5,1.5,10\n\n2026.01.05 10:01,1.5,2.1,1.0,1.8,12\n"
    )
    df = CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert df.height == 2


def test_mixed_width_file_reports_line_and_widths():
    with pytest.raises(DataError, match="line 2") as info:
        CsvProvider(FIX.with_name("mixed_width.csv"), tz_offset_hours=3).fetch(
            "XAUUSD", None, None
        )
    assert "expected 6" in str(info.value)
    assert "found 8" in str(info.value)


def test_narrowing_width_file_reports_line(tmp_path):
    path = tmp_path / "narrowing.csv"
    path.write_text(
        "2026.01.05 10:00,1,2,0.5,1.5,1.4,1.6,10\n2026.01.05 10:01,1,2,0.5,1.5,10\n"
    )
    with pytest.raises(DataError, match="line 2") as info:
        CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert "expected 8" in str(info.value)
    assert "found 6" in str(info.value)


def test_crossed_quotes_are_rejected(tmp_path):
    path = tmp_path / "crossed.csv"
    # bid_close 1.6 is above ask_close 1.4, so the spread would be negative.
    path.write_text("2026.01.05 10:00,1.0,2.0,0.5,1.5,1.6,1.4,10\n")
    with pytest.raises(DataError, match="line 1") as info:
        CsvProvider(path, tz_offset_hours=3).fetch("XAUUSD", None, None)
    assert info.value.column == "spread"
