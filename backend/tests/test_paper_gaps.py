"""Tests for the continuous paper runner's configurable gap policy.

The behaviour under test is narrow and auditable:

* out-of-order and duplicate bars are refused under every policy;
* ``strict`` (the default) keeps the historical refusal for a missing bar
  inside the open session and writes nothing;
* ``session_aware`` tolerates a real in-session gap (the PR #140 examples),
  records it in the append-only ``paper_gaps`` table and never invents a bar;
* ``skip_with_report`` does the same for an unknown symbol, where the session
  calendar cannot classify the gap;
* a jump across a weekend/daily close is not a data gap under any policy;
* restarting across a gap neither duplicates signals nor re-records the gap;
* future bars do not change decisions made before a gap (no look-ahead);
* a session's policy is part of its identity: resuming under a different one is
  refused.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import get_args

import polars as pl
import pytest

from scalping.config import RiskConfig, Settings
from scalping.data.store import ParquetStore
from scalping.paper import GAP_POLICIES, PaperLedger
from scalping.paper.continuous import (
    BarGapError,
    BarOrderError,
    ContinuousPaperRunner,
    GapPolicy,
    SessionMismatchError,
)
from scalping.paper.ledger import GapRecord, LedgerError
from scalping.strategy.dsl import (
    Compare,
    Const,
    EntryRule,
    ExitRule,
    HoldingLimit,
    StrategySpec,
)

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
SETTINGS = Settings(risk=RiskConfig(daily_loss_pct=0.9))
SPREAD = 0.20

#: PR #140's M1 example: a Monday inside the metals session.
M1_START = datetime(2021, 5, 24, 2, 0, tzinfo=UTC)
M1_GAP = datetime(2021, 5, 24, 4, 4, tzinfo=UTC)
#: PR #140's H1 example: Monday 2009-05-25, 17:00 through 21:00 missing.
H1_START = datetime(2009, 5, 25, 0, 0, tzinfo=UTC)


# --------------------------------------------------------------------------- fixtures


def holding_spec() -> StrategySpec:
    """Long every bar when flat; close when the holding limit is reached."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(5),
        max_lookback=2,
    )


def bars_at(stamps: list[datetime], *, start_price: float = 2000.0) -> pl.DataFrame:
    """Deterministic bars on the given timestamps (which need not be contiguous)."""
    closes = [start_price + 0.5 * ((index % 8) - 4) for index in range(len(stamps))]
    opens = [closes[0], *closes[:-1]]
    return pl.DataFrame(
        {
            "ts": stamps,
            "open": opens,
            "high": [max(o, c) + 0.20 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 0.20 for o, c in zip(opens, closes, strict=True)],
            "close": closes,
            "bid_close": closes,
            "ask_close": [close + SPREAD for close in closes],
            "spread": [SPREAD] * len(stamps),
            "volume": [100] * len(stamps),
        }
    )


def minute_bars(start: datetime, count: int, *, skip: tuple[int, ...] = ()) -> pl.DataFrame:
    stamps = [start + timedelta(minutes=index) for index in range(count) if index not in skip]
    return bars_at(stamps)


def hourly_bars(start: datetime, count: int, *, skip_hours: range | None = None) -> pl.DataFrame:
    skip = set(range(0)) if skip_hours is None else set(skip_hours)
    stamps = [start + timedelta(hours=index) for index in range(count) if index not in skip]
    return bars_at(stamps)


def write(
    root: Path,
    bars: pl.DataFrame,
    *,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
) -> int:
    return ParquetStore(root).write(symbol, timeframe, bars)


def make_runner(
    ledger: PaperLedger,
    root: Path,
    *,
    symbol: str = SYMBOL,
    timeframe: str = TIMEFRAME,
    window: int = 60,
    gap_policy: GapPolicy = "strict",
) -> ContinuousPaperRunner:
    return ContinuousPaperRunner(
        ledger,
        holding_spec(),
        symbol,
        timeframe,
        SETTINGS,
        store_root=root,
        cache=False,
        window=window,
        gap_policy=gap_policy,
    )


def stored_ts(root: Path, *, symbol: str = SYMBOL, timeframe: str = TIMEFRAME) -> set[datetime]:
    return set(ParquetStore(root).read(symbol, timeframe).get_column("ts").to_list())


def runner_ts(ledger: PaperLedger, session_id: str) -> set[datetime]:
    return {signal.ts for signal in ledger.signals(session_id)}


# --------------------------------------------------------------------- policy wiring


def test_gap_policy_literal_and_table_check_agree() -> None:
    assert set(get_args(GapPolicy)) == GAP_POLICIES


def test_unknown_gap_policy_is_refused(tmp_path: Path) -> None:
    write(tmp_path / "data", minute_bars(M1_START, 150))
    with (
        PaperLedger(tmp_path / "paper.db") as ledger,
        pytest.raises(ValueError, match="gap_policy"),
    ):
        ContinuousPaperRunner(
            ledger,
            holding_spec(),
            SYMBOL,
            TIMEFRAME,
            SETTINGS,
            store_root=tmp_path / "data",
            cache=False,
            gap_policy="whatever",  # type: ignore[arg-type]
        )


def test_default_policy_is_strict(tmp_path: Path) -> None:
    write(tmp_path / "data", minute_bars(M1_START, 150))
    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = ContinuousPaperRunner(
            ledger,
            holding_spec(),
            SYMBOL,
            TIMEFRAME,
            SETTINGS,
            store_root=tmp_path / "data",
            cache=False,
        )
        assert runner.gap_policy == "strict"


# ----------------------------------------------------------- real PR #140 examples


def test_real_m1_gap_is_refused_under_strict(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bars = minute_bars(M1_START, 150)
    gap_index = int((M1_GAP - M1_START).total_seconds() // 60)
    write(root, pl.concat([bars[:gap_index], bars[gap_index + 1 :]]))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root)
        with pytest.raises(BarGapError, match="2021-05-24T04:04"):
            runner.advance()
        assert ledger.session(runner.session_id) is None
        assert ledger.signals(runner.session_id) == []
        assert ledger.gaps(runner.session_id) == []


def test_real_m1_gap_is_skipped_and_recorded_under_session_aware(tmp_path: Path) -> None:
    root = tmp_path / "data"
    bars = minute_bars(M1_START, 150)
    gap_index = int((M1_GAP - M1_START).total_seconds() // 60)
    stored = pl.concat([bars[:gap_index], bars[gap_index + 1 :]])
    write(root, stored)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        summary = runner.advance()

        assert summary.gap_policy == "session_aware"
        assert summary.gaps_skipped == 1
        assert summary.missing_bars == 1
        assert summary.gaps_recorded == 1
        assert summary.new_signals > 0

        gaps = ledger.gaps(runner.session_id)
        assert len(gaps) == 1
        gap = gaps[0]
        assert isinstance(gap, GapRecord)
        assert gap.policy == "session_aware"
        assert gap.missing_bars == 1
        assert gap.from_ts == M1_GAP - timedelta(minutes=1)
        assert gap.to_ts == M1_GAP + timedelta(minutes=1)

        # No bar was invented: every decision and equity mark sits on a stored ts.
        assert runner_ts(ledger, runner.session_id) <= stored_ts(root)
        assert M1_GAP not in runner_ts(ledger, runner.session_id)
        assert all(point.ts in stored_ts(root) for point in ledger.equity(runner.session_id))


def test_real_h1_gap_is_skipped_and_recorded(tmp_path: Path) -> None:
    root = tmp_path / "data"
    # 2009-05-25 is a Monday; 17:00..21:00 missing, exactly as measured.
    skip = range(17, 22)
    bars = hourly_bars(H1_START, 96, skip_hours=skip)
    write(root, bars, timeframe="H1")

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, timeframe="H1", window=20, gap_policy="session_aware")
        summary = runner.advance()

        assert summary.new_signals > 0
        gaps = ledger.gaps(runner.session_id)
        assert len(gaps) == 1
        assert gaps[0].missing_bars == 5
        assert gaps[0].from_ts == H1_START + timedelta(hours=16)
        assert gaps[0].to_ts == H1_START + timedelta(hours=22)
        assert runner_ts(ledger, runner.session_id) <= stored_ts(root, timeframe="H1")


def test_strict_still_processes_a_weekend_jump_without_recording(tmp_path: Path) -> None:
    root = tmp_path / "data"
    # Friday 18:00..20:59 then Sunday 22:00..Monday 01:00; the metals session
    # closes at Friday 21:00 and reopens Sunday 22:00, so no market bar is lost.
    friday = datetime(2026, 1, 9, 18, 0, tzinfo=UTC)
    sunday = datetime(2026, 1, 11, 22, 0, tzinfo=UTC)
    bars = pl.concat(
        [
            minute_bars(friday, 180),
            minute_bars(sunday, 180),
        ]
    )
    write(root, bars)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="strict")
        summary = runner.advance()
        assert summary.new_signals > 0
        assert summary.gaps_skipped == 0
        assert ledger.gaps(runner.session_id) == []


def test_weekend_jump_is_also_unrecorded_under_session_aware(tmp_path: Path) -> None:
    root = tmp_path / "data"
    friday = datetime(2026, 1, 9, 18, 0, tzinfo=UTC)
    sunday = datetime(2026, 1, 11, 22, 0, tzinfo=UTC)
    write(root, pl.concat([minute_bars(friday, 180), minute_bars(sunday, 180)]))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        summary = runner.advance()
        assert summary.new_signals > 0
        assert summary.gaps_skipped == 0
        assert ledger.gaps(runner.session_id) == []


# ------------------------------------------------------------- synthetic gap cases


def test_skip_with_report_records_a_synthetic_gap_without_interpolation(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    bars = minute_bars(start, 200, skip=(150, 151, 152))
    write(root, bars)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="skip_with_report")
        summary = runner.advance()

        assert summary.gaps_skipped == 1
        assert summary.missing_bars == 3
        gaps = ledger.gaps(runner.session_id)
        assert len(gaps) == 1
        assert gaps[0].from_ts == start + timedelta(minutes=149)
        assert gaps[0].to_ts == start + timedelta(minutes=153)
        assert gaps[0].missing_bars == 3
        assert gaps[0].policy == "skip_with_report"

        # The skipped instants are absent from every recorded timestamp.
        skipped = {start + timedelta(minutes=offset) for offset in (150, 151, 152)}
        recorded = runner_ts(ledger, runner.session_id)
        assert recorded <= stored_ts(root)
        assert not recorded & skipped


def test_session_aware_records_a_synthetic_in_session_gap(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(root, minute_bars(start, 200, skip=(130,)))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        summary = runner.advance()
        assert summary.gaps_recorded == 1
        assert ledger.gaps(runner.session_id)[0].missing_bars == 1


def test_gap_inside_warmup_is_recorded_by_the_first_writing_call(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(root, minute_bars(start, 150, skip=(10,)))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="skip_with_report")
        summary = runner.advance()

        # The gap is before the first decidable bar, but it is still audited.
        assert summary.new_signals > 0
        assert summary.gaps_recorded == 1
        gaps = ledger.gaps(runner.session_id)
        assert gaps[0].from_ts == start + timedelta(minutes=9)
        assert gaps[0].to_ts == start + timedelta(minutes=11)


# ------------------------------------------------------------- refusals stay loud


@pytest.mark.parametrize("policy", ["strict", "session_aware", "skip_with_report"])
def test_out_of_order_bars_are_refused_under_every_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: GapPolicy
) -> None:
    root = tmp_path / "data"
    bars = minute_bars(M1_START, 150)
    write(root, bars)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy=policy)
        swapped = pl.concat([bars[:100], bars[101:102], bars[100:101], bars[102:]])
        monkeypatch.setattr(runner._store, "read", lambda *args, **kwargs: swapped)
        with pytest.raises(BarOrderError, match="not sorted"):
            runner.advance()
        assert ledger.session(runner.session_id) is None


@pytest.mark.parametrize("policy", ["strict", "session_aware", "skip_with_report"])
def test_duplicate_bars_are_refused_under_every_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, policy: GapPolicy
) -> None:
    root = tmp_path / "data"
    bars = minute_bars(M1_START, 150)
    write(root, bars)

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy=policy)
        duplicate = pl.concat([bars, bars[-1:]])
        monkeypatch.setattr(runner._store, "read", lambda *args, **kwargs: duplicate)
        with pytest.raises(BarOrderError, match="duplicate"):
            runner.advance()
        assert ledger.session(runner.session_id) is None


def test_off_grid_bar_is_refused_even_when_gaps_are_skipped(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    stamps = [start + timedelta(minutes=index) for index in range(150)]
    stamps[100] = stamps[100] + timedelta(seconds=30)  # half a slot late
    write(root, bars_at(stamps))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="skip_with_report")
        with pytest.raises(BarGapError, match="off the M1 grid"):
            runner.advance()
        assert ledger.session(runner.session_id) is None


# ------------------------------------------------------- unknown symbol / timeframe


def test_session_aware_refuses_an_unregistered_symbol_but_skip_reports_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(root, minute_bars(start, 150, skip=(130,)), symbol="FOOUSD")

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, symbol="FOOUSD", gap_policy="session_aware")
        with pytest.raises(BarGapError, match="not in the registry"):
            runner.advance()
        assert ledger.session(runner.session_id) is None

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, symbol="FOOUSD", gap_policy="skip_with_report")
        summary = runner.advance()
        assert summary.gaps_recorded == 1
        gap = ledger.gaps(runner.session_id)[0]
        assert gap.missing_bars == 1
        assert gap.policy == "skip_with_report"


def test_unknown_timeframe_is_ordering_only_and_never_invents_gaps(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    # Seven-minute spacing is not a registry rule, so no cadence is known.
    stamps = [start + timedelta(minutes=7 * index) for index in range(150)]
    del stamps[100]
    write(root, bars_at(stamps), timeframe="M7")

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, timeframe="M7")
        summary = runner.advance()
        assert summary.new_signals > 0
        assert summary.gaps_skipped == 0
        assert ledger.gaps(runner.session_id) == []


# ----------------------------------------------------------- resume across a gap


def test_restart_across_a_gap_records_it_once_and_keeps_history(tmp_path: Path) -> None:
    root = tmp_path / "data"
    ledger_path = tmp_path / "paper.db"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(root, minute_bars(start, 200, skip=(150,)))

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        first = runner.advance()
        assert first.gaps_recorded == 1
        before = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]
        session_id = runner.session_id

    # Restart with the same store: nothing new, nothing duplicated.
    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        second = runner.advance()
        assert second.new_signals == 0
        assert second.gaps_recorded == 0
        assert len(ledger.gaps(session_id)) == 1
        assert [(signal.ts, signal.action) for signal in ledger.signals(session_id)] == before

    # New bars after the gap; the earlier gap must not be re-recorded or
    # shifted, and the existing signal prefix must be untouched.
    write(root, minute_bars(start, 220, skip=(150,)))
    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        third = runner.advance()
        assert third.new_signals == 20
        assert third.gaps_recorded == 0
        after = [(signal.ts, signal.action) for signal in ledger.signals(session_id)]
        assert after[: len(before)] == before
        assert len(ledger.gaps(session_id)) == 1
        assert len({signal.ts for signal in ledger.signals(session_id)}) == len(after)


def test_resume_discovers_a_gap_in_newly_appended_bars(tmp_path: Path) -> None:
    root = tmp_path / "data"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(root, minute_bars(start, 130))

    with PaperLedger(tmp_path / "paper.db") as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        first = runner.advance()
        assert first.gaps_recorded == 0
        before = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]

        # The gap only exists in the newly stored segment (index 150).
        write(root, minute_bars(start, 220, skip=(150,)))
        second = runner.advance()

        assert second.new_signals > 0
        assert second.gaps_recorded == 1
        gaps = ledger.gaps(runner.session_id)
        assert gaps[0].from_ts == start + timedelta(minutes=149)
        assert gaps[0].to_ts == start + timedelta(minutes=151)
        after = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]
        assert after[: len(before)] == before


def test_policy_change_on_resume_is_refused(tmp_path: Path) -> None:
    root = tmp_path / "data"
    ledger_path = tmp_path / "paper.db"
    write(root, minute_bars(M1_START, 150, skip=(130,)))

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root, gap_policy="skip_with_report")
        runner.advance()
        record = ledger.session(runner.session_id)
        assert record is not None
        assert record.config["extra"]["gap_policy"] == "skip_with_report"

    with PaperLedger(ledger_path) as ledger:
        runner = make_runner(ledger, root, gap_policy="session_aware")
        with pytest.raises(SessionMismatchError, match="gap_policy"):
            runner.advance()


# ------------------------------------------------------------------- no look-ahead


def test_future_bars_do_not_change_decisions_before_a_gap(tmp_path: Path) -> None:
    short_root = tmp_path / "short"
    long_root = tmp_path / "long"
    start = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)
    write(short_root, minute_bars(start, 200, skip=(150,)))
    write(long_root, minute_bars(start, 260, skip=(150,)))

    with PaperLedger(tmp_path / "short.db") as ledger:
        runner = make_runner(ledger, short_root, gap_policy="session_aware")
        summary = runner.advance()
        short_signals = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]
        cutoff = summary.last_bar_ts
        assert cutoff is not None

    with PaperLedger(tmp_path / "long.db") as ledger:
        runner = make_runner(ledger, long_root, gap_policy="session_aware")
        runner.advance()
        long_signals = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]

    assert [row for row in long_signals if row[0] <= cutoff] == short_signals

    # The incremental ledger keeps its prefix when the future arrives.
    with PaperLedger(tmp_path / "short.db") as ledger:
        write(short_root, minute_bars(start, 260, skip=(150,)))
        runner = make_runner(ledger, short_root, gap_policy="session_aware")
        runner.advance()
        extended = [(signal.ts, signal.action) for signal in ledger.signals(runner.session_id)]
    assert extended[: len(short_signals)] == short_signals


# --------------------------------------------------------------- ledger behaviour


def test_gap_rows_round_trip_and_are_idempotent(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        ledger.open_session(
            "s1",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-1",
            started_at=M1_START,
        )
        first = ledger.record_gap(
            "s1",
            from_ts=M1_GAP - timedelta(minutes=1),
            to_ts=M1_GAP + timedelta(minutes=1),
            missing_bars=1,
            policy="session_aware",
        )
        again = ledger.record_gap(
            "s1",
            from_ts=M1_GAP - timedelta(minutes=1),
            to_ts=M1_GAP + timedelta(minutes=1),
            missing_bars=1,
            policy="session_aware",
        )
        assert again == first
        assert ledger.count_gaps("s1") == 1

        gap = ledger.gaps("s1")[0]
        assert gap.gap_id == first
        assert gap.from_ts == M1_GAP - timedelta(minutes=1)
        assert gap.to_ts == M1_GAP + timedelta(minutes=1)
        assert gap.missing_bars == 1
        assert gap.policy == "session_aware"
        assert gap.recorded_at.tzinfo is not None


def test_conflicting_gap_recording_is_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        ledger.open_session(
            "s1",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-1",
            started_at=M1_START,
        )
        ledger.record_gap(
            "s1",
            from_ts=M1_GAP,
            to_ts=M1_GAP + timedelta(minutes=2),
            missing_bars=1,
            policy="session_aware",
        )
        with pytest.raises(LedgerError, match="already recorded"):
            ledger.record_gap(
                "s1",
                from_ts=M1_GAP,
                to_ts=M1_GAP + timedelta(minutes=2),
                missing_bars=2,
                policy="session_aware",
            )


def test_invalid_gap_rows_are_refused(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        ledger.open_session(
            "s1",
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="hash-1",
            started_at=M1_START,
        )
        with pytest.raises(LedgerError, match="unknown session"):
            ledger.record_gap(
                "nope",
                from_ts=M1_GAP,
                to_ts=M1_GAP + timedelta(minutes=1),
                missing_bars=1,
                policy="strict",
            )
        with pytest.raises(ValueError, match="must be after"):
            ledger.record_gap(
                "s1",
                from_ts=M1_GAP,
                to_ts=M1_GAP,
                missing_bars=1,
                policy="strict",
            )
        with pytest.raises(ValueError, match="missing_bars"):
            ledger.record_gap(
                "s1",
                from_ts=M1_GAP,
                to_ts=M1_GAP + timedelta(minutes=1),
                missing_bars=0,
                policy="strict",
            )
        with pytest.raises(ValueError, match="policy"):
            ledger.record_gap(
                "s1",
                from_ts=M1_GAP,
                to_ts=M1_GAP + timedelta(minutes=1),
                missing_bars=1,
                policy="lenient",
            )
        assert ledger.gaps("s1") == []


def test_schema_creates_the_gap_table(tmp_path: Path) -> None:
    with PaperLedger(tmp_path / "paper.db") as ledger:
        names = {
            str(row["name"])
            for row in ledger.connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert "paper_gaps" in names


def test_a_version_one_ledger_is_upgraded_in_place(tmp_path: Path) -> None:
    import sqlite3

    from scalping.paper import ledger as ledger_module

    path = tmp_path / "v1.db"
    conn = sqlite3.connect(path)
    conn.execute(ledger_module._SCHEMA_VERSION_DDL)
    for statement in ledger_module._MIGRATIONS[1]:
        conn.execute(statement)
    conn.execute(
        "INSERT INTO schema_version (version, applied_at) VALUES (1, '2026-01-01T00:00:00+00:00')"
    )
    conn.commit()
    conn.close()

    with PaperLedger(path) as ledger:
        assert ledger.schema_version == 2
        row = ledger.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'paper_gaps'"
        ).fetchone()
        assert row is not None
        # The upgrade is additive: existing tables and rows are untouched.
        assert ledger.init_schema() == 2
