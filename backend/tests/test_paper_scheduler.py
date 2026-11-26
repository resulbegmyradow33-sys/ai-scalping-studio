"""The paper scheduler: an evidence champion plus a new bar becomes one job.

Every test uses a temporary Parquet store, a temporary discovery archive and a
temporary paper ledger. The properties pinned here are the ones the runtime
link depends on:

* an evidence champion with a decidable bar enqueues exactly one
  ``paper.advance``;
* the same decision bar is not enqueued twice, across a poll or a restart, and
  a crash between ``scan`` and ``seen`` converges on the existing job;
* a missing or non-evidence champion, an invalid scope, an unreadable ledger
  and an empty store are refusals with an audit event, never a legacy fallback;
* the payload carries the configured gap policy, the default stays strict and
  the cursor refuses a restart under a different policy;
* a known synthetic store is refused unless the explicit opt-in is given;
* the warm-up window and the provisional tail are respected.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from promotion_evidence import evidence_for
from promotion_support import promote_for_tests, registered_test_policies
from scalping import cli
from scalping.data.provenance import build_provenance, save_provenance
from scalping.data.store import ParquetStore
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.champion import scope_for
from scalping.discovery.promotion import PROMOTION_V1, EvidenceIdentity
from scalping.discovery.protocols import TrialResult
from scalping.orchestrator import JobStore, Orchestrator, scan_once
from scalping.orchestrator.handlers import (
    KIND_PAPER_ADVANCE as HANDLER_KIND,
)
from scalping.orchestrator.handlers import (
    MAX_PAPER_BARS,
    HandlerContext,
    register_default_handlers,
)
from scalping.orchestrator.paper_scheduler import (
    DEFAULT_GAP_POLICY,
    EVENT_DEFERRED,
    EVENT_REFUSED,
    KIND_PAPER_ADVANCE,
    MAX_ADVANCE_BARS,
    REASON_AWAITING_BAR,
    REASON_CURSOR_CORRUPT,
    REASON_INVALID_SCOPE,
    REASON_LEDGER_UNAVAILABLE,
    REASON_NO_ARCHIVE,
    REASON_NO_CHAMPION,
    REASON_NO_EVIDENCE_CHAMPION,
    REASON_PROVENANCE_UNAVAILABLE,
    REASON_SYNTHETIC,
    REASON_WARMING_UP,
    PaperAdvanceScheduler,
    SchedulerPolicyMismatch,
)
from scalping.paper.ledger import PaperLedger
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
START = datetime(2026, 1, 5, tzinfo=UTC)
DAY = START + timedelta(days=1)
DATASET_CHECKSUM = "c1" * 32
SETTINGS_DIGEST = "c2" * 32
TEST_POLICY = replace(
    PROMOTION_V1,
    policy_id="scheduler-test-policy",
    require_locked_oos=False,
)


@pytest.fixture(autouse=True)
def _register_test_policy():
    with registered_test_policies(TEST_POLICY):
        yield


# ----------------------------------------------------------------------- helpers


def trading_spec(holding: int = 10) -> StrategySpec:
    """Long every flat bar, close on the holding limit; deterministic DSL only."""
    return StrategySpec(
        entries=(EntryRule("long", Compare(Const(1.0), ">", Const(0.0))),),
        exits=(ExitRule(Compare(Const(0.0), ">", Const(1.0))),),
        holding=HoldingLimit(holding),
        max_lookback=2,
    )


def bars(end: datetime) -> pl.DataFrame:
    return SyntheticProvider(seed=3).fetch(SYMBOL, START, end)


def write_bars(root: Path, end: datetime = DAY, timeframe: str = TIMEFRAME) -> None:
    ParquetStore(root).write(SYMBOL, timeframe, bars(end))


def last_decidable_ts(root: Path, timeframe: str = TIMEFRAME) -> datetime:
    frame = ParquetStore(root).read(SYMBOL, timeframe)
    stamps = frame.get_column("ts").to_list()
    return stamps[-2]


def promote(archive: DiscoveryArchive, spec: StrategySpec, scope: str) -> str:
    """Archive, trial and promote one spec through the evidence path."""
    dataset_id = f"validation:{scope}:h1:test"
    record = archive.add_candidate(spec, code_version="test")
    archive.record_trial(
        record.candidate_id,
        TrialResult(
            metrics={"expectancy_r": 0.5, "net_profit": 3.0, "trades": 5.0},
            accepted=True,
        ),
        dataset_id=dataset_id,
        code_version="test",
        seed=0,
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
    )
    evidence = evidence_for(
        spec_hash=record.candidate_id,
        dataset_id=dataset_id,
        symbol=SYMBOL,
        timeframe=scope.partition(":")[2],
        code_version="test",
        dataset_checksum=DATASET_CHECKSUM,
        settings_digest=SETTINGS_DIGEST,
        expectancy_r=0.5,
    )
    promote_for_tests(
        archive,
        record.candidate_id,
        evidence,
        TEST_POLICY,
        scope=scope,
        reason="scheduler unit test promotion",
        expected_identity=EvidenceIdentity(
            code_version="test",
            dataset_checksum=DATASET_CHECKSUM,
            settings_digest=SETTINGS_DIGEST,
        ),
    )
    return record.candidate_id


def seed_archive(
    path: Path,
    *,
    spec: StrategySpec | None = None,
    scope: str = "XAUUSD:M1",
) -> str:
    with DiscoveryArchive(path) as archive:
        return promote(archive, spec if spec is not None else trading_spec(), scope)


def seed_ledger(path: Path, session_id: str, ts: datetime, *, window: int = 60) -> None:
    with PaperLedger(path) as ledger:
        ledger.open_session(
            session_id,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            strategy="dsl",
            model_version="champion-hash",
            config={"window": window, "extra": {"origin": "continuous"}},
        )
        ledger.record_signal(session_id, ts, "hold", model_version="champion-hash")


def session_id_for(champion_id: str) -> str:
    return f"auto-{SYMBOL}-{TIMEFRAME}-{champion_id[:16]}"


def scheduler(
    tmp_path: Path,
    *,
    scopes: tuple[str, ...] | None = None,
    max_scopes: int = 100,
    gap_policy: str = "strict",
    allow_synthetic: bool = False,
) -> PaperAdvanceScheduler:
    return PaperAdvanceScheduler(
        store=JobStore(tmp_path / "orchestrator.sqlite3"),
        data_root=tmp_path / "data",
        archive_path=tmp_path / "runs" / "discovery" / "discovery.sqlite",
        ledger_path=tmp_path / "archive" / "paper.sqlite",
        scopes=scopes,
        max_scopes=max_scopes,
        gap_policy=gap_policy,
        allow_synthetic=allow_synthetic,
    )


def event_details(store: JobStore, name: str) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for event in store.events():
        if event.event != name:
            continue
        assert event.detail is not None
        parsed = json.loads(event.detail)
        assert isinstance(parsed, dict)
        details.append(parsed)
    return details


def jobs(store: JobStore) -> list:
    return store.list_jobs(kind=KIND_PAPER_ADVANCE)


def archive_path(tmp_path: Path) -> Path:
    return tmp_path / "runs" / "discovery" / "discovery.sqlite"


# ------------------------------------------------------------------------ tests


def test_scheduler_kind_matches_the_registered_handler() -> None:
    assert KIND_PAPER_ADVANCE == HANDLER_KIND
    assert MAX_ADVANCE_BARS == MAX_PAPER_BARS


def test_new_decision_bar_enqueues_exactly_one_advance(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    champion_id = seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert len(outcome.queued) == 1
    job = outcome.queued[0]
    assert job.kind == KIND_PAPER_ADVANCE
    assert set(job.payload) == {"symbol", "timeframe", "bars", "gap_policy"}
    assert job.payload["symbol"] == SYMBOL
    assert job.payload["timeframe"] == TIMEFRAME
    assert 1 <= job.payload["bars"] <= MAX_ADVANCE_BARS
    assert job.payload["gap_policy"] == DEFAULT_GAP_POLICY == "strict"
    assert job.dedupe_key == (
        f"{KIND_PAPER_ADVANCE}:{scope_for(SYMBOL, TIMEFRAME)}:"
        f"{last_decidable_ts(tmp_path / 'data').isoformat()}"
    )

    # A second poll over the same bar is quiet: the cursor records the key.
    again = scan_once(Orchestrator(batch.store), [batch])
    assert again.queued == () and again.duplicates == ()
    assert len(jobs(batch.store)) == 1
    # A cold start (no ledger at all) still enqueues: the first advance is what
    # creates the ledger. The scheduler never writes it.
    assert not (tmp_path / "archive" / "paper.sqlite").exists()
    assert session_id_for(champion_id)  # the id the handler will use


def test_same_bar_across_a_restart_is_not_requeued(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    first = scheduler(tmp_path)
    scan_once(Orchestrator(first.store), [first])

    restarted = scheduler(tmp_path)
    outcome = scan_once(Orchestrator(restarted.store), [restarted])

    assert outcome.queued == () and outcome.duplicates == ()
    assert len(jobs(first.store)) == 1


def test_crash_between_enqueue_and_seen_converges_on_one_job(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    first = scheduler(tmp_path)
    # The scan emits, the enqueue happens, then the process dies before ``seen``.
    for trigger in first.scan():
        first.store.enqueue(
            trigger.kind,
            dedupe_key=trigger.dedupe_key,
            payload=trigger.payload,
            priority=trigger.priority,
            max_retries=trigger.max_retries,
        )

    restarted = scheduler(tmp_path)
    outcome = scan_once(Orchestrator(restarted.store), [restarted])

    assert outcome.queued == ()
    assert len(outcome.duplicates) == 1
    assert len(jobs(first.store)) == 1


def test_ledger_cursor_needs_two_bars_after_it(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_bars(data_root)
    champion_id = seed_archive(archive_path(tmp_path))
    ledger_path = tmp_path / "archive" / "paper.sqlite"
    seed_ledger(ledger_path, session_id_for(champion_id), last_decidable_ts(data_root))
    batch = scheduler(tmp_path)

    # The cursor already sits on the newest decidable bar and the stored tail is
    # provisional: nothing to advance.
    outcome = scan_once(Orchestrator(batch.store), [batch])
    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_AWAITING_BAR

    # One appended bar makes the previously provisional tail decidable. The new
    # tail stays provisional, so this is one decision, not two.
    write_bars(data_root, START + timedelta(days=1, minutes=1))
    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert len(outcome.queued) == 1
    job = outcome.queued[0]
    assert job.dedupe_key.endswith("2026-01-06T00:00:00+00:00")
    assert job.payload["bars"] == 1

    # The same provisional rule now applies to the new tail.
    again = scan_once(Orchestrator(batch.store), [batch])
    assert again.queued == ()


def test_warmup_and_window_are_respected(tmp_path: Path) -> None:
    seed_archive(archive_path(tmp_path))
    # window 60 + WARMUP 60 puts the first decision index at 119; the tail keeps
    # one bar provisional, so 120 bars are still one short.
    write_bars(tmp_path / "data", START + timedelta(minutes=119))
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])
    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_WARMING_UP

    write_bars(tmp_path / "data", START + timedelta(minutes=120))
    outcome = scan_once(Orchestrator(batch.store), [batch])
    assert len(outcome.queued) == 1


def test_exploratory_champion_is_refused_and_never_falls_back(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    path = archive_path(tmp_path)
    seed_archive(path)
    with DiscoveryArchive(path) as archive:
        archive.connection.execute(
            """
            UPDATE champions SET mode = 'exploratory', policy_id = NULL,
                policy_version = NULL, policy_digest = NULL, evidence_checksum = NULL
            """
        )
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    refusals = event_details(batch.store, EVENT_REFUSED)
    assert len(refusals) == 1
    assert refusals[0]["reason"] == REASON_NO_EVIDENCE_CHAMPION
    assert refusals[0]["scope"] == scope_for(SYMBOL, TIMEFRAME)
    assert not (tmp_path / "archive" / "paper.sqlite").exists()

    # The refusal is not repeated by every poll.
    scan_once(Orchestrator(batch.store), [batch])
    assert len(event_details(batch.store, EVENT_REFUSED)) == 1


def test_no_champion_scope_is_refused(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    path = archive_path(tmp_path)
    with DiscoveryArchive(path):
        pass  # schema exists, nothing was promoted
    batch = scheduler(tmp_path, scopes=(scope_for(SYMBOL, TIMEFRAME),))

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    refusals = event_details(batch.store, EVENT_REFUSED)
    assert refusals[-1]["reason"] == REASON_NO_CHAMPION


def test_missing_archive_is_an_audited_refusal(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_NO_ARCHIVE
    assert not archive_path(tmp_path).exists()  # the scheduler never creates it


def test_invalid_scope_is_refused(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path, scopes=("xauusd:M1",))

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_INVALID_SCOPE


def test_unreadable_ledger_is_refused_not_guessed(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    ledger_path = tmp_path / "archive" / "paper.sqlite"
    ledger_path.parent.mkdir(parents=True)
    ledger_path.write_bytes(b"this is not a sqlite database")
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_LEDGER_UNAVAILABLE


def test_unparseable_ledger_cursor_is_refused(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    write_bars(data_root)
    champion_id = seed_archive(archive_path(tmp_path))
    ledger_path = tmp_path / "archive" / "paper.sqlite"
    seed_ledger(ledger_path, session_id_for(champion_id), last_decidable_ts(data_root))
    with PaperLedger(ledger_path) as ledger:
        ledger.connection.execute("UPDATE signals SET ts = 'not-a-timestamp'")
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == () and len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == REASON_LEDGER_UNAVAILABLE


def holed_bars() -> pl.DataFrame:
    """One day of M1 bars with a single in-session bar removed."""
    return bars(DAY).with_row_index().filter(pl.col("index") != 300).drop("index")


def handler_orchestrator(batch: PaperAdvanceScheduler) -> Orchestrator:
    """The real handlers over the same roots the scheduler reads."""
    context = HandlerContext(
        data_root=batch.data_root,
        runs_root=batch.archive_path.parent.parent,
        archive_root=batch.ledger_path.parent,
        use_feature_cache=False,
    )
    orchestrator = Orchestrator(batch.store)
    register_default_handlers(orchestrator, context)
    return orchestrator


def test_payload_carries_the_default_gap_policy(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert len(outcome.queued) == 1
    payload = outcome.queued[0].payload
    assert set(payload) == {"symbol", "timeframe", "bars", "gap_policy"}
    assert payload["gap_policy"] == DEFAULT_GAP_POLICY == "strict"
    source = Path("src/scalping/orchestrator/paper_scheduler.py").read_text(encoding="utf-8")
    assert "FlowMomentumRule" not in source
    assert "scalping.strategy" not in source


def test_configured_gap_policy_is_carried_in_the_payload(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path, gap_policy="session_aware")

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert len(outcome.queued) == 1
    assert outcome.queued[0].payload["gap_policy"] == "session_aware"


@pytest.mark.parametrize("policy", ["ignore", "", "STRICT"])
def test_invalid_gap_policy_is_refused_at_construction(tmp_path: Path, policy: str) -> None:
    with pytest.raises(ValueError, match="gap_policy"):
        scheduler(tmp_path, gap_policy=policy)


def test_gap_policy_and_synthetic_opt_in_are_validated(tmp_path: Path) -> None:
    kwargs = {
        "store": JobStore(tmp_path / "orchestrator.sqlite3"),
        "data_root": tmp_path / "data",
        "archive_path": archive_path(tmp_path),
        "ledger_path": tmp_path / "archive" / "paper.sqlite",
    }
    with pytest.raises(TypeError, match="gap_policy"):
        PaperAdvanceScheduler(gap_policy=1, **kwargs)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="allow_synthetic"):
        PaperAdvanceScheduler(allow_synthetic=1, **kwargs)  # type: ignore[arg-type]


def test_cursor_persists_the_policy_and_a_restart_keeps_it(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    first = scheduler(tmp_path, gap_policy="session_aware")
    outcome = scan_once(Orchestrator(first.store), [first])
    assert len(outcome.queued) == 1

    raw = first.store.get_state(first.cursor_key)
    assert raw is not None
    document = json.loads(raw)
    assert document["version"] == 2
    assert document["gap_policy"] == "session_aware"

    restarted = scheduler(tmp_path, gap_policy="session_aware")
    again = scan_once(Orchestrator(restarted.store), [restarted])
    assert again.queued == () and again.duplicates == ()
    assert len(jobs(first.store)) == 1

    changed = scheduler(tmp_path, gap_policy="strict")
    with pytest.raises(SchedulerPolicyMismatch) as mismatch:
        changed.scan()
    message = str(mismatch.value)
    assert "recorded gap_policy 'session_aware'" in message
    assert "configured 'strict'" in message
    assert "pre-policy" not in message


def test_legacy_cursor_without_a_policy_is_read_as_strict(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path)
    batch.store.set_state(batch.cursor_key, json.dumps({"version": 1, "offset": 0}))

    with pytest.raises(SchedulerPolicyMismatch) as mismatch:
        scheduler(tmp_path, gap_policy="session_aware").scan()
    message = str(mismatch.value)
    assert "predates the gap_policy field" in message
    assert "recorded gap_policy" not in message

    outcome = scan_once(Orchestrator(batch.store), [batch])
    assert len(outcome.queued) == 1
    assert outcome.queued[0].payload["gap_policy"] == "strict"
    upgraded = json.loads(batch.store.get_state(batch.cursor_key) or "{}")
    assert upgraded["gap_policy"] == "strict"


@pytest.mark.parametrize(
    "document",
    ["{not json", "[]", "123", json.dumps({"gap_policy": "ignore"})],
)
def test_corrupt_cursor_is_refused_not_reinterpreted(tmp_path: Path, document: str) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))
    session_aware = scheduler(tmp_path, gap_policy="session_aware")
    scan_once(Orchestrator(session_aware.store), [session_aware])
    assert len(jobs(session_aware.store)) == 1
    session_aware.store.set_state(session_aware.cursor_key, document)

    strict = scheduler(tmp_path, gap_policy="strict")
    outcome = scan_once(Orchestrator(strict.store), [strict])

    assert outcome.queued == () and outcome.duplicates == ()
    assert len(jobs(strict.store)) == 1, "a corrupt cursor must not emit a new job"
    refusals = event_details(strict.store, EVENT_REFUSED)
    assert refusals[-1]["reason"] == REASON_CURSOR_CORRUPT
    assert refusals[-1]["scope"] is None


def test_known_synthetic_store_is_refused_unless_allowed(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data = bars(DAY)
    ParquetStore(data_root).write(SYMBOL, TIMEFRAME, data)
    save_provenance(
        data_root,
        build_provenance(
            data,
            symbol=SYMBOL,
            timeframe=TIMEFRAME,
            source="synthetic",
            synthetic=True,
        ),
    )
    seed_archive(archive_path(tmp_path))

    refused = scheduler(tmp_path)
    outcome = scan_once(Orchestrator(refused.store), [refused])
    assert outcome.queued == ()
    assert event_details(refused.store, EVENT_REFUSED)[-1]["reason"] == REASON_SYNTHETIC

    allowed = scheduler(tmp_path, allow_synthetic=True)
    outcome = scan_once(Orchestrator(allowed.store), [allowed])
    assert len(outcome.queued) == 1
    assert outcome.queued[0].payload["gap_policy"] == "strict"


@pytest.mark.parametrize(
    "sidecar_text",
    [
        "not json",
        "{}",
        '{"synthetic": true}',
        '{"symbol": "XAUUSD"}',
        "[]",
    ],
)
def test_unreadable_provenance_is_refused_not_guessed(tmp_path: Path, sidecar_text: str) -> None:
    data_root = tmp_path / "data"
    write_bars(data_root)
    sidecar = data_root / "_meta" / SYMBOL / f"{TIMEFRAME}.json"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text(sidecar_text, encoding="utf-8")
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path)

    outcome = scan_once(Orchestrator(batch.store), [batch])

    assert outcome.queued == ()
    assert len(jobs(batch.store)) == 0
    assert event_details(batch.store, EVENT_REFUSED)[-1]["reason"] == (
        REASON_PROVENANCE_UNAVAILABLE
    )


def test_session_aware_policy_reaches_the_handler_end_to_end(tmp_path: Path) -> None:
    ParquetStore(tmp_path / "data").write(SYMBOL, TIMEFRAME, holed_bars())
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path, gap_policy="session_aware")
    orchestrator = handler_orchestrator(batch)

    scan = scan_once(orchestrator, [batch])
    assert len(scan.queued) == 1
    job = scan.queued[0]
    assert job.payload["gap_policy"] == "session_aware"

    run = orchestrator.run_once()
    row = batch.store.get(job.id)
    assert row is not None and row.result is not None
    assert run.status == "succeeded"
    result = row.result["data"]
    assert result["gap_policy"] == "session_aware"
    assert result["gaps_skipped"] == 1
    assert result["missing_bars"] == 1
    assert result["gaps_recorded"] == 1
    assert result["bars_advanced"] > 0


def test_default_strict_policy_still_refuses_the_same_gap_end_to_end(tmp_path: Path) -> None:
    ParquetStore(tmp_path / "data").write(SYMBOL, TIMEFRAME, holed_bars())
    seed_archive(archive_path(tmp_path))
    batch = scheduler(tmp_path)
    orchestrator = handler_orchestrator(batch)

    scan = scan_once(orchestrator, [batch])
    assert len(scan.queued) == 1
    job = scan.queued[0]
    assert job.payload["gap_policy"] == "strict"

    run = orchestrator.run_once()
    row = batch.store.get(job.id)
    assert row is not None
    assert run.status == "retry"
    assert "bar_gap" in (row.error or "")


def test_bounded_scope_scan_rotates_and_audits(tmp_path: Path) -> None:
    write_bars(tmp_path / "data")
    write_bars(tmp_path / "data", timeframe="M5")
    path = archive_path(tmp_path)
    with DiscoveryArchive(path) as archive:
        promote(archive, trading_spec(), "XAUUSD:M1")
        promote(archive, trading_spec(holding=5), "XAUUSD:M5")
    batch = scheduler(tmp_path, max_scopes=1)

    first = scan_once(Orchestrator(batch.store), [batch])
    assert len(first.queued) == 1
    assert event_details(batch.store, EVENT_DEFERRED)[-1]["deferred"] == 1

    second = scan_once(Orchestrator(batch.store), [batch])
    assert len(second.queued) == 1
    assert {job.payload["timeframe"] for job in jobs(batch.store)} == {"M1", "M5"}


# -------------------------------------------------------------------------- CLI


def _cli_args(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "orchestrator",
        "run",
        "--once",
        "--profile",
        "none",
        "--db",
        str(tmp_path / "orchestrator.sqlite3"),
        "--root",
        str(tmp_path / "data"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--archive-root",
        str(tmp_path / "archive"),
        "--max-jobs-per-tick",
        "0",
        *extra,
    ]


def test_cli_schedule_paper_enqueues_one_advance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))

    code = cli.main(_cli_args(tmp_path, "--schedule-paper"))

    assert code == 0
    payload = json.loads(capsys.readouterr().out.strip())
    assert payload["tick"]["scan"]["queued"]
    store = JobStore(tmp_path / "orchestrator.sqlite3")
    try:
        listed = store.list_jobs(kind=KIND_PAPER_ADVANCE)
        assert len(listed) == 1
        assert listed[0].payload["gap_policy"] == "strict"
    finally:
        store.close()


def test_cli_paper_gap_policy_reaches_the_job(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_bars(tmp_path / "data")
    seed_archive(archive_path(tmp_path))

    code = cli.main(_cli_args(tmp_path, "--schedule-paper", "--paper-gap-policy", "session_aware"))
    assert code == 0
    capsys.readouterr()
    store = JobStore(tmp_path / "orchestrator.sqlite3")
    try:
        listed = store.list_jobs(kind=KIND_PAPER_ADVANCE)
        assert len(listed) == 1
        assert listed[0].payload["gap_policy"] == "session_aware"
    finally:
        store.close()


def test_cli_paper_gap_policy_requires_the_scheduler(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(_cli_args(tmp_path, "--paper-gap-policy", "session_aware"))

    assert code == 2
    error = json.loads(capsys.readouterr().err.strip())
    assert "paper-gap-policy" in str(error)


def test_cli_schedule_paper_cannot_skip_the_scan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = cli.main(_cli_args(tmp_path, "--schedule-paper", "--no-scan"))

    assert code == 2
    error = json.loads(capsys.readouterr().err.strip())
    assert "no-scan" in str(error)
