"""Phase 11 acceptance: no policy, spec or job payload can bypass a risk limit.

The generated DSL spec is data. It may name entry, exit, holding and sizing
nodes; it cannot name a risk limit, and the evaluator never reads its sizing
node. Every entry path — a raw simulator action, a DSL policy in
:class:`~scalping.discovery.evaluate.DslEvaluator`, or an orchestrator job —
reaches the same frozen :class:`~scalping.env.risk.RiskEngine`. These tests run
both paths on the *same* bars and settings and compare the fills, so a
divergence fails instead of passing silently.

The synthetic frames are constant-price with a single downward step, or
constant-price flat. That makes the loss that trips each kill switch a known
number of dollars instead of a statistical outcome.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields, is_dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scalping.config import ContractSpec, Settings
from scalping.data.registry import session_for
from scalping.data.synthetic import SyntheticProvider
from scalping.discovery.evaluate import DslEvaluator
from scalping.env.risk import RiskEngine
from scalping.env.simulator import WINDOW, Simulator
from scalping.features.build import FeatureFrame, build_features
from scalping.orchestrator import JobStore, Orchestrator
from scalping.orchestrator.handlers import (
    KIND_DATASET_BUILD,
    KIND_DISCOVERY_RUN,
    KIND_KNOWLEDGE_INGEST,
    KIND_PAPER_ADVANCE,
    KIND_PAPER_ANALYSE,
    HandlerContext,
    PaperAdvanceRequest,
    PayloadError,
    register_default_handlers,
)
from scalping.paper.autonomous import AutonomousPaperRunner
from scalping.strategy.dsl import (
    AtrMultiple,
    Compare,
    Const,
    DslValidationError,
    EntryRule,
    ExitRule,
    FixedFraction,
    HoldingLimit,
    StrategySpec,
)
from scalping.types import RiskConfig, Trade

SYMBOL = "XAUUSD"
TIMEFRAME = "M1"
DATASET_ID = "validation:XAUUSD:M1:risk-acceptance"
CODE_VERSION = "test-2026-11-24"
START = datetime(2026, 1, 5, tzinfo=UTC)
SESSION = session_for(SYMBOL)

#: Conditions that are always true / always false as *data*, not as code.
ALWAYS = Compare(Const(0.0), "<", Const(1.0))
NEVER = Compare(Const(0.0), ">", Const(1.0))


# --------------------------------------------------------------------------- frames


def drop_frame(
    rows: int = 160,
    *,
    drop_at: int = 60,
    drop: float = 1.0,
    price: float = 1000.0,
    spread: float = 0.20,
    atr: float = 5.0,
    start: datetime = START,
) -> FeatureFrame:
    """Constant price, then one downward step from ``drop_at`` on.

    The window is 60 bars, so the first decision bar is 59. An always-long
    entry fills at bar 59 and the step at bar 60 produces a known loss: the
    fill is ``ask + slippage``, the mark is the new bid, and ATR 5.0 keeps the
    stop (5.0 away) out of reach. That is enough to trip a 0.1% loss limit
    without ever touching SL/TP, so the kill switch is the only cause.
    """
    prices = np.full(rows, price, dtype=np.float64)
    prices[drop_at:] = price - drop
    ts = np.datetime64(start.replace(tzinfo=None), "us") + np.arange(rows).astype("timedelta64[m]")
    ohlc = np.column_stack([prices, prices + 0.001, prices - 0.001, prices])
    return FeatureFrame(
        X=np.zeros((rows, 1), dtype=np.float32),
        ts=ts.astype("datetime64[us]"),
        ohlc=ohlc,
        spread=np.full(rows, spread),
        atr=np.full(rows, atr),
        valid_from=0,
        columns=["ret_1"],
    )


def long_spec(
    *,
    sizing: FixedFraction | AtrMultiple | None = None,
    holding: int | None = 1,
) -> StrategySpec:
    """Always wants to be long; the holding limit (when set) closes it."""
    return StrategySpec(
        entries=(EntryRule("long", ALWAYS),),
        exits=(ExitRule(NEVER),),
        holding=None if holding is None else HoldingLimit(holding),
        sizing=sizing,
        max_lookback=5,
    )


def evaluator(features: FeatureFrame, settings: Settings) -> DslEvaluator:
    return DslEvaluator(
        features,
        settings,
        symbol=SYMBOL,
        timeframe=TIMEFRAME,
        dataset_id=DATASET_ID,
        window=WINDOW,
    )


def trade_key(trade: Trade) -> tuple[Any, ...]:
    return (
        trade.entry_ts,
        trade.exit_ts,
        trade.side,
        trade.lot,
        trade.entry,
        trade.exit,
        trade.pnl,
        trade.reason,
    )


def drive_always_long(
    features: FeatureFrame, settings: Settings
) -> tuple[list[Trade], list[tuple[int, str]]]:
    """Mirror the evaluator's policy loop by hand: enter flat, close after 1 bar.

    Returns the closed trades and every ``(bar_index, blocked reason)`` for a
    rejected entry. Running the evaluator on the same frame must produce the
    identical trade list, which proves both paths share the same gate.
    """
    sim = Simulator(features, settings, session=SESSION)
    sim.reset()
    trades: list[Trade] = []
    blocked: list[tuple[int, str]] = []
    while True:
        account = sim.broker.state(sim.balance, sim.day_start_equity)
        if sim.broker.position is None:
            action = 1  # BUY
        elif account.bars_in_trade >= 1:
            action = 3  # CLOSE
        else:
            action = 0  # HOLD
        result = sim.step(action)
        if action == 1 and result.info["blocked"] is not None:
            blocked.append((sim.bar_index, str(result.info["blocked"])))
        trade = result.info["trade"]
        if isinstance(trade, Trade):
            trades.append(trade)
        if result.terminated or result.truncated:
            break
    return trades, blocked


# ----------------------------------------------------- DSL spec cannot size or loosen


def test_dsl_sizing_node_never_reaches_the_broker() -> None:
    """The lot is the RiskEngine's, whatever the spec asks for."""
    features = drop_frame()
    settings = Settings()
    runner = evaluator(features, settings)

    plain = runner.simulate(long_spec())
    extreme_atr = runner.simulate(long_spec(sizing=AtrMultiple(1_000.0)))
    full_equity = runner.simulate(long_spec(sizing=FixedFraction(1.0)))

    assert plain.trades, "the control run must trade"
    key = [trade_key(trade) for trade in plain.trades]
    assert [trade_key(trade) for trade in extreme_atr.trades] == key
    assert [trade_key(trade) for trade in full_equity.trades] == key
    # Sizing is still part of the candidate identity, so the hashes must differ
    # even though the fills do not.
    assert long_spec().canonical_hash() != long_spec(sizing=AtrMultiple(1_000.0)).canonical_hash()

    engine = RiskEngine(settings.risk, settings.contract)
    balance = settings.contract.initial_balance
    for trade in plain.trades:
        sizing = engine.size(balance, trade.entry, trade.sl)
        assert trade.lot == pytest.approx(sizing.lot)
        risk_budget = balance * settings.risk.risk_pct
        assert trade.lot * abs(trade.entry - trade.sl) * settings.contract.size <= (
            risk_budget + 1e-9
        )
        balance += trade.pnl


def test_an_always_enter_spec_cannot_open_a_second_position() -> None:
    """The spec wants a position on every bar; the engine still allows one."""
    features = drop_frame(rows=200, drop=0.0)
    settings = Settings(risk=RiskConfig(max_positions=1))

    sim = Simulator(features, settings, session=SESSION)
    sim.reset()
    first = sim.step(1)
    assert first.info["opened"] is not None
    for _ in range(10):
        blocked = sim.step(1)
        assert blocked.info["blocked"] == "max_positions"
        assert blocked.info["opened"] is None

    # The same always-enter policy through the DSL evaluator: one position at a
    # time. A trade may end by the holding timeout, never by a second entry.
    spec = long_spec(holding=None)
    result = evaluator(features, settings).simulate(spec)
    assert result.trades
    previous_exit: datetime | None = None
    for trade in result.trades:
        if previous_exit is not None:
            assert trade.entry_ts >= previous_exit
        previous_exit = trade.exit_ts


def test_a_spec_cannot_exceed_the_notional_cap() -> None:
    """A tight stop makes the risk lot huge; the leverage cap still binds."""
    features = drop_frame(drop=0.0, atr=0.05)
    settings = Settings(risk=RiskConfig(max_notional_x=1.0))
    spec = long_spec(sizing=AtrMultiple(1_000.0))

    result = evaluator(features, settings).simulate(spec)
    assert result.trades
    engine = RiskEngine(settings.risk, settings.contract)
    balance = settings.contract.initial_balance
    for trade in result.trades:
        sizing = engine.size(balance, trade.entry, trade.sl)
        assert sizing.reason == "notional_cap"
        assert trade.lot == pytest.approx(sizing.lot)
        notional = trade.lot * settings.contract.size * trade.entry
        assert notional <= balance * settings.risk.max_notional_x + 1e-6
        balance += trade.pnl


def test_generated_spec_has_no_risk_field_and_cannot_smuggle_one() -> None:
    payload = long_spec(sizing=AtrMultiple(4.0)).to_dict()
    assert "risk" not in payload
    assert not any("risk" in key for key in payload)
    with pytest.raises(DslValidationError):
        StrategySpec.from_dict({**payload, "risk": {"risk_pct": 0.5}})

    names = {field.name for field in fields(StrategySpec)}
    assert not {name for name in names if "risk" in name}


def test_evaluate_and_paper_run_take_no_settings_or_risk_argument() -> None:
    evaluate = list(inspect.signature(DslEvaluator.evaluate).parameters)
    assert evaluate == ["self", "candidate", "dataset_id", "code_version", "budget"]
    assert "settings" not in evaluate and "risk" not in evaluate

    run = list(inspect.signature(AutonomousPaperRunner.run).parameters)
    assert run == ["self", "bars", "max_bars"]


# ------------------------------------------- kill switches block the same entry


KILL_SWITCHES = [
    ("daily_loss", RiskConfig(daily_loss_pct=0.001), "daily_loss_limit", 1),
    ("weekly_loss", RiskConfig(weekly_loss_pct=0.001), "weekly_loss_limit", 1),
    ("max_drawdown", RiskConfig(max_drawdown_pct=0.001), "max_drawdown", 1),
    (
        "consecutive_losses",
        RiskConfig(consecutive_loss_pause=1),
        "consecutive_loss_pause",
        1,
    ),
    ("emergency_stop", RiskConfig(min_equity_frac=0.999), "emergency_stop", 1),
    ("max_spread", RiskConfig(max_spread_points=150.0), "max_spread", 0),
]


@pytest.mark.parametrize(
    ("name", "risk", "reason", "trades_expected"),
    KILL_SWITCHES,
    ids=[case[0] for case in KILL_SWITCHES],
)
def test_kill_switch_blocks_the_same_entry_in_the_simulator_and_the_evaluator(
    name: str,
    risk: RiskConfig,
    reason: str,
    trades_expected: int,
) -> None:
    features = drop_frame()
    settings = Settings(risk=risk)

    manual_trades, blocked = drive_always_long(features, settings)
    assert reason in [item[1] for item in blocked], (
        f"{name}: the simulator must reject the post-trip entry with {reason!r}"
    )

    evaluated = evaluator(features, settings).simulate(long_spec())
    assert len(evaluated.trades) == trades_expected, f"{name}: extra entries escaped the gate"
    assert [trade_key(trade) for trade in evaluated.trades] == [
        trade_key(trade) for trade in manual_trades
    ]


def test_without_a_switch_the_same_frame_keeps_trading() -> None:
    manual_trades, blocked = drive_always_long(drop_frame(), Settings())
    assert len(manual_trades) > 1
    assert not blocked, "a disabled switch must not block anything"


@pytest.mark.parametrize("reason", ["weekly_loss_limit", "max_drawdown", "emergency_stop"])
def test_equity_kill_switches_force_close_the_open_position(reason: str) -> None:
    settings = {
        "weekly_loss_limit": Settings(risk=RiskConfig(weekly_loss_pct=0.001)),
        "max_drawdown": Settings(risk=RiskConfig(max_drawdown_pct=0.001)),
        "emergency_stop": Settings(risk=RiskConfig(min_equity_frac=0.999)),
    }[reason]
    features = drop_frame()

    manual_trades, _ = drive_always_long(features, settings)
    assert len(manual_trades) == 1
    assert manual_trades[0].reason == reason

    evaluated = evaluator(features, settings).simulate(long_spec())
    assert len(evaluated.trades) == 1
    assert evaluated.trades[0].reason == reason


# ------------------------------------------------- action masks cannot disagree


def _sim(settings: Settings | None = None, features: FeatureFrame | None = None) -> Simulator:
    sim = Simulator(features or drop_frame(), settings or Settings(), session=SESSION)
    sim.reset()
    return sim


def _daily_loss() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(daily_loss_pct=0.001)))
    sim.balance = 9_000.0
    sim._equity_value = 9_000.0
    return sim


def _weekly_loss() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(weekly_loss_pct=0.05)))
    sim.balance = 9_000.0
    sim._equity_value = 9_000.0
    sim.day_start_equity = 9_000.0
    sim.week_start_equity = 20_000.0
    return sim


def _drawdown() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(max_drawdown_pct=0.10)))
    sim.balance = 9_000.0
    sim._equity_value = 9_000.0
    sim.day_start_equity = 9_000.0
    sim._peak = 20_000.0
    return sim


def _emergency() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(min_equity_frac=0.8)))
    sim.balance = 7_000.0
    sim._equity_value = 7_000.0
    sim.day_start_equity = 7_000.0
    return sim


def _consecutive() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(consecutive_loss_pause=2)))
    sim.consecutive_losses = 2
    return sim


def _spread() -> Simulator:
    return _sim(Settings(risk=RiskConfig(max_spread_points=100.0)))


def _paused() -> Simulator:
    sim = _sim(Settings(risk=RiskConfig(pause_bars=3)))
    sim.paused_until = START + timedelta(hours=1)
    return sim


def _position_open() -> Simulator:
    sim = _sim()
    assert sim.step(1).info["position"] is not None
    return sim


def _no_atr() -> Simulator:
    return _sim(features=drop_frame(atr=0.0))


def _lot_too_small() -> Simulator:
    return _sim(Settings(contract=ContractSpec(min_lot=10.0)))


def _session_closed() -> Simulator:
    saturday = datetime(2026, 1, 3, tzinfo=UTC)
    return _sim(features=drop_frame(start=saturday))


def _flow_blocked() -> Simulator:
    features = drop_frame()
    flow = np.zeros(len(features.ts))
    sim = Simulator(
        features,
        Settings(),
        flow=flow,
        flow_thr=np.zeros(len(features.ts)),
        session=SESSION,
    )
    sim.reset()
    return sim


GATES: list[tuple[str, Callable[[], Simulator], str]] = [
    ("daily_loss", _daily_loss, "daily_loss_limit"),
    ("weekly_loss", _weekly_loss, "weekly_loss_limit"),
    ("max_drawdown", _drawdown, "max_drawdown"),
    ("emergency_stop", _emergency, "emergency_stop"),
    ("consecutive_losses", _consecutive, "consecutive_loss_pause"),
    ("max_spread", _spread, "max_spread"),
    ("kill_switch_pause", _paused, "kill_switch_pause"),
    ("position_open", _position_open, "max_positions"),
    ("no_atr", _no_atr, "no_atr"),
    ("lot_too_small", _lot_too_small, "lot_too_small"),
    ("session_closed", _session_closed, "session_closed"),
    ("flow_gate", _flow_blocked, "no_flow"),
]


@pytest.mark.parametrize(("name", "build", "reason"), GATES, ids=[case[0] for case in GATES])
def test_action_masks_agree_with_the_gate_and_the_step(
    name: str, build: Callable[[], Simulator], reason: str
) -> None:
    sim = build()
    mask = sim.action_masks()
    assert bool(mask[0]), f"{name}: HOLD must always be available"

    for action in (1, 2):  # BUY, SELL
        entry_reason = sim._entry_reason(action)
        assert bool(mask[action]) == (entry_reason is None)
        assert entry_reason == reason, f"{name}: expected {reason!r}, got {entry_reason!r}"

        probe = build()
        result = probe.step(action)
        assert result.info["blocked"] == reason
        assert result.info["opened"] is None, f"{name}: a rejected entry must open nothing"


def test_default_state_allows_entries_and_a_step_fills_one() -> None:
    sim = _sim()
    mask = sim.action_masks()
    assert mask.tolist() == [True, True, True, False]
    result = sim.step(1)
    assert result.info["blocked"] is None
    assert result.info["position"] is not None


def test_done_and_no_bar_frames_mask_every_action_but_hold() -> None:
    done = _sim()
    done._force_equity(8_000.0)
    assert done.step(0).terminated
    assert done.action_masks().tolist() == [True, False, False, False]

    balance = done.balance
    no_op = done.step(1)
    assert no_op.info["blocked"] is None
    assert done.balance == balance, "a done episode must not act on a masked entry"

    short = _sim(features=drop_frame(rows=30))
    assert short._no_bars
    assert short.action_masks().tolist() == [True, False, False, False]


def _replay(features: FeatureFrame, settings: Settings, actions: list[int]) -> Simulator:
    sim = Simulator(features, settings, session=SESSION)
    sim.reset()
    for action in actions:
        sim.step(action)
    return sim


def test_action_masks_agree_with_the_gate_across_a_synthetic_sequence() -> None:
    """Property check: on every bar of a real-ish path, mask == gate == step."""
    bars = SyntheticProvider(seed=5).fetch(SYMBOL, START, START + timedelta(hours=4))
    features = build_features(bars)
    settings = Settings(
        risk=RiskConfig(
            daily_loss_pct=0.02,
            weekly_loss_pct=0.01,
            max_drawdown_pct=0.01,
            consecutive_loss_pause=2,
            max_spread_points=250.0,
            min_equity_frac=0.9,
            pause_bars=1,
            max_notional_x=10.0,
        )
    )
    sim = Simulator(features, settings, session=SESSION)
    sim.reset()

    actions: list[int] = []
    denied_before: frozenset[int] = frozenset()
    steps = 0
    while not sim._done:
        mask = sim.action_masks()
        reasons = {action: sim._entry_reason(action) for action in (1, 2)}
        for action in (1, 2):
            assert bool(mask[action]) == (reasons[action] is None)

        denied = frozenset(action for action in (1, 2) if not mask[action])
        if denied != denied_before:
            for action in denied:
                probe = _replay(features, settings, actions)
                assert probe._entry_reason(action) == reasons[action]
                result = probe.step(action)
                assert result.info["blocked"] == reasons[action]
                assert result.info["opened"] is None
        denied_before = denied

        action = 1 if mask[1] else (3 if mask[3] else 0)
        result = sim.step(action)
        assert (result.info["blocked"] is None) == bool(mask[action])
        actions.append(action)
        steps += 1
        assert steps < 400
        if result.terminated or result.truncated:
            break

    assert steps >= 100, "the synthetic path must exercise a meaningful number of bars"


# ------------------------------------------------------- public risk API shape


def test_risk_engine_public_api_has_no_override_parameter() -> None:
    expected = {
        "__init__": ["self", "risk", "contract"],
        "size": ["self", "equity", "entry", "sl"],
        "allow_new": ["self", "account", "ts"],
        "allow_new_extended": ["self", "account", "ts"],
        "kill_switch_reason": ["self", "account"],
        "must_force_close": ["self", "bars_in_trade"],
        "sl_tp": ["self", "entry", "side", "atr"],
        "session_anchor": ["self", "ts"],
        "week_anchor": ["self", "ts"],
    }
    for name, parameters in expected.items():
        assert list(inspect.signature(getattr(RiskEngine, name)).parameters) == parameters

    assert list(inspect.signature(Simulator.step).parameters) == ["self", "action"]
    assert list(inspect.signature(Simulator.action_masks).parameters) == ["self"]


def test_risk_config_is_frozen_and_the_engine_binds_it() -> None:
    assert is_dataclass(RiskConfig)
    assert RiskConfig.__dataclass_params__.frozen  # type: ignore[attr-defined]

    config = RiskConfig()
    with pytest.raises(FrozenInstanceError):
        config.risk_pct = 0.5  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        config.max_notional_x = 500.0  # type: ignore[misc]

    settings = Settings()
    with pytest.raises(FrozenInstanceError):
        settings.risk = RiskConfig(risk_pct=0.5)  # type: ignore[misc]

    engine = RiskEngine(config, ContractSpec())
    assert type(engine).risk.fset is None  # type: ignore[attr-defined]
    assert type(engine).contract.fset is None  # type: ignore[attr-defined]
    with pytest.raises(AttributeError):
        engine.risk = RiskConfig(risk_pct=0.5)  # type: ignore[misc]
    with pytest.raises(AttributeError):
        engine.contract = ContractSpec(size=1.0)  # type: ignore[misc]


# ------------------------------------------- orchestrator payloads cannot override


RISK_JOBS = [
    (KIND_KNOWLEDGE_INGEST, {"document_path": "/tmp/does-not-matter.txt"}),
    (KIND_DATASET_BUILD, {"symbol": SYMBOL, "timeframe": TIMEFRAME}),
    (
        KIND_DISCOVERY_RUN,
        {
            "symbol": SYMBOL,
            "timeframe": TIMEFRAME,
            "max_trials": 1,
            "seed": 0,
            "stress": [2.0],
            "gate": {},
        },
    ),
    (KIND_PAPER_ADVANCE, {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 5}),
    (KIND_PAPER_ANALYSE, {}),
]


@pytest.fixture
def orchestrated(tmp_path: Path) -> tuple[Orchestrator, JobStore, HandlerContext]:
    context = HandlerContext(
        data_root=tmp_path / "data",
        runs_root=tmp_path / "runs",
        archive_root=tmp_path / "archive",
        use_feature_cache=False,
    )
    store = JobStore(tmp_path / "jobs.sqlite")
    orchestrator = Orchestrator(store)
    register_default_handlers(orchestrator, context)
    return orchestrator, store, context


def test_paper_advance_request_has_no_risk_field() -> None:
    assert [field.name for field in fields(PaperAdvanceRequest)] == ["symbol", "timeframe", "bars"]
    with pytest.raises(PayloadError):
        PaperAdvanceRequest.from_payload(
            {"symbol": SYMBOL, "timeframe": TIMEFRAME, "bars": 5, "risk": {"risk_pct": 0.5}}
        )


@pytest.mark.parametrize(("kind", "payload"), RISK_JOBS, ids=[job[0] for job in RISK_JOBS])
def test_a_risk_key_in_a_job_payload_is_a_permanent_failure(
    orchestrated: tuple[Orchestrator, JobStore, HandlerContext],
    kind: str,
    payload: dict[str, Any],
) -> None:
    orchestrator, store, context = orchestrated
    body = {**payload, "risk": {"risk_pct": 0.5, "max_notional_x": 500.0}}

    job, _ = store.enqueue(kind, dedupe_key=f"risk-{kind}", payload=body, max_retries=3)
    outcome = orchestrator.run_once()
    assert outcome.status == "failed"

    row = store.get(job.id)
    assert row is not None
    assert row.attempts == 1, "a permanent failure must not spend a retry"
    error = row.error or ""
    assert "invalid payload" in error and "risk" in error

    events = [
        json.loads(event.detail) if event.detail else {}
        for event in store.events(job.id)
        if event.detail
    ]
    assert any(event.get("permanent") for event in events)
    assert not context.paper_ledger_path.exists()
    assert not context.discovery_archive_path.exists()
