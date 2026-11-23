"""Autonomous paper session driven by a DSL :class:`StrategySpec`.

The manual :class:`~scalping.paper.runner.PaperSession` needs a hand-written
``observation -> action`` policy. This module removes that requirement: it turns
a validated strategy spec into the same session, so a spec produced by the
discovery engine can be paper-traded without a human authoring entry and exit
rules.

What this module does
---------------------
* Validates the spec with the DSL validator **before** it reads bars or touches
  the ledger. An invalid spec raises :class:`~scalping.strategy.dsl.nodes.DslValidationError`
  and no session row is written.
* Reads stored bars through :class:`~scalping.data.store.ParquetStore` and builds
  the feature matrix through :class:`~scalping.features.cache.FeatureCache`; it
  does not open a second data path.
* Drives the existing :class:`~scalping.paper.runner.PaperSession` (and therefore
  the existing :class:`~scalping.env.simulator.Simulator` and ``RiskEngine``) one
  decision bar at a time. Every signal, order, fill, trade and equity mark is
  written through :class:`~scalping.paper.ledger.PaperLedger`'s own API.

What this module deliberately does not do
-----------------------------------------
* It has no network client, no credentials and no order-routing path. Real-money
  execution stays disabled and out of scope.
* It never writes ledger rows directly. There is no second writer and no SQL in
  this file.
* It cannot raise a risk limit. The spec never sees ``Settings.risk``; the
  simulator's ``RiskEngine`` owns every entry, stop, target and size. The DSL
  sizing nodes are part of the strategy identity but do not reach the broker.
* It does not decide *which* strategy to run. The intended producer of ``spec``
  is the discovery engine; this module only executes one validated result.

Attribution
-----------
The canonical spec hash is the strategy version. It is written to the session's
``model_version`` and to every signal's ``model_version``. The session config
also stores the canonical payload and ``spec_version``, so the exact strategy is
recoverable from the ledger alone. ``spec_version`` is part of the canonical
payload, so a signal's hash pins the DSL version as well; the explicit integer is
stored once per session because the ``signals`` table has no version column.
``probabilities_json`` stays ``None``: the DSL is deterministic and has no
probability output.

Determinism
-----------
The matrix is a pure function of the stored bars and settings, the policy is a
deterministic interpreter, and the simulator is seeded. Running the same store
slice, spec and seed twice therefore produces identical signals and trades. The
default session id is derived from symbol, timeframe and spec hash, so a restart
resumes the same record instead of opening a duplicate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.features.build import FeatureFrame
from scalping.features.cache import (
    FEATURE_CACHE_DIRNAME,
    FeatureCache,
    cached_build_features,
)
from scalping.paper.ledger import PaperLedger, PositionRecord, ResumeState
from scalping.paper.runner import PaperRunResult, PaperSession
from scalping.strategy.dsl import Decision, Policy, StrategySpec, compile_policy, validate
from scalping.types import Side

__all__ = ["AutonomousPaperRunner", "AutonomousSummary"]

#: Store root used when the caller supplies neither ``store`` nor ``data_root``.
#: It matches the CLI/API default (``data``), so all three read the same bars.
_DEFAULT_DATA_ROOT: Final[Path] = Path("data")

#: DSL action -> simulator action code. ``NO_TRADE`` suppresses entries and maps
#: to HOLD, exactly as in :mod:`scalping.discovery.evaluate`.
_ACTION_CODES: Final[dict[str, int]] = {
    "HOLD": 0,
    "NO_TRADE": 0,
    "BUY": 1,
    "SELL": 2,
    "CLOSE": 3,
}


@dataclass(frozen=True, slots=True)
class AutonomousSummary:
    """Auditable state of one autonomous paper session."""

    session_id: str
    spec_hash: str
    status: str
    bars_processed: int
    trades: int
    realized_pnl: float
    balance: float | None
    equity: float | None
    position: PositionRecord | None
    last_bar_ts: datetime | None


class _SpecPolicy:
    """Adapt a compiled DSL policy to ``PaperSession``'s observation callable.

    ``PaperSession`` hands the policy the latest observation only. The DSL needs
    the current feature row and the position state instead, so this adapter reads
    both from the attached session's simulator and keeps the interpreter's
    history. It is the same state the discovery evaluator maintains; seeding the
    history from rows *before* the decision bar keeps lookbacks causal and makes
    a resumed session independent of whether replay invoked the policy.
    """

    def __init__(self, spec: StrategySpec, policy: Policy, features: FeatureFrame) -> None:
        self._spec = spec
        self._policy = policy
        self._columns = tuple(features.columns)
        self._x = features.X
        self._state = policy.new_state()
        self._cursor = -1
        self._last: Decision | None = None
        self._session: PaperSession | None = None

    def attach(self, session: PaperSession) -> None:
        """Bind the session whose simulator supplies the row and position."""
        self._session = session

    def __call__(self, obs: np.ndarray) -> int:
        del obs  # the DSL reads named features, not the flattened observation
        session = self._session
        if session is None:  # pragma: no cover - attach happens before run
            raise RuntimeError("DSL policy is not attached to a paper session")
        index = session.sim.bar_index
        if index != self._cursor + 1:
            self._seed(index)
        account = session.sim.broker.state(session.sim.balance, session.sim.day_start_equity)
        position = account.position
        self._state.position = (
            None if position is None else ("long" if position.side is Side.LONG else "short")
        )
        self._state.bars_held = int(account.bars_in_trade)
        row = self._row(index)
        decision = self._policy.decide(row, self._state)
        self._state.push(row)
        self._cursor = index
        self._last = decision
        return _ACTION_CODES[decision.action]

    def signal_reason(self, action: int) -> str:
        """The interpreter's reason for the action just taken."""
        del action
        decision = self._last
        return "unknown" if decision is None else decision.reason

    def _row(self, index: int) -> dict[str, float]:
        return dict(zip(self._columns, self._x[index].tolist(), strict=True))

    def _seed(self, index: int) -> None:
        """Rebuild history from rows strictly before ``index``.

        Called on the first decision and after a replay jumps the cursor. Only
        past rows are read, so a resume cannot introduce look-ahead.
        """
        state = self._policy.new_state()
        start = max(0, index - self._spec.max_lookback)
        for offset in range(start, index):
            state.push(self._row(offset))
        self._state = state
        self._cursor = index - 1


class AutonomousPaperRunner:
    """Paper-trade one validated DSL spec over stored bars.

    ``ledger`` is the only persistence; ``spec`` is validated before the store is
    read. ``store`` (or ``data_root``) selects the bar source. ``cache`` enables
    the on-disk feature cache under ``<store root>/.feature-cache``; pass a
    :class:`FeatureCache` to share a cache root, or ``False`` to disable it.

    The constructor creates no session row. :meth:`resume` and :meth:`run` do,
    after the spec and the frame are known to be usable.
    """

    def __init__(
        self,
        ledger: PaperLedger,
        spec: StrategySpec,
        symbol: str,
        timeframe: str,
        settings: Settings,
        *,
        window: int = 60,
        session_id: str | None = None,
        cache: bool | FeatureCache = True,
        store: ParquetStore | None = None,
        data_root: str | Path | None = None,
        start: datetime | None = None,
        end: datetime | None = None,
        seed: int = 0,
        started_at: datetime | None = None,
        max_steps: int | None = None,
    ) -> None:
        # First gate: no state may exist if the strategy is not executable.
        validate(spec)
        if not isinstance(ledger, PaperLedger):
            raise TypeError(f"ledger must be a PaperLedger, got {type(ledger).__name__}")
        if not isinstance(settings, Settings):
            raise TypeError(f"settings must be a Settings, got {type(settings).__name__}")
        if not isinstance(symbol, str) or not symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        if not isinstance(timeframe, str) or not timeframe.strip():
            raise ValueError("timeframe must be a non-empty string")
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise ValueError("session_id must be a non-empty string or None")
        if store is not None and data_root is not None:
            raise ValueError("pass either store or data_root, not both")

        self._ledger = ledger
        self._spec = spec
        self._spec_hash = spec.canonical_hash()
        self._symbol = symbol
        self._timeframe = timeframe
        self._settings = settings
        self._session_id = (
            session_id
            if session_id is not None
            else f"auto-{symbol}-{timeframe}-{self._spec_hash[:16]}"
        )
        self._store = store if store is not None else ParquetStore(data_root or _DEFAULT_DATA_ROOT)
        self._frame = self._build_frame(start, end, cache)
        self._policy = _SpecPolicy(spec, compile_policy(spec), self._frame)
        self._session = PaperSession(
            ledger,
            self._frame,
            settings,
            self._policy,
            session_id=self._session_id,
            symbol=symbol,
            timeframe=timeframe,
            strategy="dsl",
            model_version=self._spec_hash,
            config={
                "origin": "autonomous",
                "dsl": {
                    "spec_version": spec.spec_version,
                    "spec_hash": self._spec_hash,
                    "spec": spec.canonical_dict(),
                },
            },
            started_at=started_at,
            seed=seed,
            max_steps=max_steps,
            window=window,
            signal_reason=self._policy.signal_reason,
        )
        self._policy.attach(self._session)

    # ---------------------------------------------------------------- properties

    @property
    def session_id(self) -> str:
        """Ledger id of this run; derived from symbol, timeframe and spec hash."""
        return self._session_id

    @property
    def spec_hash(self) -> str:
        """Canonical sha256 of the strategy, the recorded strategy version."""
        return self._spec_hash

    @property
    def spec(self) -> StrategySpec:
        """The validated strategy this runner executes."""
        return self._spec

    @property
    def features(self) -> FeatureFrame:
        """The feature frame loaded once for this runner."""
        return self._frame

    @property
    def session(self) -> PaperSession:
        """The underlying session; it still owns every simulator call."""
        return self._session

    # ------------------------------------------------------------------ lifecycle

    def resume(self) -> ResumeState:
        """Create or attach to the ledger session and reconcile the simulator."""
        return self._session.resume()

    def run(self, bars: int | None = None, max_bars: int | None = None) -> PaperRunResult:
        """Advance decision bars and persist each one.

        ``bars`` caps this call. ``max_bars`` caps the session's total number of
        decisions, counted from the ledger, so a scheduler can enforce a horizon
        across restarts instead of per call. With neither, every remaining stored
        bar is processed. The call stops early when the episode ends.
        """
        if bars is not None and (isinstance(bars, bool) or not isinstance(bars, int)):
            raise TypeError(f"bars must be an int or None, got {type(bars).__name__}")
        if bars is not None and bars < 0:
            raise ValueError(f"bars must be >= 0, got {bars!r}")
        if max_bars is not None and (isinstance(max_bars, bool) or not isinstance(max_bars, int)):
            raise TypeError(f"max_bars must be an int or None, got {type(max_bars).__name__}")
        if max_bars is not None and max_bars < 0:
            raise ValueError(f"max_bars must be >= 0, got {max_bars!r}")

        self._session.resume()
        limit = bars
        if max_bars is not None:
            processed = len(self._ledger.signals(self._session_id))
            remaining = max(0, max_bars - processed)
            limit = remaining if limit is None else min(limit, remaining)
        if limit is None:
            limit = max(0, len(self._frame.ts) - self._session.sim.bar_index)
        return self._session.run(limit)

    def close(self) -> None:
        """Close the ledger session; further runs are refused."""
        self._session.close()

    # -------------------------------------------------------------------- summary

    def summary(self) -> AutonomousSummary:
        """Read the session's durable state; never steps the simulator."""
        record = self._ledger.session(self._session_id)
        if record is None:
            return AutonomousSummary(
                session_id=self._session_id,
                spec_hash=self._spec_hash,
                status="new",
                bars_processed=0,
                trades=0,
                realized_pnl=0.0,
                balance=None,
                equity=None,
                position=None,
                last_bar_ts=None,
            )
        state = self._ledger.resume(self._session_id)
        trades = self._ledger.trades(self._session_id)
        return AutonomousSummary(
            session_id=self._session_id,
            spec_hash=self._spec_hash,
            status=record.status,
            bars_processed=len(self._ledger.signals(self._session_id)),
            trades=len(trades),
            realized_pnl=float(sum(trade.pnl for trade in trades)),
            balance=state.balance,
            equity=state.equity,
            position=state.position,
            last_bar_ts=state.last_bar_ts,
        )

    # -------------------------------------------------------------------- internal

    def _build_frame(
        self,
        start: datetime | None,
        end: datetime | None,
        cache: bool | FeatureCache,
    ) -> FeatureFrame:
        """Read the stored slice and build/cached features for it."""
        bars = self._store.read(self._symbol, self._timeframe, start, end)
        return cached_build_features(
            self._cache(cache),
            symbol=self._symbol,
            timeframe=self._timeframe,
            frame=bars,
            settings=self._settings,
        )

    def _cache(self, cache: bool | FeatureCache) -> FeatureCache | None:
        if isinstance(cache, FeatureCache):
            return cache
        if not isinstance(cache, bool):
            raise TypeError(f"cache must be a bool or FeatureCache, got {type(cache).__name__}")
        if not cache:
            return None
        return FeatureCache(Path(self._store.root) / FEATURE_CACHE_DIRNAME)
