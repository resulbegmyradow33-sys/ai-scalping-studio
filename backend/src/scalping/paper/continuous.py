"""Continuous paper operation and matured-outcome analysis for one DSL spec.

The autonomous runner (:mod:`scalping.paper.autonomous`) processes the stored
slice it was constructed with. This module keeps that strategy contract but
makes the run *incremental*: call :meth:`ContinuousPaperRunner.advance` again
after new bars were stored, and only bars newer than the ledger's last recorded
decision are processed. The session is resumed through the existing
:class:`~scalping.paper.runner.PaperSession` replay path, so an already recorded
bar can never produce a second signal or trade.

Design decisions
----------------
* **The ledger is the cursor.** The last processed timestamp is the newest
  signal in the session, not an in-memory counter. A restart therefore resumes
  exactly where the previous process stopped, and the cursor survives a crash.
* **One provisional bar.** The newest stored bar is priced but never used as a
  decision bar. The simulator needs a following bar to evaluate stops and
  targets, and treating the current tail as a decision bar would book a
  ``data_end`` close and make a later append unreplayable. Appending ``N`` bars
  therefore advances exactly ``N`` decisions (the previous tail becomes
  decidable, the new tail stays provisional).
* **Typed refusal instead of silent skipping.** Out-of-order or duplicate bars
  raise :class:`BarOrderError`; a missing bar in the pending range raises
  :class:`BarGapError`. A jump is allowed only when the missing instants fall
  inside the registry's closed session window (a weekend or a daily close).
  Nothing is written before those checks pass.
* **Idempotency by bar timestamp.** The runner never writes the ledger itself;
  :class:`PaperSession` replays the recorded actions and only then appends. The
  stored signal timestamps are checked against the feature timeline, so a
  rewritten history fails loudly instead of shifting decisions onto other bars.

What this module deliberately does not do
-----------------------------------------
* No network client, no broker, no credentials and no real-money path.
* No scheduler or daemon. ``advance()`` is caller-driven; a long-running
  process is future work.
* No champion selection. The producer of ``spec`` is the discovery/champion
  layer; this module only executes one validated strategy.
* No legacy strategy fallback. A missing or invalid spec is a typed error, never
  a silent switch to a hand-written rule.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl

from scalping.config import Settings
from scalping.data.provenance import load_provenance
from scalping.data.registry import SessionSpec, session_for, timeframe_rule
from scalping.data.store import ParquetStore
from scalping.features.build import FeatureFrame
from scalping.features.cache import (
    FEATURE_CACHE_DIRNAME,
    FeatureCache,
    cached_build_features,
    frame_meta,
)
from scalping.paper.autonomous import _SpecPolicy
from scalping.paper.ledger import (
    PaperLedger,
    SessionRecord,
    SignalRecord,
    TradeRecord,
    provenance_config,
    session_provenance,
)
from scalping.paper.runner import PaperSession
from scalping.strategy.dsl import StrategySpec, compile_policy, validate

__all__ = [
    "AdvanceSummary",
    "BarError",
    "BarGapError",
    "BarOrderError",
    "ContinuousError",
    "ContinuousPaperRunner",
    "OutcomeBatch",
    "OutcomeError",
    "OutcomeRow",
    "SessionMismatchError",
    "matured_outcomes",
    "outcome_fingerprint",
]

#: Bar-size grammar for the registry's resampling rules (``1m``/``4h``/``1d``).
_RESAMPLE_RULE: Final[re.Pattern[str]] = re.compile(r"^(?P<count>\d+)(?P<unit>[mhd])$")
_UNIT_SECONDS: Final[dict[str, int]] = {"m": 60, "h": 3600, "d": 86400}


class ContinuousError(Exception):
    """A continuous paper operation was refused; the ledger is unchanged."""


class BarError(ContinuousError):
    """The stored bar sequence is not usable for continuous operation."""


class BarOrderError(BarError):
    """Bars are not strictly increasing or contain a duplicate timestamp."""


class BarGapError(BarError):
    """A bar is missing (or repeated) at the timeframe cadence."""


class SessionMismatchError(ContinuousError):
    """The ledger session does not match this spec, market or configuration."""


class OutcomeError(ContinuousError):
    """The session's matured outcomes cannot be attributed unambiguously."""


@dataclass(frozen=True, slots=True)
class AdvanceSummary:
    """What one :meth:`ContinuousPaperRunner.advance` call changed.

    ``new_bars``, ``new_signals`` and ``new_trades`` are the deltas written by
    this call. ``bars_processed``, ``trades`` and ``last_bar_ts`` are the
    durable totals read back from the ledger. ``status`` is ``"new"`` before the
    first session row exists.
    """

    session_id: str
    spec_hash: str
    scope: str
    status: str
    new_bars: int
    new_signals: int
    new_trades: int
    bars_processed: int
    trades: int
    last_bar_ts: datetime | None
    done: bool
    reason: str | None


class ContinuousPaperRunner:
    """Paper-trade one validated DSL spec over a store that keeps growing.

    ``store_root`` is a :class:`~scalping.data.store.ParquetStore` root. The
    constructor validates the spec and the arguments, but reads no bar and
    writes no session row; :meth:`advance` does both once it knows there is a
    decidable bar. ``cache`` is ``True`` (the default on-disk cache under
    ``<store_root>/.feature-cache``), ``False``, or a shared ``FeatureCache``.

    ``scope`` is the champion scope recorded for audit; it defaults to
    ``"<symbol>:<timeframe>"`` and does not change the session id. The id is
    derived from symbol, timeframe and the canonical spec hash, so restarting a
    process reopens the same record.
    """

    def __init__(
        self,
        ledger: PaperLedger,
        spec: StrategySpec,
        symbol: str,
        timeframe: str,
        settings: Settings,
        *,
        store_root: str | Path,
        window: int = 60,
        cache: bool | FeatureCache = True,
        scope: str | None = None,
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
        if not isinstance(store_root, (str, Path)):
            raise TypeError(f"store_root must be a str or Path, got {type(store_root).__name__}")
        if isinstance(store_root, str) and not store_root.strip():
            raise ValueError("store_root must not be empty")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        if isinstance(cache, FeatureCache):
            cache_value: bool | FeatureCache = cache
        elif isinstance(cache, bool):
            cache_value = cache
        else:
            raise TypeError(f"cache must be a bool or FeatureCache, got {type(cache).__name__}")
        if scope is None:
            scope_value = f"{symbol}:{timeframe}"
        elif isinstance(scope, str) and scope.strip():
            scope_value = scope
        else:
            raise ValueError("scope must be a non-empty string or None")

        self._ledger = ledger
        self._spec = spec
        self._spec_hash = spec.canonical_hash()
        self._symbol = symbol
        self._timeframe = timeframe
        self._settings = settings
        self._window = window
        self._cache = cache_value
        self._scope = scope_value
        self._store = ParquetStore(store_root)
        self._session_id = f"auto-{symbol}-{timeframe}-{self._spec_hash[:16]}"
        self._step = _step_delta(timeframe)
        self._session_spec = _session_spec(symbol)

    # ---------------------------------------------------------------- properties

    @property
    def session_id(self) -> str:
        """Ledger id of this run; derived from symbol, timeframe and spec hash."""
        return self._session_id

    @property
    def spec_hash(self) -> str:
        """Canonical sha256 of the strategy, recorded on every signal."""
        return self._spec_hash

    @property
    def spec(self) -> StrategySpec:
        """The validated strategy this runner executes."""
        return self._spec

    @property
    def scope(self) -> str:
        """Champion scope recorded for audit."""
        return self._scope

    @property
    def window(self) -> int:
        """Observation window the session was created with."""
        return self._window

    # ------------------------------------------------------------------ advance

    def advance(self) -> AdvanceSummary:
        """Process every newly closed bar and return what changed.

        The call is idempotent by bar timestamp: a second call without new bars
        writes nothing. A store with an out-of-order, duplicate or missing bar
        is refused before the session is touched.
        """
        bars = self._store.read(self._symbol, self._timeframe)
        _validate_ordering(bars, self._session_id)

        record = self._ledger.session(self._session_id)
        if record is not None:
            self._check_session(record)
        signals = self._ledger.signals(self._session_id) if record is not None else []
        last_ts = _last_signal_ts(signals)

        start = _pending_start(bars, last_ts)
        if start is None:
            return self._summary(
                record,
                signals,
                new_bars=0,
                new_signals=0,
                new_trades=0,
                done=False,
                reason="no_new_bars",
            )
        self._validate_cadence(bars, start, last_ts)

        frame = self._build_frame(bars)
        first_index = max(frame.valid_from + self._window - 1, self._window - 1)
        self._verify_signal_timestamps(frame, signals, first_index)
        if first_index + len(signals) > len(frame.ts) - 2:
            return self._summary(
                record,
                signals,
                new_bars=0,
                new_signals=0,
                new_trades=0,
                done=False,
                reason="warming_up" if not signals else "awaiting_closed_bar",
            )

        data = None if record is not None else self._data_block(bars)
        session = self._session(frame, data)
        session.resume()
        expected_index = first_index + len(signals)
        if session.sim.bar_index != expected_index:
            raise SessionMismatchError(
                f"session {self._session_id!r}: replayed simulator is at bar"
                f" {session.sim.bar_index}, expected {expected_index}; the stored"
                " bars or the session history changed"
            )
        limit = len(frame.ts) - 1 - session.sim.bar_index
        if limit <= 0:  # pragma: no cover - the first_index guard already refuses
            return self._summary(
                record,
                signals,
                new_bars=0,
                new_signals=0,
                new_trades=0,
                done=False,
                reason="no_closed_bar",
            )

        before_trades = self._ledger.count_trades(self._session_id)
        result = session.run(limit)
        if result.steps == 0 and result.done:
            # A previous call ended the episode; continuing would silently
            # replay a finished account on every later call.
            raise ContinuousError(
                f"session {self._session_id!r} is finished ({result.reason});"
                " close it and start a new session instead"
            )

        after_signals = self._ledger.signals(self._session_id)
        after_trades = self._ledger.count_trades(self._session_id)
        new_signals = len(after_signals) - len(signals)
        if new_signals != result.steps:  # pragma: no cover - one signal per step
            raise ContinuousError(
                f"session {self._session_id!r}: {result.steps} steps wrote {new_signals} signals"
            )
        return AdvanceSummary(
            session_id=self._session_id,
            spec_hash=self._spec_hash,
            scope=self._scope,
            status=self._status(),
            new_bars=result.steps,
            new_signals=new_signals,
            new_trades=after_trades - before_trades,
            bars_processed=len(after_signals),
            trades=after_trades,
            last_bar_ts=_last_signal_ts(after_signals),
            done=result.done,
            reason=result.reason,
        )

    # ------------------------------------------------------------------ internal

    def _build_frame(self, bars: pl.DataFrame) -> FeatureFrame:
        """Build or load the feature matrix for the whole stored slice."""
        cache: FeatureCache | None
        if isinstance(self._cache, FeatureCache):
            cache = self._cache
        elif self._cache:
            cache = FeatureCache(Path(self._store.root) / FEATURE_CACHE_DIRNAME)
        else:
            cache = None
        return cached_build_features(
            cache,
            symbol=self._symbol,
            timeframe=self._timeframe,
            frame=bars,
            settings=self._settings,
        )

    def _session(self, frame: FeatureFrame, data: dict[str, Any] | None) -> PaperSession:
        """Create the session object; the ledger row is written by ``resume``."""
        policy = _SpecPolicy(self._spec, compile_policy(self._spec), frame)
        config: dict[str, Any] = {
            "origin": "continuous",
            "scope": self._scope,
            "dsl": {
                "spec_version": self._spec.spec_version,
                "spec_hash": self._spec_hash,
                "spec": self._spec.canonical_dict(),
            },
        }
        provenance = self._dataset_provenance()
        if provenance != (None, None):
            config["provenance"] = provenance_config(provenance[0], provenance[1])
        if data is not None:
            config["data"] = data
        session = PaperSession(
            self._ledger,
            frame,
            self._settings,
            policy,
            session_id=self._session_id,
            symbol=self._symbol,
            timeframe=self._timeframe,
            strategy="dsl",
            model_version=self._spec_hash,
            config=config,
            seed=0,
            max_steps=None,
            window=self._window,
            signal_reason=policy.signal_reason,
        )
        policy.attach(session)
        return session

    def _data_block(self, bars: pl.DataFrame) -> dict[str, Any]:
        """Session-level data identity, stored once when the session is created.

        The fingerprint covers every stored bar at creation time. It is a
        session identity, not a per-trade hash: the ``signals`` table has no
        data column, so a per-bar data hash would need a schema migration.
        """
        meta = frame_meta(bars)
        return {
            "dataset_id": f"{self._symbol}:{self._timeframe}:{meta.fingerprint[:16]}",
            "rows": meta.rows,
            "first_ts": meta.first_ts,
            "last_ts": meta.last_ts,
        }

    def _stored_provenance(self, config: Mapping[str, Any]) -> tuple[str | None, bool | None]:
        """Read a recorded provenance pair; a corrupt block is a typed refusal."""
        try:
            return session_provenance(config)
        except (TypeError, ValueError) as exc:
            raise SessionMismatchError(f"session {self._session_id!r}: {exc}") from exc

    def _dataset_provenance(self) -> tuple[str | None, bool | None]:
        """The store's recorded data provenance, or ``(None, None)`` when absent.

        The sidecar is written by the ingest path (``scalping.data.provenance``)
        and names the provider and the synthetic flag. An unreadable sidecar is
        a refusal, not unknown: silently trading on a dataset whose identity
        cannot be read would erase the one fact this record exists to keep.
        """
        try:
            recorded = load_provenance(self._store.root, self._symbol, self._timeframe)
        except (OSError, TypeError, ValueError) as exc:
            raise ContinuousError(
                f"session {self._session_id!r}: unreadable provenance sidecar for"
                f" {self._symbol}/{self._timeframe}: {exc}"
            ) from exc
        if recorded is None:
            return None, None
        return recorded.source, recorded.synthetic

    def _check_session(self, record: SessionRecord) -> None:
        """Refuse to continue a session that this runner cannot reproduce."""
        if record.symbol != self._symbol or record.timeframe != self._timeframe:
            raise SessionMismatchError(
                f"session {self._session_id!r} is {record.symbol}/{record.timeframe},"
                f" not {self._symbol}/{self._timeframe}"
            )
        if record.strategy != "dsl":
            raise SessionMismatchError(
                f"session {self._session_id!r} uses strategy {record.strategy!r}, not 'dsl'"
            )
        if record.model_version != self._spec_hash:
            raise SessionMismatchError(
                f"session {self._session_id!r} records strategy {record.model_version!r},"
                f" not {self._spec_hash!r}"
            )
        if record.status != "running":
            raise SessionMismatchError(
                f"session {self._session_id!r} is {record.status}; a closed session cannot advance"
            )
        config = record.config
        if config.get("window") != self._window:
            raise SessionMismatchError(
                f"session {self._session_id!r} was created with window"
                f" {config.get('window')!r}, not {self._window!r}"
            )
        if config.get("seed") not in (None, 0):
            raise SessionMismatchError(
                f"session {self._session_id!r} was created with seed {config.get('seed')!r}"
            )
        for name in ("contract", "risk", "costs"):
            if config.get(name) != asdict(getattr(self._settings, name)):
                raise SessionMismatchError(
                    f"session {self._session_id!r} was created with different {name} settings"
                )
        stored_source, stored_synthetic = self._stored_provenance(config)
        current_source, current_synthetic = self._dataset_provenance()
        # Unknown on either side is not a conflict: a session recorded before
        # provenance existed must still resume. A known disagreement is refused,
        # because the evidence identity behind the trades has changed.
        if (
            stored_source is not None
            and current_source is not None
            and stored_source != current_source
        ):
            raise SessionMismatchError(
                f"session {self._session_id!r} records data source {stored_source!r},"
                f" but the store is now {current_source!r}"
            )
        if (
            stored_synthetic is not None
            and current_synthetic is not None
            and stored_synthetic != current_synthetic
        ):
            raise SessionMismatchError(
                f"session {self._session_id!r} records synthetic={stored_synthetic!r},"
                f" but the store is now synthetic={current_synthetic!r}"
            )

    def _validate_cadence(
        self,
        bars: pl.DataFrame,
        start: int,
        last_ts: datetime | None,
    ) -> None:
        """Require one bar per timeframe slot in the range about to be processed.

        History before the ledger cursor is not re-checked for cadence: those
        bars are already reflected in recorded decisions. Ordering is checked
        everywhere by :func:`_validate_ordering`.
        """
        delta = self._step
        if delta is None:
            return
        stamps: list[datetime] = bars.get_column("ts").to_list()
        if last_ts is not None:
            expected = last_ts + delta
            first = stamps[start]
            if first < expected:
                raise BarGapError(
                    f"session {self._session_id!r}: stored bar {first.isoformat()} is"
                    f" before the expected next slot {expected.isoformat()}"
                )
            if first > expected and not self._is_closed(expected):
                raise BarGapError(
                    f"session {self._session_id!r}: missing bar at {expected.isoformat()}"
                    f" before {first.isoformat()}"
                )
        for index in range(max(start, 1), len(stamps)):
            expected = stamps[index - 1] + delta
            actual = stamps[index]
            if actual < expected:
                raise BarGapError(
                    f"session {self._session_id!r}: bar {actual.isoformat()} is off the"
                    f" {self._timeframe} grid after {stamps[index - 1].isoformat()}"
                )
            if actual > expected and not self._is_closed(expected):
                raise BarGapError(
                    f"session {self._session_id!r}: missing bar at {expected.isoformat()}"
                    f" before {actual.isoformat()}"
                )

    def _is_closed(self, stamp: datetime) -> bool:
        """A jump is legitimate only across a closed session window."""
        spec = self._session_spec
        return spec is not None and not spec.is_open(stamp)

    def _verify_signal_timestamps(
        self,
        frame: FeatureFrame,
        signals: list[SignalRecord],
        first_index: int,
    ) -> None:
        """Replayed signals must land on the stored bars they were recorded on."""
        if not signals:
            return
        if first_index + len(signals) > len(frame.ts):
            raise SessionMismatchError(
                f"session {self._session_id!r}: {len(signals)} recorded signals do"
                f" not fit in {len(frame.ts)} stored bars"
            )
        for offset, signal in enumerate(signals):
            expected = _frame_ts(frame, first_index + offset)
            if expected != signal.ts:
                raise SessionMismatchError(
                    f"session {self._session_id!r}: recorded signal {offset} is at"
                    f" {signal.ts.isoformat()}, but the store has {expected.isoformat()}"
                    " at that decision slot; the stored bars changed"
                )

    def _status(self) -> str:
        record = self._ledger.session(self._session_id)
        return "new" if record is None else record.status

    def _summary(
        self,
        record: SessionRecord | None,
        signals: list[SignalRecord],
        *,
        new_bars: int,
        new_signals: int,
        new_trades: int,
        done: bool,
        reason: str | None,
    ) -> AdvanceSummary:
        trades = self._ledger.count_trades(self._session_id) if record is not None else 0
        return AdvanceSummary(
            session_id=self._session_id,
            spec_hash=self._spec_hash,
            scope=self._scope,
            status="new" if record is None else record.status,
            new_bars=new_bars,
            new_signals=new_signals,
            new_trades=new_trades,
            bars_processed=len(signals),
            trades=trades,
            last_bar_ts=_last_signal_ts(signals),
            done=done,
            reason=reason,
        )


# --------------------------------------------------------------------- outcomes


@dataclass(frozen=True, slots=True)
class OutcomeRow:
    """One matured (closed) trade and its attribution.

    ``source`` and ``synthetic`` repeat the session's data provenance on every
    row so a row exported on its own still says which evidence produced it.
    ``None`` is unknown; it is never a stand-in for ``False``.
    """

    trade_id: int
    entry_ts: datetime
    exit_ts: datetime
    side: str
    entry_price: float
    exit_price: float
    lot: float
    pnl: float
    r_multiple: float
    exit_reason: str
    strategy_hash: str | None
    dataset_id: str | None
    source: str | None = None
    synthetic: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection; timestamps become ISO-8601 strings."""
        return {
            "trade_id": self.trade_id,
            "entry_ts": self.entry_ts.isoformat(),
            "exit_ts": self.exit_ts.isoformat(),
            "side": self.side,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "lot": self.lot,
            "pnl": self.pnl,
            "r_multiple": self.r_multiple,
            "exit_reason": self.exit_reason,
            "strategy_hash": self.strategy_hash,
            "dataset_id": self.dataset_id,
            "source": self.source,
            "synthetic": self.synthetic,
        }


@dataclass(frozen=True, slots=True)
class OutcomeBatch:
    """An immutable snapshot of the session's matured outcomes.

    Only closed trades are rows. ``open_trades`` counts the session's open
    position rows, which are excluded because their P&L is not final.
    ``source``/``synthetic`` are the session's data provenance; unknown is
    ``None`` so a consumer can refuse "not known to be real" without guessing.
    """

    session_id: str
    strategy_hash: str | None
    dataset_id: str | None
    rows: tuple[OutcomeRow, ...]
    open_trades: int = 0
    source: str | None = None
    synthetic: bool | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.rows, tuple):
            raise TypeError("OutcomeBatch.rows must be a tuple")
        if any(not isinstance(row, OutcomeRow) for row in self.rows):
            raise TypeError("OutcomeBatch.rows must contain OutcomeRow values")
        if isinstance(self.open_trades, bool) or not isinstance(self.open_trades, int):
            raise TypeError("OutcomeBatch.open_trades must be an int")
        if self.open_trades < 0:
            raise ValueError("OutcomeBatch.open_trades must be >= 0")
        if self.source is not None and (
            not isinstance(self.source, str) or not self.source.strip()
        ):
            raise ValueError("OutcomeBatch.source must be non-empty text or None")
        if self.synthetic is not None and not isinstance(self.synthetic, bool):
            raise ValueError("OutcomeBatch.synthetic must be a bool or None")
        for row in self.rows:
            if row.source != self.source or row.synthetic != self.synthetic:
                raise ValueError(
                    f"outcome {row.trade_id} carries different provenance than its batch"
                )

    @property
    def closed_trades(self) -> int:
        """How many matured trades the batch carries."""
        return len(self.rows)

    def to_dict(self) -> dict[str, Any]:
        """JSON-safe projection of the whole batch."""
        return {
            "session_id": self.session_id,
            "strategy_hash": self.strategy_hash,
            "dataset_id": self.dataset_id,
            "closed_trades": len(self.rows),
            "open_trades": self.open_trades,
            "source": self.source,
            "synthetic": self.synthetic,
            "outcomes": [row.to_dict() for row in self.rows],
        }

    def to_json(self) -> str:
        """Canonical JSON; safe for hashing and cross-process comparison."""
        return json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )


def matured_outcomes(ledger: PaperLedger, session_id: str) -> OutcomeBatch:
    """Read the session's closed trades; exclude and count open positions.

    The strategy hash is the session's ``model_version`` (the canonical spec
    hash), cross-checked against every non-null signal version. The dataset id
    is read from the session config when a producer stored one; it is ``None``
    for sessions that predate that field. ``source``/``synthetic`` come from the
    session config through :func:`session_provenance` and are ``None`` when the
    creator recorded no provider metadata: unknown is not real evidence.
    """
    if not isinstance(ledger, PaperLedger):
        raise TypeError(f"ledger must be a PaperLedger, got {type(ledger).__name__}")
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("session_id must be a non-empty string")
    record = ledger.session(session_id)
    if record is None:
        raise OutcomeError(f"unknown session {session_id!r}")
    try:
        source, synthetic = session_provenance(record.config)
    except (TypeError, ValueError) as exc:
        raise OutcomeError(f"session {session_id!r}: {exc}") from exc

    signals = ledger.signals(session_id)
    versions = {signal.model_version for signal in signals if signal.model_version}
    if len(versions) > 1:
        raise OutcomeError(
            f"session {session_id!r} has signals from {len(versions)} strategy hashes"
        )
    signal_hash = next(iter(versions), None)
    if (
        record.model_version is not None
        and signal_hash is not None
        and record.model_version != signal_hash
    ):
        raise OutcomeError(
            f"session {session_id!r} records {record.model_version!r} but its signals"
            f" carry {signal_hash!r}"
        )
    strategy_hash = record.model_version or signal_hash
    dataset_id = _dataset_id(record)

    trades = ledger.trades(session_id)
    open_trades = sum(1 for position in ledger.positions(session_id) if position.status == "open")
    rows = tuple(
        _outcome_row(trade, strategy_hash, dataset_id, source, synthetic) for trade in trades
    )
    return OutcomeBatch(
        session_id=session_id,
        strategy_hash=strategy_hash,
        dataset_id=dataset_id,
        rows=rows,
        open_trades=open_trades,
        source=source,
        synthetic=synthetic,
    )


def outcome_fingerprint(batch: OutcomeBatch) -> str:
    """Canonical sha256 of the matured outcomes, for research-trigger dedupe.

    The payload excludes ``trade_id`` and ``open_trades``: a re-created ledger
    can renumber rows, and an open position is not a matured outcome. Rows are
    sorted by their own fields, so the fingerprint is independent of read order.
    ``source`` and ``synthetic`` are part of the payload: the same trades claimed
    against different evidence are different outcomes. Two identical batches
    always hash identically; a newly closed trade changes the hash.
    """
    if not isinstance(batch, OutcomeBatch):
        raise TypeError(f"batch must be an OutcomeBatch, got {type(batch).__name__}")
    payload = {
        "session_id": batch.session_id,
        "strategy_hash": batch.strategy_hash,
        "dataset_id": batch.dataset_id,
        "source": batch.source,
        "synthetic": batch.synthetic,
        "outcomes": [
            {key: value for key, value in row.to_dict().items() if key != "trade_id"}
            for row in sorted(batch.rows, key=_outcome_key)
        ],
    }
    canonical = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# -------------------------------------------------------------------- helpers


def _step_delta(timeframe: str) -> timedelta | None:
    """Nominal bar spacing for a registered timeframe; ``None`` when unknown."""
    try:
        rule = timeframe_rule(timeframe)
    except ValueError:
        return None
    match = _RESAMPLE_RULE.fullmatch(rule)
    if match is None:
        return None
    return timedelta(seconds=int(match.group("count")) * _UNIT_SECONDS[match.group("unit")])


def _session_spec(symbol: str) -> SessionSpec | None:
    try:
        return session_for(symbol)
    except ValueError:
        return None


def _validate_ordering(bars: pl.DataFrame, session_id: str) -> None:
    """Strictly increasing, duplicate-free timestamps across the whole store."""
    if bars.is_empty():
        return
    stamps = bars.get_column("ts")
    if not stamps.is_sorted():
        raise BarOrderError(f"session {session_id!r}: stored bars are not sorted by ts")
    if stamps.n_unique() != bars.height:
        raise BarOrderError(f"session {session_id!r}: stored bars contain a duplicate ts")


def _pending_start(bars: pl.DataFrame, last_ts: datetime | None) -> int | None:
    """Index of the first stored bar strictly after ``last_ts``, or ``None``."""
    if bars.is_empty():
        return None
    if last_ts is None:
        return 0
    stamps: list[datetime] = bars.get_column("ts").to_list()
    index = bisect.bisect_right(stamps, last_ts)
    return None if index >= len(stamps) else index


def _last_signal_ts(signals: list[SignalRecord]) -> datetime | None:
    if not signals:
        return None
    return max(signal.ts for signal in signals)


def _frame_ts(frame: FeatureFrame, index: int) -> datetime:
    stamp = np.datetime64(frame.ts[index], "us").astype(datetime)
    return stamp.replace(tzinfo=UTC)


def _dataset_id(record: SessionRecord) -> str | None:
    """Read a stored dataset id from the session config, wherever it is nested."""
    config = record.config
    candidates: list[object] = [config.get("dataset_id")]
    extra = config.get("extra")
    if isinstance(extra, Mapping):
        candidates.append(extra.get("dataset_id"))
        data = extra.get("data")
        if isinstance(data, Mapping):
            candidates.append(data.get("dataset_id"))
    data = config.get("data")
    if isinstance(data, Mapping):
        candidates.append(data.get("dataset_id"))
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    return None


def _outcome_row(
    trade: TradeRecord,
    strategy_hash: str | None,
    dataset_id: str | None,
    source: str | None,
    synthetic: bool | None,
) -> OutcomeRow:
    return OutcomeRow(
        trade_id=trade.trade_id,
        entry_ts=trade.entry_ts,
        exit_ts=trade.exit_ts,
        side=trade.side,
        entry_price=trade.entry_price,
        exit_price=trade.exit_price,
        lot=trade.lot,
        pnl=trade.pnl,
        r_multiple=trade.r_multiple,
        exit_reason=trade.exit_reason,
        strategy_hash=strategy_hash,
        dataset_id=dataset_id,
        source=source,
        synthetic=synthetic,
    )


def _outcome_key(row: OutcomeRow) -> tuple[object, ...]:
    """Order-independent identity of one outcome for the fingerprint."""
    return (
        row.entry_ts,
        row.exit_ts,
        row.side,
        row.pnl,
        row.r_multiple,
        row.exit_reason,
        row.strategy_hash,
        row.dataset_id,
        row.source or "",
        row.synthetic is True,
    )
