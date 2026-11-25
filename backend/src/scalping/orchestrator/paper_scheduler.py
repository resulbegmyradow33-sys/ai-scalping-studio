"""Evidence-champion paper advances as a trigger source (P7 slice one).

The paper handler can advance a governed champion, and the queue dedupes a job
by key, but nothing connected "bars were appended" to ``paper.advance``. This
module is that link: a :class:`~scalping.orchestrator.triggers.TriggerSource`
that reads the discovery archive, the stored bars and the ledger cursor and
enqueues exactly one advance per scope when a new decision bar exists.

Contract
--------
* **Evidence only.** A scope is scheduled only when ``ChampionProvider`` accepts
  its champion: an ``evidence`` promotion recorded by a production policy. An
  exploratory or manual champion, a missing champion, an unreadable payload or
  an old archive is *refused* with a ``paper.advance_refused`` audit event and
  no job. There is no legacy fallback: the legacy ``flow_momentum`` rule is
  never named, imported or enqueued from here.
* **Bars plus cursor.** The job is produced only when a *decidable* stored bar is
  newer than the ledger cursor. The newest stored bar is always provisional —
  the simulator needs a following bar to evaluate stops and targets — so a
  scope with one bar after the cursor still waits. The cold start (no ledger
  yet) uses the same warm-up rule the runner applies: the first decision index
  is ``max(valid_from + window - 1, window - 1)`` with
  ``valid_from = min(WARMUP, bars)``.
* **Payload is the handler's.** The trigger carries exactly ``symbol``,
  ``timeframe`` and ``bars`` — the fields ``PaperAdvanceRequest`` accepts. It
  cannot select a gap policy, raise a limit or name a strategy: a
  ``paper.advance`` job still runs the continuous runner's default ``strict``
  policy. A gap in the stored slice therefore reaches the runner unchanged and
  is refused there; the scheduler does not skip it to manufacture an advance.

Dedupe
------
Two layers, the same shape as the file and outcome triggers:

1. The cursor document in the job store's ``state`` table records the last
   emitted ``(scope, last_bar_ts)`` key per scope. A scan that sees the same
   decision bar emits nothing, so a poll every few seconds is quiet.
2. The job ``dedupe_key`` is ``paper.advance:{scope}:{last_bar_ts}``. If the
   process crashes between ``scan`` and ``seen`` the cursor is not persisted,
   the next scan recomputes the same key and the store returns the existing job
   instead of a second one. A restart therefore cannot double-advance.

``last_bar_ts`` is the newest **decidable** stored bar, not the provisional
tail. When several bars arrive between scans the target moves and a second job
may be created; the ledger cursor is the real idempotency boundary, so that
second job advances nothing already recorded. One decision bar can never
produce a second signal.

Limits
------
* **No feed, no clock.** Nothing is polled, slept on or fetched here. ``scan``
  is a call; the daemon's poll decides when. Whether a bar has closed is
  inferred from stored timestamps, never from wall time.
* **Bounded scope scan.** At most ``max_scopes`` champions are read per scan,
  with a persisted rotation offset so a larger archive is covered over
  successive scans. The deferred count is audited.
* **Bounded warm read.** With a ledger cursor the store is read from that
  timestamp, so a steady-state scan reads only the recent partitions. A cold
  start reads the stored slice once to count the warm-up and find the tail; a
  scope with fewer bars than the warm-up needs is noted and not enqueued.
* **One local process.** The cursor and the queue are SQLite files owned by one
  writer at a time, exactly like the other trigger sources.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import polars as pl

from scalping.data.registry import TIMEFRAMES, get_symbol
from scalping.data.store import ParquetStore
from scalping.discovery.archive import ArchiveError, DiscoveryArchive
from scalping.discovery.champion import (
    ChampionProvider,
    InvalidChampionSpec,
    NoChampionError,
    NoEvidenceChampion,
)
from scalping.features.build import WARMUP
from scalping.orchestrator.store import JobStore
from scalping.orchestrator.triggers import Trigger
from scalping.paper.ledger import LedgerError, PaperLedger

__all__ = [
    "DEFAULT_MAX_RETRIES",
    "EVENT_DEFERRED",
    "EVENT_REFUSED",
    "KIND_PAPER_ADVANCE",
    "MAX_ADVANCE_BARS",
    "MAX_SCOPES",
    "MAX_SCOPES_LIMIT",
    "REASON_ARCHIVE_UNAVAILABLE",
    "REASON_AWAITING_BAR",
    "REASON_BARS_UNAVAILABLE",
    "REASON_INVALID_CHAMPION",
    "REASON_INVALID_SCOPE",
    "REASON_LEDGER_UNAVAILABLE",
    "REASON_NO_ARCHIVE",
    "REASON_NO_BARS",
    "REASON_NO_CHAMPION",
    "REASON_NO_EVIDENCE_CHAMPION",
    "REASON_WARMING_UP",
    "PaperAdvanceScheduler",
]

#: The job kind this source produces. Kept as a literal so importing this module
#: does not pull the handler stack into a trigger-only process;
#: ``test_paper_scheduler`` asserts it equals the handler's constant.
KIND_PAPER_ADVANCE: Final[str] = "paper.advance"

#: Audit event for a scope that produced no job. Written once per
#: ``(scope, reason)`` transition, so a periodic scan does not flood the log.
EVENT_REFUSED: Final[str] = "paper.advance_refused"
#: Audit event when ``max_scopes`` cut a scan short; the rotation resumes.
EVENT_DEFERRED: Final[str] = "paper.advance_deferred"

#: Refusal reasons. Stable strings, not prose, so the audit page can filter.
REASON_NO_ARCHIVE: Final[str] = "no_archive"
REASON_ARCHIVE_UNAVAILABLE: Final[str] = "archive_unavailable"
REASON_NO_CHAMPION: Final[str] = "no_champion"
REASON_NO_EVIDENCE_CHAMPION: Final[str] = "no_evidence_champion"
REASON_INVALID_CHAMPION: Final[str] = "invalid_champion"
REASON_INVALID_SCOPE: Final[str] = "invalid_scope"
REASON_NO_BARS: Final[str] = "no_bars"
REASON_BARS_UNAVAILABLE: Final[str] = "bars_unavailable"
REASON_LEDGER_UNAVAILABLE: Final[str] = "ledger_unavailable"
REASON_WARMING_UP: Final[str] = "warming_up"
REASON_AWAITING_BAR: Final[str] = "awaiting_bar"

#: The handler's own ceiling on ``bars``; a budget above it would be rejected.
MAX_ADVANCE_BARS: Final[int] = 100_000
#: Scopes read per scan. A champion per symbol/timeframe is a small number; the
#: cap keeps a pathological archive from turning one tick into a long read.
MAX_SCOPES: Final[int] = 100
MAX_SCOPES_LIMIT: Final[int] = 1_000
#: One retry by default: a transient failure must not consume the only trigger
#: for a decision bar.
DEFAULT_MAX_RETRIES: Final[int] = 1
#: Default observation window; matches ``ContinuousPaperRunner``'s default.
DEFAULT_WINDOW: Final[int] = 60
_CURSOR_VERSION: Final[int] = 1


@dataclass(slots=True)
class _Cursor:
    """The durable part of a scan: emitted keys and quiet refusals."""

    #: scope -> ``last_bar_ts`` of the newest acknowledged advance key.
    emitted: dict[str, str] = field(default_factory=dict)
    #: scope -> the last refusal reason, so the same line is not repeated.
    not_queued: dict[str, str] = field(default_factory=dict)
    #: scope -> key emitted but not acknowledged; retried next scan.
    pending_ack: dict[str, str] = field(default_factory=dict)
    #: Where the next scan resumes in the scope list.
    offset: int = 0
    #: Scopes emitted by the scan that has not been acknowledged yet.
    new_scopes: set[str] = field(default_factory=set)


@dataclass(frozen=True, slots=True)
class _LedgerCursor:
    """What the ledger says about one champion session."""

    last_signal_ts: datetime | None
    signals: int
    window: int | None


class PaperAdvanceScheduler:
    """Enqueue one ``paper.advance`` per scope when a new decision bar exists.

    ``data_root``, ``archive_path`` and ``ledger_path`` are the same paths the
    handlers use (``HandlerContext`` exposes them), so the scheduler reads the
    bars a job would read and the ledger a job would write. Nothing is created:
    a missing archive or ledger is a state, not an initialisation. An existing
    archive is opened with its own schema handling; a legacy database may be
    migrated by ``DiscoveryArchive``, exactly as a handler would.

    ``scopes`` optionally names the scopes to visit instead of reading the
    archive's champion list. That is how a caller asks for an explicit market
    even before anything was promoted: the scope then gets a ``no_champion``
    refusal rather than being invisible. The default is the archive's list.

    ``window`` is the observation window used for the cold-start warm-up check.
    It matches the runner's default and does not change an existing session's
    identity.
    """

    def __init__(
        self,
        *,
        store: JobStore,
        data_root: str | Path,
        archive_path: str | Path,
        ledger_path: str | Path,
        scopes: Sequence[str] | None = None,
        window: int = DEFAULT_WINDOW,
        max_scopes: int = MAX_SCOPES,
        priority: int = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        name: str | None = None,
        cursor_key: str | None = None,
    ) -> None:
        if not isinstance(store, JobStore):
            raise TypeError(f"store must be a JobStore, got {type(store).__name__}")
        if not isinstance(data_root, (str, Path)):
            raise TypeError(f"data_root must be a str or Path, got {type(data_root).__name__}")
        if not isinstance(archive_path, (str, Path)):
            raise TypeError(
                f"archive_path must be a str or Path, got {type(archive_path).__name__}"
            )
        if not isinstance(ledger_path, (str, Path)):
            raise TypeError(f"ledger_path must be a str or Path, got {type(ledger_path).__name__}")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise ValueError(f"window must be a positive int, got {window!r}")
        if isinstance(max_scopes, bool) or not isinstance(max_scopes, int):
            raise TypeError("max_scopes must be an int")
        if not 1 <= max_scopes <= MAX_SCOPES_LIMIT:
            raise ValueError(f"max_scopes must be between 1 and {MAX_SCOPES_LIMIT}")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an int")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        explicit = _explicit_scopes(scopes)

        self.store = store
        self.data_root = Path(data_root)
        self.archive_path = Path(archive_path)
        self.ledger_path = Path(ledger_path)
        self.scopes = explicit
        self.window = window
        self.max_scopes = max_scopes
        self.priority = int(priority)
        self.max_retries = max_retries
        self.name = name or f"paper.advance:{self.archive_path}"
        self.cursor_key = cursor_key or (
            f"trigger.paper.advance:{self.archive_path.resolve()}:{self.ledger_path.resolve()}"
        )
        self._store = ParquetStore(self.data_root)
        self._pending: _Cursor | None = None

    # ------------------------------------------------------------------ protocol

    def scan(self) -> list[Trigger]:
        """Return one advance trigger per scope with a new decision bar."""
        cursor = self._load_cursor()
        self._pending = cursor
        cursor.new_scopes = set()
        triggers: list[Trigger] = []

        archive: DiscoveryArchive | None = None
        try:
            if self.scopes is None:
                archive = self._open_archive(cursor)
                scopes = () if archive is None else self._archive_scopes(archive, cursor)
            else:
                scopes = self.scopes
                # An explicit scope still needs the archive to prove its
                # champion; a missing one becomes a per-scope ``no_champion``
                # refusal instead of a global ``no_archive`` line.
                if self.archive_path.is_file():
                    archive = self._open_archive(cursor)
            for scope in scopes:
                self._process(scope, archive, cursor, triggers)
        finally:
            if archive is not None:
                archive.close()
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the keys the caller acknowledged, not the whole scan.

        ``scan_once`` passes every trigger it enqueued. A caller that
        acknowledges a subset leaves the rest in ``pending_ack`` and the next
        scan re-emits them; the job ``dedupe_key`` still prevents a second job.
        Refusal reasons carry no work and are persisted whole.
        """
        pending = self._pending
        if pending is None:
            return
        acknowledged = {scope for trigger in triggers if (scope := _scope_of(trigger)) is not None}
        for scope in acknowledged:
            key = pending.pending_ack.pop(scope, None)
            if key is not None:
                pending.emitted[scope] = key
        for scope in list(pending.new_scopes):
            if scope in acknowledged:
                pending.pending_ack.pop(scope, None)
                continue
            key = pending.emitted.pop(scope, None)
            if key is not None:
                pending.pending_ack[scope] = key
        document = {
            "version": _CURSOR_VERSION,
            "emitted": dict(sorted(pending.emitted.items())),
            "not_queued": dict(sorted(pending.not_queued.items())),
            "pending_ack": dict(sorted(pending.pending_ack.items())),
            "offset": pending.offset,
        }
        self.store.set_state(
            self.cursor_key,
            json.dumps(document, sort_keys=True, separators=(",", ":")),
        )
        self._pending = None

    # ------------------------------------------------------------------ internals

    def _open_archive(self, cursor: _Cursor) -> DiscoveryArchive | None:
        if not self.archive_path.is_file():
            self._note(
                cursor,
                "",
                REASON_NO_ARCHIVE,
                f"no discovery archive at {self.archive_path}; no champion to advance",
            )
            return None
        try:
            return DiscoveryArchive(self.archive_path)
        except (ArchiveError, sqlite3.Error) as exc:
            self._note(
                cursor,
                "",
                REASON_ARCHIVE_UNAVAILABLE,
                f"{type(exc).__name__}: {exc}",
            )
            return None

    def _archive_scopes(self, archive: DiscoveryArchive, cursor: _Cursor) -> tuple[str, ...]:
        """One page of champion scopes, with a persisted rotation offset."""
        try:
            row = archive.connection.execute("SELECT COUNT(*) FROM champions").fetchone()
            total = 0 if row is None else int(row[0])
            if total < 1:
                return ()
            start = cursor.offset % total
            take = min(self.max_scopes, total)
            rows = archive.connection.execute(
                "SELECT scope FROM champions ORDER BY scope LIMIT ? OFFSET ?",
                (take, start),
            ).fetchall()
        except sqlite3.Error as exc:
            self._note(
                cursor,
                "",
                REASON_ARCHIVE_UNAVAILABLE,
                f"{type(exc).__name__}: {exc}",
            )
            return ()
        scopes = tuple(str(row["scope"]) for row in rows)
        cursor.offset = (start + len(scopes)) % total
        deferred = total - len(scopes)
        if deferred > 0:
            self.store.append_event(
                EVENT_DEFERRED,
                detail=_json(
                    {
                        "deferred": deferred,
                        "scanned": len(scopes),
                        "total": total,
                        "resume_offset": cursor.offset,
                    }
                ),
            )
        return scopes

    def _process(
        self,
        scope: str,
        archive: DiscoveryArchive | None,
        cursor: _Cursor,
        triggers: list[Trigger],
    ) -> None:
        resolved = _resolve_scope(scope)
        if resolved is None:
            self._note(
                cursor,
                scope,
                REASON_INVALID_SCOPE,
                "scope must be '<registered symbol>:<supported timeframe>' exactly as the"
                " paper handler will resolve it",
            )
            return
        symbol, timeframe = resolved
        if archive is None:
            # Only reachable with explicit scopes and no readable archive.
            self._note(
                cursor,
                scope,
                REASON_NO_CHAMPION,
                "no discovery archive is readable; nothing was promoted for this scope",
            )
            return

        champion_id = self._champion_id(archive, scope, cursor)
        if champion_id is None:
            return

        session_id = _session_id(symbol, timeframe, champion_id)
        ledger = _read_ledger_cursor(self.ledger_path, session_id)
        if ledger is None:
            self._note(
                cursor,
                scope,
                REASON_LEDGER_UNAVAILABLE,
                f"paper ledger at {self.ledger_path} exists but cannot be read",
            )
            return

        decision = self._decision_bar(symbol, timeframe, ledger, cursor, scope)
        if decision is None:
            return
        last_bar_ts, decidable = decision

        key = f"{KIND_PAPER_ADVANCE}:{scope}:{_key_stamp(last_bar_ts)}"
        if cursor.emitted.get(scope) == key:
            # The same decision bar was emitted before and acknowledged. A quiet
            # scan is the normal state after a successful advance.
            cursor.not_queued.pop(scope, None)
            return
        triggers.append(
            Trigger(
                kind=KIND_PAPER_ADVANCE,
                dedupe_key=key,
                payload={
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "bars": min(max(decidable, 1), MAX_ADVANCE_BARS),
                },
                priority=self.priority,
                max_retries=self.max_retries,
            )
        )
        cursor.emitted[scope] = key
        cursor.new_scopes.add(scope)
        cursor.not_queued.pop(scope, None)

    def _champion_id(
        self,
        archive: DiscoveryArchive,
        scope: str,
        cursor: _Cursor,
    ) -> str | None:
        """The validated evidence champion id, or ``None`` after a refusal."""
        try:
            champion = ChampionProvider(archive, scope=scope).champion()
        except NoChampionError as exc:
            self._note(cursor, scope, REASON_NO_CHAMPION, str(exc))
            return None
        except NoEvidenceChampion as exc:
            self._note(
                cursor,
                scope,
                REASON_NO_EVIDENCE_CHAMPION,
                f"{exc}; an exploratory or manual champion is not paper-tradeable",
            )
            return None
        except InvalidChampionSpec as exc:
            self._note(cursor, scope, REASON_INVALID_CHAMPION, str(exc))
            return None
        except ArchiveError as exc:
            self._note(cursor, scope, REASON_ARCHIVE_UNAVAILABLE, str(exc))
            return None
        return champion.candidate_id

    def _decision_bar(
        self,
        symbol: str,
        timeframe: str,
        ledger: _LedgerCursor,
        cursor: _Cursor,
        scope: str,
    ) -> tuple[datetime, int] | None:
        """The newest decidable bar and how many decisions it enables, or ``None``."""
        cursor_ts = ledger.last_signal_ts
        try:
            frame = (
                self._store.read(symbol, timeframe)
                if cursor_ts is None
                else self._store.read(symbol, timeframe, start=cursor_ts)
            )
        except (OSError, ValueError, TypeError) as exc:
            self._note(
                cursor,
                scope,
                REASON_BARS_UNAVAILABLE,
                f"{type(exc).__name__}: {exc}",
            )
            return None
        try:
            stamps = _bar_stamps(frame)
        except TypeError as exc:
            self._note(cursor, scope, REASON_BARS_UNAVAILABLE, str(exc))
            return None

        if cursor_ts is None:
            total = len(stamps)
            if total < 2:
                self._note(
                    cursor,
                    scope,
                    REASON_NO_BARS if total == 0 else REASON_WARMING_UP,
                    "no stored bars" if total == 0 else "not enough bars for a decision",
                )
                return None
            window = self.window if ledger.window is None else ledger.window
            first_index = _first_decision_index(total, window)
            if total - 2 < first_index:
                self._note(
                    cursor,
                    scope,
                    REASON_WARMING_UP,
                    f"{total} stored bars are below the first decision index"
                    f" {first_index} (window {window}); nothing to advance",
                )
                return None
            return stamps[-2], total - 1 - first_index

        # A cursor exists: the session already absorbed the warm-up. The newest
        # stored bar is provisional, so the decision needs one more bar behind it.
        pending = [stamp for stamp in stamps if stamp > cursor_ts]
        if len(pending) < 2:
            self._note(
                cursor,
                scope,
                REASON_AWAITING_BAR,
                "the ledger cursor is still the newest decidable bar; the appended bar"
                " is provisional until a following bar is stored",
            )
            return None
        return pending[-2], len(pending) - 1

    def _note(self, cursor: _Cursor, scope: str, reason: str, detail: str) -> None:
        """Record a refusal once per scope until its reason changes."""
        if cursor.not_queued.get(scope) == reason:
            return
        cursor.not_queued[scope] = reason
        self.store.append_event(
            EVENT_REFUSED,
            detail=_json({"scope": scope or None, "reason": reason, "detail": detail}),
        )

    def _load_cursor(self) -> _Cursor:
        raw = self.store.get_state(self.cursor_key)
        if raw is None:
            return _Cursor()
        try:
            parsed = json.loads(raw)
        except ValueError:
            return _Cursor()  # a corrupt cursor costs a re-read; dedupe still holds
        if not isinstance(parsed, dict):
            return _Cursor()
        offset = parsed.get("offset")
        return _Cursor(
            emitted=_text_map(parsed.get("emitted")),
            not_queued=_text_map(parsed.get("not_queued")),
            pending_ack=_text_map(parsed.get("pending_ack")),
            offset=0 if isinstance(offset, bool) or not isinstance(offset, int) else max(offset, 0),
        )


# --------------------------------------------------------------------- helpers


def _read_ledger_cursor(path: Path, session_id: str) -> _LedgerCursor | None:
    """The newest signal of the champion session, or a fresh-session cursor.

    A missing file is ``_LedgerCursor(None, 0, None)``: the first advance will
    create it. An existing file that cannot be opened, has no schema or refuses
    the query is ``None`` and the scope is refused, because guessing the cursor
    would risk re-advancing a session this process cannot read.
    """
    if not path.exists():
        return _LedgerCursor(None, 0, None)
    try:
        ledger = PaperLedger.open_reader(path)
    except (FileNotFoundError, LedgerError):
        return None
    try:
        if not ledger.has_schema():
            return None
        session = ledger.session(session_id)
        if session is None:
            return _LedgerCursor(None, 0, None)
        row = ledger.connection.execute(
            "SELECT MAX(ts) AS last, COUNT(*) AS n FROM signals WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        last = None if row is None or row["last"] is None else _parse_ts(row["last"])
        count = 0 if row is None else int(row["n"])
        window = session.config.get("window")
        return _LedgerCursor(
            last_signal_ts=last,
            signals=count,
            window=window if isinstance(window, int) and not isinstance(window, bool) else None,
        )
    except (LedgerError, sqlite3.Error, ValueError, TypeError):
        return None
    finally:
        ledger.close()


def _parse_ts(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _session_id(symbol: str, timeframe: str, champion_id: str) -> str:
    """The continuous runner's session id: symbol, timeframe and spec hash."""
    return f"auto-{symbol}-{timeframe}-{champion_id[:16]}"


def _first_decision_index(total: int, window: int) -> int:
    """The runner's own first decision row for a cold slice."""
    warmup = min(WARMUP, total)
    return max(warmup + window - 1, window - 1)


def _resolve_scope(scope: str) -> tuple[str, str] | None:
    """Canonical ``(symbol, timeframe)`` for a scope key, or ``None``.

    The paper handler opens the champion with ``scope_for(payload.symbol,
    payload.timeframe)``, so a scope that does not round-trip through the
    registry would enqueue a job that can never see its own champion. Those are
    refused instead of guessed.
    """
    if not isinstance(scope, str):
        return None
    symbol_text, separator, timeframe_text = scope.partition(":")
    if not separator or ":" in timeframe_text:
        return None
    try:
        symbol_spec = get_symbol(symbol_text)
    except ValueError:
        return None
    timeframe = timeframe_text.strip().upper()
    if timeframe not in TIMEFRAMES:
        return None
    if f"{symbol_spec.symbol}:{timeframe}" != scope:
        return None
    return symbol_spec.symbol, timeframe


def _explicit_scopes(scopes: Sequence[str] | None) -> tuple[str, ...] | None:
    if scopes is None:
        return None
    if isinstance(scopes, (str, bytes)):
        raise TypeError("scopes must be a sequence of scope strings, not a string")
    if not isinstance(scopes, Iterable):  # pragma: no cover - Sequence already implies it
        raise TypeError("scopes must be a sequence")
    ordered: list[str] = []
    for scope in scopes:
        if not isinstance(scope, str) or not scope.strip():
            raise ValueError("scopes entries must be non-empty strings")
        if scope not in ordered:
            ordered.append(scope)
    if not ordered:
        raise ValueError("scopes must not be empty; omit it to read the archive")
    if len(ordered) > MAX_SCOPES_LIMIT:
        raise ValueError(f"scopes must contain at most {MAX_SCOPES_LIMIT} entries")
    return tuple(ordered)


def _scope_of(trigger: Trigger) -> str | None:
    payload = trigger.payload
    if not isinstance(payload, Mapping):
        return None
    symbol = payload.get("symbol")
    timeframe = payload.get("timeframe")
    if isinstance(symbol, str) and isinstance(timeframe, str):
        return f"{symbol}:{timeframe}"
    return None


def _bar_stamps(frame: pl.DataFrame) -> list[datetime]:
    """The ``ts`` column as aware datetimes; a foreign frame is a refusal."""
    values = frame.get_column("ts").to_list()
    stamps: list[datetime] = []
    for value in values:
        if not isinstance(value, datetime):
            raise TypeError(f"stored ts is not a datetime: {type(value).__name__}")
        stamps.append(value if value.tzinfo is not None else value.replace(tzinfo=UTC))
    return stamps


def _key_stamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _json(payload: dict[str, object]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
