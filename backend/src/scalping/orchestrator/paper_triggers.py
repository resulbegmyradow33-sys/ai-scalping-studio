"""Matured paper outcomes as the next research-cycle trigger (P5 slice 2).

The orchestrator already knows how to queue work and how to run a paper session.
This module is the seam between them: when a paper session has trades that are
*closed*, it emits one ``paper.outcomes`` trigger identified by
:func:`~scalping.paper.continuous.outcome_fingerprint`. A fingerprint that was
already enqueued is skipped, so a scan on every tick converges on exactly one
job per matured batch.

Two deliberate separations
--------------------------
* **Reading is not executing.** ``PaperOutcomeTriggerSource`` only reads the
  ledger through ``PaperLedger.open_reader`` and calls ``matured_outcomes``.
  Open positions are counted by the batch and never turned into rows, so an
  unrealised trade cannot start research.
* **A trigger is not a promotion.** The handler built by
  :func:`make_paper_outcome_handler` records what arrived and returns a
  proposal (``queue_discovery`` or ``no_action``) with ``executed: false``. It
  imports no discovery, training or champion code and has no holdout access.
  Upgrading the proposal into a real enqueue is an explicit caller decision, and
  the optional ``should_queue`` callback is where that policy lives.

Dedupe has two layers
---------------------
The trigger ``seen``/cursor document in the job store holds the fingerprints
already emitted for this ledger path (plus the last refusal reason per session).
The job's ``dedupe_key`` is ``paper.outcomes:{fingerprint}`` as a second layer:
a crash between enqueue and ``seen`` re-emits the same trigger on the next scan
and the store returns the existing job instead of a duplicate.

What this slice does not do
---------------------------
No scheduler, no market clock, no background thread. ``scan`` is a call. The
cursor and the job store live on local SQLite files; one process owns them at a
time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from scalping.orchestrator.core import Handler, HandlerResult, JobContext
from scalping.orchestrator.store import Job, JobStore
from scalping.orchestrator.triggers import Trigger
from scalping.paper.continuous import OutcomeError, matured_outcomes, outcome_fingerprint
from scalping.paper.ledger import LedgerError, PaperLedger, SessionRecord

__all__ = [
    "EVENT_NOT_QUEUED",
    "EVENT_RECEIVED",
    "KIND_PAPER_OUTCOMES",
    "PAPER_OUTCOME_FIELDS",
    "REASON_EMPTY_BATCH",
    "REASON_LEDGER_UNAVAILABLE",
    "REASON_NO_DATASET_ID",
    "REASON_NO_LEDGER",
    "REASON_NO_STRATEGY_HASH",
    "REASON_OUTCOME_ERROR",
    "PaperOutcomeTriggerSource",
    "make_paper_outcome_handler",
    "register_paper_outcome_handler",
]

#: The one kind this module produces and handles.
KIND_PAPER_OUTCOMES: Final[str] = "paper.outcomes"

#: Events written by the source and the handler; documented in
#: ``docs/orchestrator.md`` so the audit page can filter on them.
EVENT_NOT_QUEUED: Final[str] = "paper.outcome_not_queued"
EVENT_RECEIVED: Final[str] = "paper.outcome_received"

#: Why a session produced no trigger. A reason is recorded once per session and
#: rewritten only when it changes, so a periodic scan does not flood the log.
REASON_EMPTY_BATCH: Final[str] = "empty_batch"
REASON_NO_STRATEGY_HASH: Final[str] = "no_strategy_hash"
REASON_NO_DATASET_ID: Final[str] = "no_dataset_id"
REASON_OUTCOME_ERROR: Final[str] = "outcome_error"
REASON_NO_LEDGER: Final[str] = "no_ledger"
REASON_LEDGER_UNAVAILABLE: Final[str] = "ledger_unavailable"

#: Every payload field the handler accepts. Keep this in sync with the trigger
#: payload and with ``docs/orchestrator.md``; unknown fields are rejected.
PAPER_OUTCOME_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "session_id",
        "fingerprint",
        "trades",
        "pnl",
        "r_total",
        "strategy_hash",
        "dataset_id",
    }
)

#: Upper bound on sessions read per scan. A larger ledger is truncated, not
#: read for minutes on the target machine.
MAX_SCAN_SESSIONS: Final[int] = 1_000
_PAGE_SIZE: Final[int] = 200
_CURSOR_VERSION: Final[int] = 1

#: A policy hook, not an action. ``True`` means "propose discovery", not "run".
PaperOutcomePolicy = Callable[[Mapping[str, Any]], bool]

DEFAULT_APPROVED_REASON: Final[str] = (
    "matured paper outcomes are eligible for the next research cycle"
)
DEFAULT_DECLINED_REASON: Final[str] = (
    "no should_queue callback installed; the proposal remains unexecuted"
)


@dataclass(slots=True)
class _Cursor:
    """The durable part of a scan: emitted fingerprints and quiet refusals."""

    fingerprints: dict[str, str] = field(default_factory=dict)
    not_queued: dict[str, str] = field(default_factory=dict)


class PaperOutcomeTriggerSource:
    """Emit one trigger per matured, attributable paper outcome batch.

    ``ledger_path`` is read with a read-only connection, so a scan never
    migrates the schema, never creates the file and never blocks the session
    writer beyond a normal WAL read. Sessions are read newest first, capped at
    ``max_sessions``.

    The cursor document is stored in the job store under
    ``trigger.paper.outcomes:<absolute ledger path>``. It survives a restart of
    the process; pointing a second source at the same ledger path reuses it.

    A session is eligible when ``matured_outcomes`` returns at least one closed
    trade and the batch carries both a strategy hash and a dataset id. Anything
    else is recorded as a refusal reason and emits nothing. Session status
    (``running``/``closed``) is not required: a running session's closed trades
    are matured, and waiting for a session close would delay feedback.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        store: JobStore,
        kind: str = KIND_PAPER_OUTCOMES,
        priority: int = 0,
        max_retries: int = 0,
        max_sessions: int = MAX_SCAN_SESSIONS,
        name: str | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        if not str(kind).strip():
            raise ValueError("kind must not be empty")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be an int")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        self.kind = str(kind)
        self.store = store
        self.priority = int(priority)
        self.max_retries = int(max_retries)
        self.max_sessions = max_sessions
        self.name = name or f"paper.outcomes:{self.ledger_path}"
        self.cursor_key = f"trigger.paper.outcomes:{self.ledger_path.resolve()}"
        self._pending: _Cursor | None = None

    # ------------------------------------------------------------------ protocol

    def scan(self) -> list[Trigger]:
        """Read the ledger and return triggers not yet recorded in the cursor."""
        cursor = self._load_cursor()
        # The cursor is persisted only by ``seen``; a crash or an exception
        # before acknowledgement costs a re-read, never a lost job.
        self._pending = cursor
        triggers: list[Trigger] = []

        if not self.ledger_path.exists():
            self._note(cursor, "", REASON_NO_LEDGER, f"no paper ledger at {self.ledger_path}")
            return triggers
        try:
            ledger = PaperLedger.open_reader(self.ledger_path)
        except (FileNotFoundError, LedgerError) as exc:
            self._note(cursor, "", REASON_LEDGER_UNAVAILABLE, str(exc))
            return triggers

        try:
            if not ledger.has_schema():
                self._note(cursor, "", REASON_LEDGER_UNAVAILABLE, "ledger has no sessions table")
                return triggers
            for record in self._sessions(ledger):
                self._process(ledger, record, cursor, triggers)
        finally:
            ledger.close()
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the fingerprint and refusal sets computed by the last scan.

        The triggers are not required to save the cursor; the argument keeps the
        protocol uniform and lets a future source acknowledge per trigger.
        """
        pending = self._pending
        if pending is None:
            return
        document = {
            "version": _CURSOR_VERSION,
            "fingerprints": dict(sorted(pending.fingerprints.items())),
            "not_queued": dict(sorted(pending.not_queued.items())),
        }
        self.store.set_state(
            self.cursor_key,
            json.dumps(document, sort_keys=True, separators=(",", ":")),
        )
        self._pending = None

    # ------------------------------------------------------------------ internals

    def _process(
        self,
        ledger: PaperLedger,
        record: SessionRecord,
        cursor: _Cursor,
        triggers: list[Trigger],
    ) -> None:
        session_id = record.session_id
        try:
            batch = matured_outcomes(ledger, session_id)
        except OutcomeError as exc:
            self._note(cursor, session_id, REASON_OUTCOME_ERROR, str(exc))
            return

        if batch.closed_trades == 0:
            self._note(cursor, session_id, REASON_EMPTY_BATCH, "no closed trades yet")
            return
        strategy_hash = batch.strategy_hash
        if strategy_hash is None or not strategy_hash.strip():
            self._note(
                cursor,
                session_id,
                REASON_NO_STRATEGY_HASH,
                "session records no strategy hash; outcomes cannot be attributed",
            )
            return
        dataset_id = batch.dataset_id
        if dataset_id is None or not dataset_id.strip():
            self._note(
                cursor,
                session_id,
                REASON_NO_DATASET_ID,
                "session records no dataset id; research cannot be scoped to evidence",
            )
            return

        fingerprint = outcome_fingerprint(batch)
        if fingerprint in cursor.fingerprints:
            return

        payload = {
            "session_id": session_id,
            "fingerprint": fingerprint,
            "trades": batch.closed_trades,
            "pnl": math.fsum(row.pnl for row in batch.rows),
            "r_total": math.fsum(row.r_multiple for row in batch.rows),
            "strategy_hash": strategy_hash,
            "dataset_id": dataset_id,
        }
        triggers.append(
            Trigger(
                kind=self.kind,
                dedupe_key=f"{self.kind}:{fingerprint}",
                payload=payload,
                priority=self.priority,
                max_retries=self.max_retries,
            )
        )
        cursor.fingerprints[fingerprint] = session_id
        cursor.not_queued.pop(session_id, None)

    def _sessions(self, ledger: PaperLedger) -> Iterator[SessionRecord]:
        offset = 0
        read = 0
        while read < self.max_sessions:
            page, total = ledger.session_page(limit=_PAGE_SIZE, offset=offset)
            if not page:
                return
            for record in page:
                if read >= self.max_sessions:
                    return
                read += 1
                yield record
            offset += len(page)
            if offset >= total:
                return

    def _note(self, cursor: _Cursor, session_id: str, reason: str, detail: str) -> None:
        """Record a refusal once per session until its reason changes."""
        if cursor.not_queued.get(session_id) == reason:
            return
        cursor.not_queued[session_id] = reason
        self.store.append_event(
            EVENT_NOT_QUEUED,
            detail=_json(
                {
                    "session_id": session_id or None,
                    "reason": reason,
                    "detail": detail,
                }
            ),
        )

    def _load_cursor(self) -> _Cursor:
        raw = self.store.get_state(self.cursor_key)
        if raw is None:
            return _Cursor()
        try:
            parsed = json.loads(raw)
        except ValueError:
            return _Cursor()  # a corrupt cursor costs a re-read; dedupe still holds
        if not isinstance(parsed, Mapping):
            return _Cursor()
        fingerprints = parsed.get("fingerprints")
        not_queued = parsed.get("not_queued")
        return _Cursor(
            fingerprints=_text_map(fingerprints),
            not_queued=_text_map(not_queued),
        )


# --------------------------------------------------------------------- handler


class HandlerRegistry(Protocol):
    """The narrow part of ``Orchestrator`` this module registers into."""

    def register(self, kind: str, handler: Handler) -> None: ...


def make_paper_outcome_handler(
    *,
    should_queue: PaperOutcomePolicy | None = None,
) -> Handler:
    """Build a handler that records the event and proposes the next step.

    It never runs discovery, training or a promotion. ``should_queue`` is the
    policy hook: returning ``True`` proposes ``queue_discovery`` and ``False``
    proposes ``no_action``. Either way the result carries ``queued: false`` and
    ``executed: false`` because this slice has no executor behind the proposal.
    A callback exception propagates as an ordinary handler failure and is
    retried by the queue.
    """

    def handler(job: Job, context: JobContext) -> HandlerResult:
        try:
            request = _validated(job.payload)
        except _PayloadError as exc:
            return HandlerResult.failure(str(exc), permanent=True)

        context.log(
            EVENT_RECEIVED,
            _json(
                {
                    "session_id": request["session_id"],
                    "fingerprint": request["fingerprint"],
                    "trades": request["trades"],
                }
            ),
        )
        if should_queue is None:
            approved = True
            reason = DEFAULT_APPROVED_REASON
        else:
            approved = bool(should_queue(dict(request)))
            reason = (
                "should_queue approved the next research cycle"
                if approved
                else ("should_queue declined the next research cycle")
            )

        return HandlerResult.success(
            {
                "action": "queue_discovery" if approved else "no_action",
                "reason": reason,
                "queued": False,
                "executed": False,
                "session_id": request["session_id"],
                "fingerprint": request["fingerprint"],
                "trades": request["trades"],
                "pnl": request["pnl"],
                "r_total": request["r_total"],
                "strategy_hash": request["strategy_hash"],
                "dataset_id": request["dataset_id"],
            }
        )

    return handler


def register_paper_outcome_handler(
    registry: HandlerRegistry,
    *,
    should_queue: PaperOutcomePolicy | None = None,
) -> str:
    """Register the proposal-only handler for ``paper.outcomes``; returns the kind."""
    registry.register(KIND_PAPER_OUTCOMES, make_paper_outcome_handler(should_queue=should_queue))
    return KIND_PAPER_OUTCOMES


# --------------------------------------------------------------------- helpers


class _PayloadError(ValueError):
    """A ``paper.outcomes`` payload is invalid; retrying cannot change it."""


def _validated(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise _PayloadError(f"payload must be a mapping, got {type(payload).__name__}")
    unknown = set(payload) - PAPER_OUTCOME_FIELDS
    if unknown:
        raise _PayloadError(f"unknown payload fields: {sorted(unknown)}")
    missing = PAPER_OUTCOME_FIELDS - set(payload)
    if missing:
        raise _PayloadError(f"missing payload fields: {sorted(missing)}")

    return {
        "session_id": _text(payload["session_id"], "session_id"),
        "fingerprint": _text(payload["fingerprint"], "fingerprint"),
        "trades": _count(payload["trades"], "trades"),
        "pnl": _number(payload["pnl"], "pnl"),
        "r_total": _number(payload["r_total"], "r_total"),
        "strategy_hash": _text(payload["strategy_hash"], "strategy_hash"),
        "dataset_id": _text(payload["dataset_id"], "dataset_id"),
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _PayloadError(f"{name} must be a non-empty string")
    return value


def _count(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _PayloadError(f"{name} must be an int")
    if value < 1:
        raise _PayloadError(f"{name} must be >= 1")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _PayloadError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _PayloadError(f"{name} must be finite")
    return number


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }


def _json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
