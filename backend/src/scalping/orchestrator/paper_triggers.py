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

Coverage and backlog
--------------------
Sessions are ordered by last trade activity (``MAX(trades.exit_ts)``, falling
back to the session start), not by session start, so a long-running session that
keeps closing trades is never starved by newer, quiet sessions. The scan also
carries a rotation offset, so a ledger larger than ``max_sessions`` is covered
over successive scans instead of always reading the same newest window. Sessions
deferred by the cap or by the per-scan proposal limit are recorded with
:data:`REASON_TRUNCATED`; they are delayed, not forgotten. ``since`` bounds the
window for a large historical ledger, and ``max_proposals`` caps a first scan.

What this slice does not do
---------------------------
No internal scheduler, no market clock, no background thread. ``scan`` is a
call. The CLI can bind it with ``--watch-paper-ledger`` and the daemon's poll
then decides the cadence; nothing here watches a bar or a file event. The cursor
and the job store live on local SQLite files; one process owns them at a time.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Protocol

from scalping.orchestrator.core import Handler, HandlerResult, JobContext
from scalping.orchestrator.store import Job, JobStore
from scalping.orchestrator.triggers import Trigger
from scalping.paper.continuous import (
    OutcomeBatch,
    OutcomeError,
    matured_outcomes,
    outcome_fingerprint,
)
from scalping.paper.ledger import LedgerError, PaperLedger

__all__ = [
    "EVENT_NOT_QUEUED",
    "EVENT_RECEIVED",
    "EVENT_REFUSED",
    "EVENT_TRUNCATED",
    "KIND_PAPER_OUTCOMES",
    "MAX_SCAN_PROPOSALS_LIMIT",
    "MAX_SCAN_SESSIONS_LIMIT",
    "PAPER_OUTCOME_FIELDS",
    "PAPER_OUTCOME_REQUIRED_FIELDS",
    "REASON_EMPTY_BATCH",
    "REASON_LEDGER_UNAVAILABLE",
    "REASON_NO_DATASET_ID",
    "REASON_NO_LEDGER",
    "REASON_NO_STRATEGY_HASH",
    "REASON_OUTCOME_ERROR",
    "REASON_SYNTHETIC",
    "REASON_TRUNCATED",
    "REFUSAL_SYNTHETIC",
    "PaperOutcomeTriggerSource",
    "make_paper_outcome_handler",
    "register_paper_outcome_handler",
]

#: The one kind this module produces and handles. A caller cannot swap it: the
#: payload contract, the cursor key and the registered handler all assume this
#: name, so an arbitrary kind would enqueue jobs nothing can run.
KIND_PAPER_OUTCOMES: Final[str] = "paper.outcomes"

#: Events written by the source and the handler; documented in
#: ``docs/orchestrator.md`` so the audit page can filter on them.
EVENT_NOT_QUEUED: Final[str] = "paper.outcome_not_queued"
EVENT_RECEIVED: Final[str] = "paper.outcome_received"
EVENT_REFUSED: Final[str] = "paper.outcome_refused"
EVENT_TRUNCATED: Final[str] = "paper.outcome_truncated"

#: Why a session produced no trigger. A reason is recorded once per session and
#: rewritten only when it changes, so a periodic scan does not flood the log.
REASON_EMPTY_BATCH: Final[str] = "empty_batch"
REASON_NO_STRATEGY_HASH: Final[str] = "no_strategy_hash"
REASON_NO_DATASET_ID: Final[str] = "no_dataset_id"
REASON_OUTCOME_ERROR: Final[str] = "outcome_error"
REASON_NO_LEDGER: Final[str] = "no_ledger"
REASON_LEDGER_UNAVAILABLE: Final[str] = "ledger_unavailable"
#: The batch is known to come from synthetic data and the source is configured
#: not to treat that as market evidence (``allow_synthetic=False``).
REASON_SYNTHETIC: Final[str] = "synthetic_outcome"
#: The same refusal code on the handler side; a job may arrive from an
#: ``allow_synthetic=True`` source or from an older queue.
REFUSAL_SYNTHETIC: Final[str] = REASON_SYNTHETIC
#: The session was not read this scan because ``max_sessions`` or
#: ``max_proposals`` was reached. The rotation cursor resumes at it next scan.
REASON_TRUNCATED: Final[str] = "truncated"

#: Fields every payload must carry; the producer writes them, the handler
#: requires them. Kept separate from the optional provenance pair so an old
#: queued job still validates after an upgrade.
PAPER_OUTCOME_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
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

#: Every payload field the handler accepts. Keep this in sync with the trigger
#: payload and with ``docs/orchestrator.md``; unknown fields are rejected.
PAPER_OUTCOME_FIELDS: Final[frozenset[str]] = PAPER_OUTCOME_REQUIRED_FIELDS | {
    "source",
    "synthetic",
}

#: Upper bound on sessions read per scan. A larger ledger is truncated, not
#: read for minutes on the target machine. The rotation cursor visits the rest.
MAX_SCAN_SESSIONS: Final[int] = 1_000
#: Refuse a configuration above this: a scan is a synchronous call and a caller
#: asking for more than this is misconfiguring the trigger, not optimizing it.
MAX_SCAN_SESSIONS_LIMIT: Final[int] = 10_000
#: Upper bound on triggers emitted per scan, so a first scan over a large
#: historical ledger cannot propose an unbounded research backlog at once.
MAX_SCAN_PROPOSALS: Final[int] = 100
MAX_SCAN_PROPOSALS_LIMIT: Final[int] = 1_000
#: How many deferred session ids get their own ``truncated`` audit line; the
#: summary event always records the exact totals.
TRUNCATION_NOTE_LIMIT: Final[int] = 25
#: One retry by default: a transient handler failure must not silently consume
#: the only trigger for a matured batch.
DEFAULT_MAX_RETRIES: Final[int] = 1
_PAGE_SIZE: Final[int] = 200
_CURSOR_VERSION: Final[int] = 3


#: A policy hook, not an action. ``True`` means "propose discovery", not "run".
PaperOutcomePolicy = Callable[[Mapping[str, Any]], bool]

DEFAULT_APPROVED_REASON: Final[str] = (
    "matured paper outcomes are eligible for the next research cycle"
)
DEFAULT_DECLINED_REASON: Final[str] = (
    "no should_queue callback installed; the proposal remains unexecuted"
)
DEFAULT_SYNTHETIC_REASON: Final[str] = (
    "synthetic outcomes are not market evidence; allow_synthetic=False refuses them"
)


@dataclass(slots=True)
class _Cursor:
    """The durable part of a scan: emitted fingerprints and quiet refusals."""

    fingerprints: dict[str, str] = field(default_factory=dict)
    not_queued: dict[str, str] = field(default_factory=dict)
    #: Where the next scan resumes in the activity-ordered session list; keeps a
    #: ledger larger than ``max_sessions`` fully covered across scans.
    offset: int = 0
    #: Fingerprints emitted but not acknowledged yet (fingerprint -> session id).
    #: Persisted and retried at the start of the next scan, so a partial
    #: acknowledgement does not wait for the rotation to wrap.
    pending_ack: dict[str, str] = field(default_factory=dict)
    #: Fingerprints emitted by the scan that has not been acknowledged yet. Not
    #: persisted beyond ``pending_ack``: ``seen`` decides which survive.
    new_fingerprints: set[str] = field(default_factory=set)


class PaperOutcomeTriggerSource:
    """Emit one trigger per matured, attributable paper outcome batch.

    ``ledger_path`` is read with a read-only connection, so a scan never
    migrates the schema, never creates the file and never blocks the session
    writer beyond a normal WAL read. Sessions are ordered by last trade activity
    (``MAX(trades.exit_ts)``, or the start when no trade has closed) and read up
    to ``max_sessions`` per scan, starting at a persisted rotation offset so a
    larger ledger is covered over successive scans.

    ``since`` optionally bounds the window by session start, which is the way to
    keep a first scan over a long ledger cheap; ``max_proposals`` caps how many
    triggers one scan may emit. Deferred sessions are recorded with
    :data:`REASON_TRUNCATED` and one ``paper.outcome_truncated`` summary event.

    The cursor document is stored in the job store under
    ``trigger.paper.outcomes:<absolute ledger path>``. It survives a restart of
    the process; pointing a second source at the same ledger path reuses it.

    A session is eligible when ``matured_outcomes`` returns at least one closed
    trade and the batch carries both a strategy hash and a dataset id. Anything
    else is recorded as a refusal reason and emits nothing. Session status
    (``running``/``closed``) is not required: a running session's closed trades
    are matured, and waiting for a session close would delay feedback.

    ``allow_synthetic`` defaults to ``False``: a batch whose recorded
    ``synthetic`` flag is ``True`` is refused and logged with
    :data:`REASON_SYNTHETIC`, because a demo run must not become market
    evidence. ``synthetic is None`` (unknown) is not refused here; it travels in
    the payload as ``null`` so a policy hook can decide without the trigger
    inventing a ``False``.
    """

    def __init__(
        self,
        ledger_path: str | Path,
        *,
        store: JobStore,
        kind: str = KIND_PAPER_OUTCOMES,
        priority: int = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        max_sessions: int = MAX_SCAN_SESSIONS,
        max_proposals: int = MAX_SCAN_PROPOSALS,
        since: str | datetime | None = None,
        allow_synthetic: bool = False,
        name: str | None = None,
    ) -> None:
        self.ledger_path = Path(ledger_path)
        if str(kind) != KIND_PAPER_OUTCOMES:
            raise ValueError(
                f"kind must be {KIND_PAPER_OUTCOMES!r}; {kind!r} has no registered handler"
            )
        if not isinstance(allow_synthetic, bool):
            raise TypeError(f"allow_synthetic must be a bool, got {type(allow_synthetic).__name__}")
        if isinstance(max_sessions, bool) or not isinstance(max_sessions, int):
            raise TypeError("max_sessions must be an int")
        if max_sessions < 1:
            raise ValueError("max_sessions must be >= 1")
        if max_sessions > MAX_SCAN_SESSIONS_LIMIT:
            raise ValueError(f"max_sessions must be <= {MAX_SCAN_SESSIONS_LIMIT}")
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int):
            raise TypeError("max_proposals must be an int")
        if max_proposals < 1:
            raise ValueError("max_proposals must be >= 1")
        if max_proposals > MAX_SCAN_PROPOSALS_LIMIT:
            raise ValueError(f"max_proposals must be <= {MAX_SCAN_PROPOSALS_LIMIT}")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an int")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.kind = KIND_PAPER_OUTCOMES
        self.store = store
        self.priority = int(priority)
        self.max_retries = max_retries
        self.max_sessions = max_sessions
        self.max_proposals = max_proposals
        self.since = _coerce_since(since)
        self.allow_synthetic = allow_synthetic
        self.name = name or f"paper.outcomes:{self.ledger_path}"
        self.cursor_key = f"trigger.paper.outcomes:{self.ledger_path.resolve()}"
        self._pending: _Cursor | None = None

    # ------------------------------------------------------------------ protocol

    def scan(self) -> list[Trigger]:
        """Read the ledger and return triggers not yet recorded in the cursor."""
        cursor = self._load_cursor()
        cursor.new_fingerprints = set()
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
            total = self._session_count(ledger)
            if total < 1:
                return triggers
            # Unacknowledged triggers go first: a partial acknowledgement must
            # not wait for the rotation to wrap. Bounded by max_proposals, and
            # it does not consume the rotation budget.
            for fingerprint in list(cursor.pending_ack):
                if len(triggers) >= self.max_proposals:
                    break
                session_id = cursor.pending_ack.pop(fingerprint)
                self._process(ledger, session_id, cursor, triggers)
            start = cursor.offset % total
            visited = 0
            for session_id in self._sessions(ledger, total, start, self.max_sessions):
                if len(triggers) >= self.max_proposals:
                    break
                self._process(ledger, session_id, cursor, triggers)
                visited += 1
            cursor.offset = (start + visited) % total
            self._record_deferred(cursor, ledger, total, visited)
        finally:
            ledger.close()
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the fingerprints the caller acknowledged, not the whole scan.

        ``scan_once`` passes every trigger it enqueued. A caller that
        acknowledges only a subset leaves the rest in the cursor's
        ``pending_ack`` set, and the next scan re-emits them before it resumes
        the rotation. They do not wait for the offset to wrap. The job
        ``dedupe_key`` still prevents a second job for a trigger that was in
        fact enqueued. Refusal reasons carry no work and are persisted whole.
        """
        pending = self._pending
        if pending is None:
            return
        acknowledged = set()
        for trigger in triggers:
            fingerprint = _fingerprint_of(trigger)
            if isinstance(fingerprint, str):
                acknowledged.add(fingerprint)
        # A late acknowledgement (a trigger kept from an earlier scan) still
        # promotes its fingerprint from pending to durable.
        for fingerprint in acknowledged:
            session_id = pending.pending_ack.pop(fingerprint, None)
            if session_id is not None:
                pending.fingerprints[fingerprint] = session_id
        for fingerprint in list(pending.new_fingerprints):
            if fingerprint in acknowledged:
                pending.pending_ack.pop(fingerprint, None)
                continue
            session_id = pending.fingerprints.pop(fingerprint, None)
            if session_id is not None:
                pending.pending_ack[fingerprint] = session_id
        document = {
            "version": _CURSOR_VERSION,
            "fingerprints": dict(sorted(pending.fingerprints.items())),
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

    def _process(
        self,
        ledger: PaperLedger,
        session_id: str,
        cursor: _Cursor,
        triggers: list[Trigger],
    ) -> None:
        try:
            batch = matured_outcomes(ledger, session_id)
        except (OutcomeError, LedgerError, ValueError) as exc:
            self._note(cursor, session_id, REASON_OUTCOME_ERROR, str(exc))
            return
        source, synthetic = batch.source, batch.synthetic

        if batch.closed_trades == 0:
            self._note(cursor, session_id, REASON_EMPTY_BATCH, "no closed trades yet")
            return
        if synthetic is True and not self.allow_synthetic:
            self._note(
                cursor,
                session_id,
                REASON_SYNTHETIC,
                "session outcomes are synthetic; allow_synthetic=False refuses them"
                " as market evidence",
            )
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

        fingerprint = self._fingerprint(batch, session_id, cursor)
        if fingerprint is None:
            return
        if fingerprint in cursor.fingerprints:
            # Already proposed: any `truncated` note from an earlier scan is
            # stale, not a refusal, and must not survive as one.
            cursor.not_queued.pop(session_id, None)
            return

        payload = {
            "session_id": session_id,
            "fingerprint": fingerprint,
            "trades": batch.closed_trades,
            "pnl": math.fsum(row.pnl for row in batch.rows),
            "r_total": math.fsum(row.r_multiple for row in batch.rows),
            "strategy_hash": strategy_hash,
            "dataset_id": dataset_id,
            "source": source,
            "synthetic": synthetic,
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
        cursor.new_fingerprints.add(fingerprint)
        cursor.not_queued.pop(session_id, None)

    def _fingerprint(
        self,
        batch: OutcomeBatch,
        session_id: str,
        cursor: _Cursor,
    ) -> str | None:
        """Hash the batch; a non-finite value is a refused session, not a crash."""
        try:
            return outcome_fingerprint(batch)
        except (TypeError, ValueError) as exc:
            self._note(cursor, session_id, REASON_OUTCOME_ERROR, str(exc))
            return None

    def _session_count(self, ledger: PaperLedger) -> int:
        sql = "SELECT COUNT(*) FROM sessions"
        params: list[object] = []
        if self.since is not None:
            sql += " WHERE started_at >= ?"
            params.append(_since_stamp(self.since))
        row = ledger.connection.execute(sql, params).fetchone()
        return 0 if row is None else int(row[0])

    def _page_ids(
        self,
        ledger: PaperLedger,
        limit: int,
        offset: int,
    ) -> list[str]:
        """One page of session ids ordered by latest trade activity."""
        clauses: list[str] = []
        params: list[object] = []
        if self.since is not None:
            clauses.append("s.started_at >= ?")
            params.append(_since_stamp(self.since))
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        params.extend([limit, offset])
        rows = ledger.connection.execute(
            """
            SELECT s.session_id AS session_id
            FROM sessions AS s
            LEFT JOIN trades AS t ON t.session_id = s.session_id
            """
            + where
            + """
            GROUP BY s.session_id
            ORDER BY COALESCE(MAX(t.exit_ts), s.started_at) DESC, s.session_id ASC
            LIMIT ? OFFSET ?
            """,
            params,
        ).fetchall()
        return [str(row["session_id"]) for row in rows]

    def _sessions(
        self,
        ledger: PaperLedger,
        total: int,
        start: int,
        limit: int,
    ) -> Iterator[str]:
        """Session ids in activity order, wrapping from the rotation offset."""
        wanted = min(limit, total)
        read = 0
        offset = start % total
        while read < wanted:
            take = min(_PAGE_SIZE, wanted - read, total - offset)
            page = self._page_ids(ledger, take, offset)
            if not page:
                return
            yield from page
            read += len(page)
            offset = (offset + len(page)) % total

    def _record_deferred(
        self,
        cursor: _Cursor,
        ledger: PaperLedger,
        total: int,
        visited: int,
    ) -> None:
        """Audit the sessions this scan did not reach; they are never dropped."""
        deferred = total - visited
        if deferred <= 0:
            return
        detail = (
            f"{deferred} session(s) deferred this scan; the rotation cursor resumes"
            f" at offset {cursor.offset}"
        )
        proposed = set(cursor.fingerprints.values())
        for session_id in self._page_ids(
            ledger, min(deferred, TRUNCATION_NOTE_LIMIT), cursor.offset
        ):
            if session_id in proposed:
                # Its trigger already exists; being outside this scan's window
                # is not an actionable refusal and must not be logged as one.
                continue
            self._note(cursor, session_id, REASON_TRUNCATED, detail)
        self.store.append_event(
            EVENT_TRUNCATED,
            detail=_json(
                {
                    "deferred": deferred,
                    "scanned": visited,
                    "total": total,
                    "resume_offset": cursor.offset,
                }
            ),
        )

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
        pending_ack = parsed.get("pending_ack")
        offset = parsed.get("offset")
        return _Cursor(
            fingerprints=_text_map(fingerprints),
            not_queued=_text_map(not_queued),
            offset=0 if isinstance(offset, bool) or not isinstance(offset, int) else max(offset, 0),
            pending_ack=_text_map(pending_ack),
        )


# --------------------------------------------------------------------- handler


class HandlerRegistry(Protocol):
    """The narrow part of ``Orchestrator`` this module registers into."""

    def register(self, kind: str, handler: Handler) -> None: ...


def make_paper_outcome_handler(
    *,
    should_queue: PaperOutcomePolicy | None = None,
    allow_synthetic: bool = False,
) -> Handler:
    """Build a handler that records the event and proposes the next step.

    It never runs discovery, training or a promotion. ``should_queue`` is the
    policy hook: returning ``True`` proposes ``queue_discovery`` and ``False``
    proposes ``no_action``. Either way the result carries ``queued: false`` and
    ``executed: false`` because this slice has no executor behind the proposal.
    A callback exception propagates as an ordinary handler failure and is
    retried by the queue.

    ``allow_synthetic`` defaults to ``False``. A payload with
    ``synthetic: true`` is then refused with ``no_action`` and one
    ``paper.outcome_refused`` audit line instead of reaching the policy: a demo
    batch is not market evidence. ``synthetic: null`` (unknown) is passed to
    ``should_queue`` unchanged, never upgraded to ``False``.
    """
    if not isinstance(allow_synthetic, bool):
        raise TypeError(f"allow_synthetic must be a bool, got {type(allow_synthetic).__name__}")

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
                    "source": request["source"],
                    "synthetic": request["synthetic"],
                }
            ),
        )
        if request["synthetic"] is True and not allow_synthetic:
            context.log(
                EVENT_REFUSED,
                _json(
                    {
                        "session_id": request["session_id"],
                        "fingerprint": request["fingerprint"],
                        "reason": REFUSAL_SYNTHETIC,
                        "detail": DEFAULT_SYNTHETIC_REASON,
                    }
                ),
            )
            return HandlerResult.success(
                {
                    "action": "no_action",
                    "reason": DEFAULT_SYNTHETIC_REASON,
                    "queued": False,
                    "executed": False,
                    "refused": REFUSAL_SYNTHETIC,
                    "session_id": request["session_id"],
                    "fingerprint": request["fingerprint"],
                    "trades": request["trades"],
                    "pnl": request["pnl"],
                    "r_total": request["r_total"],
                    "strategy_hash": request["strategy_hash"],
                    "dataset_id": request["dataset_id"],
                    "source": request["source"],
                    "synthetic": request["synthetic"],
                }
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
                "source": request["source"],
                "synthetic": request["synthetic"],
            }
        )

    return handler


def register_paper_outcome_handler(
    registry: HandlerRegistry,
    *,
    should_queue: PaperOutcomePolicy | None = None,
    allow_synthetic: bool = False,
) -> str:
    """Register the proposal-only handler for ``paper.outcomes``; returns the kind."""
    registry.register(
        KIND_PAPER_OUTCOMES,
        make_paper_outcome_handler(should_queue=should_queue, allow_synthetic=allow_synthetic),
    )
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
    missing = PAPER_OUTCOME_REQUIRED_FIELDS - set(payload)
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
        # Absent on a job queued before provenance was added: unknown, not real.
        "source": _optional_text(payload.get("source"), "source"),
        "synthetic": _optional_bool(payload.get("synthetic"), "synthetic"),
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


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _PayloadError(f"{name} must be a non-empty string or null")
    return value


def _optional_bool(value: object, name: str) -> bool | None:
    if value is None:
        return None
    if not isinstance(value, bool):
        raise _PayloadError(f"{name} must be a bool or null")
    return value


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _PayloadError(f"{name} must be a number")
    number = float(value)
    if not math.isfinite(number):
        raise _PayloadError(f"{name} must be finite")
    return number


def _coerce_since(value: str | datetime | None) -> datetime | None:
    """Normalise the optional window bound; ``None`` means the whole ledger."""
    if value is None:
        return None
    if isinstance(value, datetime):
        moment = value
    elif isinstance(value, str):
        try:
            moment = datetime.fromisoformat(value)
        except ValueError as exc:
            raise ValueError(f"since is not an ISO-8601 timestamp: {value!r}") from exc
    else:
        raise TypeError(
            f"since must be a datetime, an ISO-8601 string or None, got {type(value).__name__}"
        )
    if moment.tzinfo is None or moment.utcoffset() is None:
        raise ValueError("since must be timezone-aware")
    return moment.astimezone(UTC)


def _since_stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).isoformat()


def _fingerprint_of(trigger: Trigger) -> object:
    payload = trigger.payload
    if not isinstance(payload, Mapping):
        return None
    return payload.get("fingerprint")


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
