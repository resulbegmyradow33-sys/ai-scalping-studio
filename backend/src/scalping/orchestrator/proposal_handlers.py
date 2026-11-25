"""Approved experiment proposals as discovery triggers (knowledge → orchestrator).

The knowledge package records an :class:`~scalping.knowledge.routing.ExperimentProposal`
only after a human calls ``ProposalStore.approve``. This module is the seam that
turns those approved records into ordinary ``discovery.run`` jobs. It is a bridge,
not an executor:

* **Only approved records are read.** ``proposals_for_queue`` returns approved
  proposals oldest first and never changes a status; rejected, proposed and
  converted records never reach this source.
* **A trigger is not an execution.** ``ProposalQueueTriggerSource.scan`` only
  enqueues one queue row per approved proposal. Nothing here calls the discovery
  runner, opens a dataset, reads a bar or writes the discovery archive. The
  orchestrator decides *when* the job runs, under the same gate the payload
  states explicitly.
* **No governance is automated.** The source never approves, rejects or converts
  a proposal. Conversion still requires the explicit
  ``ProposalStore.link_candidate`` call after a discovery run produced a
  candidate hash.
* **The holdout is not here.** This module has no import of
  ``scalping.discovery`` and no path to the final holdout. A proposal whose
  dataset hint names a test/holdout split is refused before a job exists.

The optional ``knowledge.proposals`` handler records a proposal snapshot and
returns the same proposal-only answer as the paper-outcome handler
(``queue_discovery`` with ``queued: false`` and ``executed: false``). A caller
registers it with :func:`register_proposal_handler`; it is deliberately not part
of ``register_default_handlers``, so importing this module does not change the
default queue.

Dedupe has two layers
---------------------
The cursor document in the job store keeps the identities already emitted plus
the last refusal reason per proposal. The trigger's ``dedupe_key`` is
``discovery.run:proposal:<proposal id>:<payload hash>`` as the second layer. A
crash between enqueue and ``seen`` re-emits the same trigger on the next scan and
the store returns the existing job instead of a duplicate. Changing the
discovery template changes the payload hash, which is a new, explicitly
different experiment and therefore a new job.

Bounded read
------------
One scan reads at most ``MAX_SCAN_PROPOSALS`` approved proposals and emits at
most ``max_proposals`` of them. A larger approved backlog is covered by later
scans unless it exceeds the read bound; that limit is documented rather than
hidden behind an unbounded query.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

from scalping.data.registry import SYMBOLS, TIMEFRAMES
from scalping.knowledge.routing import (
    PROPOSAL_STATUSES,
    ExperimentProposal,
    ProposalStore,
    proposals_for_queue,
)
from scalping.orchestrator.core import Handler, HandlerResult, JobContext
from scalping.orchestrator.store import Job, JobStore
from scalping.orchestrator.triggers import Trigger
from scalping.registry.gate import GateCriteria

__all__ = [
    "DEFAULT_MAX_PROPOSALS",
    "DEFAULT_MAX_RETRIES",
    "EVENT_NOT_QUEUED",
    "EVENT_QUEUED",
    "EVENT_RECEIVED",
    "EVENT_TRUNCATED",
    "KIND_DISCOVERY_RUN",
    "KIND_KNOWLEDGE_PROPOSALS",
    "MAX_DISCOVERY_GENERATIONS",
    "MAX_DISCOVERY_POPULATION",
    "MAX_DISCOVERY_TRIALS",
    "MAX_SCAN_PROPOSALS",
    "MAX_SCAN_PROPOSALS_LIMIT",
    "PROPOSAL_MESSAGE_FIELDS",
    "PROPOSAL_MESSAGE_REQUIRED_FIELDS",
    "REASON_HOLDOUT_DATA",
    "REASON_INVALID_PROPOSAL",
    "REASON_MISSING_DATA",
    "REASON_MISSING_SCOPE",
    "REASON_NOT_APPROVED",
    "REASON_TRUNCATED",
    "REASON_UNKNOWN_SCOPE",
    "DiscoveryTemplate",
    "ProposalQueueTriggerSource",
    "discovery_payload_hash",
    "make_proposal_handler",
    "proposal_message",
    "register_proposal_handler",
]

#: The queue kind this source produces. It is the same kind
#: :mod:`scalping.orchestrator.handlers` validates, pinned by a test rather than
#: imported: importing that module would pull the discovery stack into a bridge
#: that must not execute discovery.
KIND_DISCOVERY_RUN: Final[str] = "discovery.run"

#: The optional proposal-recording handler. Not registered by default.
KIND_KNOWLEDGE_PROPOSALS: Final[str] = "knowledge.proposals"

#: Audit events written by the source and the handler.
EVENT_QUEUED: Final[str] = "proposal.discovery_queued"
EVENT_NOT_QUEUED: Final[str] = "proposal.not_queued"
EVENT_RECEIVED: Final[str] = "proposal.received"
EVENT_TRUNCATED: Final[str] = "proposal.truncated"

#: Machine-readable reasons a proposal produced no trigger. Recorded once per
#: proposal and rewritten only when the reason changes.
REASON_NOT_APPROVED: Final[str] = "not_approved"
REASON_MISSING_SCOPE: Final[str] = "missing_scope"
REASON_UNKNOWN_SCOPE: Final[str] = "unknown_scope"
REASON_MISSING_DATA: Final[str] = "missing_data"
REASON_HOLDOUT_DATA: Final[str] = "holdout_data"
REASON_INVALID_PROPOSAL: Final[str] = "invalid_proposal"
REASON_TRUNCATED: Final[str] = "truncated"

#: One scan reads at most this many approved proposals. A larger approved
#: backlog is drained over successive scans until it exceeds this bound.
MAX_SCAN_PROPOSALS: Final[int] = 1_000
#: Refuse an emission cap above the read bound: asking for more than a scan can
#: see is a misconfiguration, not an optimization.
MAX_SCAN_PROPOSALS_LIMIT: Final[int] = 1_000
#: Default triggers per scan, so a first scan over years of proposals cannot
#: enqueue an unbounded research backlog at once.
DEFAULT_MAX_PROPOSALS: Final[int] = 100
#: One retry by default: a transient queue failure must not consume the only
#: trigger for an approval the owner already made.
DEFAULT_MAX_RETRIES: Final[int] = 1

#: Bounds copied from ``scalping.orchestrator.handlers`` so this module can
#: validate a template without importing the discovery stack. A test pins them
#: to the handler's constants.
MAX_DISCOVERY_TRIALS: Final[int] = 2_000
MAX_DISCOVERY_POPULATION: Final[int] = 256
MAX_DISCOVERY_GENERATIONS: Final[int] = 100

#: Tokens that name an untouched test/holdout split in a dataset id. Mirrors
#: the evaluator's guard; this module never opens a dataset, it only refuses to
#: forward a hint that would ask someone else to.
_HOLDOUT_TOKENS: Final[frozenset[str]] = frozenset(
    {"test", "tests", "holdout", "final", "locked", "oos", "unseen", "outofsample"}
)

_CURSOR_VERSION: Final[int] = 1
_DEDUPE_PREFIX: Final[str] = f"{KIND_DISCOVERY_RUN}:proposal:"

#: Fields every proposal message carries. The optional handler requires them.
PROPOSAL_MESSAGE_REQUIRED_FIELDS: Final[frozenset[str]] = frozenset(
    {"proposal_id", "hypothesis_id", "symbol", "timeframe", "status"}
)

#: Every field the optional handler accepts. Unknown fields are rejected so a
#: typo cannot silently drop provenance.
PROPOSAL_MESSAGE_FIELDS: Final[frozenset[str]] = PROPOSAL_MESSAGE_REQUIRED_FIELDS | {
    "approval_actor",
    "approval_reason",
    "created_at",
    "dataset_id",
    "hypothesis_kind",
    "kind_source",
    "payload_hash",
    "required_columns",
    "required_groups",
    "source_refs",
    "updated_at",
}


@dataclass(frozen=True, slots=True)
class DiscoveryTemplate:
    """The search policy every emitted ``discovery.run`` payload states.

    The gate and stress policy are required rather than inherited: a proposal
    authorizes an experiment, it does not choose the evidence policy. ``seed``
    and ``max_trials`` make the run reproducible; ``population``,
    ``generations``, ``patience`` and ``horizon`` are the same bounded knobs the
    discovery handler exposes.
    """

    max_trials: int
    seed: int
    gate: GateCriteria
    stress: tuple[float, ...] = (1.5, 2.0)
    population: int = 16
    generations: int = 5
    patience: int | None = 5
    horizon: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_trials, bool) or not isinstance(self.max_trials, int):
            raise TypeError("max_trials must be an int")
        if not 1 <= self.max_trials <= MAX_DISCOVERY_TRIALS:
            raise ValueError(f"max_trials must be in 1..{MAX_DISCOVERY_TRIALS}")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an int")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if not isinstance(self.gate, GateCriteria):
            raise TypeError("gate must be a GateCriteria")
        if isinstance(self.stress, list):  # caller convenience; store a tuple
            object.__setattr__(self, "stress", tuple(self.stress))
        if not isinstance(self.stress, tuple) or not self.stress:
            raise ValueError("stress must be a non-empty tuple of multipliers")
        for multiplier in self.stress:
            if (
                isinstance(multiplier, bool)
                or not isinstance(multiplier, (int, float))
                or not math.isfinite(float(multiplier))
                or float(multiplier) <= 0.0
            ):
                raise ValueError("stress multipliers must be finite and > 0")
        if isinstance(self.population, bool) or not isinstance(self.population, int):
            raise TypeError("population must be an int")
        if not 2 <= self.population <= MAX_DISCOVERY_POPULATION:
            raise ValueError(f"population must be in 2..{MAX_DISCOVERY_POPULATION}")
        if isinstance(self.generations, bool) or not isinstance(self.generations, int):
            raise TypeError("generations must be an int")
        if not 1 <= self.generations <= MAX_DISCOVERY_GENERATIONS:
            raise ValueError(f"generations must be in 1..{MAX_DISCOVERY_GENERATIONS}")
        if self.patience is not None:
            if isinstance(self.patience, bool) or not isinstance(self.patience, int):
                raise TypeError("patience must be an int or None")
            if self.patience < 1:
                raise ValueError("patience must be >= 1 or None")
        if isinstance(self.horizon, bool) or not isinstance(self.horizon, int):
            raise TypeError("horizon must be an int")
        if self.horizon < 1:
            raise ValueError("horizon must be >= 1")

    def payload(self, *, symbol: str, timeframe: str) -> dict[str, Any]:
        """A payload the current ``discovery.run`` handler accepts unchanged."""
        gate: dict[str, Any] = {
            "min_trades": self.gate.min_trades,
            "min_expectancy_r": self.gate.min_expectancy_r,
            "min_profit_factor": self.gate.min_profit_factor,
            "max_drawdown": self.gate.max_drawdown,
            "max_ece": self.gate.max_ece,
            "min_regime_coverage": self.gate.min_regime_coverage,
        }
        # The handler rejects an empty stress list, so an empty requirement is
        # omitted rather than serialised as ``[]``.
        if self.gate.require_positive_cost_stress:
            gate["require_positive_cost_stress"] = list(self.gate.require_positive_cost_stress)
        return {
            "symbol": symbol,
            "timeframe": timeframe,
            "max_trials": self.max_trials,
            "seed": self.seed,
            "stress": list(self.stress),
            "gate": gate,
            "population": self.population,
            "generations": self.generations,
            "patience": self.patience,
            "horizon": self.horizon,
        }


@dataclass(slots=True)
class _Cursor:
    """The durable part of a scan: emitted identities and quiet refusals.

    ``emitted`` maps ``<proposal id>:<payload hash>`` to the proposal id.
    ``pending_ack`` holds identities emitted by the scan that has not been
    acknowledged yet, so a partial acknowledgement is retried before anything
    else. ``not_queued`` keeps the last refusal reason per proposal.
    """

    emitted: dict[str, str] = field(default_factory=dict)
    not_queued: dict[str, str] = field(default_factory=dict)
    pending_ack: dict[str, str] = field(default_factory=dict)


class ProposalQueueTriggerSource:
    """Turn approved experiment proposals into ``discovery.run`` triggers.

    ``proposals`` is read through
    :func:`~scalping.knowledge.routing.proposals_for_queue`, so only approved
    records are visible and no status is changed. The cursor lives in the job
    store under ``trigger.knowledge.proposals:<proposal store path>`` and
    survives a process restart.

    A proposal is eligible when its status is ``approved``, its symbol and
    timeframe are in the project registry, it still carries a non-empty data
    requirement, and its optional dataset hint (if any) does not name a locked
    test/holdout split. Anything else writes one ``proposal.not_queued`` event
    with a machine-readable reason and emits nothing.

    The source does not import or call discovery. It emits a payload that the
    existing ``discovery.run`` handler validates; whether and when that job runs
    remains the orchestrator's decision.
    """

    def __init__(
        self,
        proposals: ProposalStore,
        *,
        store: JobStore,
        template: DiscoveryTemplate,
        max_proposals: int = DEFAULT_MAX_PROPOSALS,
        priority: int = 0,
        max_retries: int = DEFAULT_MAX_RETRIES,
        name: str | None = None,
    ) -> None:
        if not isinstance(proposals, ProposalStore):
            raise TypeError("proposals must be a ProposalStore")
        if not isinstance(store, JobStore):
            raise TypeError("store must be a JobStore")
        if not isinstance(template, DiscoveryTemplate):
            raise TypeError("template must be a DiscoveryTemplate")
        if isinstance(max_proposals, bool) or not isinstance(max_proposals, int):
            raise TypeError("max_proposals must be an int")
        if not 1 <= max_proposals <= MAX_SCAN_PROPOSALS_LIMIT:
            raise ValueError(f"max_proposals must be in 1..{MAX_SCAN_PROPOSALS_LIMIT}")
        if isinstance(max_retries, bool) or not isinstance(max_retries, int):
            raise TypeError("max_retries must be an int")
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.proposals = proposals
        self.store = store
        self.template = template
        self.max_proposals = max_proposals
        self.priority = int(priority)
        self.max_retries = max_retries
        self.name = name or f"proposals:{self._proposal_path}"
        self.cursor_key = f"trigger.knowledge.proposals:{self._proposal_path}"
        self._pending: _Cursor | None = None
        self._identity_by_dedupe: dict[str, str] = {}

    # ------------------------------------------------------------------ protocol

    def scan(self) -> list[Trigger]:
        """Read the approved queue and return triggers not yet acknowledged."""
        cursor = self._load_cursor()
        self._identity_by_dedupe = {}
        self._pending = cursor
        triggers: list[Trigger] = []

        approved = proposals_for_queue(self.proposals, MAX_SCAN_PROPOSALS)
        for index, proposal in enumerate(approved):
            if len(triggers) >= self.max_proposals:
                self._record_deferred(len(approved) - index)
                break
            trigger = self._process(proposal, cursor)
            if trigger is not None:
                triggers.append(trigger)
        return triggers

    def seen(self, triggers: Sequence[Trigger]) -> None:
        """Persist the identities the caller acknowledged, not the whole scan.

        ``scan_once`` passes every trigger it enqueued. A caller that
        acknowledges only a subset leaves the rest in ``pending_ack`` and the
        next scan re-emits them; the store key still prevents a second job for a
        trigger that was in fact enqueued. Refusal reasons carry no work and are
        persisted whole.
        """
        pending = self._pending
        if pending is None:
            return
        acknowledged: set[str] = set()
        for trigger in triggers:
            identity = self._identity_by_dedupe.get(trigger.dedupe_key)
            if identity is None:
                identity = _identity_from_dedupe(trigger.dedupe_key)
            if identity is not None:
                acknowledged.add(identity)

        for identity in acknowledged:
            proposal_id = pending.pending_ack.pop(identity, None)
            if proposal_id is not None:
                pending.emitted[identity] = proposal_id

        document = {
            "version": _CURSOR_VERSION,
            "emitted": dict(sorted(pending.emitted.items())),
            "not_queued": dict(sorted(pending.not_queued.items())),
            "pending_ack": dict(sorted(pending.pending_ack.items())),
        }
        self.store.set_state(
            self.cursor_key,
            json.dumps(document, sort_keys=True, separators=(",", ":")),
        )
        self._pending = None

    # ------------------------------------------------------------------ internals

    @property
    def _proposal_path(self) -> str:
        path = self.proposals.path
        if path == ":memory:":
            return path
        return str(Path(path).resolve())

    def _process(self, proposal: object, cursor: _Cursor) -> Trigger | None:
        if not isinstance(proposal, ExperimentProposal):
            self._note(cursor, "", REASON_INVALID_PROPOSAL, "queue returned a non-proposal record")
            return None
        if proposal.status != "approved":
            self._note(
                cursor,
                proposal.proposal_id,
                REASON_NOT_APPROVED,
                f"status is {proposal.status!r}; only approved proposals may run",
            )
            return None

        symbol = _cleaned(proposal.symbol).upper()
        timeframe = _cleaned(proposal.timeframe).upper()
        if not symbol or not timeframe:
            self._note(
                cursor,
                proposal.proposal_id,
                REASON_MISSING_SCOPE,
                "proposal records no symbol/timeframe scope",
            )
            return None
        if symbol not in SYMBOLS or timeframe not in TIMEFRAMES:
            self._note(
                cursor,
                proposal.proposal_id,
                REASON_UNKNOWN_SCOPE,
                f"scope {symbol}/{timeframe} is not in the project registry",
            )
            return None

        if not proposal.required_columns or not proposal.required_groups:
            self._note(
                cursor,
                proposal.proposal_id,
                REASON_MISSING_DATA,
                "proposal records no required bar columns or feature groups",
            )
            return None
        dataset_hint = proposal.dataset_id
        if dataset_hint is not None:
            if not isinstance(dataset_hint, str) or not dataset_hint.strip():
                self._note(
                    cursor,
                    proposal.proposal_id,
                    REASON_MISSING_DATA,
                    "proposal records an empty dataset hint",
                )
                return None
            dataset_hint = dataset_hint.strip()
            if _names_holdout(dataset_hint):
                self._note(
                    cursor,
                    proposal.proposal_id,
                    REASON_HOLDOUT_DATA,
                    f"dataset hint {dataset_hint!r} names a locked test/holdout split",
                )
                return None

        try:
            payload = self.template.payload(symbol=symbol, timeframe=timeframe)
            payload_hash = discovery_payload_hash(payload)
        except (TypeError, ValueError) as exc:
            self._note(
                cursor,
                proposal.proposal_id,
                REASON_INVALID_PROPOSAL,
                f"cannot build a discovery payload: {exc}",
            )
            return None

        identity = f"{proposal.proposal_id}:{payload_hash}"
        if identity in cursor.emitted or identity in cursor.pending_ack:
            cursor.not_queued.pop(proposal.proposal_id, None)
            return None

        dedupe_key = f"{_DEDUPE_PREFIX}{identity}"
        cursor.pending_ack[identity] = proposal.proposal_id
        self._identity_by_dedupe[dedupe_key] = identity
        cursor.not_queued.pop(proposal.proposal_id, None)
        self.store.append_event(
            EVENT_QUEUED,
            detail=_json(
                {
                    "proposal_id": proposal.proposal_id,
                    "hypothesis_id": proposal.hypothesis_id,
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "status": proposal.status,
                    "dataset_id": dataset_hint,
                    "required_columns": list(proposal.required_columns),
                    "required_groups": list(proposal.required_groups),
                    "payload_hash": payload_hash,
                    "dedupe_key": dedupe_key,
                    "approval_actor": None
                    if proposal.approval is None
                    else proposal.approval.actor,
                    "approval_reason": (
                        None if proposal.approval is None else proposal.approval.reason
                    ),
                }
            ),
        )
        return Trigger(
            kind=KIND_DISCOVERY_RUN,
            dedupe_key=dedupe_key,
            payload=payload,
            priority=self.priority,
            max_retries=self.max_retries,
        )

    def _record_deferred(self, deferred: int) -> None:
        self.store.append_event(
            EVENT_TRUNCATED,
            detail=_json(
                {
                    "deferred": deferred,
                    "reason": REASON_TRUNCATED,
                    "detail": (
                        f"{deferred} approved proposal(s) not read this scan;"
                        " the next scan resumes at the oldest unacknowledged record"
                    ),
                }
            ),
        )

    def _note(self, cursor: _Cursor, proposal_id: str, reason: str, detail: str) -> None:
        """Record a refusal once per proposal until its reason changes."""
        if cursor.not_queued.get(proposal_id) == reason:
            return
        cursor.not_queued[proposal_id] = reason
        self.store.append_event(
            EVENT_NOT_QUEUED,
            detail=_json(
                {
                    "proposal_id": proposal_id or None,
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
        return _Cursor(
            emitted=_text_map(parsed.get("emitted")),
            not_queued=_text_map(parsed.get("not_queued")),
            pending_ack=_text_map(parsed.get("pending_ack")),
        )


# --------------------------------------------------------------------- handler


class HandlerRegistry(Protocol):
    """The narrow part of ``Orchestrator`` this module registers into."""

    def register(self, kind: str, handler: Handler) -> None: ...


def proposal_message(
    proposal: ExperimentProposal,
    *,
    payload_hash: str | None = None,
) -> dict[str, Any]:
    """The payload :func:`make_proposal_handler` accepts, built from a record.

    This is a snapshot, not a reference: the handler never opens the proposal
    store and never re-reads governance state. Callers that want a fresh status
    should build the message again from the current record.
    """
    if not isinstance(proposal, ExperimentProposal):
        raise TypeError("proposal must be an ExperimentProposal")
    approval = proposal.approval
    return {
        "proposal_id": proposal.proposal_id,
        "hypothesis_id": proposal.hypothesis_id,
        "status": proposal.status,
        "symbol": proposal.symbol,
        "timeframe": proposal.timeframe,
        "hypothesis_kind": proposal.hypothesis_kind,
        "kind_source": proposal.kind_source,
        "dataset_id": proposal.dataset_id,
        "payload_hash": payload_hash,
        "approval_actor": None if approval is None else approval.actor,
        "approval_reason": None if approval is None else approval.reason,
        "created_at": proposal.created_at,
        "updated_at": proposal.updated_at,
        "source_refs": [ref.to_dict() for ref in proposal.source_refs],
        "required_columns": list(proposal.required_columns),
        "required_groups": list(proposal.required_groups),
    }


def make_proposal_handler() -> Handler:
    """Build the optional ``knowledge.proposals`` handler.

    It validates the snapshot, writes one ``proposal.received`` audit line and
    returns a proposal-only answer. It does not import discovery, read a bar,
    open a holdout, approve a proposal or enqueue work. ``JobContext.store`` is
    the read-only facade, so it could not enqueue even if it tried.
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
                    "proposal_id": request["proposal_id"],
                    "hypothesis_id": request["hypothesis_id"],
                    "status": request["status"],
                    "symbol": request["symbol"],
                    "timeframe": request["timeframe"],
                    "dataset_id": request["dataset_id"],
                    "payload_hash": request["payload_hash"],
                }
            ),
        )

        if request["status"] != "approved":
            action = "no_action"
            reason = "only an approved proposal may propose a discovery run"
        elif request["dataset_id"] is not None and _names_holdout(request["dataset_id"]):
            action = "no_action"
            reason = "dataset hint names a locked test/holdout split"
        else:
            action = "queue_discovery"
            reason = "approved proposal is eligible for a discovery run"

        data = dict(request)
        data.update(
            {
                "action": action,
                "reason": reason,
                "queued": False,
                "executed": False,
            }
        )
        return HandlerResult.success(data)

    return handler


def register_proposal_handler(registry: HandlerRegistry) -> str:
    """Register the optional ``knowledge.proposals`` handler; returns its kind."""
    registry.register(KIND_KNOWLEDGE_PROPOSALS, make_proposal_handler())
    return KIND_KNOWLEDGE_PROPOSALS


# --------------------------------------------------------------------- helpers


class _PayloadError(ValueError):
    """A ``knowledge.proposals`` payload is invalid; a retry cannot change it."""


def _validated(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise _PayloadError(f"payload must be a mapping, got {type(payload).__name__}")
    unknown = set(payload) - PROPOSAL_MESSAGE_FIELDS
    if unknown:
        raise _PayloadError(f"unknown payload fields: {sorted(unknown)}")
    missing = PROPOSAL_MESSAGE_REQUIRED_FIELDS - set(payload)
    if missing:
        raise _PayloadError(f"missing payload fields: {sorted(missing)}")

    status = _text(payload["status"], "status")
    if status not in PROPOSAL_STATUSES:
        raise _PayloadError(f"status must be one of {sorted(PROPOSAL_STATUSES)}, got {status!r}")
    symbol = _text(payload["symbol"], "symbol").upper()
    timeframe = _text(payload["timeframe"], "timeframe").upper()
    if symbol not in SYMBOLS:
        raise _PayloadError(f"symbol {symbol!r} is not in the project registry")
    if timeframe not in TIMEFRAMES:
        raise _PayloadError(f"timeframe {timeframe!r} is not in the project registry")

    return {
        "proposal_id": _text(payload["proposal_id"], "proposal_id"),
        "hypothesis_id": _text(payload["hypothesis_id"], "hypothesis_id"),
        "status": status,
        "symbol": symbol,
        "timeframe": timeframe,
        "hypothesis_kind": _optional_text(payload.get("hypothesis_kind"), "hypothesis_kind"),
        "kind_source": _optional_text(payload.get("kind_source"), "kind_source"),
        "dataset_id": _optional_text(payload.get("dataset_id"), "dataset_id"),
        "payload_hash": _optional_text(payload.get("payload_hash"), "payload_hash"),
        "approval_actor": _optional_text(payload.get("approval_actor"), "approval_actor"),
        "approval_reason": _optional_text(payload.get("approval_reason"), "approval_reason"),
        "created_at": _optional_text(payload.get("created_at"), "created_at"),
        "updated_at": _optional_text(payload.get("updated_at"), "updated_at"),
        "source_refs": _source_refs(payload.get("source_refs")),
        "required_columns": _string_list(payload.get("required_columns"), "required_columns"),
        "required_groups": _string_list(payload.get("required_groups"), "required_groups"),
    }


def _text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _PayloadError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_text(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise _PayloadError(f"{name} must be a non-empty string or null")
    return value.strip()


def _string_list(value: object, name: str) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, list):
        raise _PayloadError(f"{name} must be a list of strings")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise _PayloadError(f"{name} must contain non-empty strings")
        result.append(item.strip())
    return result


def _source_refs(value: object) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise _PayloadError("source_refs must be a list of objects")
    refs: list[dict[str, Any]] = []
    for index, item in enumerate(value):
        if not isinstance(item, Mapping):
            raise _PayloadError(f"source_refs[{index}] must be an object")
        unknown = set(item) - {"doc_id", "section_ordinal", "quote"}
        if unknown:
            raise _PayloadError(f"source_refs[{index}] has unknown fields: {sorted(unknown)}")
        ordinal = item.get("section_ordinal")
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 0:
            raise _PayloadError(f"source_refs[{index}].section_ordinal must be an int >= 0")
        refs.append(
            {
                "doc_id": _text(item.get("doc_id"), f"source_refs[{index}].doc_id"),
                "section_ordinal": ordinal,
                "quote": _text(item.get("quote"), f"source_refs[{index}].quote"),
            }
        )
    return refs


def discovery_payload_hash(payload: Mapping[str, Any]) -> str:
    """Lowercase sha256 of the canonical payload; the experiment's identity."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json(payload: Mapping[str, Any]) -> str:
    return _canonical_json(payload)


def _cleaned(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _names_holdout(dataset_id: str) -> bool:
    tokens = re.split(r"[^a-z0-9]+", dataset_id.casefold())
    return any(token in _HOLDOUT_TOKENS for token in tokens if token)


def _identity_from_dedupe(dedupe_key: str) -> str | None:
    if not dedupe_key.startswith(_DEDUPE_PREFIX):
        return None
    identity = dedupe_key[len(_DEDUPE_PREFIX) :]
    return identity or None


def _text_map(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        return {}
    return {
        str(key): str(item)
        for key, item in value.items()
        if isinstance(key, str) and isinstance(item, str)
    }
