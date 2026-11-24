"""Hypothesis to experiment-proposal routing (Phase 17, P4 slice 3).

A proposal is a **record**, not an execution instruction. It restates one
hypothesis as the scope and data a quantitative experiment would need, keeps the
original citations, and waits for an explicit human approval. Nothing here
builds a DSL AST, evaluates a candidate, opens a dataset, calls a broker or
touches an order path: the suggested strategy skeleton is stored as text and is
never parsed by this module.

The flow is deliberately narrow::

    hypothesis (proposed/testing, with sources)
        -> ExperimentProposal (status="proposed")
        -> approve(reason, actor)  or  reject(reason, actor)
        -> approved proposal enters the orchestrator queue
        -> link_candidate(candidate_hash, dataset_id) marks it "converted"

"Converted" means a discovery run represented by ``candidate_hash`` on
``dataset_id`` was linked to the proposal. It does not mean the idea was
profitable, validated or paper-traded.

Proposal ids are derived from the immutable proposal payload, so the same
hypothesis, scope and skeleton always produce the same id. Status, approval and
conversion are mutable governance fields and are not part of the hash.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, Self

from scalping.data.registry import SYMBOLS, TIMEFRAMES
from scalping.knowledge.classify import classify_sentence
from scalping.knowledge.hypothesis import Hypothesis, SourceRef
from scalping.knowledge.index import KnowledgeIndex, KnowledgeIndexError

__all__ = [
    "APPROVAL_STATUSES",
    "PROPOSAL_BAR_COLUMNS",
    "PROPOSAL_GROUPS",
    "PROPOSAL_SCHEMA_VERSION",
    "PROPOSAL_STATUSES",
    "ROUTING_ERROR_CODES",
    "Approval",
    "ExperimentProposal",
    "ProposalStore",
    "RoutingError",
    "canonical_proposal_id",
    "proposals_for_queue",
    "propose_from_hypothesis",
]

#: Every status an experiment proposal may hold.
PROPOSAL_STATUSES: frozenset[str] = frozenset({"proposed", "approved", "rejected", "converted"})

#: Statuses that carry a recorded human decision.
APPROVAL_STATUSES: frozenset[str] = frozenset({"approved", "rejected", "converted"})

#: Machine-readable refusal codes. ``RoutingError.code`` is always one of these.
ROUTING_ERROR_CODES: frozenset[str] = frozenset(
    {
        "missing_source",
        "bad_status",
        "unknown_scope",
        "already_converted",
        "not_approved",
        "unknown_proposal",
        "unsupported_kind",
        "invalid_proposal",
    }
)

#: Raw bar columns a proposal's dataset must provide. Costs need the spread.
PROPOSAL_BAR_COLUMNS: Final[tuple[str, ...]] = (
    "open",
    "high",
    "low",
    "close",
    "spread",
    "volume",
)

#: Feature groups a proposal may name. Kept as text so this module does not
#: import the feature builder (and therefore numpy/polars) just to record a
#: requirement. A test pins these names to the live feature registry.
PROPOSAL_GROUPS: Final[tuple[str, ...]] = (
    "price",
    "momentum",
    "volatility",
    "volume",
    "structure",
    "session",
    "multi_timeframe",
)

_BASE_GROUPS: Final[tuple[str, ...]] = ("price", "volatility")

#: Deterministic keyword -> feature-group routing. A match only records that the
#: experiment will need the group; it is not an interpretation of the prose.
_GROUP_TERMS: Final[dict[str, tuple[str, ...]]] = {
    "price": ("breakout", "high", "low", "range", "retrace", "swing", "wick"),
    "momentum": ("momentum", "trend", "rsi", "macd", "ema", "moving average"),
    "volatility": ("volatility", "atr", "compression", "expansion", "spread"),
    "volume": ("volume", "order flow", "flow", "imbalance"),
    "structure": ("structure", "liquidity", "sweep", "order block", "fvg", "fair value", "bos"),
    "session": ("session", "asia", "london", "new york", "ny", "vwap", "time of day"),
    "multi_timeframe": (
        "multi-timeframe",
        "multi timeframe",
        "higher timeframe",
        "htf",
        "m5",
        "m15",
    ),
}

PROPOSAL_SCHEMA_VERSION: int = 1

_WAL_ATTEMPTS = 5
_WAL_BACKOFF_SECONDS = 0.05
_HASH_RE: Final[re.Pattern[str]] = re.compile(r"^[0-9a-f]{64}$")

_SCHEMA_VERSION_DDL = """
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL
)
"""

_MIGRATIONS: dict[int, tuple[str, ...]] = {
    1: (
        """
        CREATE TABLE IF NOT EXISTS proposals (
            proposal_id TEXT PRIMARY KEY,
            hypothesis_id TEXT NOT NULL,
            symbol TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            hypothesis_kind TEXT NOT NULL
                CHECK (hypothesis_kind IN ('claim', 'observation')),
            required_columns_json TEXT NOT NULL DEFAULT '[]',
            required_groups_json TEXT NOT NULL DEFAULT '[]',
            dsl_observation TEXT NOT NULL,
            dsl_condition TEXT NOT NULL,
            dsl_expected_effect TEXT NOT NULL,
            falsification_test TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('proposed', 'approved', 'rejected', 'converted')),
            approval_actor TEXT,
            approval_reason TEXT,
            approval_at TEXT,
            converted_candidate_hash TEXT,
            dataset_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            CHECK (
                (status = 'proposed'
                    AND approval_actor IS NULL
                    AND approval_reason IS NULL
                    AND approval_at IS NULL)
                OR (status IN ('approved', 'rejected', 'converted')
                    AND approval_actor IS NOT NULL
                    AND approval_reason IS NOT NULL
                    AND approval_at IS NOT NULL)
            ),
            CHECK (
                (status = 'converted'
                    AND converted_candidate_hash IS NOT NULL
                    AND dataset_id IS NOT NULL)
                OR (status <> 'converted'
                    AND converted_candidate_hash IS NULL
                    AND dataset_id IS NULL)
            )
        )
        """,
        "CREATE INDEX IF NOT EXISTS proposals_status_idx ON proposals(status, created_at)",
        (
            "CREATE INDEX IF NOT EXISTS proposals_hypothesis_idx"
            " ON proposals(hypothesis_id, created_at)"
        ),
        """
        CREATE TABLE IF NOT EXISTS proposal_sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposal_id TEXT NOT NULL
                REFERENCES proposals(proposal_id) ON DELETE CASCADE,
            position INTEGER NOT NULL CHECK (position >= 0),
            doc_id TEXT NOT NULL,
            section_ordinal INTEGER NOT NULL CHECK (section_ordinal >= 0),
            quote TEXT NOT NULL,
            UNIQUE (proposal_id, position)
        )
        """,
        (
            "CREATE INDEX IF NOT EXISTS proposal_sources_proposal_idx"
            " ON proposal_sources(proposal_id, position)"
        ),
    )
}


class RoutingError(KnowledgeIndexError):
    """A proposal is invalid or the requested governance move is not allowed."""

    def __init__(self, code: str, message: str) -> None:
        if code not in ROUTING_ERROR_CODES:
            raise ValueError(f"unknown routing error code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True, slots=True)
class Approval:
    """The recorded human gate decision for one proposal."""

    actor: str
    reason: str
    decided_at: str

    @property
    def time(self) -> str:
        """Alias for the decision timestamp."""
        return self.decided_at

    def to_dict(self) -> dict[str, str]:
        return {
            "actor": self.actor,
            "reason": self.reason,
            "decided_at": self.decided_at,
        }


@dataclass(frozen=True, slots=True)
class ExperimentProposal:
    """A hypothesis restated as an experiment scope, never as an order.

    ``dsl_observation``/``dsl_condition``/``dsl_expected_effect`` are a
    suggested skeleton **as text**. They are copied from the hypothesis and are
    not compiled here; a discovery slice that wants a DSL candidate must build
    and validate its own typed nodes from these hints.
    """

    proposal_id: str
    hypothesis_id: str
    source_refs: tuple[SourceRef, ...]
    symbol: str
    timeframe: str
    hypothesis_kind: str
    required_columns: tuple[str, ...]
    required_groups: tuple[str, ...]
    dsl_observation: str
    dsl_condition: str
    dsl_expected_effect: str
    falsification_test: str
    status: str
    approval: Approval | None
    converted_candidate_hash: str | None
    dataset_id: str | None
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "hypothesis_id": self.hypothesis_id,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "hypothesis_kind": self.hypothesis_kind,
            "required_columns": list(self.required_columns),
            "required_groups": list(self.required_groups),
            "dsl_observation": self.dsl_observation,
            "dsl_condition": self.dsl_condition,
            "dsl_expected_effect": self.dsl_expected_effect,
            "falsification_test": self.falsification_test,
            "status": self.status,
            "approval": None if self.approval is None else self.approval.to_dict(),
            "converted_candidate_hash": self.converted_candidate_hash,
            "dataset_id": self.dataset_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a str, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise ValueError(f"{field} must not be empty")
    return text


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _payload(
    *,
    hypothesis_id: str,
    source_refs: tuple[SourceRef, ...],
    symbol: str,
    timeframe: str,
    hypothesis_kind: str,
    required_columns: tuple[str, ...],
    required_groups: tuple[str, ...],
    dsl_observation: str,
    dsl_condition: str,
    dsl_expected_effect: str,
    falsification_test: str,
) -> dict[str, Any]:
    return {
        "schema": "scalping.knowledge.experiment_proposal.v1",
        "hypothesis_id": hypothesis_id,
        "source_refs": [
            {
                "doc_id": ref.doc_id,
                "section_ordinal": ref.section_ordinal,
                "quote": ref.quote,
            }
            for ref in source_refs
        ],
        "symbol": symbol,
        "timeframe": timeframe,
        "hypothesis_kind": hypothesis_kind,
        "required_data": {
            "columns": list(required_columns),
            "groups": list(required_groups),
        },
        "dsl": {
            "observation": dsl_observation,
            "condition": dsl_condition,
            "expected_effect": dsl_expected_effect,
        },
        "falsification_test": falsification_test,
    }


def canonical_proposal_id(
    *,
    hypothesis_id: str,
    source_refs: tuple[SourceRef, ...],
    symbol: str,
    timeframe: str,
    hypothesis_kind: str,
    required_columns: tuple[str, ...],
    required_groups: tuple[str, ...],
    dsl_observation: str,
    dsl_condition: str,
    dsl_expected_effect: str,
    falsification_test: str,
) -> str:
    """Lowercase sha256 of the immutable proposal payload.

    Status, approval and conversion fields are excluded: the id names the
    experiment scope, while governance changes are recorded separately.
    """
    payload = _payload(
        hypothesis_id=hypothesis_id,
        source_refs=source_refs,
        symbol=symbol,
        timeframe=timeframe,
        hypothesis_kind=hypothesis_kind,
        required_columns=required_columns,
        required_groups=required_groups,
        dsl_observation=dsl_observation,
        dsl_condition=dsl_condition,
        dsl_expected_effect=dsl_expected_effect,
        falsification_test=falsification_test,
    )
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _required_groups(text: str) -> tuple[str, ...]:
    haystack = text.casefold()
    selected = set(_BASE_GROUPS)
    for group in PROPOSAL_GROUPS:
        if any(term in haystack for term in _GROUP_TERMS[group]):
            selected.add(group)
    return tuple(group for group in PROPOSAL_GROUPS if group in selected)


def _check_sources(index: KnowledgeIndex, refs: tuple[SourceRef, ...]) -> None:
    conn = index.connection
    seen: set[tuple[str, int, str]] = set()
    for ref in refs:
        key = (ref.doc_id, ref.section_ordinal, ref.quote)
        if key in seen:
            continue
        seen.add(key)
        document = conn.execute(
            "SELECT status FROM documents WHERE doc_id = ?", (ref.doc_id,)
        ).fetchone()
        if document is None:
            raise RoutingError("missing_source", f"unknown source document {ref.doc_id!r}")
        if str(document["status"]) != "indexed":
            raise RoutingError(
                "missing_source",
                f"source document {ref.doc_id!r} has status {document['status']!r}, not 'indexed'",
            )
        section = conn.execute(
            "SELECT 1 FROM sections WHERE doc_id = ? AND ordinal = ?",
            (ref.doc_id, ref.section_ordinal),
        ).fetchone()
        if section is None:
            raise RoutingError(
                "missing_source",
                f"source document {ref.doc_id!r} has no section {ref.section_ordinal}",
            )


def propose_from_hypothesis(
    hypothesis: Hypothesis,
    *,
    symbol: str,
    timeframe: str,
    index: KnowledgeIndex,
    hypothesis_kind: str | None = None,
) -> ExperimentProposal:
    """Build a ``proposed`` experiment record from one hypothesis.

    The hypothesis must still be ``proposed`` or ``testing`` and must carry at
    least one citation that resolves in ``index``. Supported/rejected ideas are
    history: re-testing them requires a new hypothesis. ``symbol`` and
    ``timeframe`` are normalised and checked against the project registry.

    Nothing is persisted here; pass the record to :meth:`ProposalStore.create`.
    No DSL node, order or dataset is created.
    """
    if not isinstance(hypothesis, Hypothesis):
        raise TypeError("hypothesis must be a Hypothesis")
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")

    if hypothesis.status not in {"proposed", "testing"}:
        raise RoutingError(
            "bad_status",
            f"hypothesis {hypothesis.hypothesis_id!r} is {hypothesis.status!r};"
            " only proposed or testing hypotheses can be routed",
        )
    if not hypothesis.source_refs:
        raise RoutingError(
            "missing_source",
            f"hypothesis {hypothesis.hypothesis_id!r} has no source refs",
        )
    _check_sources(index, hypothesis.source_refs)

    normalized_symbol = _text(symbol, "symbol").upper()
    if normalized_symbol not in SYMBOLS:
        raise RoutingError(
            "unknown_scope",
            f"unknown symbol {symbol!r}; available: {', '.join(sorted(SYMBOLS))}",
        )
    normalized_timeframe = _text(timeframe, "timeframe").upper()
    if normalized_timeframe not in TIMEFRAMES:
        raise RoutingError(
            "unknown_scope",
            f"unknown timeframe {timeframe!r}; available: {', '.join(TIMEFRAMES)}",
        )

    if hypothesis_kind is None:
        derived = classify_sentence(hypothesis.statement)
        if derived not in {"claim", "observation"}:
            raise RoutingError(
                "unsupported_kind",
                "hypothesis statement is not a claim or observation;"
                " pass hypothesis_kind explicitly only for an imported record",
            )
        kind: str = derived
    else:
        if hypothesis_kind not in {"claim", "observation"}:
            raise RoutingError(
                "unsupported_kind",
                f"hypothesis_kind must be 'claim' or 'observation', got {hypothesis_kind!r}",
            )
        kind = hypothesis_kind

    dsl_observation = _text(hypothesis.observed, "observed")
    dsl_condition = _text(hypothesis.proposed_condition, "proposed_condition")
    dsl_expected_effect = _text(hypothesis.expected_effect, "expected_effect")
    falsification_test = _text(hypothesis.falsification_test, "falsification_test")
    required_groups = _required_groups(
        f"{hypothesis.statement} {dsl_observation} {dsl_condition} {dsl_expected_effect}"
    )
    required_columns = PROPOSAL_BAR_COLUMNS
    source_refs = tuple(hypothesis.source_refs)
    proposal_id = canonical_proposal_id(
        hypothesis_id=hypothesis.hypothesis_id,
        source_refs=source_refs,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        hypothesis_kind=kind,
        required_columns=required_columns,
        required_groups=required_groups,
        dsl_observation=dsl_observation,
        dsl_condition=dsl_condition,
        dsl_expected_effect=dsl_expected_effect,
        falsification_test=falsification_test,
    )
    stamp = _now()
    return ExperimentProposal(
        proposal_id=proposal_id,
        hypothesis_id=hypothesis.hypothesis_id,
        source_refs=source_refs,
        symbol=normalized_symbol,
        timeframe=normalized_timeframe,
        hypothesis_kind=kind,
        required_columns=required_columns,
        required_groups=required_groups,
        dsl_observation=dsl_observation,
        dsl_condition=dsl_condition,
        dsl_expected_effect=dsl_expected_effect,
        falsification_test=falsification_test,
        status="proposed",
        approval=None,
        converted_candidate_hash=None,
        dataset_id=None,
        created_at=stamp,
        updated_at=stamp,
    )


def _enable_wal(conn: sqlite3.Connection) -> None:
    row = conn.execute("PRAGMA journal_mode").fetchone()
    if row is not None and str(row[0]).lower() == "wal":
        return
    for attempt in range(1, _WAL_ATTEMPTS + 1):
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            return
        except sqlite3.OperationalError:
            if attempt == _WAL_ATTEMPTS:
                raise
            time.sleep(_WAL_BACKOFF_SECONDS * attempt)


def _parse_strings(raw: str, field: str) -> tuple[str, ...]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise KnowledgeIndexError(f"stored {field} are not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise KnowledgeIndexError(f"stored {field} must be a JSON array of strings")
    return tuple(data)


def _load(store: ProposalStore, row: sqlite3.Row) -> ExperimentProposal:
    proposal_id = str(row["proposal_id"])
    source_rows = store.connection.execute(
        """
        SELECT doc_id, section_ordinal, quote
        FROM proposal_sources
        WHERE proposal_id = ?
        ORDER BY position
        """,
        (proposal_id,),
    ).fetchall()
    approval: Approval | None = None
    if row["approval_actor"] is not None:
        approval = Approval(
            actor=str(row["approval_actor"]),
            reason=str(row["approval_reason"]),
            decided_at=str(row["approval_at"]),
        )
    return ExperimentProposal(
        proposal_id=proposal_id,
        hypothesis_id=str(row["hypothesis_id"]),
        source_refs=tuple(
            SourceRef(
                doc_id=str(source["doc_id"]),
                section_ordinal=int(source["section_ordinal"]),
                quote=str(source["quote"]),
            )
            for source in source_rows
        ),
        symbol=str(row["symbol"]),
        timeframe=str(row["timeframe"]),
        hypothesis_kind=str(row["hypothesis_kind"]),
        required_columns=_parse_strings(str(row["required_columns_json"]), "required columns"),
        required_groups=_parse_strings(str(row["required_groups_json"]), "required groups"),
        dsl_observation=str(row["dsl_observation"]),
        dsl_condition=str(row["dsl_condition"]),
        dsl_expected_effect=str(row["dsl_expected_effect"]),
        falsification_test=str(row["falsification_test"]),
        status=str(row["status"]),
        approval=approval,
        converted_candidate_hash=(
            None
            if row["converted_candidate_hash"] is None
            else str(row["converted_candidate_hash"])
        ),
        dataset_id=None if row["dataset_id"] is None else str(row["dataset_id"]),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


class ProposalStore:
    """One local SQLite file holding experiment proposals and their audit trail.

    Conventions match the knowledge index: WAL, foreign keys, a versioned
    ``schema_version`` table and ``BEGIN IMMEDIATE`` writes. Not thread-safe;
    one connection belongs to the thread that opened it.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = sqlite3.connect(
            self.path,
            isolation_level=None,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA synchronous = NORMAL")
        try:
            _enable_wal(self._conn)
            self.init_schema()
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ lifecycle

    @property
    def connection(self) -> sqlite3.Connection:
        return self._connection()

    def _connection(self) -> sqlite3.Connection:
        if self._conn is None:
            raise KnowledgeIndexError("proposal store is closed")
        return self._conn

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        conn = self._connection()
        if conn.in_transaction:
            raise KnowledgeIndexError("nested proposal store transaction")
        try:
            conn.execute("BEGIN IMMEDIATE")
        except sqlite3.OperationalError as exc:
            raise KnowledgeIndexError(f"could not acquire the proposal write lock: {exc}") from exc
        try:
            yield conn
        except BaseException:
            conn.rollback()
            raise
        else:
            conn.commit()

    def _read_schema_version(self) -> int:
        try:
            row = self._connection().execute("SELECT MAX(version) FROM schema_version").fetchone()
        except sqlite3.OperationalError as exc:
            if "no such table" not in str(exc):
                raise
            return 0
        return 0 if row is None or row[0] is None else int(row[0])

    def init_schema(self) -> int:
        current = self._read_schema_version()
        if current > PROPOSAL_SCHEMA_VERSION:
            raise KnowledgeIndexError(
                f"proposal schema version {current} is newer than supported"
                f" {PROPOSAL_SCHEMA_VERSION}"
            )
        if current == PROPOSAL_SCHEMA_VERSION:
            return current
        with self.transaction() as tx:
            tx.execute(_SCHEMA_VERSION_DDL)
            row = tx.execute("SELECT MAX(version) FROM schema_version").fetchone()
            current = 0 if row is None or row[0] is None else int(row[0])
            if current > PROPOSAL_SCHEMA_VERSION:
                raise KnowledgeIndexError(
                    f"proposal schema version {current} is newer than supported"
                    f" {PROPOSAL_SCHEMA_VERSION}"
                )
            for version in sorted(_MIGRATIONS):
                if version <= current:
                    continue
                for statement in _MIGRATIONS[version]:
                    tx.execute(statement)
                tx.execute(
                    "INSERT INTO schema_version (version, applied_at) VALUES (?, ?)",
                    (version, _now()),
                )
        return self.schema_version

    @property
    def schema_version(self) -> int:
        return self._read_schema_version()

    def close(self) -> None:
        conn = self._conn
        if conn is not None:
            self._conn = None
            conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ------------------------------------------------------------------- writes

    def create(self, proposal: ExperimentProposal) -> ExperimentProposal:
        """Persist a ``proposed`` record; repeating the same id returns it.

        Only the ``proposed`` state may be inserted. Approval is a separate,
        audited call, so importing a hand-built approved record cannot bypass
        the gate.
        """
        if not isinstance(proposal, ExperimentProposal):
            raise TypeError("proposal must be an ExperimentProposal")
        self._check_payload(proposal)
        existing = self.get(proposal.proposal_id)
        if existing is not None:
            return existing
        try:
            with self.transaction() as tx:
                tx.execute(
                    """
                    INSERT INTO proposals (
                        proposal_id, hypothesis_id, symbol, timeframe,
                        hypothesis_kind, required_columns_json, required_groups_json,
                        dsl_observation, dsl_condition, dsl_expected_effect,
                        falsification_test, status, approval_actor, approval_reason,
                        approval_at, converted_candidate_hash, dataset_id,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL,
                              NULL, NULL, ?, ?)
                    """,
                    (
                        proposal.proposal_id,
                        proposal.hypothesis_id,
                        proposal.symbol,
                        proposal.timeframe,
                        proposal.hypothesis_kind,
                        json.dumps(list(proposal.required_columns), separators=(",", ":")),
                        json.dumps(list(proposal.required_groups), separators=(",", ":")),
                        proposal.dsl_observation,
                        proposal.dsl_condition,
                        proposal.dsl_expected_effect,
                        proposal.falsification_test,
                        proposal.status,
                        proposal.created_at,
                        proposal.updated_at,
                    ),
                )
                for position, ref in enumerate(proposal.source_refs):
                    tx.execute(
                        """
                        INSERT INTO proposal_sources (
                            proposal_id, position, doc_id, section_ordinal, quote
                        ) VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            proposal.proposal_id,
                            position,
                            ref.doc_id,
                            ref.section_ordinal,
                            ref.quote,
                        ),
                    )
        except sqlite3.IntegrityError as exc:
            concurrent = self.get(proposal.proposal_id)
            if concurrent is not None:
                return concurrent
            raise RoutingError("invalid_proposal", f"proposal could not be stored: {exc}") from exc
        stored = self.get(proposal.proposal_id)
        if stored is None:
            raise KnowledgeIndexError("proposal insert committed but the row is missing")
        return stored

    def _check_payload(self, proposal: ExperimentProposal) -> None:
        if proposal.status != "proposed":
            raise RoutingError(
                "bad_status",
                "create() accepts only proposed records; approve/reject are separate calls",
            )
        if proposal.approval is not None:
            raise RoutingError("invalid_proposal", "a proposed record cannot carry an approval")
        if proposal.converted_candidate_hash is not None or proposal.dataset_id is not None:
            raise RoutingError("invalid_proposal", "a proposed record cannot be converted")
        for value, field in (
            (proposal.hypothesis_id, "hypothesis_id"),
            (proposal.symbol, "symbol"),
            (proposal.timeframe, "timeframe"),
            (proposal.hypothesis_kind, "hypothesis_kind"),
            (proposal.dsl_observation, "dsl_observation"),
            (proposal.dsl_condition, "dsl_condition"),
            (proposal.dsl_expected_effect, "dsl_expected_effect"),
            (proposal.falsification_test, "falsification_test"),
            (proposal.created_at, "created_at"),
            (proposal.updated_at, "updated_at"),
        ):
            _text(value, field)
        if proposal.hypothesis_kind not in {"claim", "observation"}:
            raise RoutingError(
                "unsupported_kind",
                f"hypothesis_kind must be 'claim' or 'observation', got"
                f" {proposal.hypothesis_kind!r}",
            )
        if proposal.symbol not in SYMBOLS:
            raise RoutingError("unknown_scope", f"unknown symbol {proposal.symbol!r}")
        if proposal.timeframe not in TIMEFRAMES:
            raise RoutingError("unknown_scope", f"unknown timeframe {proposal.timeframe!r}")
        if not proposal.source_refs:
            raise RoutingError("missing_source", "a proposal needs at least one source ref")
        if not _HASH_RE.fullmatch(proposal.proposal_id):
            raise RoutingError(
                "invalid_proposal",
                f"proposal_id is not a sha256 hex digest: {proposal.proposal_id!r}",
            )
        expected = canonical_proposal_id(
            hypothesis_id=proposal.hypothesis_id,
            source_refs=proposal.source_refs,
            symbol=proposal.symbol,
            timeframe=proposal.timeframe,
            hypothesis_kind=proposal.hypothesis_kind,
            required_columns=proposal.required_columns,
            required_groups=proposal.required_groups,
            dsl_observation=proposal.dsl_observation,
            dsl_condition=proposal.dsl_condition,
            dsl_expected_effect=proposal.dsl_expected_effect,
            falsification_test=proposal.falsification_test,
        )
        if expected != proposal.proposal_id:
            raise RoutingError(
                "invalid_proposal",
                f"proposal_id {proposal.proposal_id!r} does not match its payload",
            )

    def approve(self, proposal_id: str, reason: str, actor: str) -> ExperimentProposal:
        """Record the human approval that lets a proposal enter the queue.

        Only ``proposed`` may be approved. Re-approving an approved proposal is
        an idempotent no-op; a rejected or converted proposal cannot be
        approved because that history would be rewritten.
        """
        current = self._require(proposal_id)
        decision_reason = _text(reason, "reason")
        decision_actor = _text(actor, "actor")
        if current.status == "approved":
            return current
        if current.status == "converted":
            raise RoutingError(
                "already_converted", f"proposal {current.proposal_id!r} is already converted"
            )
        if current.status != "proposed":
            raise RoutingError(
                "bad_status",
                f"cannot approve proposal {current.proposal_id!r} from {current.status!r}",
            )
        stamp = _now()
        with self.transaction() as tx:
            tx.execute(
                """
                UPDATE proposals
                SET status = 'approved', approval_actor = ?, approval_reason = ?,
                    approval_at = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                (decision_actor, decision_reason, stamp, stamp, current.proposal_id),
            )
        return self._require(current.proposal_id)

    def reject(self, proposal_id: str, reason: str, actor: str) -> ExperimentProposal:
        """Record a rejection. An approved but not-yet-converted proposal may be
        rejected; conversion is terminal."""
        current = self._require(proposal_id)
        decision_reason = _text(reason, "reason")
        decision_actor = _text(actor, "actor")
        if current.status == "rejected":
            return current
        if current.status == "converted":
            raise RoutingError(
                "already_converted", f"proposal {current.proposal_id!r} is already converted"
            )
        stamp = _now()
        with self.transaction() as tx:
            tx.execute(
                """
                UPDATE proposals
                SET status = 'rejected', approval_actor = ?, approval_reason = ?,
                    approval_at = ?, updated_at = ?
                WHERE proposal_id = ?
                """,
                (decision_actor, decision_reason, stamp, stamp, current.proposal_id),
            )
        return self._require(current.proposal_id)

    def link_candidate(
        self, proposal_id: str, candidate_hash: str, dataset_id: str
    ) -> ExperimentProposal:
        """Mark an approved proposal converted by one candidate on one dataset.

        The call is one-way and may happen once. It records identity, not
        evidence: conversion is not promotion, validation or profitability.
        """
        current = self._require(proposal_id)
        if current.status == "converted":
            raise RoutingError(
                "already_converted", f"proposal {current.proposal_id!r} is already converted"
            )
        if current.status != "approved":
            raise RoutingError(
                "not_approved",
                f"proposal {current.proposal_id!r} is {current.status!r}, not 'approved'",
            )
        candidate = _text(candidate_hash, "candidate_hash")
        dataset = _text(dataset_id, "dataset_id")
        stamp = _now()
        with self.transaction() as tx:
            tx.execute(
                """
                UPDATE proposals
                SET status = 'converted', converted_candidate_hash = ?, dataset_id = ?,
                    updated_at = ?
                WHERE proposal_id = ?
                """,
                (candidate, dataset, stamp, current.proposal_id),
            )
        return self._require(current.proposal_id)

    # -------------------------------------------------------------------- reads

    def _require(self, proposal_id: str) -> ExperimentProposal:
        if not isinstance(proposal_id, str) or not proposal_id.strip():
            raise ValueError("proposal_id must be a non-empty str")
        row = self.connection.execute(
            "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id,)
        ).fetchone()
        if row is None:
            raise RoutingError("unknown_proposal", f"unknown proposal {proposal_id!r}")
        return _load(self, row)

    def get(self, proposal_id: str) -> ExperimentProposal | None:
        """One proposal, or ``None`` when the id is unknown."""
        if not isinstance(proposal_id, str):
            raise TypeError("proposal_id must be a str")
        row = self.connection.execute(
            "SELECT * FROM proposals WHERE proposal_id = ?", (proposal_id.strip(),)
        ).fetchone()
        return None if row is None else _load(self, row)

    def list(
        self,
        *,
        status: str | None = None,
        hypothesis_id: str | None = None,
    ) -> list[ExperimentProposal]:
        """Proposals oldest first, optionally filtered by status or hypothesis."""
        if status is not None and status not in PROPOSAL_STATUSES:
            raise ValueError(f"unknown proposal status {status!r}")
        clauses: list[str] = []
        params: list[str] = []
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if hypothesis_id is not None:
            clauses.append("hypothesis_id = ?")
            params.append(_text(hypothesis_id, "hypothesis_id"))
        where = "" if not clauses else " WHERE " + " AND ".join(clauses)
        rows = self.connection.execute(
            f"SELECT * FROM proposals{where} ORDER BY created_at, proposal_id",
            params,
        ).fetchall()
        return [_load(self, row) for row in rows]


def proposals_for_queue(store: ProposalStore, limit: int) -> list[ExperimentProposal]:
    """Approved proposals ready for the orchestrator, oldest first.

    This function never changes status; an unapproved proposal stays out of the
    queue until :meth:`ProposalStore.approve` is called explicitly. Converted
    and rejected proposals never appear.
    """
    if not isinstance(store, ProposalStore):
        raise TypeError("store must be a ProposalStore")
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise TypeError("limit must be an int")
    if limit < 1:
        raise ValueError("limit must be >= 1")
    return store.list(status="approved")[:limit]
