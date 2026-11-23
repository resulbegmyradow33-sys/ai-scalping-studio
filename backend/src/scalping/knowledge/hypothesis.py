"""Structured, falsifiable hypotheses linked to document passages.

A hypothesis is a record, not an instruction. ``create_hypothesis`` stores the
statement, what was observed, the proposed condition, the expected effect, its
market/timeframe scope, assumptions, a falsification test and at least the
source passages the author cites. Later experiments are linked by reference.
Nothing here parses or executes prose, and no field is read by the trading or
order path.

Status is monotonic for auditability: ``proposed -> testing`` and either
outcome -> ``supported``/``rejected`` from ``testing``; ``proposed`` may be
rejected directly. Terminal states are not reopened; a re-test is a new
hypothesis that cites the old one in its statement.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from scalping.knowledge.index import KnowledgeIndex, KnowledgeIndexError

__all__ = [
    "HYPOTHESIS_STATUSES",
    "STATUS_TRANSITIONS",
    "Hypothesis",
    "SourceRef",
    "create_hypothesis",
    "get_hypothesis",
    "link_experiment",
    "list_hypotheses",
    "set_status",
]

#: Every status a hypothesis may hold.
HYPOTHESIS_STATUSES: frozenset[str] = frozenset({"proposed", "testing", "supported", "rejected"})

#: Allowed status moves. A repeated status is an idempotent no-op.
STATUS_TRANSITIONS: dict[str, frozenset[str]] = {
    "proposed": frozenset({"testing", "rejected"}),
    "testing": frozenset({"supported", "rejected"}),
    "supported": frozenset(),
    "rejected": frozenset(),
}


class HypothesisError(KnowledgeIndexError):
    """The hypothesis record is invalid or the requested move is not allowed."""


@dataclass(frozen=True, slots=True)
class SourceRef:
    """A citation: document, section ordinal and the quoted passage."""

    doc_id: str
    section_ordinal: int
    quote: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "section_ordinal": self.section_ordinal,
            "quote": self.quote,
        }


@dataclass(frozen=True, slots=True)
class Hypothesis:
    """One research hypothesis and its traceable evidence links."""

    hypothesis_id: str
    statement: str
    observed: str
    proposed_condition: str
    expected_effect: str
    market_scope: str
    timeframe_scope: str
    assumptions: tuple[str, ...]
    falsification_test: str
    source_refs: tuple[SourceRef, ...]
    status: str
    experiment_refs: tuple[str, ...]
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "statement": self.statement,
            "observed": self.observed,
            "proposed_condition": self.proposed_condition,
            "expected_effect": self.expected_effect,
            "market_scope": self.market_scope,
            "timeframe_scope": self.timeframe_scope,
            "assumptions": list(self.assumptions),
            "falsification_test": self.falsification_test,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "status": self.status,
            "experiment_refs": list(self.experiment_refs),
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


def _assumptions(values: object) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, str):
        raise TypeError("assumptions must be a sequence of str, not a str")
    if not isinstance(values, Sequence):
        raise TypeError(f"assumptions must be a sequence, got {type(values).__name__}")
    return tuple(_text(item, "assumptions item") for item in values)


def _source_ref(value: object) -> SourceRef:
    if isinstance(value, SourceRef):
        ref = value
    elif isinstance(value, Mapping):
        ordinal = value.get(
            "section_ordinal", value.get("section", value.get("ordinal"))
        )
        try:
            ref = SourceRef(
                doc_id=value["doc_id"],
                section_ordinal=ordinal,
                quote=value["quote"],
            )
        except KeyError as exc:
            raise TypeError(f"source ref is missing {exc.args[0]!r}") from exc
    elif isinstance(value, (tuple, list)) and len(value) == 3:
        ref = SourceRef(doc_id=value[0], section_ordinal=value[1], quote=value[2])
    else:
        raise TypeError(
            "source refs must be SourceRef, a mapping with doc_id/section_ordinal/quote, "
            f"or a 3-tuple, got {type(value).__name__}"
        )
    if not isinstance(ref.doc_id, str):
        raise TypeError("source ref doc_id must be a str")
    if isinstance(ref.section_ordinal, bool) or not isinstance(ref.section_ordinal, int):
        raise TypeError("source ref section_ordinal must be an int")
    if ref.section_ordinal < 0:
        raise ValueError("source ref section_ordinal must be >= 0")
    return SourceRef(
        doc_id=_text(ref.doc_id, "source ref doc_id"),
        section_ordinal=ref.section_ordinal,
        quote=_text(ref.quote, "source ref quote"),
    )


def _source_refs(values: object) -> tuple[SourceRef, ...]:
    if values is None:
        return ()
    if isinstance(values, (SourceRef, Mapping)):
        return (_source_ref(values),)
    if (
        isinstance(values, tuple)
        and len(values) == 3
        and isinstance(values[0], str)
        and isinstance(values[1], int)
    ):
        return (_source_ref(values),)
    if isinstance(values, str):
        raise TypeError("source_refs must be a sequence, not a str")
    if not isinstance(values, Sequence):
        raise TypeError(f"source_refs must be a sequence, got {type(values).__name__}")
    return tuple(_source_ref(value) for value in values)


def _source_id(ref: SourceRef) -> tuple[str, int, str]:
    return (ref.doc_id, ref.section_ordinal, ref.quote)


def _canonical_assumptions(values: tuple[str, ...]) -> str:
    return json.dumps(list(values), ensure_ascii=True, separators=(",", ":"))


def _parse_assumptions(raw: str) -> tuple[str, ...]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise HypothesisError(f"stored assumptions are not valid JSON: {exc}") from exc
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise HypothesisError("stored assumptions must be a JSON array of strings")
    return tuple(data)


def _load(index: KnowledgeIndex, row: Any) -> Hypothesis:
    hypothesis_id = str(row["hypothesis_id"])
    conn = index.connection
    source_rows = conn.execute(
        """
        SELECT doc_id, section_ordinal, quote
        FROM hypothesis_sources
        WHERE hypothesis_id = ?
        ORDER BY position
        """,
        (hypothesis_id,),
    ).fetchall()
    experiment_rows = conn.execute(
        """
        SELECT experiment_ref
        FROM hypothesis_experiments
        WHERE hypothesis_id = ?
        ORDER BY position
        """,
        (hypothesis_id,),
    ).fetchall()
    return Hypothesis(
        hypothesis_id=hypothesis_id,
        statement=str(row["statement"]),
        observed=str(row["observed"]),
        proposed_condition=str(row["proposed_condition"]),
        expected_effect=str(row["expected_effect"]),
        market_scope=str(row["market_scope"]),
        timeframe_scope=str(row["timeframe_scope"]),
        assumptions=_parse_assumptions(str(row["assumptions_json"])),
        falsification_test=str(row["falsification_test"]),
        source_refs=tuple(
            SourceRef(
                doc_id=str(source["doc_id"]),
                section_ordinal=int(source["section_ordinal"]),
                quote=str(source["quote"]),
            )
            for source in source_rows
        ),
        status=str(row["status"]),
        experiment_refs=tuple(str(item["experiment_ref"]) for item in experiment_rows),
        created_at=str(row["created_at"]),
        updated_at=str(row["updated_at"]),
    )


def _check_source_refs(index: KnowledgeIndex, refs: tuple[SourceRef, ...]) -> None:
    """Every citation must name an indexed section; otherwise it is not traceable."""
    conn = index.connection
    seen: set[tuple[str, int, str]] = set()
    for ref in refs:
        if _source_id(ref) in seen:
            continue
        seen.add(_source_id(ref))
        doc = conn.execute(
            "SELECT status FROM documents WHERE doc_id = ?", (ref.doc_id,)
        ).fetchone()
        if doc is None:
            raise HypothesisError(f"unknown source document {ref.doc_id!r}")
        if str(doc["status"]) != "indexed":
            raise HypothesisError(
                f"source document {ref.doc_id!r} has status {doc['status']!r}, not 'indexed'"
            )
        section = conn.execute(
            "SELECT id FROM sections WHERE doc_id = ? AND ordinal = ?",
            (ref.doc_id, ref.section_ordinal),
        ).fetchone()
        if section is None:
            raise HypothesisError(
                f"source document {ref.doc_id!r} has no section {ref.section_ordinal}"
            )


def create_hypothesis(
    index: KnowledgeIndex,
    statement: str,
    observed: str,
    proposed_condition: str,
    expected_effect: str,
    market_scope: str,
    timeframe_scope: str,
    falsification_test: str,
    assumptions: Sequence[str] = (),
    source_refs: Sequence[object] = (),
    *,
    status: str = "proposed",
    hypothesis_id: str | None = None,
    created_at: str | None = None,
) -> Hypothesis:
    """Store one hypothesis and return the persisted record.

    Source refs are checked against the index before the transaction: a
    citation that does not resolve is rejected, not stored as decoration.
    """
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")
    values = {
        "statement": _text(statement, "statement"),
        "observed": _text(observed, "observed"),
        "proposed_condition": _text(proposed_condition, "proposed_condition"),
        "expected_effect": _text(expected_effect, "expected_effect"),
        "market_scope": _text(market_scope, "market_scope"),
        "timeframe_scope": _text(timeframe_scope, "timeframe_scope"),
        "falsification_test": _text(falsification_test, "falsification_test"),
    }
    assumption_values = _assumptions(assumptions)
    ref_values = _source_refs(source_refs)
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"status must be one of {sorted(HYPOTHESIS_STATUSES)}, got {status!r}")
    identifier = (
        uuid.uuid4().hex if hypothesis_id is None else _text(hypothesis_id, "hypothesis_id")
    )
    stamp = _now() if created_at is None else _text(created_at, "created_at")

    _check_source_refs(index, ref_values)
    try:
        with index.transaction() as tx:
            tx.execute(
                """
                INSERT INTO hypotheses (
                    hypothesis_id, statement, observed, proposed_condition,
                    expected_effect, market_scope, timeframe_scope, assumptions_json,
                    falsification_test, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    identifier,
                    values["statement"],
                    values["observed"],
                    values["proposed_condition"],
                    values["expected_effect"],
                    values["market_scope"],
                    values["timeframe_scope"],
                    _canonical_assumptions(assumption_values),
                    values["falsification_test"],
                    status,
                    stamp,
                    stamp,
                ),
            )
            for position, ref in enumerate(ref_values):
                tx.execute(
                    """
                    INSERT INTO hypothesis_sources (
                        hypothesis_id, position, doc_id, section_ordinal, quote
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (identifier, position, ref.doc_id, ref.section_ordinal, ref.quote),
                )
    except KnowledgeIndexError:
        raise
    except Exception as exc:
        raise HypothesisError(f"hypothesis could not be stored: {exc}") from exc
    stored = get_hypothesis(index, identifier)
    if stored is None:
        raise HypothesisError("hypothesis insert committed but the row is missing")
    return stored


def get_hypothesis(index: KnowledgeIndex, hypothesis_id: str) -> Hypothesis | None:
    """One hypothesis with its sources and experiments, or ``None``."""
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")
    hypothesis_id = _text(hypothesis_id, "hypothesis_id")
    row = index.connection.execute(
        "SELECT * FROM hypotheses WHERE hypothesis_id = ?", (hypothesis_id,)
    ).fetchone()
    return None if row is None else _load(index, row)


def list_hypotheses(index: KnowledgeIndex, *, status: str | None = None) -> list[Hypothesis]:
    """Hypotheses oldest first, optionally filtered by one status."""
    if not isinstance(index, KnowledgeIndex):
        raise TypeError("index must be a KnowledgeIndex")
    conn = index.connection
    if status is None:
        rows = conn.execute(
            "SELECT * FROM hypotheses ORDER BY created_at, hypothesis_id"
        ).fetchall()
    else:
        if status not in HYPOTHESIS_STATUSES:
            raise ValueError(f"unknown hypothesis status {status!r}")
        rows = conn.execute(
            "SELECT * FROM hypotheses WHERE status = ? ORDER BY created_at, hypothesis_id",
            (status,),
        ).fetchall()
    return [_load(index, row) for row in rows]


def set_status(index: KnowledgeIndex, hypothesis_id: str, status: str) -> Hypothesis:
    """Move a hypothesis along the allowed transitions and return the record."""
    if status not in HYPOTHESIS_STATUSES:
        raise ValueError(f"status must be one of {sorted(HYPOTHESIS_STATUSES)}, got {status!r}")
    current = get_hypothesis(index, hypothesis_id)
    if current is None:
        raise HypothesisError(f"unknown hypothesis {hypothesis_id!r}")
    if status == current.status:
        return current
    if status not in STATUS_TRANSITIONS[current.status]:
        raise HypothesisError(
            f"cannot move hypothesis {hypothesis_id!r} from {current.status!r} to {status!r}"
        )
    with index.transaction() as tx:
        tx.execute(
            "UPDATE hypotheses SET status = ?, updated_at = ? WHERE hypothesis_id = ?",
            (status, _now(), current.hypothesis_id),
        )
    updated = get_hypothesis(index, current.hypothesis_id)
    if updated is None:
        raise HypothesisError("hypothesis disappeared during the status update")
    return updated


def link_experiment(index: KnowledgeIndex, hypothesis_id: str, experiment_ref: str) -> Hypothesis:
    """Link an experiment id/reference; repeating the same link is a no-op."""
    experiment_ref = _text(experiment_ref, "experiment_ref")
    current = get_hypothesis(index, hypothesis_id)
    if current is None:
        raise HypothesisError(f"unknown hypothesis {hypothesis_id!r}")
    if experiment_ref in current.experiment_refs:
        return current
    with index.transaction() as tx:
        position = tx.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM hypothesis_experiments"
            " WHERE hypothesis_id = ?",
            (current.hypothesis_id,),
        ).fetchone()
        next_position = 0 if position is None else int(position[0])
        tx.execute(
            """
            INSERT INTO hypothesis_experiments (
                hypothesis_id, experiment_ref, position, linked_at
            ) VALUES (?, ?, ?, ?)
            """,
            (current.hypothesis_id, experiment_ref, next_position, _now()),
        )
        tx.execute(
            "UPDATE hypotheses SET updated_at = ? WHERE hypothesis_id = ?",
            (_now(), current.hypothesis_id),
        )
    updated = get_hypothesis(index, current.hypothesis_id)
    if updated is None:
        raise HypothesisError("hypothesis disappeared during the experiment link")
    return updated
