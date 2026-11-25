"""Read-only evidence API over the discovery archive.

The panel needs to review what the search produced and why a promotion was or was
not justified. This module exposes that archive: summary counts, the candidate
list with its trial evidence, one candidate's trials/lineage/decisions, and a
bounded trial window. Every route is a ``GET`` and nothing here can promote,
evaluate, schedule or write.

**No invented numbers.** A missing archive answers ``configured: false``; a file
that exists but cannot be opened or lacks the candidate/trial tables answers
``configured: true, readable: false`` and omits every count. "No candidates" and
"the archive could not be read" are different answers and the API keeps them
apart. A real zero means the archive was read and nothing matched.

**Read-only connections.** SQLite is opened with a ``mode=ro`` URI and
``PRAGMA query_only = ON``. When no WAL sidecar exists the file is opened
``immutable=1`` as well, which stops SQLite from creating ``-wal``/``-shm`` for a
read: a review poll leaves the archive directory byte-for-byte unchanged. If a
writer left a non-empty WAL, a plain ``mode=ro`` connection is used so its
committed rows are visible instead of silently reading a stale main file.

**Schema compatibility.** The archive is read with plain SQL, never through
:class:`~scalping.discovery.archive.DiscoveryArchive`, because constructing that
class runs migrations. A v1 archive without ``champions`` answers with an empty
champion list; a table that exists but lacks the required columns is reported as
unreadable rather than empty.

**No request paths and no query injection.** The database path comes from
``app.state.discovery_archive_path``; no request field selects a file. Candidate
ids are 64 lowercase hex characters, metric names use the archive's own
conservative alphabet, and every value is a bound SQL parameter. Scope and
dataset filters are data-id text only; they never name a filesystem path.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Final, Literal, cast

from fastapi import APIRouter, HTTPException, Query, Request

from scalping.api.schemas import (
    DiscoveryCandidateDetail,
    DiscoveryCandidateId,
    DiscoveryCandidateItem,
    DiscoveryCandidatePage,
    DiscoveryChampion,
    DiscoveryDatasetId,
    DiscoveryDecisionItem,
    DiscoveryLineageEdge,
    DiscoveryMetric,
    DiscoveryScope,
    DiscoverySummary,
    DiscoveryTrialItem,
    DiscoveryTrialPage,
)

__all__ = ["PAGE_DEFAULT", "PAGE_MAX", "router"]

router = APIRouter(prefix="/api/discovery", tags=["discovery"])

#: A page is bounded on both ends: enough for review, small enough that a
#: hostile ``limit`` cannot ask the process to materialise the whole archive.
PAGE_DEFAULT: Final[int] = 100
PAGE_MAX: Final[int] = 500

#: The default ranking metric matches ``SearchConfig.selection_metric``. A caller
#: reviewing another metric passes ``?metric=``; the API never guesses a winner.
DEFAULT_METRIC: Final[str] = "net_profit"

#: A scope is split into lowercase alphanumeric tokens exactly like
#: ``scalping.discovery.archive._scope_tokens``, so the read side and the write
#: side agree on what "this trial belongs to the scope" means.
_TOKEN_SPLIT: Final[re.Pattern[str]] = re.compile(r"[^a-z0-9]+")

_CANDIDATE_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "candidate_id",
        "canonical_json",
        "complexity",
        "created_at",
        "parent_a",
        "parent_b",
        "mutation",
        "code_version",
        "trials",
    }
)
_TRIAL_COLUMNS: Final[frozenset[str]] = frozenset(
    {
        "id",
        "candidate_id",
        "dataset_id",
        "code_version",
        "seed",
        "metrics_json",
        "accepted",
        "rejections_json",
        "created_at",
    }
)
_DECISION_COLUMNS: Final[frozenset[str]] = frozenset(
    {"id", "candidate_id", "decision", "reason", "champion_id", "created_at"}
)
_LINEAGE_COLUMNS: Final[frozenset[str]] = frozenset({"candidate_id", "parent_id", "relation"})
#: Only what a reader needs to identify a champion. Governance columns arrived in
#: schema v3/v4 and are read when present, never invented when absent.
_CHAMPION_COLUMNS: Final[frozenset[str]] = frozenset({"scope", "candidate_id", "promoted_at"})
_CHAMPION_OPTIONAL: Final[tuple[str, ...]] = (
    "reason",
    "mode",
    "policy_id",
    "policy_version",
    "policy_digest",
    "evidence_checksum",
)

#: The archive's own decision vocabulary (``decisions.decision`` CHECK).
_Decision = Literal["promoted", "rejected", "no_change", "no_trade", "rolled_back"]


# ------------------------------------------------------------------ connections


@contextmanager
def _read_only(path: Path) -> Iterator[sqlite3.Connection | None]:
    """A ``mode=ro`` connection, or ``None`` when there is no readable file.

    A missing path, a directory where a file was expected, a corrupt header and a
    permission failure all yield ``None``: the route then reports the archive as
    unreadable instead of turning a review poll into a 500. The URI comes from
    server configuration only; no request value reaches this function.
    """
    if not path.is_file():
        yield None
        return
    conn: sqlite3.Connection | None = None
    try:
        target = path.resolve()
        wal = Path(f"{target}-wal")
        # ``immutable=1`` keeps a read from creating the WAL sidecars, so a GET
        # on a closed archive changes nothing on disk. A non-empty WAL means a
        # writer (or a crash) holds committed rows that the main file does not;
        # read those through the ordinary read-only URI instead of hiding them.
        params = "mode=ro"
        if not (wal.is_file() and wal.stat().st_size > 0):
            params = "mode=ro&immutable=1"
        conn = sqlite3.connect(f"{target.as_uri()}?{params}", uri=True, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 5000")
        conn.execute("PRAGMA query_only = ON")
    except (OSError, ValueError, sqlite3.Error):
        if conn is not None:
            conn.close()
        yield None
        return
    try:
        yield conn
    finally:
        conn.close()


def _tables(conn: sqlite3.Connection) -> frozenset[str]:
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[0]) for row in rows)


def _columns(conn: sqlite3.Connection, table: str) -> frozenset[str]:
    """Column names for an internal table name; never a request value."""
    try:
        rows = conn.execute(f'PRAGMA table_info("{table}")').fetchall()
    except sqlite3.Error:
        return frozenset()
    return frozenset(str(row[1]) for row in rows)


def _count(conn: sqlite3.Connection, sql: str, params: Sequence[object] = ()) -> int:
    row = conn.execute(sql, tuple(params)).fetchone()
    return 0 if row is None else int(row[0])


def _archive_readable(conn: sqlite3.Connection, tables: frozenset[str]) -> bool:
    """Whether both tables every candidate/trial answer depends on can be read."""
    return (
        "candidates" in tables
        and "trials" in tables
        and _CANDIDATE_COLUMNS <= _columns(conn, "candidates")
        and _TRIAL_COLUMNS <= _columns(conn, "trials")
    )


# ---------------------------------------------------------------------- filters


def _scope_tokens(scope: str) -> frozenset[str]:
    """Lowercase alphanumeric tokens of a scope or dataset id."""
    return frozenset(token for token in _TOKEN_SPLIT.split(scope.lower()) if token)


def _scope_sql(tokens: frozenset[str], alias: str = "t") -> tuple[str, list[str]]:
    """An ``AND`` fragment requiring every scope token inside ``alias.dataset_id``.

    Tokens are ``[a-z0-9]+`` by construction and are bound, not interpolated, so
    the only SQL text that varies is the number of placeholders.
    """
    ordered = sorted(tokens)
    if not ordered:
        return "", []
    parts = [f"instr(lower({alias}.dataset_id), ?) > 0" for _ in ordered]
    return " AND " + " AND ".join(parts), ordered


# ---------------------------------------------------------------------- mapping


def _numeric_metrics(raw: object) -> dict[str, float]:
    """Decode one stored metric object; a non-object or non-number is unreadable."""
    if not isinstance(raw, str):
        raise TypeError("metrics payload is not text")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("metrics payload is not a JSON object")
    metrics: dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise TypeError(f"metric {key!r} is not a number")
        metrics[str(key)] = float(value)
    return metrics


def _rejections(raw: object) -> list[str]:
    if not isinstance(raw, str):
        raise TypeError("rejections payload is not text")
    data = json.loads(raw)
    if not isinstance(data, list):
        raise TypeError("rejections payload is not a JSON array")
    return [str(reason) for reason in data]


def _json_object(raw: object) -> dict[str, object] | None:
    """A stored detail object, or ``None`` when the column is NULL."""
    if raw is None:
        return None
    if not isinstance(raw, str):
        raise TypeError("detail payload is not text")
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise TypeError("detail payload is not a JSON object")
    return data


def _candidate_item(row: sqlite3.Row, metric: str) -> DiscoveryCandidateItem:
    best_metrics: dict[str, float] | None = None
    best_value: float | None = None
    raw = row["best_metrics_json"]
    if raw is not None:
        best_metrics = _numeric_metrics(raw)
        best_value = best_metrics.get(metric)
    return DiscoveryCandidateItem(
        candidate_id=str(row["candidate_id"]),
        complexity=int(row["complexity"]),
        created_at=str(row["created_at"]),
        code_version=str(row["code_version"]),
        mutation=None if row["mutation"] is None else str(row["mutation"]),
        parent_a=None if row["parent_a"] is None else str(row["parent_a"]),
        parent_b=None if row["parent_b"] is None else str(row["parent_b"]),
        trials=int(row["trial_count"]),
        accepted_trials=int(row["accepted_trials"]),
        best_value=best_value,
        best_metrics=best_metrics,
    )


def _trial_select(conn: sqlite3.Connection) -> str:
    """The trial columns this reader exposes, with absent v4 columns as NULL."""
    columns = _columns(conn, "trials")
    checksum = "dataset_checksum" if "dataset_checksum" in columns else "NULL AS dataset_checksum"
    settings = "settings_digest" if "settings_digest" in columns else "NULL AS settings_digest"
    return (
        "id, candidate_id, dataset_id, code_version, seed, metrics_json, accepted, "
        f"rejections_json, created_at, {checksum}, {settings}"
    )


def _trial_item(row: sqlite3.Row) -> DiscoveryTrialItem:
    return DiscoveryTrialItem(
        id=int(row["id"]),
        candidate_id=str(row["candidate_id"]),
        dataset_id=str(row["dataset_id"]),
        dataset_checksum=(
            None if row["dataset_checksum"] is None else str(row["dataset_checksum"])
        ),
        settings_digest=(None if row["settings_digest"] is None else str(row["settings_digest"])),
        code_version=str(row["code_version"]),
        seed=int(row["seed"]),
        metrics=_numeric_metrics(row["metrics_json"]),
        accepted=bool(row["accepted"]),
        rejections=_rejections(row["rejections_json"]),
        created_at=str(row["created_at"]),
    )


def _decision_item(row: sqlite3.Row) -> DiscoveryDecisionItem:
    return DiscoveryDecisionItem(
        id=int(row["id"]),
        candidate_id=str(row["candidate_id"]),
        decision=cast(_Decision, str(row["decision"])),
        reason=str(row["reason"]),
        champion_id=None if row["champion_id"] is None else str(row["champion_id"]),
        created_at=str(row["created_at"]),
        detail=_json_object(row["detail_json"]),
    )


# ------------------------------------------------------------------ champions


def _champions(
    conn: sqlite3.Connection,
    tables: frozenset[str],
    scope: str | None,
    limit: int,
) -> tuple[bool, list[DiscoveryChampion] | None]:
    """``(readable, items)`` for the champion table.

    An archive with no ``champions`` table is readable and empty: nothing was
    promoted under that schema. A table that exists but lacks the required
    columns is unreadable, because an empty answer would be a lie. Governance
    columns are included only when the archive has them; a v2 row reports
    ``mode=None`` instead of a guessed value.
    """
    if "champions" not in tables:
        return True, []
    columns = _columns(conn, "champions")
    if not _CHAMPION_COLUMNS <= columns:
        return False, None
    select = ", ".join(
        column if column in columns else f"NULL AS {column}"
        for column in ("scope", "candidate_id", "promoted_at", *_CHAMPION_OPTIONAL)
    )
    where = ""
    params: list[object] = []
    if scope is not None:
        where = "WHERE scope = ?"
        params.append(scope)
    params.append(limit)
    rows = conn.execute(
        f"SELECT {select} FROM champions {where} ORDER BY scope LIMIT ?", params
    ).fetchall()
    items: list[DiscoveryChampion] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        tokens = _scope_tokens(str(row["scope"]))
        scope_sql, scope_params = _scope_sql(tokens)
        trials = _count(
            conn,
            f"SELECT COUNT(*) FROM trials t WHERE t.candidate_id = ?{scope_sql}",
            [candidate_id, *scope_params],
        )
        accepted = _count(
            conn,
            f"SELECT COUNT(*) FROM trials t WHERE t.candidate_id = ? AND t.accepted = 1{scope_sql}",
            [candidate_id, *scope_params],
        )
        policy_version = row["policy_version"]
        items.append(
            DiscoveryChampion(
                scope=str(row["scope"]),
                candidate_id=candidate_id,
                mode=None if row["mode"] is None else str(row["mode"]),
                policy_id=None if row["policy_id"] is None else str(row["policy_id"]),
                policy_version=None if policy_version is None else int(policy_version),
                policy_digest=(None if row["policy_digest"] is None else str(row["policy_digest"])),
                evidence_checksum=(
                    None if row["evidence_checksum"] is None else str(row["evidence_checksum"])
                ),
                promoted_at=(None if row["promoted_at"] is None else str(row["promoted_at"])),
                reason=None if row["reason"] is None else str(row["reason"]),
                trials=trials,
                accepted_trials=accepted,
            )
        )
    return True, items


# ----------------------------------------------------------------------- routes


@router.get(
    "/summary",
    response_model=DiscoverySummary,
    response_model_exclude_none=True,
)
def discovery_summary(
    request: Request,
    scope: Annotated[DiscoveryScope | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_DEFAULT,
) -> DiscoverySummary:
    """Archive totals, optionally restricted to one scope.

    ``limit`` bounds the champion list. With ``scope`` the trial-derived counts
    describe only trials whose ``dataset_id`` carries every scope token, and
    ``candidates``/``decisions`` cover that same population. An unreadable
    archive has no counts at all, not zeros.
    """
    path = Path(request.app.state.discovery_archive_path)
    configured = path.is_file()
    if not configured:
        return DiscoverySummary(configured=False, readable=False, limit=limit)
    with _read_only(path) as conn:
        if conn is None:
            return DiscoverySummary(configured=True, readable=False, limit=limit)
        try:
            tables = _tables(conn)
            if not _archive_readable(conn, tables):
                return DiscoverySummary(configured=True, readable=False, limit=limit)
            decisions_readable = "decisions" in tables and _DECISION_COLUMNS <= _columns(
                conn, "decisions"
            )
            tokens = _scope_tokens(scope) if scope is not None else frozenset()
            scope_sql, scope_params = _scope_sql(tokens)
            if tokens:
                trials = _count(
                    conn,
                    f"SELECT COUNT(*) FROM trials t WHERE 1 = 1{scope_sql}",
                    scope_params,
                )
                accepted = _count(
                    conn,
                    f"SELECT COUNT(*) FROM trials t WHERE t.accepted = 1{scope_sql}",
                    scope_params,
                )
                candidates = _count(
                    conn,
                    f"SELECT COUNT(DISTINCT t.candidate_id) FROM trials t WHERE 1 = 1{scope_sql}",
                    scope_params,
                )
                datasets = _count(
                    conn,
                    f"SELECT COUNT(DISTINCT t.dataset_id) FROM trials t WHERE 1 = 1{scope_sql}",
                    scope_params,
                )
                decisions = (
                    _count(
                        conn,
                        "SELECT COUNT(*) FROM decisions WHERE candidate_id IN"
                        " (SELECT t.candidate_id FROM trials t WHERE 1 = 1"
                        f"{scope_sql})",
                        scope_params,
                    )
                    if decisions_readable
                    else None
                )
            else:
                candidates = _count(conn, "SELECT COUNT(*) FROM candidates")
                trials = _count(conn, "SELECT COUNT(*) FROM trials")
                accepted = _count(conn, "SELECT COUNT(*) FROM trials WHERE accepted = 1")
                datasets = _count(conn, "SELECT COUNT(DISTINCT dataset_id) FROM trials")
                decisions = (
                    _count(conn, "SELECT COUNT(*) FROM decisions") if decisions_readable else None
                )
            champions_readable, champions = _champions(conn, tables, scope, limit)
        except (sqlite3.Error, ValueError, TypeError):
            return DiscoverySummary(configured=True, readable=False, limit=limit)
    return DiscoverySummary(
        configured=True,
        readable=True,
        scope=scope,
        candidates=candidates,
        trials=trials,
        accepted_trials=accepted,
        rejected_trials=trials - accepted,
        datasets=datasets,
        decisions=decisions,
        champions_readable=champions_readable,
        champions=champions,
        limit=limit,
    )


@router.get(
    "/candidates",
    response_model=DiscoveryCandidatePage,
    response_model_exclude_none=True,
)
def discovery_candidates(
    request: Request,
    scope: Annotated[DiscoveryScope | None, Query()] = None,
    metric: Annotated[DiscoveryMetric, Query()] = DEFAULT_METRIC,
    limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_DEFAULT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DiscoveryCandidatePage:
    """Candidates newest first with their spec hash, complexity and trial evidence.

    ``trials``/``accepted_trials`` are counted from trial rows, so a scope filter
    narrows them too. ``best_metrics`` is the metric object of the accepted trial
    with the highest ``metric`` value; a candidate whose accepted trials never
    publish that metric reports ``None`` instead of an invented score.
    """
    path = Path(request.app.state.discovery_archive_path)
    configured = path.is_file()
    if not configured:
        return DiscoveryCandidatePage(
            configured=False,
            readable=False,
            metric=metric,
            limit=limit,
            offset=offset,
        )
    with _read_only(path) as conn:
        if conn is None:
            return DiscoveryCandidatePage(
                configured=True,
                readable=False,
                metric=metric,
                limit=limit,
                offset=offset,
            )
        try:
            tables = _tables(conn)
            if not _archive_readable(conn, tables):
                return DiscoveryCandidatePage(
                    configured=True,
                    readable=False,
                    metric=metric,
                    limit=limit,
                    offset=offset,
                )
            tokens = _scope_tokens(scope) if scope is not None else frozenset()
            scope_sql, scope_params = _scope_sql(tokens)
            where = ""
            where_params: list[str] = []
            if tokens:
                where = (
                    "WHERE EXISTS (SELECT 1 FROM trials t "
                    f"WHERE t.candidate_id = c.candidate_id{scope_sql})"
                )
                where_params = list(scope_params)
            metric_path = f'$."{metric}"'
            rows = conn.execute(
                f"""
                SELECT c.candidate_id, c.complexity, c.created_at, c.code_version,
                       c.mutation, c.parent_a, c.parent_b,
                       (SELECT COUNT(*) FROM trials t
                         WHERE t.candidate_id = c.candidate_id{scope_sql})
                           AS trial_count,
                       (SELECT COUNT(*) FROM trials t
                         WHERE t.candidate_id = c.candidate_id
                           AND t.accepted = 1{scope_sql})
                           AS accepted_trials,
                       (SELECT t.metrics_json FROM trials t
                         WHERE t.candidate_id = c.candidate_id
                           AND t.accepted = 1{scope_sql}
                           AND json_extract(t.metrics_json, ?) IS NOT NULL
                         ORDER BY json_extract(t.metrics_json, ?) IS NULL,
                                  json_extract(t.metrics_json, ?) DESC,
                                  t.id DESC
                         LIMIT 1) AS best_metrics_json
                FROM candidates c
                {where}
                ORDER BY c.created_at DESC, c.candidate_id ASC
                LIMIT ? OFFSET ?
                """,
                [
                    *scope_params,
                    *scope_params,
                    *scope_params,
                    metric_path,
                    metric_path,
                    metric_path,
                    *where_params,
                    limit,
                    offset,
                ],
            ).fetchall()
            if tokens:
                total = _count(
                    conn,
                    f"SELECT COUNT(DISTINCT t.candidate_id) FROM trials t WHERE 1 = 1{scope_sql}",
                    scope_params,
                )
            else:
                total = _count(conn, "SELECT COUNT(*) FROM candidates")
            items = [_candidate_item(row, metric) for row in rows]
        except (sqlite3.Error, ValueError, TypeError):
            return DiscoveryCandidatePage(
                configured=True,
                readable=False,
                metric=metric,
                limit=limit,
                offset=offset,
            )
    return DiscoveryCandidatePage(
        configured=True,
        readable=True,
        items=items,
        total=total,
        metric=metric,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/candidates/{candidate_id}",
    response_model=DiscoveryCandidateDetail,
    response_model_exclude_none=True,
)
def discovery_candidate(
    candidate_id: DiscoveryCandidateId,
    request: Request,
    metric: Annotated[DiscoveryMetric, Query()] = DEFAULT_METRIC,
    limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_DEFAULT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DiscoveryCandidateDetail:
    """One candidate's spec, evidence and audit trail.

    ``404`` means no archive is configured or the id is unknown. ``409`` means an
    archive exists but cannot be read (corrupt file, unrelated schema, or a row
    whose stored JSON is not decodable). A malformed id is a ``422`` from request
    validation: it never reaches a lookup.
    """
    path = Path(request.app.state.discovery_archive_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="no discovery archive is configured")
    with _read_only(path) as conn:
        if conn is None:
            raise HTTPException(status_code=409, detail="discovery archive is not readable")
        try:
            tables = _tables(conn)
            if not _archive_readable(conn, tables):
                raise HTTPException(status_code=409, detail="discovery archive is not readable")
            row = conn.execute(
                "SELECT * FROM candidates WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            if row is None:
                raise HTTPException(
                    status_code=404, detail=f"unknown discovery candidate: {candidate_id}"
                )
            spec = json.loads(str(row["canonical_json"]))
            if not isinstance(spec, dict):
                raise TypeError("candidate payload is not a JSON object")
            metric_path = f'$."{metric}"'
            trial_total = _count(
                conn,
                "SELECT COUNT(*) FROM trials WHERE candidate_id = ?",
                (candidate_id,),
            )
            accepted_total = _count(
                conn,
                "SELECT COUNT(*) FROM trials WHERE candidate_id = ? AND accepted = 1",
                (candidate_id,),
            )
            best_row = conn.execute(
                """
                SELECT metrics_json FROM trials
                WHERE candidate_id = ? AND accepted = 1
                  AND json_extract(metrics_json, ?) IS NOT NULL
                ORDER BY json_extract(metrics_json, ?) IS NULL,
                         json_extract(metrics_json, ?) DESC,
                         id DESC
                LIMIT 1
                """,
                (candidate_id, metric_path, metric_path, metric_path),
            ).fetchone()
            best_metrics = None if best_row is None else _numeric_metrics(best_row["metrics_json"])
            item = DiscoveryCandidateItem(
                candidate_id=str(row["candidate_id"]),
                complexity=int(row["complexity"]),
                created_at=str(row["created_at"]),
                code_version=str(row["code_version"]),
                mutation=None if row["mutation"] is None else str(row["mutation"]),
                parent_a=None if row["parent_a"] is None else str(row["parent_a"]),
                parent_b=None if row["parent_b"] is None else str(row["parent_b"]),
                trials=trial_total,
                accepted_trials=accepted_total,
                best_value=None if best_metrics is None else best_metrics.get(metric),
                best_metrics=best_metrics,
            )
            trial_select = _trial_select(conn)
            trial_rows = conn.execute(
                f"SELECT {trial_select} FROM trials WHERE candidate_id = ?"
                " ORDER BY id DESC LIMIT ? OFFSET ?",
                (candidate_id, limit, offset),
            ).fetchall()
            trials = [_trial_item(trial_row) for trial_row in trial_rows]
            if "lineage" in tables and _LINEAGE_COLUMNS <= _columns(conn, "lineage"):
                lineage_rows = conn.execute(
                    "SELECT parent_id, relation FROM lineage WHERE candidate_id = ?"
                    " ORDER BY parent_id, relation",
                    (candidate_id,),
                ).fetchall()
                lineage: list[DiscoveryLineageEdge] | None = [
                    DiscoveryLineageEdge(
                        parent_id=str(edge["parent_id"]),
                        relation=str(edge["relation"]),
                    )
                    for edge in lineage_rows
                ]
            else:
                lineage = None
            if "decisions" in tables and _DECISION_COLUMNS <= _columns(conn, "decisions"):
                decisions_columns = _columns(conn, "decisions")
                detail = (
                    "detail_json" if "detail_json" in decisions_columns else "NULL AS detail_json"
                )
                decision_total = _count(
                    conn,
                    "SELECT COUNT(*) FROM decisions WHERE candidate_id = ?",
                    (candidate_id,),
                )
                decision_rows = conn.execute(
                    "SELECT id, candidate_id, decision, reason, champion_id,"
                    f" created_at, {detail} FROM decisions"
                    " WHERE candidate_id = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                    (candidate_id, limit, offset),
                ).fetchall()
                decisions: list[DiscoveryDecisionItem] | None = [
                    _decision_item(decision_row) for decision_row in decision_rows
                ]
            else:
                decisions = None
                decision_total = None
        except HTTPException:
            raise
        except (sqlite3.Error, ValueError, TypeError) as exc:
            raise HTTPException(
                status_code=409, detail="discovery archive is not readable"
            ) from exc
    return DiscoveryCandidateDetail(
        configured=True,
        readable=True,
        candidate=item,
        spec=spec,
        trials=trials,
        trial_total=trial_total,
        lineage=lineage,
        decisions=decisions,
        decision_total=decision_total,
        limit=limit,
        offset=offset,
    )


@router.get(
    "/trials",
    response_model=DiscoveryTrialPage,
    response_model_exclude_none=True,
)
def discovery_trials(
    request: Request,
    candidate_id: Annotated[DiscoveryCandidateId | None, Query()] = None,
    dataset_id: Annotated[DiscoveryDatasetId | None, Query()] = None,
    scope: Annotated[DiscoveryScope | None, Query()] = None,
    accepted: Annotated[bool | None, Query()] = None,
    limit: Annotated[int, Query(ge=1, le=PAGE_MAX)] = PAGE_DEFAULT,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> DiscoveryTrialPage:
    """A bounded, newest-first trial window; every filter is optional.

    ``scope`` matches the archive's dataset-token rule; ``dataset_id`` is an
    exact match; ``accepted`` selects the gate verdict. An unknown but
    well-formed candidate id is an empty page, not a ``404``: this is a query,
    not a resource lookup.
    """
    path = Path(request.app.state.discovery_archive_path)
    configured = path.is_file()
    if not configured:
        return DiscoveryTrialPage(configured=False, readable=False, limit=limit, offset=offset)
    with _read_only(path) as conn:
        if conn is None:
            return DiscoveryTrialPage(configured=True, readable=False, limit=limit, offset=offset)
        try:
            tables = _tables(conn)
            if not _archive_readable(conn, tables):
                return DiscoveryTrialPage(
                    configured=True, readable=False, limit=limit, offset=offset
                )
            conditions: list[str] = []
            params: list[object] = []
            if candidate_id is not None:
                conditions.append("candidate_id = ?")
                params.append(candidate_id)
            if dataset_id is not None:
                conditions.append("dataset_id = ?")
                params.append(dataset_id)
            if scope is not None:
                scope_sql, scope_params = _scope_sql(_scope_tokens(scope), alias="trials")
                conditions.append(scope_sql.removeprefix(" AND "))
                params.extend(scope_params)
            if accepted is not None:
                conditions.append("accepted = ?")
                params.append(1 if accepted else 0)
            where = f" WHERE {' AND '.join(conditions)}" if conditions else ""
            total = _count(conn, f"SELECT COUNT(*) FROM trials{where}", params)
            trial_select = _trial_select(conn)
            rows = conn.execute(
                f"SELECT {trial_select} FROM trials{where} ORDER BY id DESC LIMIT ? OFFSET ?",
                [*params, limit, offset],
            ).fetchall()
            items = [_trial_item(row) for row in rows]
        except (sqlite3.Error, ValueError, TypeError):
            return DiscoveryTrialPage(configured=True, readable=False, limit=limit, offset=offset)
    return DiscoveryTrialPage(
        configured=True,
        readable=True,
        items=items,
        total=total,
        limit=limit,
        offset=offset,
    )
