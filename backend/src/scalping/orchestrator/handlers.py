"""Production-style job handlers registered with the existing orchestrator.

This module is the wiring the queue deliberately left out: it maps the five job
kinds of the first autonomous research cycle to real code, keeps every path in
an explicit :class:`HandlerContext` (no module globals), validates each payload
before touching the disk and classifies invalid input as a permanent failure so
the queue does not retry it forever.

What a handler may do
---------------------
* ``knowledge.ingest`` — preserve one document, index its sections and propose
  deterministic hypothesis drafts. Duplicate bytes are recognised by SHA-256
  before extraction, so a second run is cheap and creates no second row.
* ``dataset.build`` — build the chronological, purged dataset manifest for one
  stored symbol/timeframe from the same feature cache every other command uses.
* ``discovery.run`` — run one bounded, seeded discovery experiment with an
  explicit gate and cost-stress policy and record its decision.
* ``paper.advance`` — read the **governed champion** of a scope and advance its
  continuous paper session to the newest closed stored bar. No champion is NO
  TRADE with a reason; there is no fallback to the legacy rule.
* ``paper.analyse`` — summarise closed paper trades only, attributed by spec
  hash and a fingerprint of the bars the session actually saw.

What no handler here does
-------------------------
* No handler promotes, demotes or rolls back a champion. ``discovery.run`` only
  records ``no_trade``/``no_change``/``rejected``; ``paper.advance`` reads the
  champion another reviewed path promoted.
* No handler starts a thread, daemon or timer. The orchestrator decides *when*
  to call one.
* No handler raises a risk limit, writes an order or opens a network client.
  Paper execution goes through the existing simulator and ledger unchanged.
* No handler falls back to a handwritten strategy. A missing or unreadable DSL
  champion makes the paper job a recorded NO TRADE.

Failure classification
----------------------
``PermanentJobError`` is raised for a payload the handler cannot serve: an
unknown field, a missing required field, a wrong type, a nonexistent document, a
symbol/timeframe with no stored bars, an unsupported format or a gate value that
does not parse. A retry would see the same input, so the queue must not spend
attempts on it. Everything else — an I/O error, a locked database, an
unexpected bug — propagates and stays retryable. NO TRADE is a success result,
not a failure.

``paper.advance`` refusals follow the same split with a typed prefix in the job
error: ``bar_gap`` is retryable because a later ingestion can close the gap,
while ``bar_order``, ``session_conflict`` and ``continuous_refused`` are
permanent.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Protocol

from scalping.config import Settings
from scalping.data.store import ParquetStore
from scalping.datasets.builder import build_dataset, save_manifest
from scalping.discovery.archive import DiscoveryArchive
from scalping.discovery.champion import (
    ChampionProvider,
    InvalidChampionSpec,
    NoChampionError,
    NoEvidenceChampion,
    scope_for,
)
from scalping.discovery.run import (
    DiscoveryRunConfig,
    DiscoveryRunError,
    DiscoveryRunResult,
    run_discovery,
)
from scalping.features.cache import (
    FEATURE_CACHE_DIRNAME,
    FeatureCache,
    cached_build_features,
    frame_meta,
)
from scalping.knowledge.extract import extract_hypotheses
from scalping.knowledge.index import KnowledgeIndex
from scalping.knowledge.ingest import DocumentIngestor
from scalping.orchestrator.core import (
    Handler,
    HandlerResult,
    JobContext,
    PermanentJobError,
)
from scalping.orchestrator.store import Job
from scalping.paper.continuous import (
    BarGapError,
    BarOrderError,
    ContinuousError,
    ContinuousPaperRunner,
    SessionMismatchError,
)
from scalping.paper.ledger import PaperLedger, PositionRecord
from scalping.paper.runner import ResumeError
from scalping.registry.gate import GateCriteria

__all__ = [
    "DEFAULT_HANDLER_KINDS",
    "HEAVY_HANDLER_KINDS",
    "KIND_DATASET_BUILD",
    "KIND_DISCOVERY_RUN",
    "KIND_KNOWLEDGE_INGEST",
    "KIND_PAPER_ADVANCE",
    "KIND_PAPER_ANALYSE",
    "HandlerContext",
    "HandlerRegistry",
    "register_default_handlers",
]

#: Namespaced job kinds. The prefix groups the pipeline stage, so a queue dump
#: can be filtered without a mapping table.
KIND_KNOWLEDGE_INGEST: Final[str] = "knowledge.ingest"
KIND_DATASET_BUILD: Final[str] = "dataset.build"
KIND_DISCOVERY_RUN: Final[str] = "discovery.run"
KIND_PAPER_ADVANCE: Final[str] = "paper.advance"
KIND_PAPER_ANALYSE: Final[str] = "paper.analyse"

DEFAULT_HANDLER_KINDS: Final[tuple[str, ...]] = (
    KIND_KNOWLEDGE_INGEST,
    KIND_DATASET_BUILD,
    KIND_DISCOVERY_RUN,
    KIND_PAPER_ADVANCE,
    KIND_PAPER_ANALYSE,
)

#: Kinds a caller should pass as ``heavy_kinds``: building features over a full
#: symbol slice and running a search are the two jobs that must not overlap on
#: an 8 GB machine. The queue enforces it; this tuple only names the default.
HEAVY_HANDLER_KINDS: Final[frozenset[str]] = frozenset({KIND_DATASET_BUILD, KIND_DISCOVERY_RUN})

KNOWLEDGE_DIRNAME: Final[str] = "knowledge"
DISCOVERY_DIRNAME: Final[str] = "discovery"
DATASETS_DIRNAME: Final[str] = "datasets"
DISCOVERY_ARCHIVE_FILENAME: Final[str] = "discovery.sqlite"
PAPER_LEDGER_FILENAME: Final[str] = "paper.sqlite"

#: Upper bounds that keep one job bounded on the target machine. A payload above
#: them is invalid, not an invitation to run for hours.
MAX_DISCOVERY_TRIALS: Final[int] = 2000
MAX_DISCOVERY_POPULATION: Final[int] = 256
MAX_DISCOVERY_GENERATIONS: Final[int] = 100
MAX_PAPER_BARS: Final[int] = 100_000
MAX_ANALYSED_SESSIONS: Final[int] = 1_000
MAX_INGEST_HYPOTHESES: Final[int] = 500

#: An outcome attribution reads the bars a session's signals span. A span wider
#: than this is reported as unavailable instead of pulling months of M1 into
#: memory on the 8 GB target.
MAX_FINGERPRINT_SPAN: Final[timedelta] = timedelta(days=31)

_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,31}$")
_HASH_CHUNK_BYTES: Final[int] = 1024 * 1024


class PayloadError(ValueError):
    """A job payload is invalid; a retry cannot change it."""


class PaperAdvanceError(Exception):
    """A continuous paper advance was refused; ``reason`` names the typed cause.

    A ``bar_gap`` is retryable: the missing bar may still be ingested, so the
    queue may spend a bounded retry after its backoff. ``bar_order``,
    ``session_conflict`` and ``continuous_refused`` cannot heal on their own and
    are raised as :class:`PermanentJobError` with the same prefix, so the queue
    does not repeat them.
    """

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason


# --------------------------------------------------------------------------- context


class DiscoveryRunner(Protocol):
    """The narrow slice of :func:`run_discovery` a handler needs."""

    def __call__(
        self,
        config: DiscoveryRunConfig,
        *,
        root: str | Path,
        should_stop: Callable[[], bool] | None = None,
    ) -> DiscoveryRunResult: ...


class HandlerRegistry(Protocol):
    """Anything that can bind a kind to a handler; ``Orchestrator`` qualifies."""

    def register(self, kind: str, handler: Handler) -> None: ...


@dataclass(frozen=True, slots=True)
class HandlerContext:
    """Explicit paths and settings for every handler; no module-level state.

    * ``data_root`` — the Parquet bar store (read only) and its feature cache.
    * ``runs_root`` — reproducible run products: dataset manifests and the
      discovery archive.
    * ``archive_root`` — durable evidence: the knowledge index/sources and the
      paper ledger. Losing a run product is re-computable; losing this is not.
    * ``settings`` — the same ``Settings`` every other entry point uses, so a
      paper job and a CLI backtest price the same costs and risk.
    * ``use_feature_cache`` — pass ``False`` in tests or when the cache must not
      be written.
    * ``discovery_runner`` — optional injection for tests or a future
      walk-forward runner with the same contract; ``None`` means the real
      :func:`run_discovery`.
    """

    data_root: Path
    runs_root: Path
    archive_root: Path
    settings: Settings = field(default_factory=Settings)
    use_feature_cache: bool = True
    discovery_runner: DiscoveryRunner | None = None

    def __post_init__(self) -> None:
        for name in ("data_root", "runs_root", "archive_root"):
            value = getattr(self, name)
            if not isinstance(value, (str, Path)):
                raise TypeError(f"{name} must be a str or Path, got {type(value).__name__}")
            object.__setattr__(self, name, Path(value))
        if not isinstance(self.settings, Settings):
            raise TypeError(f"settings must be a Settings, got {type(self.settings).__name__}")
        if not isinstance(self.use_feature_cache, bool):
            raise TypeError("use_feature_cache must be a bool")
        if self.discovery_runner is not None and not callable(self.discovery_runner):
            raise TypeError("discovery_runner must be callable or None")

    @property
    def knowledge_root(self) -> Path:
        """Directory holding the knowledge index and preserved originals."""
        return self.archive_root / KNOWLEDGE_DIRNAME

    @property
    def knowledge_index_path(self) -> Path:
        """The SQLite knowledge index for this context."""
        return self.knowledge_root / "index.sqlite"

    @property
    def discovery_archive_path(self) -> Path:
        """The discovery archive ``discovery.run`` writes and paper reads."""
        return self.runs_root / DISCOVERY_DIRNAME / DISCOVERY_ARCHIVE_FILENAME

    @property
    def datasets_root(self) -> Path:
        """Directory for dataset manifests written by ``dataset.build``."""
        return self.runs_root / DATASETS_DIRNAME

    @property
    def paper_ledger_path(self) -> Path:
        """The single paper ledger every advance and analysis shares."""
        return self.archive_root / PAPER_LEDGER_FILENAME

    @property
    def feature_cache(self) -> FeatureCache | None:
        """A cache rooted under the bar store, or ``None`` when disabled."""
        if not self.use_feature_cache:
            return None
        return FeatureCache(self.data_root / FEATURE_CACHE_DIRNAME)


# ----------------------------------------------------------------- payload helpers


def _strict(payload: Mapping[str, Any], allowed: frozenset[str]) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise PayloadError(f"unknown field(s) {unknown}; allowed fields are {sorted(allowed)}")


def _required(payload: Mapping[str, Any], key: str) -> Any:
    if key not in payload:
        raise PayloadError(f"missing required field {key!r}")
    return payload[key]


def _optional(payload: Mapping[str, Any], key: str, default: Any) -> Any:
    return payload.get(key, default)


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise PayloadError(f"{name!r} must be a string, got {type(value).__name__}")
    text = value.strip()
    if not text:
        raise PayloadError(f"{name!r} must not be empty")
    return text


def _token(value: object, name: str) -> str:
    text = _text(value, name)
    if not _SYMBOL_RE.match(text):
        raise PayloadError(f"{name!r} must match {_SYMBOL_RE.pattern!r}, got {text!r}")
    return text


def _optional_text(payload: Mapping[str, Any], key: str) -> str:
    """A free-text field that may be empty but must still be a string."""
    if key not in payload:
        return ""
    value = payload[key]
    if not isinstance(value, str):
        raise PayloadError(f"{key!r} must be a string, got {type(value).__name__}")
    return value.strip()


def _integer(
    value: object,
    name: str,
    *,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PayloadError(f"{name!r} must be an integer, got {type(value).__name__}")
    if minimum is not None and value < minimum:
        raise PayloadError(f"{name!r} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise PayloadError(f"{name!r} must be <= {maximum}, got {value}")
    return value


def _number(
    value: object,
    name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PayloadError(f"{name!r} must be a number, got {type(value).__name__}")
    number = float(value)
    if not math.isfinite(number):
        raise PayloadError(f"{name!r} must be finite, got {number!r}")
    if minimum is not None:
        if exclusive_minimum and not number > minimum:
            raise PayloadError(f"{name!r} must be > {minimum:g}, got {number:g}")
        if not exclusive_minimum and number < minimum:
            raise PayloadError(f"{name!r} must be >= {minimum:g}, got {number:g}")
    if maximum is not None and number > maximum:
        raise PayloadError(f"{name!r} must be <= {maximum:g}, got {number:g}")
    return number


def _numbers(value: object, name: str) -> tuple[float, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PayloadError(f"{name!r} must be a list of numbers")
    values = tuple(
        _number(item, f"{name}[{index}]", minimum=0.0, exclusive_minimum=True)
        for index, item in enumerate(value)
    )
    if not values:
        raise PayloadError(f"{name!r} must not be empty")
    return values


_GATE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "min_trades",
        "min_expectancy_r",
        "min_profit_factor",
        "max_drawdown",
        "max_ece",
        "require_positive_cost_stress",
        "min_regime_coverage",
    }
)


def _gate(value: object) -> GateCriteria:
    """Build a gate from an explicit payload block; no criterion is inherited."""
    if not isinstance(value, Mapping):
        raise PayloadError(f"'gate' must be a mapping, got {type(value).__name__}")
    _strict(value, _GATE_FIELDS)
    try:
        return GateCriteria(
            min_trades=_integer(_optional(value, "min_trades", 0), "gate.min_trades", minimum=0),
            min_expectancy_r=_number(
                _optional(value, "min_expectancy_r", 0.0), "gate.min_expectancy_r"
            ),
            min_profit_factor=_number(
                _optional(value, "min_profit_factor", 0.0), "gate.min_profit_factor"
            ),
            max_drawdown=_number(
                _optional(value, "max_drawdown", 0.0),
                "gate.max_drawdown",
                minimum=0.0,
                maximum=1.0,
            ),
            max_ece=_number(
                _optional(value, "max_ece", 0.0),
                "gate.max_ece",
                minimum=0.0,
                maximum=1.0,
            ),
            require_positive_cost_stress=_numbers(
                _optional(value, "require_positive_cost_stress", []),
                "gate.require_positive_cost_stress",
            )
            if "require_positive_cost_stress" in value
            else (),
            min_regime_coverage=_number(
                _optional(value, "min_regime_coverage", 0.0),
                "gate.min_regime_coverage",
                minimum=0.0,
                maximum=1.0,
            ),
        )
    except (TypeError, ValueError) as exc:
        raise PayloadError(f"invalid gate: {exc}") from exc


def _parse[T](request: type[T], payload: Mapping[str, Any], kind: str) -> T:
    """Run a request validator and turn any rejection into a permanent error."""
    try:
        if not isinstance(payload, Mapping):
            raise PayloadError(f"payload must be a mapping, got {type(payload).__name__}")
        return request.from_payload(payload)  # type: ignore[attr-defined,no-any-return]
    except (PayloadError, TypeError, ValueError) as exc:
        raise PermanentJobError(f"{kind}: invalid payload: {exc}") from exc


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _log_json(context: JobContext, event: str, payload: Mapping[str, Any]) -> None:
    context.log(
        event,
        json.dumps(dict(payload), sort_keys=True, separators=(",", ":"), ensure_ascii=True),
    )


def _jsonable(value: Any) -> Any:
    """Reject a non-JSON value at handler time, not after the queue stored it."""
    json.dumps(value, allow_nan=False)
    return value


# ---------------------------------------------------------------------- requests


@dataclass(frozen=True, slots=True)
class IngestRequest:
    """Payload of ``knowledge.ingest``."""

    document_path: Path
    market_scope: str = "unspecified"
    timeframe_scope: str = "unspecified"
    max_hypotheses: int = 50

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> IngestRequest:
        _strict(
            payload,
            frozenset({"document_path", "market_scope", "timeframe_scope", "max_hypotheses"}),
        )
        raw = _text(_required(payload, "document_path"), "document_path")
        if "\x00" in raw:
            raise PayloadError("'document_path' must not contain a NUL byte")
        given = Path(raw)
        if any(part == ".." for part in given.parts):
            raise PayloadError("'document_path' must not contain '..'")
        if not given.exists():
            raise PayloadError(f"document does not exist: {raw}")
        if not given.is_file():
            raise PayloadError(f"document is not a regular file: {raw}")
        return cls(
            document_path=given.resolve(),
            market_scope=_text(_optional(payload, "market_scope", "unspecified"), "market_scope"),
            timeframe_scope=_text(
                _optional(payload, "timeframe_scope", "unspecified"), "timeframe_scope"
            ),
            max_hypotheses=_integer(
                _optional(payload, "max_hypotheses", 50),
                "max_hypotheses",
                minimum=1,
                maximum=MAX_INGEST_HYPOTHESES,
            ),
        )


@dataclass(frozen=True, slots=True)
class DatasetRequest:
    """Payload of ``dataset.build``."""

    symbol: str
    timeframe: str
    horizon: int = 1
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    purge_bars: int = 0
    embargo_bars: int = 0
    notes: str = ""

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DatasetRequest:
        _strict(
            payload,
            frozenset(
                {
                    "symbol",
                    "timeframe",
                    "horizon",
                    "train_fraction",
                    "validation_fraction",
                    "purge_bars",
                    "embargo_bars",
                    "notes",
                }
            ),
        )
        return cls(
            symbol=_token(_required(payload, "symbol"), "symbol"),
            timeframe=_token(_required(payload, "timeframe"), "timeframe"),
            horizon=_integer(_optional(payload, "horizon", 1), "horizon", minimum=1),
            train_fraction=_number(
                _optional(payload, "train_fraction", 0.70),
                "train_fraction",
                minimum=0.0,
                maximum=1.0,
                exclusive_minimum=True,
            ),
            validation_fraction=_number(
                _optional(payload, "validation_fraction", 0.15),
                "validation_fraction",
                minimum=0.0,
                maximum=1.0,
            ),
            purge_bars=_integer(_optional(payload, "purge_bars", 0), "purge_bars", minimum=0),
            embargo_bars=_integer(_optional(payload, "embargo_bars", 0), "embargo_bars", minimum=0),
            notes=_optional_text(payload, "notes"),
        )


@dataclass(frozen=True, slots=True)
class DiscoveryRequest:
    """Payload of ``discovery.run``; the gate and stress policy are explicit."""

    symbol: str
    timeframe: str
    max_trials: int
    seed: int
    stress: tuple[float, ...]
    gate: GateCriteria
    population: int = 16
    generations: int = 5
    patience: int | None = 5
    horizon: int = 1

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> DiscoveryRequest:
        _strict(
            payload,
            frozenset(
                {
                    "symbol",
                    "timeframe",
                    "max_trials",
                    "seed",
                    "stress",
                    "gate",
                    "population",
                    "generations",
                    "patience",
                    "horizon",
                }
            ),
        )
        if "gate" not in payload:
            raise PayloadError(
                "'gate' is required: a run must state its evidence policy, even when "
                "every threshold is disabled"
            )
        patience_raw = _optional(payload, "patience", 5)
        patience = None if patience_raw is None else _integer(patience_raw, "patience", minimum=1)
        return cls(
            symbol=_token(_required(payload, "symbol"), "symbol"),
            timeframe=_token(_required(payload, "timeframe"), "timeframe"),
            max_trials=_integer(
                _required(payload, "max_trials"),
                "max_trials",
                minimum=1,
                maximum=MAX_DISCOVERY_TRIALS,
            ),
            seed=_integer(_required(payload, "seed"), "seed", minimum=0),
            stress=_numbers(_optional(payload, "stress", [1.5, 2.0]), "stress"),
            gate=_gate(_required(payload, "gate")),
            population=_integer(
                _optional(payload, "population", 16),
                "population",
                minimum=2,
                maximum=MAX_DISCOVERY_POPULATION,
            ),
            generations=_integer(
                _optional(payload, "generations", 5),
                "generations",
                minimum=1,
                maximum=MAX_DISCOVERY_GENERATIONS,
            ),
            patience=patience,
            horizon=_integer(_optional(payload, "horizon", 1), "horizon", minimum=1),
        )


@dataclass(frozen=True, slots=True)
class PaperAdvanceRequest:
    """Payload of ``paper.advance``."""

    symbol: str
    timeframe: str
    bars: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> PaperAdvanceRequest:
        _strict(payload, frozenset({"symbol", "timeframe", "bars"}))
        return cls(
            symbol=_token(_required(payload, "symbol"), "symbol"),
            timeframe=_token(_required(payload, "timeframe"), "timeframe"),
            bars=_integer(_required(payload, "bars"), "bars", minimum=1, maximum=MAX_PAPER_BARS),
        )


@dataclass(frozen=True, slots=True)
class AnalysisRequest:
    """Payload of ``paper.analyse``; an absent scope means every session."""

    symbol: str | None = None
    timeframe: str | None = None
    max_sessions: int = 100

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> AnalysisRequest:
        _strict(payload, frozenset({"symbol", "timeframe", "max_sessions"}))
        symbol = None if "symbol" not in payload else _token(payload["symbol"], "symbol")
        timeframe = (
            None if "timeframe" not in payload else _token(payload["timeframe"], "timeframe")
        )
        return cls(
            symbol=symbol,
            timeframe=timeframe,
            max_sessions=_integer(
                _optional(payload, "max_sessions", 100),
                "max_sessions",
                minimum=1,
                maximum=MAX_ANALYSED_SESSIONS,
            ),
        )


# ---------------------------------------------------------------------- handlers


def _ingest_handler(context: HandlerContext) -> Handler:
    def handler(job: Job, job_context: JobContext) -> HandlerResult:
        request = _parse(IngestRequest, job.payload, KIND_KNOWLEDGE_INGEST)
        checksum = _sha256_file(request.document_path)
        index_path = context.knowledge_index_path

        with KnowledgeIndex(index_path) as index:
            existing = index.document_by_checksum(checksum)
            if existing is not None:
                if existing.status != "indexed":
                    raise PermanentJobError(
                        f"{KIND_KNOWLEDGE_INGEST}: document {existing.doc_id} was already "
                        f"recorded with status {existing.status!r}: {existing.error or 'no reason'}"
                    )
                # Re-extract even for a known checksum: the dedupe in
                # ``extract_hypotheses`` makes this create nothing new, and a
                # crash between indexing and extraction still converges on a
                # retry instead of leaving a document with missing drafts. The
                # same pass backfills table rows added by schema version 2.
                record = DocumentIngestor(context.knowledge_root).ingest(request.document_path)
                backfilled = index.backfill_tables(record)
                sections = len(index.sections(existing.doc_id))
                extraction = extract_hypotheses(
                    index,
                    existing.doc_id,
                    max_hypotheses=request.max_hypotheses,
                    market_scope=request.market_scope,
                    timeframe_scope=request.timeframe_scope,
                )
                _log_json(
                    job_context,
                    "knowledge_duplicate",
                    {
                        "doc_id": existing.doc_id,
                        "checksum": checksum,
                        "sections": sections,
                        "new_drafts": len(extraction.created),
                        "tables_backfilled": backfilled,
                    },
                )
                return HandlerResult.success(
                    _jsonable(
                        {
                            "doc_id": existing.doc_id,
                            "checksum": checksum,
                            "status": existing.status,
                            "sections": sections,
                            "drafts": len(extraction.created),
                            "skipped": extraction.skipped,
                            "scanned_sentences": extraction.scanned_sentences,
                            "duplicate": True,
                            "stored_path": existing.stored_path,
                            "warnings": list(record.warnings),
                            "index": str(index_path),
                        }
                    )
                )

            ingestor = DocumentIngestor(context.knowledge_root)
            record = ingestor.ingest(request.document_path)
            index.add_document(record)
            if record.status != "indexed":
                raise PermanentJobError(
                    f"{KIND_KNOWLEDGE_INGEST}: cannot extract text from "
                    f"{request.document_path.name!r}: status={record.status!r} "
                    f"({record.error or 'no reason'})"
                )
            extraction = extract_hypotheses(
                index,
                record.doc_id,
                max_hypotheses=request.max_hypotheses,
                market_scope=request.market_scope,
                timeframe_scope=request.timeframe_scope,
            )
            _log_json(
                job_context,
                "knowledge_ingested",
                {
                    "doc_id": record.doc_id,
                    "sections": len(record.sections),
                    "drafts": len(extraction.created),
                    "skipped": extraction.skipped,
                },
            )
            return HandlerResult.success(
                _jsonable(
                    {
                        "doc_id": record.doc_id,
                        "checksum": record.checksum,
                        "status": record.status,
                        "sections": len(record.sections),
                        "drafts": len(extraction.created),
                        "skipped": extraction.skipped,
                        "scanned_sentences": extraction.scanned_sentences,
                        "duplicate": record.duplicate,
                        "stored_path": record.stored_path,
                        "warnings": list(record.warnings),
                        "index": str(index_path),
                    }
                )
            )

    return handler


def _dataset_handler(context: HandlerContext) -> Handler:
    def handler(job: Job, job_context: JobContext) -> HandlerResult:
        request = _parse(DatasetRequest, job.payload, KIND_DATASET_BUILD)
        store = ParquetStore(context.data_root)
        bars = store.read(request.symbol, request.timeframe)
        if bars.is_empty():
            raise PermanentJobError(
                f"{KIND_DATASET_BUILD}: no stored bars for {request.symbol} "
                f"{request.timeframe} under {context.data_root}; refusing to build a dataset "
                "for an unknown symbol/timeframe"
            )
        try:
            features = cached_build_features(
                context.feature_cache,
                symbol=request.symbol,
                timeframe=request.timeframe,
                frame=bars,
                settings=context.settings,
            )
            bundle = build_dataset(
                features,
                symbol=request.symbol,
                timeframe=request.timeframe,
                horizon=request.horizon,
                train_fraction=request.train_fraction,
                validation_fraction=request.validation_fraction,
                purge_bars=request.purge_bars,
                embargo_bars=request.embargo_bars,
                notes=request.notes,
            )
        except ValueError as exc:
            raise PermanentJobError(
                f"{KIND_DATASET_BUILD}: cannot build a chronological dataset for "
                f"{request.symbol} {request.timeframe}: {exc}"
            ) from exc

        manifest = bundle.manifest
        filename = (
            f"{request.symbol}-{request.timeframe}-h{request.horizon}-{manifest.checksum[:12]}.json"
        )
        path = context.datasets_root / filename
        save_manifest(path, bundle)
        payload = dict(manifest.to_dict())
        payload["bars_rows"] = int(bars.height)
        payload["purged_rows"] = int((bundle.split == -1).sum())
        payload["manifest_path"] = str(path)
        _log_json(
            job_context,
            "dataset_built",
            {
                "dataset_id": manifest.dataset_id,
                "rows": manifest.rows,
                "validation_rows": manifest.validation.rows,
                "checksum": manifest.checksum,
            },
        )
        return HandlerResult.success(_jsonable(payload))

    return handler


def _discovery_handler(context: HandlerContext) -> Handler:
    def handler(job: Job, job_context: JobContext) -> HandlerResult:
        request = _parse(DiscoveryRequest, job.payload, KIND_DISCOVERY_RUN)
        try:
            config = DiscoveryRunConfig(
                symbol=request.symbol,
                timeframe=request.timeframe,
                max_trials=request.max_trials,
                population=request.population,
                generations=request.generations,
                seed=request.seed,
                stress=request.stress,
                gate=request.gate,
                horizon=request.horizon,
                patience=request.patience,
                archive=context.discovery_archive_path,
                use_cache=context.use_feature_cache,
            )
        except (TypeError, ValueError) as exc:
            raise PermanentJobError(
                f"{KIND_DISCOVERY_RUN}: invalid search configuration: {exc}"
            ) from exc
        runner: DiscoveryRunner = (
            run_discovery if context.discovery_runner is None else context.discovery_runner
        )
        try:
            result = runner(
                config,
                root=context.data_root,
                should_stop=job_context.cancelled,
            )
        except DiscoveryRunError as exc:
            raise PermanentJobError(
                f"{KIND_DISCOVERY_RUN}: discovery cannot run as asked: {exc}"
            ) from exc

        if result.decision == "promoted":  # pragma: no cover - the runner forbids it
            raise RuntimeError(
                "discovery returned 'promoted'; this handler never stores a promotion"
            )
        data = {
            "decision": result.decision,
            "reason": result.reason,
            "promoted": False,
            "trials": result.trials,
            "accepted": result.accepted,
            "skipped": result.skipped,
            "deduped": result.deduped,
            "best_candidate_id": result.best_id,
            "dataset_id": result.dataset_id,
            "decision_id": result.decision_id,
            "stop_reason": result.stop_reason,
            "seed": result.seed,
            "code_version": result.code_version,
            "archive": str(config.archive),
        }
        _log_json(
            job_context,
            "discovery_decision",
            {
                "decision": result.decision,
                "trials": result.trials,
                "accepted": result.accepted,
                "best_candidate_id": result.best_id,
            },
        )
        return HandlerResult.success(_jsonable(data))

    return handler


def _no_trade(
    job_context: JobContext,
    *,
    scope: str,
    bars: int,
    reason: str,
    reason_class: str,
) -> HandlerResult:
    """A successful job whose recorded outcome is NO TRADE, not a failure."""
    _log_json(
        job_context,
        "paper_no_trade",
        {"scope": scope, "reason_class": reason_class, "reason": reason},
    )
    return HandlerResult.success(
        _jsonable(
            {
                "action": "no_trade",
                "mode": "continuous",
                "reason": reason,
                "reason_class": reason_class,
                "scope": scope,
                "bars_requested": bars,
                "bars_advanced": 0,
                "touched": False,
                "ledger_created": False,
                "champion": None,
            }
        )
    )


def _position_payload(position: PositionRecord | None) -> dict[str, Any] | None:
    if position is None:
        return None
    return {
        "side": position.side,
        "lot": position.lot,
        "entry_ts": position.entry_ts.isoformat(),
        "entry_price": position.entry_price,
    }


def _paper_advance_handler(context: HandlerContext) -> Handler:
    def handler(job: Job, job_context: JobContext) -> HandlerResult:
        request = _parse(PaperAdvanceRequest, job.payload, KIND_PAPER_ADVANCE)
        scope = scope_for(request.symbol, request.timeframe)
        archive_path = context.discovery_archive_path
        if not archive_path.exists():
            return _no_trade(
                job_context,
                scope=scope,
                bars=request.bars,
                reason=(
                    f"no discovery archive at {archive_path}; scope {scope!r} has no "
                    "governed champion, so the autonomous path does not trade"
                ),
                reason_class="no_champion",
            )

        with DiscoveryArchive(archive_path) as archive:
            provider = ChampionProvider(archive, scope=scope)
            try:
                champion = provider.champion()
            except NoChampionError as exc:
                return _no_trade(
                    job_context,
                    scope=scope,
                    bars=request.bars,
                    reason=str(exc),
                    reason_class="no_champion",
                )
            except NoEvidenceChampion as exc:
                return _no_trade(
                    job_context,
                    scope=scope,
                    bars=request.bars,
                    reason=str(exc),
                    reason_class="no_evidence_champion",
                )
            except InvalidChampionSpec as exc:
                return _no_trade(
                    job_context,
                    scope=scope,
                    bars=request.bars,
                    reason=str(exc),
                    reason_class="invalid_champion",
                )

        if ParquetStore(context.data_root).read(request.symbol, request.timeframe).is_empty():
            raise PermanentJobError(
                f"{KIND_PAPER_ADVANCE}: champion {champion.candidate_id[:12]} exists for "
                f"scope {scope!r} but there are no stored bars for "
                f"{request.symbol} {request.timeframe} under {context.data_root}"
            )

        ledger_path = context.paper_ledger_path
        with PaperLedger(ledger_path) as ledger:
            try:
                runner = ContinuousPaperRunner(
                    ledger,
                    champion.spec,
                    request.symbol,
                    request.timeframe,
                    context.settings,
                    store_root=context.data_root,
                    cache=context.use_feature_cache,
                    scope=scope,
                )
            except ValueError as exc:
                raise PermanentJobError(
                    f"{KIND_PAPER_ADVANCE}: the continuous paper runner refused the "
                    f"scope or the spec: {exc}"
                ) from exc
            try:
                advance = runner.advance()
            except BarGapError as exc:
                # Retryable: the missing bar may still be ingested. The typed
                # reason is part of the job error so the queue log names it.
                raise PaperAdvanceError("bar_gap", str(exc)) from exc
            except BarOrderError as exc:
                raise PermanentJobError(f"{KIND_PAPER_ADVANCE}: bar_order: {exc}") from exc
            except SessionMismatchError as exc:
                raise PermanentJobError(f"{KIND_PAPER_ADVANCE}: session_conflict: {exc}") from exc
            except ResumeError as exc:
                # The ledger cannot reproduce this session; a retry would invent state.
                raise PermanentJobError(f"{KIND_PAPER_ADVANCE}: session_conflict: {exc}") from exc
            except ContinuousError as exc:
                raise PermanentJobError(f"{KIND_PAPER_ADVANCE}: continuous_refused: {exc}") from exc

            state = ledger.resume(runner.session_id) if advance.status != "new" else None
            trades = ledger.trades(runner.session_id) if state is not None else []
            realized_pnl = float(sum(trade.pnl for trade in trades))
            position = None if state is None else _position_payload(state.position)
            data = {
                "action": "advanced",
                "mode": "continuous",
                "reason": (
                    f"governed champion {champion.candidate_id[:12]} advanced"
                    f" {advance.new_bars} bar(s) in continuous mode"
                ),
                "status": advance.status,
                "scope": scope,
                "session_id": runner.session_id,
                "spec_hash": champion.spec_hash,
                "candidate_id": champion.candidate_id,
                "promotion_id": champion.promotion_id,
                "bars_requested": request.bars,
                "bars_advanced": advance.new_bars,
                "bars_this_call": advance.new_bars,
                "new_signals": advance.new_signals,
                "new_trades": advance.new_trades,
                "bars_processed": advance.bars_processed,
                "trades": advance.trades,
                "realized_pnl": realized_pnl,
                "balance": None if state is None else state.balance,
                "equity": None if state is None else state.equity,
                "position": position,
                "last_bar_ts": (
                    None if advance.last_bar_ts is None else advance.last_bar_ts.isoformat()
                ),
                "done": advance.done,
                "done_reason": advance.reason,
                "ledger": str(ledger_path),
                "ledger_created": True,
            }
            _log_json(
                job_context,
                "paper_advanced",
                {
                    "scope": scope,
                    "session_id": runner.session_id,
                    "spec_hash": champion.spec_hash,
                    "mode": "continuous",
                    "bars": advance.new_bars,
                    "bars_processed": advance.bars_processed,
                },
            )
            return HandlerResult.success(_jsonable(data))

    return handler


def _data_fingerprint(
    context: HandlerContext,
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
) -> tuple[str | None, str]:
    """Fingerprint the stored bars the session's signals span.

    ``frame_meta`` hashes every bar column, so two different tapes that produce
    the same outcome cannot be confused. The window ends at the session's last
    signal, which keeps the attribution stable when new bars are appended later.
    A span above :data:`MAX_FINGERPRINT_SPAN` is not read at all; the caller
    records why the attribution is missing instead of inventing a hash.
    """
    if end - start > MAX_FINGERPRINT_SPAN:
        return None, "window_too_large"
    bars = ParquetStore(context.data_root).read(symbol, timeframe, start, end)
    if bars.is_empty():
        return None, "unavailable"
    return frame_meta(bars).fingerprint, "bar_fingerprint"


def _paper_analyse_handler(context: HandlerContext) -> Handler:
    def handler(job: Job, job_context: JobContext) -> HandlerResult:
        request = _parse(AnalysisRequest, job.payload, KIND_PAPER_ANALYSE)
        ledger_path = context.paper_ledger_path
        if not ledger_path.exists():
            _log_json(job_context, "outcomes_empty", {"reason": "no paper ledger"})
            return HandlerResult.success(
                _jsonable(
                    {
                        "status": "no_ledger",
                        "reason": f"no paper ledger at {ledger_path}",
                        "sessions_total": 0,
                        "sessions_in_scope": 0,
                        "sessions_analysed": 0,
                        "trades": 0,
                        "open_positions_excluded": 0,
                        "pnl_total": 0.0,
                        "pnl_mean": 0.0,
                        "r_total": 0.0,
                        "r_mean": 0.0,
                        "win_rate": 0.0,
                        "wins": 0,
                        "losses": 0,
                        "breakeven": 0,
                        "strategy_hashes": [],
                        "data_fingerprints": [],
                        "sessions": [],
                    }
                )
            )

        ledger = PaperLedger.open_reader(ledger_path)
        try:
            page, total = ledger.session_page(limit=request.max_sessions, offset=0)
            rows: list[dict[str, Any]] = []
            trades_total = 0
            open_excluded = 0
            pnl_total = 0.0
            r_total = 0.0
            wins = 0
            losses = 0
            breakeven = 0
            fingerprints: dict[tuple[str, str, str, str], tuple[str | None, str]] = {}

            for session in page:
                if request.symbol is not None and session.symbol != request.symbol:
                    continue
                if request.timeframe is not None and session.timeframe != request.timeframe:
                    continue
                signals = ledger.signals(session.session_id)
                trades = ledger.trades(session.session_id)
                open_positions = sum(
                    1 for item in ledger.positions(session.session_id) if item.status == "open"
                )
                session_pnl = float(sum(trade.pnl for trade in trades))
                session_r = float(sum(trade.r_multiple for trade in trades))
                session_wins = sum(1 for trade in trades if trade.pnl > 0.0)
                session_losses = sum(1 for trade in trades if trade.pnl < 0.0)
                config = session.config
                extra = config.get("extra")
                dsl = extra.get("dsl") if isinstance(extra, Mapping) else None
                config_spec_hash = dsl.get("spec_hash") if isinstance(dsl, Mapping) else None

                fingerprint: str | None = None
                attribution = "unavailable"
                first_bar = signals[0].ts if signals else None
                last_bar = signals[-1].ts if signals else None
                if first_bar is not None and last_bar is not None:
                    key = (
                        session.symbol,
                        session.timeframe,
                        first_bar.isoformat(),
                        last_bar.isoformat(),
                    )
                    if key not in fingerprints:
                        fingerprints[key] = _data_fingerprint(
                            context, session.symbol, session.timeframe, first_bar, last_bar
                        )
                    fingerprint, attribution = fingerprints[key]

                trades_total += len(trades)
                open_excluded += open_positions
                pnl_total += session_pnl
                r_total += session_r
                wins += session_wins
                losses += session_losses
                breakeven += len(trades) - session_wins - session_losses
                rows.append(
                    {
                        "session_id": session.session_id,
                        "symbol": session.symbol,
                        "timeframe": session.timeframe,
                        "status": session.status,
                        "strategy": session.strategy,
                        "spec_hash": session.model_version,
                        "config_spec_hash": (
                            None if config_spec_hash is None else str(config_spec_hash)
                        ),
                        "signals": len(signals),
                        "closed_trades": len(trades),
                        "open_positions_excluded": open_positions,
                        "pnl_total": session_pnl,
                        "r_total": session_r,
                        "r_mean": session_r / len(trades) if trades else 0.0,
                        "wins": session_wins,
                        "losses": session_losses,
                        "win_rate": session_wins / len(trades) if trades else 0.0,
                        "first_bar_ts": None if first_bar is None else first_bar.isoformat(),
                        "last_bar_ts": None if last_bar is None else last_bar.isoformat(),
                        "data_fingerprint": fingerprint,
                        "data_attribution": attribution,
                    }
                )

            strategy_hashes = sorted(
                {
                    row["spec_hash"]
                    for row in rows
                    if isinstance(row["spec_hash"], str) and row["spec_hash"]
                }
            )
            data_fingerprints = sorted(
                {
                    row["data_fingerprint"]
                    for row in rows
                    if isinstance(row["data_fingerprint"], str) and row["data_fingerprint"]
                }
            )
            data = {
                "status": "analysed",
                "sessions_total": total,
                "sessions_in_scope": len(rows),
                "sessions_analysed": sum(1 for row in rows if row["closed_trades"] > 0),
                "truncated": total > len(page),
                "trades": trades_total,
                "open_positions_excluded": open_excluded,
                "pnl_total": pnl_total,
                "pnl_mean": pnl_total / trades_total if trades_total else 0.0,
                "r_total": r_total,
                "r_mean": r_total / trades_total if trades_total else 0.0,
                "win_rate": wins / trades_total if trades_total else 0.0,
                "wins": wins,
                "losses": losses,
                "breakeven": breakeven,
                "strategy_hashes": strategy_hashes,
                "data_fingerprints": data_fingerprints,
                "sessions": rows,
            }
            _log_json(
                job_context,
                "outcomes_analysed",
                {
                    "trades": trades_total,
                    "open_positions_excluded": open_excluded,
                    "sessions": len(rows),
                },
            )
            return HandlerResult.success(_jsonable(data))
        finally:
            ledger.close()

    return handler


def register_default_handlers(
    registry: HandlerRegistry, context: HandlerContext
) -> tuple[str, ...]:
    """Bind the five pipeline kinds to their handlers; idempotent per registry.

    ``registry`` is anything with ``register(kind, handler)`` — an
    :class:`~scalping.orchestrator.core.Orchestrator` in production. The context
    is captured by the closures; a second registration replaces the first, and
    registering two different contexts into one registry is the caller's error,
    not a hidden global.
    """
    if not isinstance(context, HandlerContext):
        raise TypeError(f"context must be a HandlerContext, got {type(context).__name__}")
    handlers: dict[str, Handler] = {
        KIND_KNOWLEDGE_INGEST: _ingest_handler(context),
        KIND_DATASET_BUILD: _dataset_handler(context),
        KIND_DISCOVERY_RUN: _discovery_handler(context),
        KIND_PAPER_ADVANCE: _paper_advance_handler(context),
        KIND_PAPER_ANALYSE: _paper_analyse_handler(context),
    }
    for kind, handler in handlers.items():
        registry.register(kind, handler)
    return tuple(sorted(handlers))
