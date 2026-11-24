"""Request and response models for the API.

Every field the brief names is part of a model here, so FastAPI rejects a
malformed body with a 422 instead of an endpoint crashing on it. Timestamps are
plain datetimes: providers treat a naive value as already-UTC.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Annotated, Any, Final, Literal

from pydantic import BaseModel, Field, StringConstraints, model_validator

from scalping.serialize import metrics_payload

__all__ = [
    "IMPORT_NAME_PATTERN",
    "ML_RUN_ID_PATTERN",
    "PAPER_SESSION_PATTERN",
    "SYMBOL_PATTERN",
    "TRAIN_RUN_ID_PATTERN",
    "AutonomyChampion",
    "AutonomyChampionList",
    "AutonomyDataStatus",
    "AutonomyDiscoveries",
    "AutonomyEventItem",
    "AutonomyEventPage",
    "AutonomyJobItem",
    "AutonomyJobPage",
    "AutonomyLastEvent",
    "AutonomyOrchestratorStatus",
    "AutonomyPaperStatus",
    "AutonomyStatus",
    "BacktestRequest",
    "BacktestResponse",
    "BanditWeight",
    "CalibrateRequest",
    "ControlRequest",
    "DataQualityResponse",
    "DataStatusResponse",
    "DownloadRequest",
    "DownloadResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "FoldSummary",
    "HealthResponse",
    "ImportFileName",
    "MLCalibration",
    "MLControlRequest",
    "MLEvaluateRequest",
    "MLEvaluateResponse",
    "MLRunDetail",
    "MLRunId",
    "MLRunSummary",
    "MLTimeframe",
    "MLTrainRequest",
    "MLTrainResponse",
    "OkResponse",
    "PaperEquityPage",
    "PaperEquityPoint",
    "PaperOrder",
    "PaperOrderPage",
    "PaperPosition",
    "PaperPositionPage",
    "PaperSessionCreate",
    "PaperSessionCreated",
    "PaperSessionId",
    "PaperSessionItem",
    "PaperSessionPage",
    "PaperSessionSummary",
    "PaperTimeframe",
    "PaperTrade",
    "PaperTradePage",
    "ProviderName",
    "RegimeCount",
    "RegimeCurrent",
    "RegimePoint",
    "RegimeResponse",
    "RunControlRequest",
    "RunDetail",
    "RunSummary",
    "StartRequest",
    "StartResponse",
    "StressSummary",
    "SymbolId",
    "TimeframeSpec",
    "TrainRequest",
    "TrainResponse",
    "TrainRunId",
    "TrainingPoint",
    "is_ml_run_id",
    "is_symbol_id",
    "metrics_payload",
]

ProviderName = Literal["synthetic", "csv", "dukascopy"]
ControlAction = Literal["pause", "resume", "stop", "reset"]

#: A symbol becomes a directory name under the data root, so the HTTP boundary
#: only admits a single path segment made of letters, digits, dot, dash and
#: underscore. The first character is alphanumeric, which rules out ``.`` and
#: ``..``; separators are never accepted, so no value can traverse upwards.
#: Validating here means a bad id is a 422 from request parsing, before any
#: store or provenance path is built.
SYMBOL_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

_SYMBOL_RE: Final[re.Pattern[str]] = re.compile(SYMBOL_PATTERN)
#: Upper bound keeps a hostile 10k-character id from reaching the filesystem,
#: where it would surface as ``ENAMETOOLONG`` (a 500) instead of a 422.
_MAX_SYMBOL_LENGTH: Final[int] = 32

#: Constrained symbol for every request model and query parameter.
SymbolId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=_MAX_SYMBOL_LENGTH,
        pattern=SYMBOL_PATTERN,
    ),
]

#: An HTTP CSV import names a file, never a path: one segment, no separators,
#: no leading dot and no drive letter. The server looks the name up in
#: ``<data_root>/imports`` and never joins it to anything.
IMPORT_NAME_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

ImportFileName = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=255,
        pattern=IMPORT_NAME_PATTERN,
    ),
]

#: A paper session id is an opaque token. It is never a path segment: no
#: separator, no leading dot and a bounded length, so a client cannot steer a
#: filesystem lookup with it even if a future caller forgets to keep it opaque.
PAPER_SESSION_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9_-]*$"

PaperSessionId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=PAPER_SESSION_PATTERN,
    ),
]

#: Bar sizes a paper session may be opened on; mirrors ``data/registry.py``.
PaperTimeframe = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]

#: An ML run id is an opaque token, never a path: alphanumeric only, bounded.
#: The runner resolves it under ``runs/ml`` and the routes reject anything else
#: before a filesystem lookup, so a separator or ``..`` cannot be expressed.
ML_RUN_ID_PATTERN: Final[str] = r"^[A-Za-z0-9]{1,64}$"

_ML_RUN_ID_RE: Final[re.Pattern[str]] = re.compile(ML_RUN_ID_PATTERN)

MLRunId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=64,
        pattern=ML_RUN_ID_PATTERN,
    ),
]

#: Bar sizes an ML run may be trained on; mirrors ``data/registry.py``.
MLTimeframe = Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"]


#: A training run id becomes a directory name under ``runs_root``. The runner
#: already refuses separators and dot-only values; the HTTP boundary applies the
#: same rule plus a length bound, so an absurd id is a 422 from request parsing
#: instead of an ``OSError`` (``ENAMETOOLONG``) raised while probing the disk.
#: The generated ids are 12 hex characters; the wider class keeps hand-written
#: run directories (``odd``, ``dead``) addressable.
TRAIN_RUN_ID_PATTERN: Final[str] = r"^[A-Za-z0-9][A-Za-z0-9._-]*$"

TrainRunId = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=TRAIN_RUN_ID_PATTERN,
    ),
]


def is_ml_run_id(value: str) -> bool:
    """Whether ``value`` is a safe, opaque run token (no path component)."""
    return len(value) <= 64 and _ML_RUN_ID_RE.fullmatch(value) is not None


def is_symbol_id(value: str) -> bool:
    """Whether ``value`` is a safe single-segment symbol id.

    Query parameters are plain ``str`` to FastAPI, so routes call this before
    touching the store. It applies exactly the constraint :data:`SymbolId`
    puts on request bodies.
    """
    return len(value) <= _MAX_SYMBOL_LENGTH and _SYMBOL_RE.fullmatch(value) is not None


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str


class TimeframeSpec(BaseModel):
    """One supported bar size and the Polars rule that builds it from M1."""

    timeframe: str
    rule: str


class DataStatusResponse(BaseModel):
    symbol: str
    timeframe: str = "M1"
    rows: int
    first_ts: str | None
    last_ts: str | None
    providers: list[str]
    provenance: dict[str, Any] | None = None


class DataQualityResponse(BaseModel):
    """Read-only quality report for one stored slice.

    Violation counts are ``None`` when the check did not run: every count is
    null for an unconfigured slice, and ``outliers`` is null when the slice is
    shorter than the robust-z floor or the outlier pass was skipped. A zero
    therefore means "checked and clean", never "not checked".
    """

    symbol: str
    timeframe: str
    configured: bool
    status: Literal["ok", "warn", "fail"]
    status_rule: str
    status_detail: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    duplicates: int | None
    gaps: int | None
    ohlc_violations: int | None
    crossed_quotes: int | None
    negative_spread: int | None
    negative_volume: int | None
    outliers: int | None
    outliers_checked: bool
    provenance: dict[str, Any] | None
    thresholds: dict[str, Any]
    issues: list[dict[str, Any]]


class DownloadRequest(BaseModel):
    """One synchronous provider fetch.

    With ``provider='csv'``, ``path`` is a bare file **name**, not a filesystem
    path: the export must already sit directly in ``<data_root>/imports`` and
    the request names it there. The server looks the name up in that directory,
    so the opened path is never built from request data.
    """

    provider: ProviderName
    symbol: SymbolId
    start: datetime
    end: datetime
    seed: int | None = None
    path: ImportFileName | None = None
    tz_offset: int | None = None

    @model_validator(mode="after")
    def _csv_needs_a_path(self) -> DownloadRequest:
        if self.provider == "csv" and not self.path:
            raise ValueError(
                "path (a file name in <data_root>/imports) is required when provider is 'csv'"
            )
        return self


class RegimePoint(BaseModel):
    """One bar on the regime timeline."""

    ts: str
    regime: str
    trend: str
    volatility: str


class RegimeCount(BaseModel):
    """Bars spent in one summary regime."""

    regime: str
    bars: int
    share: float


class RegimeCurrent(BaseModel):
    """The latest bar's axes; every field comes from the rule classifier."""

    ts: str
    trend: str
    trend_strength: str
    volatility: str
    expansion: str
    breakout: str
    unusual: bool
    regime: str


class RegimeResponse(BaseModel):
    """Descriptive regime state for one stored slice, never a forecast."""

    symbol: str
    timeframe: str
    rows: int
    current: RegimeCurrent | None
    counts: list[RegimeCount]
    series: list[RegimePoint]


class DownloadResponse(BaseModel):
    rows: int


class StartRequest(BaseModel):
    speed: float = Field(default=1.0, ge=0.0)
    mode: Literal["replay"] = "replay"


class StartResponse(BaseModel):
    session_id: str


class ControlRequest(BaseModel):
    action: ControlAction


class OkResponse(BaseModel):
    ok: bool


class TrainRequest(BaseModel):
    """Defaults match :class:`~scalping.rl.train.TrainConfig`.

    ``train_end`` cuts the raw stored bars at that instant; the run trains only
    on bars at or before it and records the first later bar as ``eval_start``.
    """

    total_steps: int = Field(default=200_000, ge=1)
    n_envs: int = Field(default=2, ge=1)
    seed: int = Field(default=0, ge=0)
    symbol: SymbolId
    use_strategy: bool = True
    entry_penalty: float = Field(default=0.05, ge=0.0, le=1.0)
    gate_signals: bool = True
    window: int = Field(default=60, ge=1, le=600)
    timeframe: str = "M1"
    sl_atr_mult: float = Field(default=1.0, gt=0.0, le=20.0)
    tp_atr_mult: float = Field(default=2.0, gt=0.0, le=20.0)
    max_bars_in_trade: int = Field(default=120, ge=1, le=100_000)
    train_end: datetime | None = None


class TrainResponse(BaseModel):
    run_id: str


class RunSummary(BaseModel):
    """One row of the run registry, as the panel lists it."""

    run_id: str
    status: str
    created_at: str | None
    steps: int
    reward_mean: float | None
    reward_mean_normalized: float | None = None
    degenerate: bool | None = None
    baseline_hold_net_profit: float | None = None


class TrainingPoint(BaseModel):
    """One checkpoint on a training curve; ``step`` is the x axis."""

    step: int
    value: float


class BanditWeight(BaseModel):
    """One arm of a stored LinUCB: the norm of its coefficient vector.

    ``weight`` is not a reward. It is ``||theta_arm||`` over the context
    coefficients and says how strongly that strategy's context moves the arm's
    predicted return; the intercept is not included.
    """

    strategy: str
    weight: float | None


class RunDetail(BaseModel):
    config: dict[str, Any]
    metrics: dict[str, Any]
    status: str
    step: int = 0
    total_steps: int | None = None
    series: dict[str, list[TrainingPoint]] | None = None
    bandit: list[BanditWeight] | None = None


class RunControlRequest(BaseModel):
    action: Literal["pause", "resume", "stop"]


class BacktestRequest(BaseModel):
    run_id: TrainRunId
    symbol: SymbolId


class BacktestResponse(BaseModel):
    report_path: str
    metrics: dict[str, Any]


class EvaluateRequest(BaseModel):
    run_id: TrainRunId
    symbol: SymbolId
    folds: int = Field(default=3, ge=1, le=12)


class FoldSummary(BaseModel):
    """One walk-forward window and the metrics of its test run."""

    train_range: list[str]
    test_range: list[str]
    evaluated_range: list[str]
    metrics: dict[str, Any]
    train_expectancy: float | None = None


class StressSummary(BaseModel):
    """One cost multiplier and the metrics it produced."""

    multiplier: float
    metrics: dict[str, Any]


class EvaluateResponse(BaseModel):
    report_path: str
    train_end: str | None = None
    eval_start: str | None = None
    data_hash: str
    eval_data_hash: str
    in_sample: bool = False
    folds: list[FoldSummary]
    overall: dict[str, Any]
    overfit_warning: bool
    stress: list[StressSummary]
    method: str = "walk_forward"


# --- classical ML API --------------------------------------------------------
#
# A training request names a model and a stored slice; it never names a path or
# a run directory. The server generates the opaque run id. Metrics stay as
# dicts: the ML report contains nested precision/recall/confusion/calibration
# tables whose shape is owned by :mod:`scalping.ml.train`, not by this model.


class MLTrainRequest(BaseModel):
    """One classical classifier run and the trading evaluation it includes.

    ``groups=None`` means every registered feature group. ``stress`` is a list
    of cost multipliers for the optional stress table; ``regime`` adds the
    per-regime breakdown of the base evaluation.
    """

    symbol: SymbolId
    timeframe: MLTimeframe = "H1"
    model: str = Field(default="logistic", min_length=1, max_length=32)
    horizon: int = Field(default=5, ge=1, le=1000)
    groups: list[str] | None = Field(default=None, max_length=16)
    train_fraction: float = Field(default=0.70, gt=0.0, lt=1.0)
    validation_fraction: float = Field(default=0.15, ge=0.0, lt=1.0)
    purge_bars: int = Field(default=5, ge=0, le=100_000)
    embargo_bars: int = Field(default=5, ge=0, le=100_000)
    seed: int = Field(default=0, ge=0)
    evaluate_test: bool = False
    include_flat: bool = False
    split: Literal["train", "validation", "test"] = "validation"
    threshold: float = Field(default=0.55, ge=0.5, lt=1.0)
    exit_threshold: float = Field(default=0.55, ge=0.5, le=1.0)
    window: int = Field(default=60, ge=1, le=600)
    stress: list[Annotated[float, Field(gt=0.0, le=10.0)]] = Field(
        default_factory=list, max_length=5
    )
    regime: bool = True

    @model_validator(mode="after")
    def _fractions_fit(self) -> MLTrainRequest:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError("train_fraction + validation_fraction must be < 1")
        return self


class MLTrainResponse(BaseModel):
    run_id: str


class MLRunSummary(BaseModel):
    """One row of ``runs/ml``, as a run list shows it."""

    run_id: str
    status: str
    created_at: str | None = None
    model: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    dataset_id: str | None = None


class MLCalibration(BaseModel):
    """Flattened calibration of the validation split."""

    brier: float | None = None
    log_loss: float | None = None
    ece: float | None = None
    mce: float | None = None


class MLRunDetail(BaseModel):
    """Everything the panel reads about one run; predictions, not trading."""

    run_id: str
    status: str
    created_at: str | None = None
    finished_at: str | None = None
    model: str | None = None
    symbol: str | None = None
    timeframe: str | None = None
    dataset_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)
    spec: dict[str, Any] = Field(default_factory=dict)
    columns: list[str] = Field(default_factory=list)
    validation: dict[str, Any] | None = None
    test: dict[str, Any] | None = None
    baseline: dict[str, Any] | None = None
    split_counts: dict[str, int] = Field(default_factory=dict)
    calibration: MLCalibration | None = None
    error: str | None = None


class MLControlRequest(BaseModel):
    action: Literal["cancel"]


class MLEvaluateRequest(BaseModel):
    """Trade a finished run's artefact on one split; no symbol or path field."""

    run_id: MLRunId
    split: Literal["train", "validation", "test"] = "validation"
    threshold: float = Field(default=0.55, ge=0.5, lt=1.0)
    exit_threshold: float = Field(default=0.55, ge=0.5, le=1.0)
    window: int = Field(default=60, ge=1, le=600)
    stress: list[Annotated[float, Field(gt=0.0, le=10.0)]] = Field(
        default_factory=list, max_length=5
    )
    regime: bool = True


class MLEvaluateResponse(BaseModel):
    """Headline, hold baseline, cost stress and the descriptive regime table."""

    run_id: str
    symbol: str
    timeframe: str
    dataset_id: str
    split: str
    threshold: float
    exit_threshold: float
    metrics: dict[str, Any]
    hold: dict[str, Any]
    stress: dict[str, dict[str, Any]]
    regime: dict[str, Any]


class CalibrateRequest(BaseModel):
    """Parameters for a nightly rule recalibration."""

    symbol: SymbolId = "XAUUSD"
    timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"] = "H1"
    holdout_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    min_train_trades: int = Field(default=100, ge=10)
    min_holdout_trades: int = Field(default=30, ge=5)
    dry_run: bool = False


# --- paper trading -----------------------------------------------------------
#
# The paper API reads and writes the ledger; it never names a file. The ledger
# path is configuration on the app, so no field here can move it. Timestamps are
# serialised from the ledger's aware datetimes and are always UTC.


class PaperSessionCreate(BaseModel):
    """Open one paper session record.

    ``config`` is stored under ``extra`` in the session's ``config_json`` next to
    ``{"origin": "api"}``; it is not used to configure a runner in this slice.
    """

    symbol: SymbolId
    timeframe: PaperTimeframe = "M1"
    strategy: str = Field(default="manual", min_length=1, max_length=64)
    model_version: str | None = Field(default=None, max_length=128)
    config: dict[str, Any] | None = None


class PaperSessionCreated(BaseModel):
    """The id the server generated; the caller never chooses it."""

    session_id: str
    status: str
    started_at: datetime


class PaperSessionItem(BaseModel):
    """One row of the session list."""

    session_id: str
    symbol: str
    timeframe: str
    strategy: str
    model_version: str | None
    status: str
    started_at: datetime
    ended_at: datetime | None


class PaperPosition(BaseModel):
    """A position lifecycle; ``closed_ts`` is set when a trade settles it."""

    position_id: int
    side: str
    lot: float
    entry_ts: datetime
    entry_price: float
    sl: float
    tp: float
    status: str
    closed_ts: datetime | None
    closed_price: float | None


class PaperOrder(BaseModel):
    """One order attempt; a rejection always carries its reason."""

    order_id: int
    signal_id: int | None
    ts: datetime
    side: str
    lot: float
    requested_price: float | None
    status: str
    rejection_reason: str | None


class PaperTrade(BaseModel):
    """One closed trade, exactly as the broker settled it."""

    trade_id: int
    side: str
    entry_ts: datetime
    entry_price: float
    exit_ts: datetime
    exit_price: float
    lot: float
    pnl: float
    r_multiple: float
    exit_reason: str


class PaperEquityPoint(BaseModel):
    """One account mark."""

    ts: datetime
    balance: float
    equity: float


class PaperSessionSummary(BaseModel):
    """Everything the session panel shows above the tables.

    ``balance`` and ``equity`` are ``None`` until the first equity mark;
    ``trade_count`` counts closed trades only.
    """

    session_id: str
    symbol: str
    timeframe: str
    strategy: str
    model_version: str | None
    status: str
    started_at: datetime
    ended_at: datetime | None
    balance: float | None
    equity: float | None
    open_position: PaperPosition | None
    trade_count: int
    last_bar_ts: datetime | None
    config: dict[str, Any]


class PaperSessionPage(BaseModel):
    """A bounded page of sessions plus the total behind it."""

    items: list[PaperSessionItem]
    total: int
    limit: int
    offset: int


class PaperPositionPage(BaseModel):
    items: list[PaperPosition]
    total: int
    limit: int
    offset: int


class PaperOrderPage(BaseModel):
    items: list[PaperOrder]
    total: int
    limit: int
    offset: int


class PaperTradePage(BaseModel):
    items: list[PaperTrade]
    total: int
    limit: int
    offset: int


class PaperEquityPage(BaseModel):
    """The newest equity window in chronological order; ``offset`` pages back."""

    items: list[PaperEquityPoint]
    total: int
    limit: int
    offset: int


# ------------------------------------------------------------------ autonomy


class AutonomyPaperStatus(BaseModel):
    """Paper ledger counts.

    A missing ledger is ``configured=False``. A file that exists but cannot be
    opened or has no schema is ``configured=True, readable=False``; in both cases
    the counts are absent instead of zeros that were never read.
    """

    configured: bool
    readable: bool
    sessions: int | None = None
    open_positions: int | None = None
    closed_trades: int | None = None


class AutonomyLastEvent(BaseModel):
    """The newest audit line in the orchestrator database."""

    id: int
    ts: str
    event: str
    job_id: int | None = None


class AutonomyOrchestratorStatus(BaseModel):
    """Queue counts and persisted controls.

    ``readable=False`` means the file could not be opened or the jobs table lacks
    a column this reader needs; every count and flag is then absent.
    """

    configured: bool
    readable: bool
    queued: int | None = None
    running: int | None = None
    failed: int | None = None
    paused: bool | None = None
    safe_mode: bool | None = None
    research_disabled: bool | None = None
    last_event: AutonomyLastEvent | None = None


class AutonomyChampion(BaseModel):
    """One scope's champion.

    ``candidate_id`` is the archive's sha256 identity. ``policy_digest`` repeats
    it unless the payload records a more specific digest; ``policy_id`` is only
    present when the payload names one.
    """

    scope: str
    candidate_id: str
    mode: str | None = None
    promoted_at: str | None = None
    policy_id: str | None = None
    policy_digest: str | None = None


class AutonomyChampionList(BaseModel):
    """Every scope with a champion.

    ``items`` is present only when the archive and its champion table are
    readable; an unreadable archive omits it rather than answering "no champions".
    """

    configured: bool
    readable: bool
    items: list[AutonomyChampion] | None = None


class AutonomyDiscoveries(BaseModel):
    """Archived evaluations (``runs``) and gate decisions; absent tables are ``None``."""

    configured: bool
    readable: bool
    runs: int | None = None
    decisions: int | None = None


class AutonomyDataStatus(BaseModel):
    """Data symbols with at least one month partition on disk."""

    configured: bool
    readable: bool
    symbols: list[str] | None = None


class AutonomyStatus(BaseModel):
    """The aggregate dashboard answer; each section owns its own readable flag."""

    paper: AutonomyPaperStatus
    orchestrator: AutonomyOrchestratorStatus
    champions: list[AutonomyChampion]
    champions_readable: bool
    discoveries: AutonomyDiscoveries
    data: AutonomyDataStatus


class AutonomyJobItem(BaseModel):
    """One recent job row, including the deferral counter that explains a stall."""

    id: int
    kind: str
    status: str
    attempts: int
    defer_count: int
    priority: int
    created_at: str
    scheduled_at: str
    started_at: str | None = None
    finished_at: str | None = None


class AutonomyJobPage(BaseModel):
    """Newest jobs first; bounded by ``limit``; ``items`` is absent if unreadable."""

    configured: bool
    readable: bool
    items: list[AutonomyJobItem] | None = None
    limit: int


class AutonomyEventItem(BaseModel):
    """One audit event; a control event has no ``job_id``.

    The free-form ``detail`` column is deliberately not exposed: it can contain
    local paths or handler errors that are not part of the status contract.
    """

    id: int
    job_id: int | None = None
    ts: str
    event: str


class AutonomyEventPage(BaseModel):
    """Newest audit events first; bounded by ``limit``; ``items`` is absent if unreadable."""

    configured: bool
    readable: bool
    items: list[AutonomyEventItem] | None = None
    limit: int
