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
    "SYMBOL_PATTERN",
    "BacktestRequest",
    "BacktestResponse",
    "BanditWeight",
    "CalibrateRequest",
    "ControlRequest",
    "DataStatusResponse",
    "DownloadRequest",
    "DownloadResponse",
    "EvaluateRequest",
    "EvaluateResponse",
    "FoldSummary",
    "HealthResponse",
    "OkResponse",
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
    "TrainingPoint",
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


class DownloadRequest(BaseModel):
    provider: ProviderName
    symbol: SymbolId
    start: datetime
    end: datetime
    seed: int | None = None
    path: str | None = None
    tz_offset: int | None = None

    @model_validator(mode="after")
    def _csv_needs_a_path(self) -> DownloadRequest:
        if self.provider == "csv" and not self.path:
            raise ValueError("path is required when provider is 'csv'")
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
    run_id: str = Field(min_length=1)
    symbol: SymbolId


class BacktestResponse(BaseModel):
    report_path: str
    metrics: dict[str, Any]


class EvaluateRequest(BaseModel):
    run_id: str = Field(min_length=1)
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


class CalibrateRequest(BaseModel):
    """Parameters for a nightly rule recalibration."""

    symbol: SymbolId = "XAUUSD"
    timeframe: Literal["M1", "M5", "M15", "M30", "H1", "H4", "D1"] = "H1"
    holdout_fraction: float = Field(default=0.2, gt=0.0, lt=1.0)
    min_train_trades: int = Field(default=100, ge=10)
    min_holdout_trades: int = Field(default=30, ge=5)
    dry_run: bool = False
