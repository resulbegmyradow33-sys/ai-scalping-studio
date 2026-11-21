"""Request and response models for the API.

Every field the brief names is part of a model here, so FastAPI rejects a
malformed body with a 422 instead of an endpoint crashing on it. Timestamps are
plain datetimes: providers treat a naive value as already-UTC.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scalping.serialize import metrics_payload

__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "BanditWeight",
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
    "RunControlRequest",
    "RunDetail",
    "RunSummary",
    "StartRequest",
    "StartResponse",
    "StressSummary",
    "TrainRequest",
    "TrainResponse",
    "TrainingPoint",
    "metrics_payload",
]

ProviderName = Literal["synthetic", "csv", "dukascopy"]
ControlAction = Literal["pause", "resume", "stop", "reset"]


class HealthResponse(BaseModel):
    status: str
    version: str
    device: str


class DataStatusResponse(BaseModel):
    symbol: str
    rows: int
    first_ts: str | None
    last_ts: str | None
    providers: list[str]


class DownloadRequest(BaseModel):
    provider: ProviderName
    symbol: str = Field(min_length=1)
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
    symbol: str = Field(min_length=1)
    use_strategy: bool = True
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
    symbol: str = Field(min_length=1)


class BacktestResponse(BaseModel):
    report_path: str
    metrics: dict[str, Any]


class EvaluateRequest(BaseModel):
    run_id: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


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
    folds: list[FoldSummary]
    overall: dict[str, Any]
    overfit_warning: bool
    stress: list[StressSummary]
