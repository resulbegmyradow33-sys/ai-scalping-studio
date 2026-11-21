"""Request and response models for the API.

Every field the brief names is part of a model here, so FastAPI rejects a
malformed body with a 422 instead of an endpoint crashing on it. Timestamps are
plain datetimes: providers treat a naive value as already-UTC.
"""

from __future__ import annotations

import math
from dataclasses import fields
from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from scalping.backtest.metrics import Metrics

__all__ = [
    "ControlRequest",
    "DataStatusResponse",
    "DownloadRequest",
    "DownloadResponse",
    "HealthResponse",
    "OkResponse",
    "ProviderName",
    "StartRequest",
    "StartResponse",
    "metrics_payload",
]

ProviderName = Literal["synthetic", "csv", "dukascopy"]
ControlAction = Literal["pause", "resume", "stop", "reset"]


def metrics_payload(metrics: Metrics) -> dict[str, Any]:
    """Metrics by field name, with ``inf``/``nan`` mapped to JSON ``null``.

    ``Metrics.profit_factor`` is legitimately ``inf`` for a win-only run, and a
    JSON response with a raw ``inf`` fails ``allow_nan=False`` serialisers.
    """
    payload: dict[str, Any] = {}
    for field_ in fields(metrics):
        value = getattr(metrics, field_.name)
        if isinstance(value, float) and not math.isfinite(value):
            value = None
        payload[field_.name] = value
    return payload


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
