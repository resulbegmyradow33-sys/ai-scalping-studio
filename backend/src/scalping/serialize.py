"""JSON-safe projections of run results shared by the API and the CLI.

Keeping this outside ``scalping.api`` lets a data-only CLI command serialise a
backtest report without importing FastAPI.
"""

from __future__ import annotations

import math
from dataclasses import fields
from typing import Any

from scalping.backtest.metrics import Metrics

__all__ = ["metrics_payload"]


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
