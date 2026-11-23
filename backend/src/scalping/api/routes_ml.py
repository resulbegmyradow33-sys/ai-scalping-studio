"""Classical ML HTTP API.

Every heavy call goes through :class:`~scalping.api.ml_runner.MLRunner`, which
owns the one-job slot and the worker thread. The handlers only translate the
runner's errors into the API's status codes: 422 for a bad request field, 409
for a job that cannot be served now, 404 for an unknown run or a run without an
artefact. No request field names a filesystem path.
"""

from __future__ import annotations

import asyncio
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from scalping.api.ml_runner import (
    MLControlError,
    MLError,
    MLRequest,
    MLRunner,
    UnknownMLRunError,
)
from scalping.api.schemas import (
    MLControlRequest,
    MLEvaluateRequest,
    MLEvaluateResponse,
    MLRunDetail,
    MLRunId,
    MLRunSummary,
    MLTrainRequest,
    MLTrainResponse,
    OkResponse,
)
from scalping.features.registry import DEFAULT_GROUPS
from scalping.ml.models import MODEL_NAMES
from scalping.ml.train import TrainSpec

__all__ = ["router"]

router = APIRouter(prefix="/api/ml", tags=["ml"])


def _ml(request: Request) -> MLRunner:
    runner: MLRunner = request.app.state.ml
    return runner


@router.get("/models", response_model=list[str])
def ml_models() -> list[str]:
    """Model names the factory can build, in registry order.

    The UI reads this instead of hard-coding the list, so a new estimator or a
    removed optional backend reaches the panel without a frontend edit. The
    names are exactly the strings ``POST /api/ml/train`` accepts; whether an
    optional backend (lightgbm/xgboost) actually imports is a separate,
    per-run error.
    """
    return list(MODEL_NAMES)


@router.post("/train", response_model=MLTrainResponse, status_code=202)
async def ml_train(body: MLTrainRequest, request: Request) -> MLTrainResponse:
    """Prepare a run off-loop and return its id; the fit runs on a thread.

    Validation, the store read and the feature build happen before the 202, so
    a bad field is 422 and an unusable store is 409 instead of a run that only
    fails later. Preparation runs in a worker thread because it is CPU bound.
    """
    spec = TrainSpec(
        model=body.model,
        horizon=body.horizon,
        groups=tuple(body.groups) if body.groups is not None else DEFAULT_GROUPS,
        train_fraction=body.train_fraction,
        validation_fraction=body.validation_fraction,
        purge_bars=body.purge_bars,
        embargo_bars=body.embargo_bars,
        seed=body.seed,
        evaluate_test=body.evaluate_test,
        include_flat=body.include_flat,
    )
    ml_request = MLRequest(
        symbol=body.symbol,
        timeframe=body.timeframe,
        spec=spec,
        split=body.split,
        threshold=body.threshold,
        exit_threshold=body.exit_threshold,
        window=body.window,
        stress=tuple(body.stress),
        regime=body.regime,
    )
    try:
        run_id = await asyncio.to_thread(_ml(request).start, ml_request)
    except MLError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MLTrainResponse(run_id=run_id)


@router.get("/runs", response_model=list[MLRunSummary])
def ml_runs(request: Request) -> list[MLRunSummary]:
    """Every ``runs/ml`` entry, newest first; an absent root is an empty list."""
    return [MLRunSummary(**item) for item in _ml(request).list_runs()]


@router.get("/runs/{run_id}", response_model=MLRunDetail)
def ml_run_detail(run_id: MLRunId, request: Request) -> MLRunDetail:
    try:
        detail: dict[str, Any] = _ml(request).detail(run_id)
    except UnknownMLRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ML run: {run_id}") from exc
    return MLRunDetail(**detail)


@router.post("/runs/{run_id}/control", response_model=OkResponse)
def ml_run_control(run_id: MLRunId, body: MLControlRequest, request: Request) -> OkResponse:
    """Cancel a live run. 404 means this process never started it; 409 means the
    action does not fit its status (already cancelling or terminal)."""
    try:
        ok = _ml(request).cancel(run_id)
    except UnknownMLRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown ML run: {run_id}") from exc
    except MLControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(ok=ok)


@router.post("/evaluate", response_model=MLEvaluateResponse)
async def ml_evaluate(body: MLEvaluateRequest, request: Request) -> MLEvaluateResponse:
    """Trade a finished run on one split, with optional stress and regime rows.

    The run's own symbol and timeframe come from its artefact, not the body, so
    the evaluation cannot be pointed at another instrument. Cost stress is a
    simulator pass per multiplier, hence the worker thread and the shared slot.
    """
    try:
        result = await asyncio.to_thread(
            _ml(request).evaluate,
            body.run_id,
            split=body.split,
            threshold=body.threshold,
            exit_threshold=body.exit_threshold,
            window=body.window,
            stress=tuple(body.stress),
            regime=body.regime,
        )
    except UnknownMLRunError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except MLError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return MLEvaluateResponse(**result)
