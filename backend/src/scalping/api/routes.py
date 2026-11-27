"""HTTP and WebSocket routes.

Handlers read their dependencies from ``app.state`` so the app factory can
install a fresh store, settings and runner for every test without module-level
state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Final, Literal

import polars as pl
from anyio import ClosedResourceError
from fastapi import APIRouter, HTTPException, Query, Request, WebSocket, WebSocketDisconnect

from scalping import __version__
from scalping.api.errors import public_error_text
from scalping.api.runner import (
    TIMEFRAME,
    RunnerError,
    SimConnection,
    SimulationRunner,
)
from scalping.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    CalibrateRequest,
    ControlRequest,
    DataQualityResponse,
    DataStatusResponse,
    DownloadRequest,
    DownloadResponse,
    EvaluateRequest,
    EvaluateResponse,
    HealthResponse,
    OkResponse,
    RegimeCount,
    RegimeCurrent,
    RegimePoint,
    RegimeResponse,
    RunControlRequest,
    RunDetail,
    RunSummary,
    StartRequest,
    StartResponse,
    TimeframeSpec,
    TrainRequest,
    TrainResponse,
    TrainRunId,
    is_symbol_id,
)
from scalping.api.train_runner import (
    RunControlError,
    TrainError,
    TrainRunner,
    UnknownRunError,
)
from scalping.data.base import DataProvider
from scalping.data.csv_provider import CsvProvider
from scalping.data.dukascopy import DataUnavailable, DukascopyProvider
from scalping.data.provenance import build_provenance, load_provenance, save_provenance
from scalping.data.quality import OutlierConfig
from scalping.data.registry import SYMBOLS, TIMEFRAMES
from scalping.data.synthetic import SyntheticProvider
from scalping.data.validate import ValidationReport, validate_bars
from scalping.regime import classify
from scalping.strategy.calibrate import (
    AUDIT_FILENAME,
    calibrate,
    card_from_report,
    record_calibration,
)
from scalping.strategy.card import save_card
from scalping.strategy.flow import flow_series

__all__ = ["router"]

logger = logging.getLogger(__name__)

_DEFAULT_SEED = 7

router = APIRouter()


def _device() -> str:
    """Best available torch device, or ``cpu`` when torch is absent."""
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _runner(request: Request) -> SimulationRunner:
    runner: SimulationRunner = request.app.state.sim
    return runner


def _trains(request: Request) -> TrainRunner:
    runner: TrainRunner = request.app.state.trains
    return runner


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, device=_device())


@router.get("/api/symbols")
def symbol_registry() -> list[dict[str, Any]]:
    """Research contract metadata for every registered instrument."""
    return [asdict(spec) for spec in SYMBOLS.values()]


@router.get("/api/timeframes", response_model=list[TimeframeSpec])
def timeframe_registry() -> list[TimeframeSpec]:
    """Supported bar sizes and the rule that resamples each one from M1.

    The panel reads this instead of hard-coding its timeframe labels, so a
    change in ``data/registry.py`` reaches the UI without a frontend edit.
    """
    return [TimeframeSpec(timeframe=name, rule=rule) for name, rule in TIMEFRAMES.items()]


#: Symbol and timeframe values end up as filesystem path segments, so they are
#: validated at the HTTP boundary before any store or provenance path is built.
#: The same rule is :data:`scalping.api.schemas.SymbolId` for request bodies.
def _checked_ids(symbol: str, timeframe: str) -> None:
    """Reject ids that could leave the data root; 422, never a silent rewrite."""
    if not is_symbol_id(symbol):
        raise HTTPException(status_code=422, detail=f"invalid symbol {symbol!r}")
    if timeframe not in TIMEFRAMES:
        raise HTTPException(status_code=422, detail=f"unknown timeframe {timeframe!r}")


#: Floor for the gap tolerance in the quality report, in minutes. The report
#: uses the larger of this floor and one bar width, so an H1 slice is not
#: "all gaps" just because its bars are an hour apart. A missing
#: higher-timeframe bar still exceeds the tolerance (one bar = threshold, a
#: hole = at least two bars).
_MIN_GAP_MINUTES: Final[int] = 10

#: Thresholds the validator uses for its statistical warnings. Kept here so the
#: report can state them instead of leaving the caller to read the source.
_OUTLIERS: Final[OutlierConfig] = OutlierConfig()


def _bar_minutes(timeframe: str) -> int:
    """Bar width in minutes from the registry rule (``30m``/``4h``/``1d``)."""
    rule = TIMEFRAMES[timeframe]
    factors = {"m": 1, "h": 60, "d": 1440}
    return int(rule[:-1]) * factors[rule[-1]]


def _quality_status(
    report: ValidationReport, provenance: dict[str, Any] | None
) -> tuple[Literal["ok", "warn", "fail"], str, str]:
    """Map validator severity onto ``ok|warn|fail`` and name the deciding rule.

    Errors are structural and fail the slice. Warnings (gaps, statistical
    outliers) only warn; the historical validator already treats them as
    legitimate market structure. A slice with no sidecar warns because source
    and synthetic flag are unknown, which is a provenance gap, not a data
    error.
    """
    error = next((issue for issue in report.issues if issue.severity == "error"), None)
    if error is not None:
        return "fail", error.code, error.detail
    warning = next((issue for issue in report.issues if issue.severity == "warning"), None)
    if warning is not None:
        return "warn", warning.code, warning.detail
    if provenance is None:
        return (
            "warn",
            "provenance_missing",
            "no provenance sidecar; source and synthetic flag are unknown",
        )
    return "ok", "clean", "no validation issues"


@router.get("/api/data/status", response_model=DataStatusResponse)
def data_status(request: Request, symbol: str, timeframe: str = TIMEFRAME) -> DataStatusResponse:
    _checked_ids(symbol, timeframe)
    store = request.app.state.store
    coverage = store.coverage(symbol, timeframe)
    provenance = load_provenance(store.root, symbol, timeframe)
    if coverage.rows == 0:
        return DataStatusResponse(
            symbol=symbol,
            timeframe=timeframe,
            rows=0,
            first_ts=None,
            last_ts=None,
            providers=[],
            provenance=None if provenance is None else provenance.to_dict(),
        )
    first = coverage.first_ts
    last = coverage.last_ts
    providers = store.providers(symbol, timeframe)
    return DataStatusResponse(
        symbol=symbol,
        timeframe=timeframe,
        rows=coverage.rows,
        first_ts=first.isoformat() if first is not None else None,
        last_ts=last.isoformat() if last is not None else None,
        providers=providers,
        provenance=None if provenance is None else provenance.to_dict(),
    )


@router.get("/api/data/validate")
def data_validate(request: Request, symbol: str, timeframe: str = TIMEFRAME) -> dict[str, Any]:
    """Structural validation report; explicit because it reads the full slice."""
    _checked_ids(symbol, timeframe)
    store = request.app.state.store
    frame = store.read(symbol, timeframe)
    return validate_bars(frame, symbol=symbol, timeframe=timeframe).to_dict()


@router.get("/api/data/quality", response_model=DataQualityResponse)
def data_quality(
    request: Request,
    symbol: str,
    timeframe: str = TIMEFRAME,
    include_outliers: bool = Query(
        default=True, description="run the robust-z outlier pass over the slice"
    ),
) -> DataQualityResponse:
    """Read-only quality report for one stored slice.

    This wraps :func:`scalping.data.validate.validate_bars`; it does not add a
    second validator. The category counts and the status are a mapping from the
    validator's issue codes, so every number is the validator's own count. The
    one setting the report adjusts is the gap tolerance: it uses the larger of
    the validator's 10-minute default and one bar width, so an H1 or D1 slice is
    not reported as gap-ridden for its normal spacing. The effective value is
    returned in ``thresholds``. Nothing is written, repaired or backfilled.

    An unregistered ``symbol``/``timeframe`` is a 422. A registered slice with
    no stored bars is a 200 with ``configured: false`` and null counts, which is
    explicit about "not checked" instead of reporting a clean zero.

    The validator runs over the whole stored slice, like ``/api/data/validate``;
    ``include_outliers=false`` skips only the robust-z pass for large stores.
    """
    _checked_ids(symbol, timeframe)
    if symbol not in SYMBOLS:
        raise HTTPException(status_code=422, detail=f"unknown symbol {symbol!r}")

    store = request.app.state.store
    provenance = load_provenance(store.root, symbol, timeframe)
    provenance_payload = None if provenance is None else provenance.to_dict()
    gap_minutes = max(_MIN_GAP_MINUTES, _bar_minutes(timeframe))
    thresholds: dict[str, Any] = {
        "gap_threshold_minutes": gap_minutes,
        "outlier_return_z": _OUTLIERS.return_z,
        "outlier_volume_z": _OUTLIERS.volume_z,
        "outlier_min_rows": _OUTLIERS.min_rows,
    }

    frame = store.read(symbol, timeframe)
    if frame.height == 0:
        return DataQualityResponse(
            symbol=symbol,
            timeframe=timeframe,
            configured=False,
            status="fail",
            status_rule="store_empty",
            status_detail=f"no stored {timeframe} bars for {symbol}",
            rows=0,
            first_ts=None,
            last_ts=None,
            duplicates=None,
            gaps=None,
            ohlc_violations=None,
            crossed_quotes=None,
            negative_spread=None,
            negative_volume=None,
            outliers=None,
            outliers_checked=False,
            provenance=provenance_payload,
            thresholds=thresholds,
            issues=[],
        )

    report = validate_bars(
        frame,
        symbol=symbol,
        timeframe=timeframe,
        gap_threshold_minutes=gap_minutes,
        include_outliers=include_outliers,
    )
    counts = {issue.code: issue.count for issue in report.issues}
    outliers_checked = include_outliers and report.rows >= _OUTLIERS.min_rows
    outliers = (
        counts.get("price_outliers", 0) + counts.get("volume_outliers", 0)
        if outliers_checked
        else None
    )
    status, rule, detail = _quality_status(report, provenance_payload)
    return DataQualityResponse(
        symbol=symbol,
        timeframe=timeframe,
        configured=True,
        status=status,
        status_rule=rule,
        status_detail=detail,
        rows=report.rows,
        first_ts=report.first_ts,
        last_ts=report.last_ts,
        duplicates=counts.get("duplicate_timestamps", 0),
        gaps=counts.get("time_gaps", 0),
        ohlc_violations=counts.get("ohlc_inconsistent", 0),
        crossed_quotes=counts.get("crossed_quotes", 0),
        negative_spread=counts.get("negative_spread", 0),
        negative_volume=counts.get("negative_volume", 0),
        outliers=outliers,
        outliers_checked=outliers_checked,
        provenance=provenance_payload,
        thresholds=thresholds,
        issues=[issue.to_dict() for issue in report.issues],
    )


@router.get("/api/regime", response_model=RegimeResponse)
def market_regime(
    request: Request,
    symbol: str,
    timeframe: str = TIMEFRAME,
    limit: int = Query(default=500, ge=1, le=5000, description="timeline bars to return"),
) -> RegimeResponse:
    """Descriptive regime labels for one stored slice.

    The summary counts cover every stored bar; only the timeline is capped by
    ``limit``. An empty slice is a normal 200 with ``rows: 0`` and no current
    label, matching ``/api/data/status``. These labels describe the past path
    and are not a price forecast.
    """
    _checked_ids(symbol, timeframe)
    frame = request.app.state.store.read(symbol, timeframe)
    labels = classify(frame)
    if labels.is_empty():
        return RegimeResponse(
            symbol=symbol,
            timeframe=timeframe,
            rows=0,
            current=None,
            counts=[],
            series=[],
        )

    last = labels.tail(1).to_dicts()[0]
    grouped = (
        labels.group_by("regime")
        .agg(pl.len().alias("bars"))
        .sort(["bars", "regime"], descending=[True, False])
    )
    return RegimeResponse(
        symbol=symbol,
        timeframe=timeframe,
        rows=labels.height,
        current=RegimeCurrent(
            ts=last["ts"].isoformat(),
            trend=last["trend"],
            trend_strength=last["trend_strength"],
            volatility=last["volatility"],
            expansion=last["expansion"],
            breakout=last["breakout"],
            unusual=bool(last["unusual"]),
            regime=last["regime"],
        ),
        counts=[
            RegimeCount(
                regime=str(row["regime"]),
                bars=int(row["bars"]),
                share=int(row["bars"]) / labels.height,
            )
            for row in grouped.to_dicts()
        ],
        series=[
            RegimePoint(
                ts=row["ts"].isoformat(),
                regime=row["regime"],
                trend=row["trend"],
                volatility=row["volatility"],
            )
            for row in labels.tail(limit).to_dicts()
        ],
    )


#: HTTP CSV imports live here, directly under the data root. The request names
#: a file in this directory; it never names a path.
_IMPORTS_DIRNAME: Final[str] = "imports"


def _import_file(name: str, data_root: Path) -> Path:
    """Return the file called ``name`` from ``<data_root>/imports``.

    The request value is only compared against directory entries; it is never
    joined to a base or handed to a filesystem API. The returned path is the
    entry produced by :meth:`Path.iterdir`, so the path expression that reaches
    ``open`` is not derived from request data (CodeQL ``py/path-injection``).
    Separators and absolute paths are already rejected by ``ImportFileName``; a
    name that is absent, is not a regular file, or is a symlink is a 422 here,
    so a link pointing outside the directory cannot be read through.
    """
    imports_dir = Path(data_root) / _IMPORTS_DIRNAME
    imports_dir.mkdir(parents=True, exist_ok=True)
    for entry in imports_dir.iterdir():
        if entry.name == name and entry.is_file() and not entry.is_symlink():
            return entry
    raise HTTPException(
        status_code=422,
        detail=(
            f"no csv file named {name!r} in the imports directory; "
            f"place the export in <data_root>/imports and pass its bare file name"
        ),
    )


def _provider(body: DownloadRequest, data_root: Path) -> DataProvider:
    if body.provider == "synthetic":
        return SyntheticProvider(seed=_DEFAULT_SEED if body.seed is None else body.seed)
    if body.provider == "csv":
        if body.path is None:  # pragma: no cover - the schema validator rejects this
            raise HTTPException(
                status_code=422,
                detail=(
                    "path (a file name in <data_root>/imports) is required when provider is 'csv'"
                ),
            )
        return CsvProvider(_import_file(body.path, data_root), tz_offset_hours=body.tz_offset)
    return DukascopyProvider(cache_dir=data_root / "dukascopy", symbol=body.symbol)


@router.post("/api/data/download", response_model=DownloadResponse)
def download(body: DownloadRequest, request: Request) -> DownloadResponse:
    """Fetch one range synchronously and merge it into the store.

    Dukascopy can take a while; v1 keeps the request open rather than
    pretending a job queue exists. A feed that cannot serve the range maps to
    503 with the provider's message. Construction reads no data but validates
    the request (an unknown dukascopy symbol) and a CSV open happens inside
    ``fetch``, so both live in the same guarded block.
    """
    store = request.app.state.store
    provider: DataProvider | None = None
    try:
        provider = _provider(body, Path(request.app.state.data_root))
        frame = provider.fetch(body.symbol, body.start, body.end)
    except DataUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                "cannot read data source: "
                f"{public_error_text(exc, root=Path(request.app.state.data_root))}"
            ),
        ) from exc
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
    rows = store.write(body.symbol, TIMEFRAME, frame)
    if not frame.is_empty():
        save_provenance(
            store.root,
            build_provenance(
                frame,
                symbol=body.symbol,
                timeframe=TIMEFRAME,
                source=f"provider:{body.provider}",
                synthetic=body.provider == "synthetic",
            ),
        )
    return DownloadResponse(rows=rows)


@router.post("/api/train", response_model=TrainResponse, status_code=202)
async def train_start(body: TrainRequest, request: Request) -> TrainResponse:
    """Prepare the run off-loop and return its id; learning runs on a thread.

    Feature building is done before the 202 so a store below the bar floor gets
    the trainer's 409 instead of a background failure. A data problem raised by
    the feature layer is a 422, matching ``/api/sim/start``.
    """
    try:
        run_id = await asyncio.to_thread(
            _trains(request).start,
            total_steps=body.total_steps,
            n_envs=body.n_envs,
            seed=body.seed,
            symbol=body.symbol,
            use_strategy=body.use_strategy,
            entry_penalty=body.entry_penalty,
            gate_signals=body.gate_signals,
            window=body.window,
            timeframe=body.timeframe,
            sl_atr_mult=body.sl_atr_mult,
            tp_atr_mult=body.tp_atr_mult,
            max_bars_in_trade=body.max_bars_in_trade,
            train_end=body.train_end,
        )
    except TrainError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrainResponse(run_id=run_id)


@router.get("/api/runs", response_model=list[RunSummary])
def list_runs(request: Request) -> list[RunSummary]:
    """The registry directory, newest first; absent ``runs_root`` means empty."""
    return [RunSummary(**item) for item in _trains(request).list_runs()]


@router.get(
    "/api/runs/{run_id}",
    response_model=RunDetail,
    # Optional blocks (series, bandit, total_steps) are omitted, not null, when
    # the run has no history or no bandit; the panel treats absence as empty.
    response_model_exclude_unset=True,
)
def run_detail(run_id: TrainRunId, request: Request) -> RunDetail:
    try:
        detail = _trains(request).detail(run_id)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from exc
    return RunDetail(**detail)


@router.post("/api/runs/{run_id}/control", response_model=OkResponse)
def run_control(run_id: TrainRunId, body: RunControlRequest, request: Request) -> OkResponse:
    """Map pause/resume/stop onto the run's events.

    404 means this process never started the run; 409 means it is known but the
    action does not fit its status (for example a completed run).
    """
    try:
        ok = _trains(request).control(run_id, body.action)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown run: {run_id}") from exc
    except RunControlError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(ok=ok)


@router.post("/api/backtest", response_model=BacktestResponse)
async def backtest(body: BacktestRequest, request: Request) -> BacktestResponse:
    """Replay a stored policy over the stored bars; metrics are JSON-safe."""
    try:
        result = await asyncio.to_thread(_trains(request).backtest, body.run_id, body.symbol)
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown run: {body.run_id}") from exc
    except TrainError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return BacktestResponse(**result)


@router.post("/api/evaluate", response_model=EvaluateResponse)
async def evaluate(body: EvaluateRequest, request: Request) -> EvaluateResponse:
    """Walk-forward and cost-stress a finished run on out-of-sample bars."""
    try:
        result = await asyncio.to_thread(
            _trains(request).evaluate, body.run_id, body.symbol, body.folds
        )
    except UnknownRunError as exc:
        raise HTTPException(status_code=404, detail=f"unknown run: {body.run_id}") from exc
    except TrainError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EvaluateResponse(**result)


@router.post("/api/sim/start", response_model=StartResponse)
async def sim_start(body: StartRequest, request: Request) -> StartResponse:
    """Install a replay run; connected WebSocket pumps drive it.

    Reading history and building features is CPU bound, so it runs in a worker
    thread: the runner's state lock is a threading lock and safe off-loop. A
    store the feature layer rejects (for example a non-finite quote) is a data
    problem, not a server failure: ``build_features`` raises ``ValueError`` with
    the offending columns and it maps to 422.
    """
    try:
        session_id = await asyncio.to_thread(_runner(request).start, body.speed, body.mode)
    except RunnerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return StartResponse(session_id=session_id)


@router.post("/api/sim/control", response_model=OkResponse)
async def sim_control(body: ControlRequest, request: Request) -> OkResponse:
    try:
        ok = _runner(request).control(body.action)
    except RunnerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(ok=ok)


@router.get("/api/strategy/card")
def strategy_card(request: Request) -> dict[str, Any]:
    """The promoted rule the panel replays, or ``active: false``."""
    return _runner(request).strategy_card()


@router.post("/api/strategy/calibrate")
async def strategy_calibrate(body: CalibrateRequest, request: Request) -> dict[str, Any]:
    """Re-select the rule on the training slice and gate it on the holdout.

    Heavy CPU work (labels over years of bars), so it runs in a worker thread.
    A card is written only when the holdout gate passes and ``dry_run`` is
    false; the response always carries the report so a rejected calibration is
    inspectable instead of silent.
    """
    runner = _runner(request)
    flow_root = runner.flow_root
    if flow_root is None:
        raise HTTPException(status_code=409, detail="no flow store configured")
    card_path = runner.card_path
    if card_path is None:
        raise HTTPException(status_code=409, detail="no strategy card path configured")

    def work() -> dict[str, Any]:
        frame = request.app.state.store.read(body.symbol, body.timeframe)
        if frame.is_empty():
            raise ValueError(f"no {body.symbol} {body.timeframe} bars in the store")
        flow = flow_series(flow_root, body.symbol, body.timeframe, frame)
        incumbent = runner.card_params()
        report = calibrate(
            frame,
            flow,
            symbol=body.symbol,
            timeframe=body.timeframe,
            base=request.app.state.settings,
            holdout_fraction=body.holdout_fraction,
            min_train_trades=body.min_train_trades,
            min_holdout_trades=body.min_holdout_trades,
            incumbent=incumbent,
        )
        payload = report.to_dict()
        card_written = report.promoted and not body.dry_run
        if card_written:
            card = card_from_report(
                report,
                symbol=body.symbol,
                note="holdout reused across nightly runs; treat later promotions as weaker evidence",
            )
            save_card(card_path, card)
        audit = record_calibration(
            card_path.parent / AUDIT_FILENAME,
            actor=f"api:{request.client.host}" if request.client is not None else "api",
            dry_run=body.dry_run,
            report=report,
            symbol=body.symbol,
            timeframe=body.timeframe,
            card_written=card_written,
            card_path=card_path,
        )
        payload["audit"] = {
            "at": audit["at"],
            "actor": audit["actor"],
            "dry_run": audit["dry_run"],
        }
        payload["card"] = runner.strategy_card()
        return payload

    try:
        return await asyncio.to_thread(work)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.websocket("/ws/sim")
async def sim_socket(websocket: WebSocket) -> None:
    """Send the status frame first, then stream sequenced simulator events."""
    await websocket.accept()
    runner: SimulationRunner = websocket.app.state.sim
    conn = runner.connect()
    await websocket.send_json(runner.status_message())

    sender = asyncio.create_task(_pump(runner, conn, websocket))
    reader = asyncio.create_task(_read(websocket, runner, conn))
    try:
        await asyncio.wait({sender, reader}, return_when=asyncio.FIRST_COMPLETED)
    finally:
        runner.disconnect(conn)
        for task in (sender, reader):
            task.cancel()
        try:
            await asyncio.gather(sender, reader, return_exceptions=True)
        except asyncio.CancelledError:
            # The test client / server shuts the app task down while this
            # cleanup runs. The socket is gone either way, so let the task
            # finish instead of surfacing a teardown-only cancellation.
            pass


async def _pump(runner: SimulationRunner, conn: SimConnection, websocket: WebSocket) -> None:
    try:
        await runner.pump(conn, websocket.send_json)
    except (WebSocketDisconnect, ClosedResourceError, RuntimeError, OSError):
        # The peer left between two sends; the reader task notices too.
        return
    except Exception:
        # A bug outside the step path must not disappear into a cancelled
        # task: log it, end the episode visibly, try to deliver that frame.
        logger.exception("sim websocket pump failed")
        runner.fail("error")
        for message in runner.drain(conn):
            try:
                await websocket.send_json(message)
            except (WebSocketDisconnect, ClosedResourceError, RuntimeError, OSError):
                return


async def _read(websocket: WebSocket, runner: SimulationRunner, conn: SimConnection) -> None:
    while True:
        try:
            raw = await websocket.receive_text()
        except WebSocketDisconnect:
            return
        except RuntimeError:
            return
        try:
            message = json.loads(raw)
        except ValueError:
            continue
        if not isinstance(message, dict) or message.get("type") != "resume":
            continue
        seq = message.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            continue
        runner.resume(conn, seq)
