"""HTTP and WebSocket routes.

Handlers read their dependencies from ``app.state`` so the app factory can
install a fresh store, settings and runner for every test without module-level
state.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from anyio import ClosedResourceError
from fastapi import APIRouter, HTTPException, Request, WebSocket, WebSocketDisconnect

from scalping import __version__
from scalping.api.runner import (
    TIMEFRAME,
    RunnerError,
    SimConnection,
    SimulationRunner,
)
from scalping.api.schemas import (
    ControlRequest,
    DataStatusResponse,
    DownloadRequest,
    DownloadResponse,
    HealthResponse,
    OkResponse,
    StartRequest,
    StartResponse,
)
from scalping.data.base import DataProvider
from scalping.data.csv_provider import CsvProvider
from scalping.data.dukascopy import DataUnavailable, DukascopyProvider
from scalping.data.synthetic import SyntheticProvider

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


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, device=_device())


@router.get("/api/data/status", response_model=DataStatusResponse)
def data_status(request: Request, symbol: str) -> DataStatusResponse:
    store = request.app.state.store
    frame = store.read(symbol, TIMEFRAME)
    rows = frame.height
    if rows == 0:
        return DataStatusResponse(symbol=symbol, rows=0, first_ts=None, last_ts=None, providers=[])
    first = frame.get_column("ts").min()
    last = frame.get_column("ts").max()
    providers = sorted(frame.get_column("provider").unique().to_list())
    return DataStatusResponse(
        symbol=symbol,
        rows=rows,
        first_ts=first.isoformat() if first is not None else None,
        last_ts=last.isoformat() if last is not None else None,
        providers=providers,
    )


def _provider(body: DownloadRequest, data_root: Path) -> DataProvider:
    if body.provider == "synthetic":
        return SyntheticProvider(seed=_DEFAULT_SEED if body.seed is None else body.seed)
    if body.provider == "csv":
        if body.path is None:  # pragma: no cover - the schema validator rejects this
            raise HTTPException(status_code=422, detail="path is required when provider is 'csv'")
        return CsvProvider(Path(body.path), tz_offset_hours=body.tz_offset)
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
        raise HTTPException(status_code=400, detail=f"cannot read data source: {exc}") from exc
    finally:
        if provider is not None:
            close = getattr(provider, "close", None)
            if callable(close):
                close()
    rows = store.write(body.symbol, TIMEFRAME, frame)
    return DownloadResponse(rows=rows)


@router.post("/api/sim/start", response_model=StartResponse)
async def sim_start(body: StartRequest, request: Request) -> StartResponse:
    """Install a replay run; connected WebSocket pumps drive it.

    Reading history and building features is CPU bound, so it runs in a worker
    thread: the runner's state lock is a threading lock and safe off-loop.
    """
    try:
        session_id = await asyncio.to_thread(_runner(request).start, body.speed, body.mode)
    except RunnerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StartResponse(session_id=session_id)


@router.post("/api/sim/control", response_model=OkResponse)
async def sim_control(body: ControlRequest, request: Request) -> OkResponse:
    try:
        ok = _runner(request).control(body.action)
    except RunnerError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return OkResponse(ok=ok)


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
