"""FastAPI application factory.

Everything mutable lives on ``app.state`` and is created per factory call, so
two apps in one process never share stored data, sequence numbers or a running
simulator. Paths are only read when a route needs them: an absent ``runs_root``
must not stop the app from starting.

The store, the runner and the message ring are process-local. Run the API with
a single uvicorn worker (``uvicorn scalping.api.app:app``); extra workers would
each serve their own copy of the state and a WebSocket would only see the
simulator living in the worker that accepted it.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from scalping import __version__
from scalping.api.routes import router
from scalping.api.routes_paper import router as paper_router
from scalping.api.runner import SimulationRunner
from scalping.api.train_runner import TrainRunner
from scalping.config import Settings
from scalping.data.store import ParquetStore

__all__ = [
    "ALLOWED_ORIGINS",
    "DEFAULT_DATA_ROOT",
    "DEFAULT_RUNS_ROOT",
    "app",
    "create_app",
]

#: Defaults mirror the CLI's ``--root data`` and the plan's ``runs/`` folder.
DEFAULT_DATA_ROOT = Path("data")
DEFAULT_RUNS_ROOT = Path("runs")

#: The paper ledger sits directly under the data root unless the factory is
#: given an explicit path. It is configuration, never a request field.
DEFAULT_PAPER_LEDGER_NAME = "paper.sqlite"

#: Local panel origins. Never ``*``: ``/api/data/download`` reads a path from
#: the request body, so a wildcard would let any page ask this process to read a
#: local file.
ALLOWED_ORIGINS = (
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    # 3001 is used when another local app already holds 3000.
    "http://localhost:3001",
    "http://127.0.0.1:3001",
)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Let the app finish serving, then stop and join training threads."""
    yield
    app.state.trains.shutdown()


def create_app(
    data_root: Path | None = None,
    settings_path: Path | None = None,
    runs_root: Path | None = None,
    flow_root: Path | None = None,
    paper_ledger: Path | None = None,
) -> FastAPI:
    """Build an app whose store, settings and runner belong to this call only.

    ``paper_ledger`` is the only way to move the paper database; leaving it out
    puts it at ``<data_root>/paper.sqlite``. Neither path is created here: the
    paper router opens the ledger on the first write, so importing this module
    has no filesystem side effect.
    """
    root = Path(data_root) if data_root is not None else DEFAULT_DATA_ROOT
    runs = Path(runs_root) if runs_root is not None else DEFAULT_RUNS_ROOT
    settings = Settings.load(settings_path) if settings_path is not None else Settings()

    app = FastAPI(title="AI Scalping Studio", version=__version__, lifespan=_lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(ALLOWED_ORIGINS),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.data_root = root
    app.state.runs_root = runs
    app.state.settings = settings
    app.state.store = ParquetStore(root)
    app.state.paper_ledger_path = (
        Path(paper_ledger) if paper_ledger is not None else root / DEFAULT_PAPER_LEDGER_NAME
    )
    app.state.sim = SimulationRunner(
        app.state.store,
        settings,
        flow_root=flow_root,
        card_path=runs / "strategy_card.json",
    )
    # Training threads belong to this app: the runner map is per factory call
    # and the lifespan stops and joins them before the process exits.
    app.state.trains = TrainRunner(app.state.store, settings, runs)
    app.include_router(router)
    app.include_router(paper_router)
    return app


#: Import target for ``uvicorn scalping.api.app:app``.
app = create_app()
