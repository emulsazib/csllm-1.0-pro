"""FastAPI application.

    uvicorn gateway.main:app --port 8000        (or: make serve)

Routers:
    inference   POST /generate, GET /health
    training    POST /train/{start,stop}, GET /train/status, WS /ws/train
    config      POST /configure_model, GET /configs, POST /export
    datasets    GET /datasets, /datasets/{name}[/preview]

The model and tokenizer load ONCE in the lifespan handler — loading per request
would dominate latency and defeat the mmap-able checkpoint format.

The served model is a loaded checkpoint, deliberately independent of any training
run the supervisor is driving: a crashing trainer must not take inference down,
and a mid-run checkpoint must not silently swap under live requests.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer

from .engine import AtCapacity, Engine
from .routes import config as config_routes
from .routes import datasets as dataset_routes
from .routes import inference as inference_routes
from .routes import inspect as inspect_routes
from .routes import training as training_routes
from .settings import Settings, get_settings
from .telemetry import TrainingSupervisor
from .versioning import ConfigStore

logger = logging.getLogger("csllm.gateway")

#: Built frontend, mounted when present so production is a single process.
WEB_DIST = Path("web/dist")


def build_engine(settings: Settings) -> Engine:
    logger.info("loading tokenizer from %s", settings.tokenizer_dir)
    tokenizer = BPETokenizer.load(settings.tokenizer_dir)
    logger.info("loading checkpoint %s", settings.checkpoint)
    model = core.Model.load(settings.checkpoint)

    if tokenizer.vocab_size != model.config.vocab_size:
        raise RuntimeError(
            f"tokenizer vocab_size={tokenizer.vocab_size} != model "
            f"{model.config.vocab_size}; they must come from the same training run"
        )
    logger.info("ready: %s params", f"{model.num_params():,}")
    return Engine(model, tokenizer, settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = getattr(app.state, "settings", None) or get_settings()
    app.state.settings = settings
    # Allows tests to inject a pre-built engine and skip disk I/O.
    if not getattr(app.state, "engine", None):
        app.state.engine = build_engine(settings)
    if not getattr(app.state, "supervisor", None):
        app.state.supervisor = TrainingSupervisor()
    if not getattr(app.state, "config_store", None):
        app.state.config_store = ConfigStore()
    yield
    supervisor = getattr(app.state, "supervisor", None)
    if supervisor is not None:
        # Never leave an orphaned trainer burning cores after the gateway exits.
        await supervisor.shutdown()
    app.state.engine = None


app = FastAPI(
    title="CSLLM Gateway",
    version="1.0.0",
    description="Streaming inference, training telemetry, and diagnostics for a "
    "Transformer built from scratch in C++",
    lifespan=lifespan,
)

app.include_router(inference_routes.router)
app.include_router(inspect_routes.router)
app.include_router(training_routes.router)
app.include_router(config_routes.router)
app.include_router(dataset_routes.router)


@app.exception_handler(AtCapacity)
async def _at_capacity_handler(request: Request, exc: AtCapacity) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
        headers={"Retry-After": "5"},
    )


if WEB_DIST.is_dir():
    # Mounted last so it cannot shadow an API route.
    app.mount("/", StaticFiles(directory=str(WEB_DIST), html=True), name="web")
