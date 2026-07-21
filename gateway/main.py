"""FastAPI application.

    uvicorn gateway.main:app --port 8000        (or: make serve)

Endpoints:
    POST /generate   prompt, max_tokens, temperature, top_k, top_p, seed, stream
                     -> SSE token stream, or a single JSON body when stream=false
    GET  /health     model metadata and live session count

The model and tokenizer load ONCE in the lifespan handler — loading per request
would dominate latency and defeat the mmap-able checkpoint format.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer

from .engine import AtCapacity, Engine
from .schemas import GenerateRequest, GenerateResponse, HealthResponse
from .settings import Settings, get_settings

logger = logging.getLogger("csllm.gateway")

SSE_DONE = "[DONE]"


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
    yield
    app.state.engine = None


app = FastAPI(
    title="CSLLM Gateway",
    version="1.0.0",
    description="Streaming inference for a Transformer built from scratch in C++",
    lifespan=lifespan,
)


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model is not loaded")
    return engine


@app.exception_handler(AtCapacity)
async def _at_capacity_handler(request: Request, exc: AtCapacity) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content={"detail": str(exc)},
        headers={"Retry-After": "5"},
    )


@app.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    engine = get_engine(request)
    cfg = engine.model.config
    build = core.build_info()
    return HealthResponse(
        status="ok",
        version=build.version,
        num_params=engine.model.num_params(),
        vocab_size=cfg.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        blas_backend=build.blas_backend,
        max_concurrent_sessions=engine.settings.max_concurrent_sessions,
        sessions_in_flight=engine.sessions_in_flight,
        kv_cache_bytes_per_session=engine.kv_cache_bytes(),
    )


@app.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    engine = get_engine(request)

    if not req.stream:
        try:
            result = await engine.complete(req, request.is_disconnected)
        except AtCapacity as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
        return GenerateResponse(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            finish_reason=result.finish_reason,
        )

    async def event_stream() -> AsyncIterator[dict]:
        try:
            async for chunk in engine.stream(req, request.is_disconnected):
                yield {"data": chunk.model_dump_json()}
            yield {"data": SSE_DONE}
        except AtCapacity as exc:
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        except ValueError as exc:
            yield {"event": "error", "data": json.dumps({"detail": str(exc)})}
        except Exception:  # pragma: no cover - defensive
            logger.exception("generation failed")
            yield {"event": "error", "data": json.dumps({"detail": "internal error"})}

    # sse-starlette sets the SSE headers and cancels the generator on disconnect;
    # the explicit is_disconnected() poll above stops the C++ work promptly too,
    # so an abandoned browser tab cannot keep a CPU core busy.
    return EventSourceResponse(event_stream())
