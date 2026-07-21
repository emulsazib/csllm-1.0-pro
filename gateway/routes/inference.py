"""Generation and health — the hot path.

Kept independent of the training, config, and dataset routers so a fault there
cannot take generation down.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator

import numpy as np
from fastapi import APIRouter, HTTPException, Request, status
from sse_starlette.sse import EventSourceResponse

from csllm import _csllm_core as core

from ..engine import AtCapacity, Engine
from ..inspection import distribution_stats, top_candidates
from ..schemas import (
    EmbeddingsRequest,
    EmbeddingsResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    InspectRequest,
    InspectResponse,
    TokenInfo,
    TokenizeRequest,
    TokenizeResponse,
)

logger = logging.getLogger("csllm.gateway.inference")
router = APIRouter(tags=["inference"])

SSE_DONE = "[DONE]"


def get_engine(request: Request) -> Engine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "model is not loaded")
    return engine


@router.get("/health", response_model=HealthResponse)
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


@router.post("/generate")
async def generate(req: GenerateRequest, request: Request):
    engine = get_engine(request)

    if not req.stream:
        try:
            result = await engine.complete(req, request.is_disconnected)
        except AtCapacity as exc:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
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
    # the explicit is_disconnected() poll inside engine.stream stops the C++ work
    # promptly too, so an abandoned browser tab cannot keep a CPU core busy.
    return EventSourceResponse(event_stream())


# ── inspection (diagnostics UI) ───────────────────────────────────────────────


@router.post("/tokenize", response_model=TokenizeResponse)
async def tokenize(req: TokenizeRequest, request: Request) -> TokenizeResponse:
    """Segment text into tokens with their byte spans.

    Byte offsets rather than character offsets: this is a BYTE-level BPE, and a
    token can end mid-code-point. Reporting character spans would imply
    boundaries that do not exist.
    """
    engine = get_engine(request)
    tokenizer = engine.tokenizer
    ids = tokenizer.encode(req.text)

    tokens: list[TokenInfo] = []
    offset = 0
    for index, token_id in enumerate(ids):
        raw = tokenizer.decode_bytes([token_id])
        try:
            text = raw.decode("utf-8")
            partial = False
        except UnicodeDecodeError:
            # Surfaced so the UI can render it as a fragment rather than mojibake.
            text = raw.decode("utf-8", errors="replace")
            partial = True
        tokens.append(
            TokenInfo(
                index=index,
                id=token_id,
                text=text,
                bytes=list(raw),
                start=offset,
                end=offset + len(raw),
                partial_utf8=partial,
            )
        )
        offset += len(raw)

    num_bytes = len(req.text.encode("utf-8"))
    return TokenizeResponse(
        tokens=tokens,
        count=len(ids),
        num_chars=len(req.text),
        num_bytes=num_bytes,
        compression=(num_bytes / len(ids)) if ids else 0.0,
        vocab_size=engine.model.config.vocab_size,
    )


@router.post("/embeddings", response_model=EmbeddingsResponse)
async def embeddings(req: EmbeddingsRequest, request: Request) -> EmbeddingsResponse:
    """Embedding rows for the given tokens, plus an optional PCA projection.

    Rows come from the zero-copy ``get_param("tok_emb")`` view, so this reads the
    live weights rather than a copy that could go stale.
    """
    engine = get_engine(request)
    if (req.text is None) == (req.ids is None):
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, "provide exactly one of 'text' or 'ids'"
        )

    ids = req.ids if req.ids is not None else engine.tokenizer.encode(req.text or "")
    if not ids:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "no tokens to embed")

    vocab_size = engine.model.config.vocab_size
    bad = [i for i in ids if not 0 <= i < vocab_size]
    if bad:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT,
            f"token ids out of range [0,{vocab_size}): {bad[:5]}",
        )

    table = engine.model.get_param("tok_emb")
    vectors = np.asarray(table[ids], dtype=np.float32)

    projection: list[list[float]] = []
    explained: list[float] = []
    if req.project and len(ids) >= 3:
        projection, explained = _pca3(vectors)

    return EmbeddingsResponse(
        ids=list(ids),
        labels=[engine.tokenizer.decode_bytes([i]).decode("utf-8", errors="replace") for i in ids],
        n_embd=int(vectors.shape[1]),
        vectors=vectors.tolist(),
        vmin=float(vectors.min()),
        vmax=float(vectors.max()),
        projection=projection,
        explained_variance=explained,
    )


def _pca3(vectors: np.ndarray) -> tuple[list[list[float]], list[float]]:
    """Project to 3D via SVD.

    NumPy here is display-only post-processing, not the model path — the C++
    engine still does every forward/backward computation (rules.md #2).
    """
    centered = vectors - vectors.mean(axis=0, keepdims=True)
    # full_matrices=False: we only need the top components, and the full basis
    # would be [n_embd, n_embd] — 384x384 for a handful of tokens.
    _, singular, components = np.linalg.svd(centered, full_matrices=False)
    k = min(3, components.shape[0])
    coords = centered @ components[:k].T
    total = float((singular**2).sum()) or 1.0
    return coords.tolist(), [float(s**2 / total) for s in singular[:k]]


@router.post("/inspect/next_token", response_model=InspectResponse)
async def inspect_next_token(req: InspectRequest, request: Request) -> InspectResponse:
    """Next-token distribution before and after sampling filters.

    The filtered probabilities come from the C++ ``filtered_distribution``, which
    ``Sampler::sample`` is itself built on — so what this shows and what the model
    would actually draw from cannot diverge.
    """
    engine = get_engine(request)
    cfg = engine.model.config

    ids = engine.tokenizer.encode(req.prompt)
    if not ids:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "prompt encoded to zero tokens")
    ids = ids[-(cfg.block_size - 1) :] if len(ids) >= cfg.block_size else ids

    session = core.GenerationSession(engine.model, 0)
    logits = await asyncio.to_thread(
        session.prefill, np.asarray(ids, dtype=np.int32), cfg.vocab_size
    )

    params = core.SamplingParams(
        temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
    )
    candidates, raw, filtered = top_candidates(logits, engine.tokenizer, params, req.top_n)
    raw_entropy, filtered_entropy, kept = distribution_stats(raw, filtered)

    return InspectResponse(
        prompt_tokens=len(ids),
        candidates=candidates,
        kept_count=kept,
        raw_entropy=raw_entropy,
        filtered_entropy=filtered_entropy,
        vocab_size=cfg.vocab_size,
    )
