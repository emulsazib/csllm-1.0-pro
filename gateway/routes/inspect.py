"""Live generation with real attention weights, over WebSocket.

**Why not JSON.** Attention is `n_layer x n_head x keys` float32. At the 12M
config with a full context that is 6*6*256 = 9,216 values — ~36 KB raw, and
roughly 100 KB as JSON numbers. At several hundred tokens per second, JSON alone
would saturate the socket and the decode cost would stall the browser's main
thread. So each token is sent as **two frames**:

    1. a JSON text frame  — token, position, top candidates, and the shape of
       what follows
    2. a binary frame     — the attention block as raw little-endian float32,
       row-major [layer, head, key]

Clients pair them in order. The client may also narrow `layers` / `heads` at
subscribe time, which cuts the payload proportionally before it is ever sent.

The weights are the ones the model actually used: `GenerationSession` copies them
out of the decode loop's softmax (see core/src/model.cpp), so nothing here is a
reconstruction.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from csllm import _csllm_core as core

from ..engine import AtCapacity, Engine
from ..inspection import distribution_stats, top_candidates
from ..schemas import InspectStreamRequest

logger = logging.getLogger("csllm.gateway.inspect")
router = APIRouter(tags=["inspect"])


def _resolve(selected: list[int] | None, total: int) -> list[int]:
    """Validate a layer/head selection, defaulting to all."""
    if not selected:
        return list(range(total))
    chosen = sorted({i for i in selected if 0 <= i < total})
    return chosen or list(range(total))


@router.websocket("/ws/inspect")
async def inspect_socket(websocket: WebSocket) -> None:
    await websocket.accept()
    engine: Engine | None = getattr(websocket.app.state, "engine", None)
    if engine is None:
        await websocket.send_json({"type": "error", "message": "model is not loaded"})
        await websocket.close()
        return

    try:
        raw = await websocket.receive_json()
        req = InspectStreamRequest.model_validate(raw)
    except ValidationError as exc:
        await websocket.send_json({"type": "error", "message": exc.errors()[0]["msg"]})
        await websocket.close()
        return
    except (WebSocketDisconnect, RuntimeError):
        return

    cfg = engine.model.config
    layers = _resolve(req.layers, cfg.n_layer)
    heads = _resolve(req.heads, cfg.n_head)

    prompt_ids = engine.tokenizer.encode(req.prompt)
    if not prompt_ids:
        await websocket.send_json({"type": "error", "message": "prompt encoded to zero tokens"})
        await websocket.close()
        return
    if len(prompt_ids) >= cfg.block_size:
        prompt_ids = prompt_ids[-(cfg.block_size - 1) :]

    budget = min(req.max_tokens, cfg.block_size - len(prompt_ids))
    if budget <= 0:
        await websocket.send_json({"type": "error", "message": "prompt fills the context"})
        await websocket.close()
        return

    def label(token_id: int) -> str:
        return engine.tokenizer.decode_bytes([token_id]).decode("utf-8", errors="replace")

    try:
        # Shares the generation semaphore with /generate: an inspect stream costs
        # the same KV cache and CPU core as a normal request.
        async with engine._slot():
            session = core.GenerationSession(engine.model, req.seed or 0)
            session.set_capture_attention(req.attention)
            params = core.SamplingParams(
                temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
            )

            logits = await asyncio.to_thread(
                session.prefill, np.asarray(prompt_ids, dtype=np.int32), cfg.vocab_size
            )

            await websocket.send_json(
                {
                    "type": "start",
                    "prompt": [{"id": int(i), "text": label(int(i))} for i in prompt_ids],
                    "config": {
                        "n_layer": cfg.n_layer,
                        "n_head": cfg.n_head,
                        "n_embd": cfg.n_embd,
                        "block_size": cfg.block_size,
                        "vocab_size": cfg.vocab_size,
                    },
                    "layers": layers,
                    "heads": heads,
                    "max_tokens": budget,
                    "attention": req.attention,
                }
            )

            # `logits` are always the distribution the NEXT token is drawn from,
            # and last_attention() is what the model attended to while producing
            # them — so each frame pairs "what it looked at" with "what it then
            # predicted". step() is avoided here because it samples without
            # handing back the logits the UI needs to chart.
            emitted = 0
            while True:
                token = await asyncio.to_thread(session.sample_last, params)

                payload: dict = {
                    "type": "token",
                    "index": emitted,
                    "id": int(token),
                    "text": label(int(token)),
                    "position": session.position,
                }
                if req.top_n > 0:
                    candidates, raw, filtered = top_candidates(
                        logits, engine.tokenizer, params, req.top_n
                    )
                    raw_h, filt_h, kept = distribution_stats(raw, filtered)
                    payload["top"] = [c.model_dump() for c in candidates]
                    payload["kept_count"] = kept
                    payload["raw_entropy"] = raw_h
                    payload["filtered_entropy"] = filt_h

                if req.attention:
                    block = np.ascontiguousarray(
                        session.last_attention()[np.ix_(layers, heads)], dtype=np.float32
                    )
                    payload["attn"] = {"shape": list(block.shape), "bytes": int(block.nbytes)}
                    await websocket.send_json(payload)
                    await websocket.send_bytes(block.tobytes())
                else:
                    await websocket.send_json(payload)

                emitted += 1
                if emitted >= budget:
                    break
                # Advances the cache by one and returns the next distribution.
                logits = await asyncio.to_thread(session.decode, int(token), cfg.vocab_size)

            await websocket.send_json({"type": "done", "reason": "length", "tokens": emitted})
    except AtCapacity as exc:
        await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        # The client hung up mid-generation; the session and its KV cache are
        # released by the context manager on the way out.
        return
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("inspect stream failed")
        try:
            await websocket.send_json({"type": "error", "message": str(exc)})
        except RuntimeError:
            return

    # The socket may already be closed if the client hung up first.
    with contextlib.suppress(RuntimeError):
        await websocket.close()
