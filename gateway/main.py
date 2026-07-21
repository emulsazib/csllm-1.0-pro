"""FastAPI application — Phase 4.

Planned endpoints:
    POST /generate   prompt, max_tokens, temperature, top_k, top_p, seed, stream
                     -> SSE token stream (or a single JSON body when stream=False)
    GET  /health     model loaded, param count, config

Concurrency design (the part that is easy to get wrong):
  * Weights load ONCE in the lifespan handler, mmapped from the .csllm checkpoint.
  * Each request gets its own C++ GenerationSession with a private KV cache
    (~4.5 MiB at the 12M config); weights are shared read-only.
  * Per-token compute is dispatched via ``asyncio.to_thread``. Because the
    bindings release the GIL, the event loop keeps serving other streams.
  * A Semaphore caps concurrent sessions.
  * ``await request.is_disconnected()`` aborts generation on client hangup, so an
    abandoned browser tab cannot pin a CPU core.
"""

from __future__ import annotations


def create_app():
    raise NotImplementedError("The FastAPI gateway arrives in Phase 4")
