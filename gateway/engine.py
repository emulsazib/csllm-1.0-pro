"""Async wrapper around the C++ generation engine.

Three things make this safe under concurrent load:

1. **Weights load once, sessions are per request.** ``Model`` is shared read-only;
   each request gets its own ``GenerationSession`` with a private KV cache
   (~4.7 MB at the 12M config), so requests cannot corrupt each other's state.
2. **Per-token compute runs in a worker thread.** ``asyncio.to_thread`` plus the
   GIL release inside the pybind11 bindings means the event loop keeps serving
   other streams while one is decoding. A direct call would stall every
   concurrent request.
3. **A semaphore bounds concurrency.** Each in-flight stream costs a CPU core and
   a KV cache; past the limit requests wait, then fail with 503 rather than
   queueing without bound.

It also owns the streaming detokenization buffer. A byte-level BPE token can end
mid-UTF-8-sequence, so bytes are accumulated and flushed only at code-point
boundaries — decoding per token would emit mojibake.
"""

from __future__ import annotations

import asyncio
import contextlib
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

import numpy as np

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer

from .schemas import FinishReason, GenerateRequest, StreamChunk
from .settings import Settings

__all__ = ["AtCapacity", "Engine", "flush_utf8"]

# Longest valid UTF-8 sequence. Once this many bytes still fail to decode, the
# input is genuinely invalid rather than merely incomplete.
_MAX_UTF8_SEQUENCE = 4


class AtCapacity(RuntimeError):
    """All session slots are busy and the wait timed out."""


def flush_utf8(pending: bytes) -> tuple[str, bytes]:
    """Split buffered bytes into (decodable text, bytes to carry forward).

    Returns as much complete text as possible, keeping any trailing partial
    code point for the next token. Never buffers unboundedly: bytes that are
    invalid rather than incomplete are emitted with U+FFFD.
    """
    try:
        return pending.decode("utf-8"), b""
    except UnicodeDecodeError as exc:
        head = pending[: exc.start].decode("utf-8")
        tail = pending[exc.start :]
        if len(tail) >= _MAX_UTF8_SEQUENCE:
            return head + tail.decode("utf-8", errors="replace"), b""
        return head, tail


@dataclass
class Completion:
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


class Engine:
    def __init__(self, model, tokenizer: BPETokenizer, settings: Settings) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.settings = settings
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_sessions)
        self._in_flight = 0

    # ── introspection ────────────────────────────────────────────────────────

    @property
    def sessions_in_flight(self) -> int:
        return self._in_flight

    def kv_cache_bytes(self) -> int:
        cfg = self.model.config
        return 2 * cfg.n_layer * cfg.n_head * cfg.block_size * cfg.head_dim * 4

    # ── concurrency ──────────────────────────────────────────────────────────

    @contextlib.asynccontextmanager
    async def _slot(self) -> AsyncIterator[None]:
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self.settings.acquire_timeout_s
            )
        except (TimeoutError, asyncio.TimeoutError) as exc:
            raise AtCapacity(
                f"all {self.settings.max_concurrent_sessions} session slots are busy"
            ) from exc
        self._in_flight += 1
        try:
            yield
        finally:
            self._in_flight -= 1
            self._semaphore.release()

    # ── generation ───────────────────────────────────────────────────────────

    def _prepare(self, req: GenerateRequest) -> tuple[list[int], int, FinishReason]:
        """Encode and truncate the prompt; decide the token budget."""
        block = self.model.config.block_size
        prompt_ids = self.tokenizer.encode(req.prompt)
        if not prompt_ids:
            raise ValueError("prompt encoded to zero tokens")
        if len(prompt_ids) >= block:
            # Keep the tail: the most recent context is what conditions generation.
            prompt_ids = prompt_ids[-(block - 1) :]

        context_budget = block - len(prompt_ids)
        budget = min(req.max_tokens, context_budget)
        if budget <= 0:
            raise ValueError(
                f"prompt fills the {block}-token context; no room left to generate"
            )
        stop_reason: FinishReason = "length" if budget == req.max_tokens else "context_full"
        return prompt_ids, budget, stop_reason

    async def stream(
        self,
        req: GenerateRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield chunks of decoded text as tokens are produced."""
        prompt_ids, budget, stop_reason = self._prepare(req)
        seed = req.seed if req.seed is not None else secrets.randbits(63)
        params = core.SamplingParams(
            temperature=req.temperature, top_k=req.top_k, top_p=req.top_p
        )
        vocab_size = self.model.config.vocab_size

        async with self._slot():
            session = core.GenerationSession(self.model, seed)
            await asyncio.to_thread(
                session.prefill, np.asarray(prompt_ids, dtype=np.int32), vocab_size
            )

            # prefill() has consumed the whole prompt, so the first token comes
            # from sample_last(). step(prompt[-1]) would feed the prompt's last
            # token twice and generate from a corrupted context.
            token = await asyncio.to_thread(session.sample_last, params)

            pending = b""
            emitted = 0
            finish: FinishReason = stop_reason

            while True:
                pending += self.tokenizer.decode_bytes([token])
                text, pending = flush_utf8(pending)
                emitted += 1
                if text:
                    yield StreamChunk(text=text, index=emitted - 1)

                if emitted >= budget:
                    break
                if is_disconnected is not None and await is_disconnected():
                    finish = "disconnected"
                    break

                token = await asyncio.to_thread(session.step, token, params)

            # Anything still buffered is an incomplete sequence at the very end.
            if pending:
                yield StreamChunk(text=pending.decode("utf-8", errors="replace"), index=emitted)

            yield StreamChunk(text="", index=emitted, finish_reason=finish)

    async def complete(
        self,
        req: GenerateRequest,
        is_disconnected: Callable[[], Awaitable[bool]] | None = None,
    ) -> Completion:
        """Non-streaming variant — same code path, collected."""
        parts: list[str] = []
        finish: FinishReason = "length"
        count = 0
        async for chunk in self.stream(req, is_disconnected):
            if chunk.finish_reason is not None:
                finish = chunk.finish_reason
                count = chunk.index
            parts.append(chunk.text)
        return Completion(
            text="".join(parts),
            prompt_tokens=len(self.tokenizer.encode(req.prompt)),
            completion_tokens=count,
            finish_reason=finish,
        )
