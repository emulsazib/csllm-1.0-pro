"""Pydantic v2 request/response models.

Sampling bounds are enforced here so invalid parameters can never reach the C++
sampler — the edge is the right place for it, and it turns a would-be crash into
a 422 with a readable message.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ErrorResponse",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "StreamChunk",
]

FinishReason = Literal["length", "context_full", "disconnected"]


class GenerateRequest(BaseModel):
    model_config = {"extra": "forbid"}  # typo in a field name should 422, not be ignored

    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(128, ge=1, le=1024)
    # 0.0 means greedy argmax — the C++ sampler short-circuits rather than
    # dividing by zero.
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, description="0 disables top-k")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="1.0 disables top-p")
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    stream: bool = True


class StreamChunk(BaseModel):
    """One SSE `data:` payload."""

    text: str
    index: int
    finish_reason: FinishReason | None = None


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    num_params: int
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    blas_backend: str
    max_concurrent_sessions: int
    sessions_in_flight: int
    kv_cache_bytes_per_session: int


class ErrorResponse(BaseModel):
    detail: str
