"""Pydantic v2 request/response models — Phase 4.

Planned:
    class GenerateRequest(BaseModel):
        prompt: str          = Field(min_length=1, max_length=8192)
        max_tokens: int      = Field(128, ge=1, le=2048)
        temperature: float   = Field(1.0, ge=0.0, le=2.0)   # 0 => greedy
        top_k: int           = Field(0, ge=0)               # 0 => disabled
        top_p: float         = Field(1.0, gt=0.0, le=1.0)   # 1 => disabled
        seed: int | None     = None
        stream: bool         = True

Bounds are enforced at the edge so invalid sampling parameters can never reach
the C++ sampler.
"""

from __future__ import annotations
