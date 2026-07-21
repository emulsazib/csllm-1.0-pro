"""Gateway configuration, driven by environment variables.

Plain ``os.environ`` rather than pydantic-settings: it is one small dataclass and
not worth another dependency.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

__all__ = ["Settings", "get_settings"]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


@dataclass(frozen=True)
class Settings:
    checkpoint: str = "data/model.csllm"
    tokenizer_dir: str = "data/tokenizer"

    # Each concurrent stream owns a KV cache (~4.7 MB at the 12M config) and
    # occupies a CPU core while decoding, so this bounds both memory and
    # contention. Requests beyond it wait, then 503 rather than queue forever.
    max_concurrent_sessions: int = 4
    acquire_timeout_s: float = 30.0

    # Bounds enforced at the edge so nothing unreasonable reaches the C++ engine.
    max_prompt_chars: int = 8192
    max_new_tokens: int = 1024

    @classmethod
    def from_env(cls) -> Settings:
        return cls(
            checkpoint=os.environ.get("CSLLM_CHECKPOINT", cls.checkpoint),
            tokenizer_dir=os.environ.get("CSLLM_TOKENIZER_DIR", cls.tokenizer_dir),
            max_concurrent_sessions=_env_int(
                "CSLLM_MAX_CONCURRENT", cls.max_concurrent_sessions
            ),
            acquire_timeout_s=_env_float("CSLLM_ACQUIRE_TIMEOUT", cls.acquire_timeout_s),
            max_prompt_chars=_env_int("CSLLM_MAX_PROMPT_CHARS", cls.max_prompt_chars),
            max_new_tokens=_env_int("CSLLM_MAX_NEW_TOKENS", cls.max_new_tokens),
        )


def get_settings() -> Settings:
    return Settings.from_env()
