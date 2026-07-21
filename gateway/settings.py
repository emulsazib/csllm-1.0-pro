"""Gateway settings — Phase 4.

Environment-driven configuration: checkpoint path, tokenizer directory, max
concurrent sessions, and default sampling parameters.
"""

from __future__ import annotations

DEFAULT_CHECKPOINT = "data/model.csllm"
DEFAULT_TOKENIZER_DIR = "data/tokenizer"
DEFAULT_MAX_CONCURRENT_SESSIONS = 4
