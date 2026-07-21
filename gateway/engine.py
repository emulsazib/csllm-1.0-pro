"""Async wrapper around the C++ generation engine — Phase 4.

Owns the loaded model, the concurrency semaphore, and the async generator that
turns C++ ``GenerationSession.step()`` calls into an awaitable token stream.

Also owns the streaming-detokenization buffer: byte-level BPE can emit a token
that is only part of a UTF-8 sequence, so bytes are accumulated and flushed at
code-point boundaries rather than decoded per token.
"""

from __future__ import annotations
