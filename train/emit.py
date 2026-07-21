"""Structured progress emission, shared by the training and prepare entrypoints.

One JSON object per line on stdout. The gateway's ``TrainingSupervisor`` treats
a parseable line as a metric row and everything else as log text, so the
human-readable prints in both scripts stay exactly as they are and the scripts
remain usable on their own with the gateway down.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

__all__ = ["make_emitter"]


def make_emitter(enabled: bool, run_id: str) -> Callable[..., None]:
    """Return an ``emit(kind, **fields)`` writing one JSON object per line."""

    def emit(kind: str, **fields) -> None:
        if not enabled:
            return
        payload = {"type": kind, "run_id": run_id, "t": time.time(), **fields}
        # flush: the supervisor reads this pipe live, and block buffering would
        # stall the dashboard until the run ended.
        print(json.dumps(payload), flush=True)

    return emit
