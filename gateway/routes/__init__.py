"""FastAPI routers, split by concern.

``inference`` is the hot path and stays independent of the rest, so a broken
training or dataset route cannot take generation down with it.
"""

from __future__ import annotations

from . import config, datasets, inference, inspect, training

__all__ = ["config", "datasets", "inference", "inspect", "training"]
