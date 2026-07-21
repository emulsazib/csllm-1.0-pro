"""Dataset plugin registry.

Drop a ``.txt``, ``.jsonl``, or ``.csv`` file into ``datasets/raw/`` and the
loader picks the right reader by extension:

    from datasets import iter_documents, discover
    for doc in iter_documents(discover()):
        ...

Adding a format means subclassing ``DatasetPlugin`` in ``datasets/builtin/`` (or
anywhere imported before use) — registration happens on class definition.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from pathlib import Path

from . import builtin as _builtin  # noqa: F401  (import registers the built-ins)
from .base import REGISTERED_PLUGINS, DatasetError, DatasetInfo, DatasetPlugin

__all__ = [
    "RAW_DIR",
    "DatasetError",
    "DatasetInfo",
    "DatasetPlugin",
    "describe",
    "discover",
    "iter_documents",
    "plugin_for",
    "supported_extensions",
]

RAW_DIR = Path(__file__).parent / "raw"


def _extension_map() -> dict[str, type[DatasetPlugin]]:
    """Extension -> plugin class. Later registrations win, so a user plugin can
    deliberately override a built-in for the same extension."""
    mapping: dict[str, type[DatasetPlugin]] = {}
    for cls in REGISTERED_PLUGINS:
        for ext in cls.extensions:
            mapping[ext.lower()] = cls
    return mapping


def supported_extensions() -> list[str]:
    return sorted(_extension_map())


def plugin_for(path: str | Path, **options) -> DatasetPlugin:
    """Instantiate the plugin registered for this file's extension."""
    path = Path(path)
    cls = _extension_map().get(path.suffix.lower())
    if cls is None:
        supported = ", ".join(supported_extensions())
        raise DatasetError(
            f"no dataset plugin handles {path.suffix or '(no extension)'!r} "
            f"for {path.name}; supported: {supported}"
        )
    return cls(**options)


def discover(directory: str | Path = RAW_DIR) -> list[Path]:
    """List files in ``directory`` that some plugin can read, sorted by name."""
    root = Path(directory)
    if not root.is_dir():
        return []
    known = set(_extension_map())
    return sorted(p for p in root.iterdir() if p.is_file() and p.suffix.lower() in known)


def describe(path: str | Path, **options) -> DatasetInfo:
    path = Path(path)
    return plugin_for(path, **options).describe(path)


def iter_documents(paths: Iterable[str | Path], **options) -> Iterator[str]:
    """Stream documents across several files, in the order given.

    Lazy on purpose: a corpus can exceed memory, and the BPE trainer only needs
    pre-token frequencies rather than one concatenated string.
    """
    for path in paths:
        path = Path(path)
        yield from plugin_for(path, **options).documents(path)
