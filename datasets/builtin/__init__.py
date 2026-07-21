"""Built-in dataset readers.

Importing this package is what registers them: ``DatasetPlugin.__init_subclass__``
fires on class definition, so a module dropped in here is picked up with no
registration call. See ``datasets/__init__.py``.
"""

from __future__ import annotations

from .csv import CsvDataset
from .jsonl import JsonlDataset
from .text import TextDataset

__all__ = ["CsvDataset", "JsonlDataset", "TextDataset"]
