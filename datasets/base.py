"""The dataset plugin contract.

A plugin turns a file into a stream of *documents*. Everything downstream —
tokenizer training, binarization, the gateway's dataset browser — consumes that
stream, so adding a new format means implementing one method.

Documents are yielded lazily. A corpus can be far larger than memory, and the
BPE trainer only needs pre-token frequencies, not the whole string at once.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

__all__ = ["REGISTERED_PLUGINS", "DatasetError", "DatasetInfo", "DatasetPlugin"]

#: Every concrete DatasetPlugin subclass, in definition order. Populated by
#: ``__init_subclass__`` so a new plugin module needs no registration call.
REGISTERED_PLUGINS: list[type[DatasetPlugin]] = []


class DatasetError(ValueError):
    """A dataset file is malformed or unreadable.

    Distinct from a bare ValueError so the gateway can map it to a 422 rather
    than a 500 — a bad upload is the caller's problem, not a server fault.
    """


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    path: Path
    plugin: str
    num_documents: int
    num_chars: int
    num_bytes: int
    sample: str

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "path": str(self.path),
            "plugin": self.plugin,
            "num_documents": self.num_documents,
            "num_chars": self.num_chars,
            "num_bytes": self.num_bytes,
            "sample": self.sample,
        }


class DatasetPlugin(ABC):
    """Base class for dataset readers.

    Subclasses are auto-registered by extension (see ``datasets/__init__.py``),
    so dropping a module into ``datasets/builtin/`` is enough to support a new
    format — no registration call required.
    """

    #: Short identifier, e.g. "jsonl".
    name: ClassVar[str] = ""
    #: Lower-case extensions this plugin claims, e.g. (".jsonl", ".ndjson").
    extensions: ClassVar[tuple[str, ...]] = ()

    def __init__(self, **options) -> None:
        self.options = options

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        # Abstract intermediates (no documents() implementation) are skipped.
        if getattr(cls.documents, "__isabstractmethod__", False):
            return
        if not cls.name:
            raise TypeError(f"{cls.__name__} must set a class-level `name`")
        if not cls.extensions:
            raise TypeError(f"{cls.__name__} must set class-level `extensions`")
        REGISTERED_PLUGINS.append(cls)

    @abstractmethod
    def documents(self, path: Path) -> Iterator[str]:
        """Yield each document's text. Empty documents must be skipped."""

    def describe(self, path: Path, sample_chars: int = 400) -> DatasetInfo:
        """Stream the file once to collect counts and a preview.

        Deliberately streaming rather than reading into memory: this runs on
        user-supplied files of unknown size, often just to populate a UI list.
        """
        num_documents = 0
        num_chars = 0
        sample = ""
        for doc in self.documents(path):
            num_documents += 1
            num_chars += len(doc)
            if len(sample) < sample_chars:
                sample = (sample + doc)[:sample_chars]
        return DatasetInfo(
            name=path.name,
            path=path,
            plugin=self.name,
            num_documents=num_documents,
            num_chars=num_chars,
            num_bytes=path.stat().st_size,
            sample=sample,
        )

    # ── helpers for subclasses ───────────────────────────────────────────────

    @staticmethod
    def _read_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise DatasetError(f"no such dataset file: {path}") from exc
        except UnicodeDecodeError as exc:
            raise DatasetError(
                f"{path.name} is not valid UTF-8 (byte {exc.start}); "
                "convert it to UTF-8 before use"
            ) from exc

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} extensions={self.extensions}>"
