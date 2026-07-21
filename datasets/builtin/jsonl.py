"""JSON Lines datasets (``.jsonl``, ``.ndjson``) — one JSON object per line.

Reads the ``text`` field by default; override with ``field=``. Bare JSON strings
are also accepted, since some corpora ship as one quoted string per line.

Parsed line by line rather than loaded whole: these files routinely run to
gigabytes, and a malformed line should name its own line number instead of
failing the entire file anonymously.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from ..base import DatasetError, DatasetPlugin


class JsonlDataset(DatasetPlugin):
    name = "jsonl"
    extensions = (".jsonl", ".ndjson")

    def documents(self, path: Path) -> Iterator[str]:
        field = self.options.get("field", "text")
        strict = self.options.get("strict", True)

        try:
            handle = path.open("r", encoding="utf-8")
        except FileNotFoundError as exc:
            raise DatasetError(f"no such dataset file: {path}") from exc

        with handle:
            for lineno, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise DatasetError(
                        f"{path.name}:{lineno} is not valid JSON: {exc.msg}"
                    ) from exc

                if isinstance(record, str):
                    text = record
                elif isinstance(record, dict):
                    if field not in record:
                        if strict:
                            keys = ", ".join(sorted(record)[:8]) or "<none>"
                            raise DatasetError(
                                f"{path.name}:{lineno} has no {field!r} field (found: {keys}); "
                                f"pass field=<name> to select a different one"
                            )
                        continue
                    value = record[field]
                    if not isinstance(value, str):
                        raise DatasetError(
                            f"{path.name}:{lineno} field {field!r} is "
                            f"{type(value).__name__}, expected a string"
                        )
                    text = value
                else:
                    raise DatasetError(
                        f"{path.name}:{lineno} is a {type(record).__name__}; "
                        "expected an object or a string"
                    )

                if text.strip():
                    yield text
