"""CSV / TSV datasets.

Reads the ``text`` column by default; override with ``column=`` (a name or a
0-based index). Uses the stdlib ``csv`` module, so quoted fields containing
commas and embedded newlines are handled correctly — a hand-rolled ``split(",")``
would silently corrupt exactly the prose-heavy rows this is meant to read.
"""

from __future__ import annotations

import csv as _csv
from collections.abc import Iterator
from pathlib import Path

from ..base import DatasetError, DatasetPlugin

# Some corpora ship rows far larger than csv's default 128 KiB field cap.
_csv.field_size_limit(16 * 1024 * 1024)


class CsvDataset(DatasetPlugin):
    name = "csv"
    extensions = (".csv", ".tsv")

    def documents(self, path: Path) -> Iterator[str]:
        column = self.options.get("column", "text")
        delimiter = self.options.get("delimiter") or ("\t" if path.suffix == ".tsv" else ",")

        try:
            handle = path.open("r", encoding="utf-8", newline="")
        except FileNotFoundError as exc:
            raise DatasetError(f"no such dataset file: {path}") from exc

        with handle:
            if isinstance(column, int):
                for lineno, row in enumerate(_csv.reader(handle, delimiter=delimiter), start=1):
                    if column >= len(row):
                        raise DatasetError(
                            f"{path.name}:{lineno} has {len(row)} columns; "
                            f"index {column} is out of range"
                        )
                    if row[column].strip():
                        yield row[column]
                return

            reader = _csv.DictReader(handle, delimiter=delimiter)
            if reader.fieldnames is None:
                raise DatasetError(f"{path.name} is empty or has no header row")
            if column not in reader.fieldnames:
                available = ", ".join(reader.fieldnames)
                raise DatasetError(
                    f"{path.name} has no {column!r} column (found: {available}); "
                    f"pass column=<name> or a 0-based index"
                )
            for row in reader:
                value = row.get(column) or ""
                if value.strip():
                    yield value
