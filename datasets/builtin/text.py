"""Plain-text datasets (``.txt``, ``.md``).

Two modes:

* ``split="whole"`` (default) — the file is one document. Correct for a
  continuous corpus like TinyShakespeare, where every line follows from the last
  and inserting document boundaries would destroy real context.
* ``split="blank_line"`` — blank-line-separated paragraphs become documents.
  Use when a ``.txt`` file is really a concatenation of unrelated snippets.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

from ..base import DatasetError, DatasetPlugin

_BLANK_LINE = re.compile(r"\n\s*\n")


class TextDataset(DatasetPlugin):
    name = "text"
    extensions = (".txt", ".md", ".text")

    def documents(self, path: Path) -> Iterator[str]:
        mode = self.options.get("split", "whole")
        text = self._read_text(path)

        if mode == "whole":
            if text:
                yield text
            return

        if mode == "blank_line":
            for chunk in _BLANK_LINE.split(text):
                chunk = chunk.strip()
                if chunk:
                    yield chunk
            return

        raise DatasetError(f"unknown split mode {mode!r}; expected 'whole' or 'blank_line'")
