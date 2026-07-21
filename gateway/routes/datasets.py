"""Dataset browsing, backed by the plugin registry."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status

import datasets as ds

from ..schemas import PrepareRequest, TrainStatusResponse
from ..telemetry import RunAlreadyActive

router = APIRouter(tags=["datasets"])


def _resolve(name: str) -> Path:
    """Map a bare dataset name to a real file under ``datasets/raw/``.

    ``name`` comes straight from the URL, so neither a traversal
    (``../../etc/passwd``) nor a symlink planted in ``raw/`` may escape.
    Resolving the file itself rather than its parent is what closes the symlink
    case: a link's parent is ``raw/`` no matter where it points.
    """
    root = ds.RAW_DIR.resolve()
    path = (ds.RAW_DIR / name).resolve()
    if path.parent != root or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such dataset: {name}")
    return path


@router.get("/datasets")
async def list_datasets() -> dict:
    """Everything in ``datasets/raw/`` that a plugin can read."""
    items = []
    for path in ds.discover():
        try:
            items.append(ds.describe(path).to_dict())
        except ds.DatasetError as exc:
            # Report the bad file rather than failing the whole listing.
            items.append({"name": path.name, "path": str(path), "error": str(exc)})
    return {"supported_extensions": ds.supported_extensions(), "datasets": items}


#: Prepared artifacts live here, one directory per source dataset. Deliberately
#: NOT keyed by config alone: `data/debug` is the checked-in debug corpus, and a
#: prepare that wrote there would silently destroy it.
PREPARED_DIR = Path("data/prepared")


def prepared_dir_for(dataset: str) -> Path:
    return PREPARED_DIR / Path(dataset).stem


@router.get("/prepared")
async def list_prepared() -> dict:
    """Datasets that have been tokenized and binarized, ready to train on."""
    items = []
    if PREPARED_DIR.is_dir():
        for directory in sorted(PREPARED_DIR.iterdir()):
            train, val = directory / "train.bin", directory / "val.bin"
            tokenizer = directory / "tokenizer"
            if not (train.is_file() and val.is_file() and tokenizer.is_dir()):
                continue  # a half-finished or interrupted prepare
            vocab_size = None
            # vocab.json is {vocab_size, pattern, special_tokens, tokens} — read
            # the field, do not count top-level keys (that reports 4 every time).
            # A hand-edited or truncated file should leave the row listed without
            # its vocab size, not drop the dataset from the listing.
            with contextlib.suppress(OSError, ValueError, TypeError, KeyError):
                vocab_size = json.loads((tokenizer / "vocab.json").read_text())["vocab_size"]
            items.append(
                {
                    "name": directory.name,
                    "data_dir": str(directory),
                    "tokenizer_dir": str(tokenizer),
                    # uint16 on disk, so two bytes per token.
                    "train_tokens": train.stat().st_size // 2,
                    "val_tokens": val.stat().st_size // 2,
                    "vocab_size": vocab_size,
                    "prepared_at": train.stat().st_mtime,
                }
            )
    return {"prepared": items}


@router.get("/datasets/{name}")
async def describe_dataset(name: str) -> dict:
    try:
        return ds.describe(_resolve(name)).to_dict()
    except ds.DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.post("/datasets/{name}/prepare", response_model=TrainStatusResponse)
async def prepare_dataset_route(
    name: str, req: PrepareRequest, request: Request
) -> TrainStatusResponse:
    """Train a tokenizer on this dataset and binarize it into train/val splits.

    Runs as a supervised subprocess on the same WebSocket as training: BPE over a
    real corpus takes minutes, so this cannot be a request that blocks until it
    finishes. One job at a time — a prepare that rewrites ``data/*.bin`` while a
    run is reading them would corrupt that run.
    """
    path = _resolve(name)
    supervisor = getattr(request.app.state, "supervisor", None)
    if supervisor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "training supervisor unavailable")

    # Default the destination to data/prepared/<dataset>/ rather than data/.
    # The caller may override, but the DEFAULT must never point at a checked-in
    # corpus — `data/` and `data/debug/` both hold shipped splits.
    target = prepared_dir_for(path.name)
    options = req.model_dump()
    options["data_dir"] = req.data_dir or str(target)
    options["out"] = req.out or str(target / "tokenizer")

    try:
        await supervisor.start({**options, "kind": "prepare", "dataset": path.name})
    except RunAlreadyActive as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return TrainStatusResponse(**supervisor.status())


@router.get("/datasets/{name}/preview")
async def preview_dataset(name: str, limit: int = 5) -> dict:
    path = _resolve(name)
    limit = max(1, min(limit, 50))
    try:
        docs = []
        for i, doc in enumerate(ds.iter_documents([path])):
            if i >= limit:
                break
            docs.append(doc[:2000])
    except ds.DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return {"name": Path(name).name, "documents": docs}
