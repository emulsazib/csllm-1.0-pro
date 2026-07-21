"""Dataset browsing, backed by the plugin registry."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, status

import datasets as ds

router = APIRouter(tags=["datasets"])


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


@router.get("/datasets/{name}")
async def describe_dataset(name: str) -> dict:
    path = ds.RAW_DIR / name
    # Reject traversal: `name` comes straight from the URL.
    if path.parent.resolve() != ds.RAW_DIR.resolve() or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such dataset: {name}")
    try:
        return ds.describe(path).to_dict()
    except ds.DatasetError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.get("/datasets/{name}/preview")
async def preview_dataset(name: str, limit: int = 5) -> dict:
    path = ds.RAW_DIR / name
    if path.parent.resolve() != ds.RAW_DIR.resolve() or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such dataset: {name}")
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
