"""Model configuration versioning and export."""

from __future__ import annotations

import asyncio
import contextlib
import json
import tempfile
import zipfile
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

from ..schemas import (
    ConfigureModelRequest,
    ConfigVersionResponse,
    EstimateRequest,
    EstimateResponse,
    ExportRequest,
    ExportResponse,
    ExportSummary,
)
from ..versioning import ConfigStore

router = APIRouter(tags=["config"])


def get_store(request: Request) -> ConfigStore:
    store = getattr(request.app.state, "config_store", None)
    return store or ConfigStore()


@router.post("/configure_model/estimate", response_model=EstimateResponse)
async def estimate_model(req: EstimateRequest) -> EstimateResponse:
    """Cost an architecture without building or persisting anything.

    Deliberately side-effect free. The configurator calls this on every slider
    movement; routing that through ``/configure_model`` would bury the real
    versions under thousands of throwaway files.
    """
    from csllm.params import calculate_model_params
    from csllm.resources import probe_device

    fields = req.model_dump(exclude={"batch_size", "seq_len"})
    try:
        breakdown, memory = calculate_model_params(fields, req.batch_size, req.seq_len)
    except RuntimeError as exc:  # csllm::Error from ModelConfig::validate()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    device = probe_device()
    return EstimateResponse(
        config=fields,
        num_params=breakdown.total,
        params=breakdown.to_dict(),
        memory=memory.to_dict(),
        device=device.to_dict(),
        # total_bytes is 0 when the host could not be probed; do not claim a
        # config does not fit just because we failed to measure the machine.
        fits=device.total_bytes == 0 or memory.total <= device.total_bytes,
    )


@router.post("/configure_model", response_model=ConfigVersionResponse)
async def configure_model(req: ConfigureModelRequest, request: Request) -> ConfigVersionResponse:
    """Validate hyperparameters, persist a new version, optionally build a model.

    Validation runs through the C++ ``ModelConfig.validate()``, so a config the
    engine would refuse (odd head_dim, n_embd not divisible by n_head) is a 422
    here rather than a crash at build time.
    """
    store = get_store(request)
    fields = req.model_dump(exclude={"note", "batch_size", "initialize", "out"})

    try:
        version, created = store.create(fields, note=req.note, batch_size=req.batch_size)
    except RuntimeError as exc:  # csllm::Error from ModelConfig::validate()
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    checkpoint = None
    if req.initialize:
        out = Path(req.out or f"data/{version.version_id}.csllm")
        checkpoint = str(store.build_model(version, out))

    return ConfigVersionResponse(
        **version.to_dict() | {"created": created, "checkpoint": checkpoint}
    )


@router.get("/configs", response_model=list[ConfigVersionResponse])
async def list_configs(request: Request) -> list[ConfigVersionResponse]:
    return [
        ConfigVersionResponse(**v.to_dict() | {"created": False})
        for v in get_store(request).list()
    ]


@router.get("/configs/{version_id}", response_model=ConfigVersionResponse)
async def get_config(version_id: str, request: Request) -> ConfigVersionResponse:
    try:
        version = get_store(request).get(version_id)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    return ConfigVersionResponse(**version.to_dict() | {"created": False})


EXPORTS_DIR = Path("exports")


def _bundle_files(directory: Path) -> dict[str, int]:
    """Every file in the bundle, keyed by its path relative to the bundle root.

    Recursive: `runtime/` and `cpp/` are part of what was exported, and a flat
    listing would report a deployment package as three files.
    """
    return {
        str(p.relative_to(directory)): p.stat().st_size
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def _resolve_export(name: str) -> Path:
    """Map a bundle name to a directory under ``exports/``.

    ``name`` comes from the URL, so resolve the whole path and check its parent —
    the same guard the dataset routes use, which also closes symlink escapes.
    """
    root = EXPORTS_DIR.resolve()
    path = (EXPORTS_DIR / name).resolve()
    if path.parent != root or not path.is_dir():
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"no such export: {name}")
    return path


@router.post("/export", response_model=ExportResponse)
async def export_model(req: ExportRequest) -> ExportResponse:
    """Write a portable safetensors bundle (no torch involved).

    Export reads every weight and writes ~49 MB for the 12M model, so it runs on
    a worker thread — doing it inline would stall every concurrent stream.
    """
    from csllm.export import export_bundle

    try:
        manifest = await asyncio.to_thread(
            export_bundle,
            req.checkpoint,
            req.tokenizer_dir,
            req.out,
            req.include_runtime,
            req.include_cpp,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    out = Path(req.out)
    files = _bundle_files(out)
    return ExportResponse(
        out_dir=str(out),
        name=out.name,
        num_params=manifest["num_params"],
        files=files,
        total_bytes=sum(files.values()),
        includes=manifest.get("includes", []),
    )


@router.get("/exports", response_model=list[ExportSummary])
async def list_exports() -> list[ExportSummary]:
    """Bundles already written under ``exports/``."""
    summaries: list[ExportSummary] = []
    if not EXPORTS_DIR.is_dir():
        return summaries

    for directory in sorted(EXPORTS_DIR.iterdir()):
        if not directory.is_dir() or not (directory / "config.json").is_file():
            continue  # not a bundle
        files = _bundle_files(directory)
        manifest: dict = {}
        # A hand-edited config.json should leave the bundle listed without its
        # metadata rather than dropping it from the listing entirely.
        with contextlib.suppress(OSError, ValueError):
            manifest = json.loads((directory / "config.json").read_text())
        summaries.append(
            ExportSummary(
                name=directory.name,
                path=str(directory),
                num_params=manifest.get("num_params"),
                total_bytes=sum(files.values()),
                file_count=len(files),
                exported_at=manifest.get("exported_at"),
                includes=manifest.get("includes", []),
            )
        )
    return summaries


def _zip_bundle(directory: Path) -> tempfile.SpooledTemporaryFile:
    """Zip a bundle into a spooled buffer (memory until 8 MB, then disk).

    Deliberately not built entirely in memory: a 12M-parameter bundle is ~49 MB
    of safetensors, and a handful of concurrent downloads would be enough to
    matter. ZIP_STORED, not DEFLATE — safetensors is raw float32 that does not
    compress meaningfully, so deflating it burns CPU for ~1% off the size.
    """
    # noqa SIM115: the buffer deliberately outlives this function — the streaming
    # response reads from it and closes it when the download ends or aborts.
    buffer = tempfile.SpooledTemporaryFile(max_size=8 * 1024 * 1024)  # noqa: SIM115
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in sorted(directory.rglob("*")):
            if path.is_file():
                archive.write(path, arcname=str(Path(directory.name) / path.relative_to(directory)))
    buffer.seek(0)
    return buffer


@router.get("/export/{name}/download")
async def download_export(name: str) -> StreamingResponse:
    """Stream a bundle as a zip."""
    directory = _resolve_export(name)
    buffer = await asyncio.to_thread(_zip_bundle, directory)

    def chunks():
        try:
            while data := buffer.read(64 * 1024):
                yield data
        finally:
            # Closing releases the spill file; without this an aborted download
            # leaks one temp file per attempt.
            buffer.close()

    return StreamingResponse(
        chunks(),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}.zip"'},
    )
