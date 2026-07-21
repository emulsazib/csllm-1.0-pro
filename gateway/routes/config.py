"""Model configuration versioning and export."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, status

from ..schemas import (
    ConfigureModelRequest,
    ConfigVersionResponse,
    ExportRequest,
    ExportResponse,
)
from ..versioning import ConfigStore

router = APIRouter(tags=["config"])


def get_store(request: Request) -> ConfigStore:
    store = getattr(request.app.state, "config_store", None)
    return store or ConfigStore()


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


@router.post("/export", response_model=ExportResponse)
async def export_model(req: ExportRequest) -> ExportResponse:
    """Write a portable safetensors bundle (no torch involved)."""
    from csllm.export import export_bundle

    try:
        manifest = export_bundle(req.checkpoint, req.tokenizer_dir, req.out)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (ValueError, RuntimeError) as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc

    out = Path(req.out)
    return ExportResponse(
        out_dir=str(out),
        num_params=manifest["num_params"],
        files={p.name: p.stat().st_size for p in sorted(out.iterdir()) if p.is_file()},
    )
