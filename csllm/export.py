"""Export a trained checkpoint to a portable, self-contained bundle.

    python -m csllm.export --checkpoint data/model.csllm --out exports/v1

Produces three files:

===================== ==========================================================
``model.safetensors``  weights, flat dotted names (``blocks.0.attn.wq``)
``tokenizer.json``     merges, vocab, pattern, special tokens
``config.json``        ModelConfig + provenance
===================== ==========================================================

**Why safetensors and not ``.pt``.** ``.pt`` is a pickle and requires torch both
to write and to read. This project must never depend on torch
(``memory-bank/rules.md`` rule #1), and pickles execute arbitrary code on load.
safetensors is a JSON header plus raw little-endian tensor bytes — writable from
numpy alone, readable by torch/JAX/JS users without any of them being our
dependency, and memory-mappable like our own ``.csllm`` format.

Weights are read through ``Model.get_param()``, which returns **zero-copy views**
into the C++ arenas, so exporting a 12M model costs one buffer copy per tensor
rather than a second full model in memory.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import _csllm_core as core
from .config import config_to_dict
from .tokenizer import BPETokenizer

__all__ = ["export_bundle", "export_config", "export_tokenizer", "export_weights"]

FORMAT_VERSION = 1


def export_weights(model, out_path: str | Path) -> dict[str, tuple[int, ...]]:
    """Write ``model.safetensors``. Returns {name: shape} for the manifest."""
    try:
        from safetensors.numpy import save_file
    except ImportError as exc:  # pragma: no cover - dependency guidance
        raise ImportError(
            "safetensors is required for export: pip install safetensors\n"
            "(the numpy backend is used — torch is NOT required)"
        ) from exc

    # np.array(..., copy=True): get_param hands back a view into the C++ arena,
    # and safetensors needs contiguous owned buffers it can serialize.
    tensors = {
        name: np.array(model.get_param(name), dtype=np.float32, copy=True)
        for name in model.param_names()
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out_path), metadata={"format": "csllm", "version": str(FORMAT_VERSION)})
    return {name: tuple(t.shape) for name, t in tensors.items()}


def export_tokenizer(tokenizer: BPETokenizer, out_path: str | Path) -> None:
    """Write ``tokenizer.json`` — everything needed to rebuild the tokenizer."""
    payload = {
        "format": "csllm-bpe",
        "version": FORMAT_VERSION,
        "pattern": tokenizer.pattern,
        "vocab_size": tokenizer.vocab_size,
        "special_tokens": tokenizer.special_tokens,
        # Ordered: index in this list IS the merge rank, which encoding depends on.
        "merges": [
            [a, b] for (a, b), _ in sorted(tokenizer.merges.items(), key=lambda kv: kv[1])
        ],
        # Byte lists, not strings: many tokens are not valid UTF-8 on their own.
        "vocab": {str(i): list(b) for i, b in sorted(tokenizer.vocab.items())},
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))


def export_config(model, out_path: str | Path, extra: dict | None = None) -> dict:
    cfg = config_to_dict(model.config)
    build = core.build_info()
    payload = {
        "format": "csllm",
        "version": FORMAT_VERSION,
        "architecture": {
            "type": "csllm-transformer",
            "norm": "rmsnorm",
            "position": "rope-interleaved",
            "ffn": "swiglu",
            "residual": "pre-norm",
            "tied_embeddings": True,
        },
        "config": cfg,
        "num_params": model.num_params(),
        "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "engine_version": build.version,
        **(extra or {}),
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    return payload


def export_bundle(
    checkpoint: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
) -> dict:
    """Export weights + tokenizer + config into ``out_dir``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    model = core.Model.load(str(checkpoint))
    tokenizer = BPETokenizer.load(tokenizer_dir)

    # Catching this here is the difference between a clear failure now and a
    # bundle that produces garbage for whoever deploys it.
    if tokenizer.vocab_size != model.config.vocab_size:
        raise ValueError(
            f"tokenizer vocab_size={tokenizer.vocab_size} != model "
            f"{model.config.vocab_size}; they must come from the same training run"
        )

    shapes = export_weights(model, out / "model.safetensors")
    export_tokenizer(tokenizer, out / "tokenizer.json")
    return export_config(
        model,
        out / "config.json",
        extra={
            "source_checkpoint": str(checkpoint),
            "tensors": {name: list(shape) for name, shape in shapes.items()},
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/model.csllm")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--out", default="exports/latest")
    args = parser.parse_args()

    manifest = export_bundle(args.checkpoint, args.tokenizer_dir, args.out)
    out = Path(args.out)
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"exported {manifest['num_params']:,} params -> {out}/ ({total / 1e6:.1f} MB)")
    for item in sorted(out.iterdir()):
        print(f"  {item.name:<20} {item.stat().st_size / 1e6:7.2f} MB")


if __name__ == "__main__":
    main()
