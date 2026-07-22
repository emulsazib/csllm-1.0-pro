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

__all__ = [
    "export_bundle",
    "export_config",
    "export_cpp",
    "export_readme",
    "export_runtime",
    "export_tokenizer",
    "export_weights",
]

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


#: Engine sources copied into a C++ deployment package. The bindings are
#: deliberately excluded — they exist to reach Python, which a C++ consumer of
#: this package does not want.
CPP_TREES = ("include", "src")


def export_runtime(manifest: dict, out_dir: str | Path) -> list[Path]:
    """Write ``runtime/`` — a loader that works without this repository.

    The loader is emitted from a template in ``csllm/runtime_template.py`` rather
    than copied from a module here, because anything that ships must not import
    ``csllm``: a deployment package that needs the repo it came from is not one.
    """
    from .runtime_template import LOADER_SOURCE, REQUIREMENTS

    runtime = Path(out_dir) / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    written = [runtime / "load.py", runtime / "requirements.txt"]
    (runtime / "load.py").write_text(LOADER_SOURCE)
    (runtime / "requirements.txt").write_text(REQUIREMENTS)
    manifest.setdefault("includes", []).append("python-runtime")
    return written


def export_cpp(out_dir: str | Path, root: str | Path | None = None) -> list[Path]:
    """Copy the C++20 engine (headers + sources) plus a standalone CMakeLists.

    Headers alone would not build, so the sources come too — the engine is
    dependency-free C++20 (Accelerate is used when present, otherwise a portable
    GEMM), which is what makes shipping it as a buildable unit reasonable.
    """
    import shutil

    source_root = Path(root) if root else Path(__file__).resolve().parent.parent / "core"
    target = Path(out_dir) / "cpp"
    target.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    for tree in CPP_TREES:
        src = source_root / tree
        if not src.is_dir():
            continue
        dst = target / tree
        shutil.copytree(src, dst, dirs_exist_ok=True)
        written.extend(sorted(p for p in dst.rglob("*") if p.is_file()))

    (target / "CMakeLists.txt").write_text(CPP_CMAKE.format(version=core.build_info().version))
    written.append(target / "CMakeLists.txt")
    return written


#: Mirrors the definitions the project's own CMakeLists supplies. `build_info.cpp`
#: and `gemm.cpp` reference CSLLM_VERSION / CSLLM_BLAS_BACKEND / CSLLM_USE_ACCELERATE
#: unconditionally, so omitting any of them turns the shipped package into one
#: that does not compile. ACCELERATE_NEW_LAPACK is equally load bearing: without
#: it, recent macOS SDKs reject the cblas_* prototypes as deprecated.
CPP_CMAKE = """\
# Standalone build of the CSLLM C++ engine, extracted from an export bundle.
#
#   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
#
# No third-party dependencies. Accelerate provides BLAS on Apple platforms;
# elsewhere the portable GEMM in src/gemm.cpp is used instead.
cmake_minimum_required(VERSION 3.20)
project(csllm_engine VERSION {version} LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CSLLM_BLAS_BACKEND "naive")
if(APPLE)
  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    set(CSLLM_BLAS_BACKEND "Accelerate")
  endif()
endif()

file(GLOB CSLLM_SOURCES ${{CMAKE_CURRENT_SOURCE_DIR}}/src/*.cpp)
add_library(csllm_engine STATIC ${{CSLLM_SOURCES}})
target_include_directories(csllm_engine PUBLIC ${{CMAKE_CURRENT_SOURCE_DIR}}/include)

find_package(Threads REQUIRED)
target_link_libraries(csllm_engine PUBLIC Threads::Threads)

target_compile_definitions(csllm_engine PUBLIC
  CSLLM_VERSION="${{PROJECT_VERSION}}"
  CSLLM_BLAS_BACKEND="${{CSLLM_BLAS_BACKEND}}")

if(CSLLM_BLAS_BACKEND STREQUAL "Accelerate")
  target_link_libraries(csllm_engine PUBLIC ${{ACCELERATE_FRAMEWORK}})
  # ACCELERATE_NEW_LAPACK selects the modern (non-deprecated) cblas_* prototypes;
  # without it recent SDKs fail the build on deprecation.
  target_compile_definitions(csllm_engine PUBLIC
    CSLLM_USE_ACCELERATE=1 ACCELERATE_NEW_LAPACK=1)
else()
  target_compile_definitions(csllm_engine PUBLIC CSLLM_USE_ACCELERATE=0)
endif()

# -ffast-math is deliberately NOT set: it licenses reassociation and breaks the
# engine's NaN/Inf guards and reproducibility.
target_compile_options(csllm_engine PRIVATE -O3 -funroll-loops -fno-math-errno -Wall -Wextra)

message(STATUS "csllm_engine: BLAS backend = ${{CSLLM_BLAS_BACKEND}}")
"""


def export_readme(manifest: dict, out_dir: str | Path) -> Path:
    """Write README.md documenting the format for whoever receives the bundle."""
    from .runtime_template import README_TEMPLATE

    out = Path(out_dir)
    cfg = manifest["config"]
    includes = manifest.get("includes", [])

    extra_files = ""
    runtime_section = ""
    if "python-runtime" in includes:
        extra_files += (
            "| `runtime/load.py` | Standalone loader — no torch, no dependency on the "
            "source repo. |\n"
        )
        runtime_section = (
            "\n### Bundled loader\n\n"
            "```bash\n"
            "pip install -r runtime/requirements.txt\n"
            "python runtime/load.py          # self-check: loads, encodes, round-trips\n"
            "```\n\n"
            "```python\n"
            "from runtime.load import CSLLMBundle\n"
            "bundle = CSLLMBundle('.')\n"
            "ids = bundle.tokenizer.encode('KING RICHARD:')\n"
            "```\n"
        )
    if "cpp" in includes:
        extra_files += "| `cpp/` | The C++20 engine (headers + sources) and a CMakeLists. |\n"
        runtime_section += (
            "\n### C++ engine\n\n"
            "```bash\n"
            "cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release\n"
            "cmake --build cpp/build\n"
            "```\n\n"
            "Builds `libcsllm_engine.a`. Dependency-free C++20; Accelerate is used for BLAS "
            "on Apple platforms and a portable GEMM elsewhere.\n"
        )

    text = README_TEMPLATE.format(
        name=out.name,
        num_params=manifest["num_params"],
        n_layer=cfg["n_layer"],
        n_head=cfg["n_head"],
        n_embd=cfg["n_embd"],
        block_size=cfg["block_size"],
        vocab_size=cfg["vocab_size"],
        head_dim=cfg["n_embd"] // cfg["n_head"],
        norm_eps=cfg["norm_eps"],
        rope_theta=cfg["rope_theta"],
        exported_at=manifest["exported_at"],
        engine_version=manifest["engine_version"],
        extra_files=extra_files,
        runtime_section=runtime_section,
    )
    path = out / "README.md"
    path.write_text(text)
    return path


def export_bundle(
    checkpoint: str | Path,
    tokenizer_dir: str | Path,
    out_dir: str | Path,
    include_runtime: bool = False,
    include_cpp: bool = False,
) -> dict:
    """Export weights + tokenizer + config into ``out_dir``.

    ``include_runtime`` adds a torch-free Python loader; ``include_cpp`` adds the
    C++ engine sources. Both default to off so the bundle stays the three
    documented files unless a deployment package was explicitly asked for.
    """
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
    manifest = export_config(
        model,
        out / "config.json",
        extra={
            "source_checkpoint": str(checkpoint),
            "tensors": {name: list(shape) for name, shape in shapes.items()},
        },
    )

    if include_runtime:
        export_runtime(manifest, out)
    if include_cpp:
        export_cpp(out)
        manifest.setdefault("includes", []).append("cpp")
    if include_runtime or include_cpp:
        export_readme(manifest, out)
        # Rewrite config.json so `includes` is recorded in the shipped manifest,
        # not only in the value returned to the caller.
        (out / "config.json").write_text(json.dumps(manifest, indent=1))

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="data/model.csllm")
    parser.add_argument("--tokenizer-dir", default="data/tokenizer")
    parser.add_argument("--out", default="exports/latest")
    parser.add_argument(
        "--runtime", action="store_true", help="include the standalone Python loader"
    )
    parser.add_argument("--cpp", action="store_true", help="include the C++ engine sources")
    args = parser.parse_args()

    manifest = export_bundle(
        args.checkpoint,
        args.tokenizer_dir,
        args.out,
        include_runtime=args.runtime,
        include_cpp=args.cpp,
    )
    out = Path(args.out)
    total = sum(p.stat().st_size for p in out.iterdir() if p.is_file())
    print(f"exported {manifest['num_params']:,} params -> {out}/ ({total / 1e6:.1f} MB)")
    for item in sorted(out.iterdir()):
        print(f"  {item.name:<20} {item.stat().st_size / 1e6:7.2f} MB")


if __name__ == "__main__":
    main()
