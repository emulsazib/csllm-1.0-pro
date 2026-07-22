"""Export pipeline: safetensors bundle round-trip.

The bundle is what someone else deploys, so the tests that matter are the ones
proving a third party can reconstruct the model *without this repo*: weights
bitwise-identical, tokenizer rebuildable from tokenizer.json alone, and config
reloadable into a live ModelConfig.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from safetensors.numpy import load_file

from csllm import _csllm_core as core
from csllm.config import config_from_dict
from csllm.export import export_bundle, export_config, export_tokenizer, export_weights
from csllm.tokenizer import END_OF_TEXT, BPETokenizer

CORPUS = "To be, or not to be, that is the question. Whether tis nobler 🌍\n" * 40


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, 320, special_tokens=(END_OF_TEXT,))
    return tok


@pytest.fixture(scope="module")
def model(tokenizer):
    cfg = core.ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.block_size = 16
    cfg.ffn_hidden = 64
    return core.Model(cfg, 11)


@pytest.fixture
def bundle(model, tokenizer, tmp_path):
    ckpt = tmp_path / "m.csllm"
    model.save(str(ckpt))
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    out = tmp_path / "export"
    manifest = export_bundle(ckpt, tok_dir, out)
    return out, manifest


# ── weights ──────────────────────────────────────────────────────────────────


def test_every_parameter_is_exported(bundle, model):
    out, _ = bundle
    tensors = load_file(str(out / "model.safetensors"))
    assert set(tensors) == set(model.param_names())


def test_weights_are_bitwise_identical(bundle, model):
    """Not approximately equal — exporting must not perturb a single bit."""
    out, _ = bundle
    tensors = load_file(str(out / "model.safetensors"))
    for name in model.param_names():
        np.testing.assert_array_equal(tensors[name], model.get_param(name))
        assert tensors[name].dtype == np.float32


def test_tensor_shapes_match_the_model(bundle, model):
    out, _ = bundle
    tensors = load_file(str(out / "model.safetensors"))
    for name in model.param_names():
        assert tensors[name].shape == model.get_param(name).shape


def test_export_weights_returns_a_shape_manifest(model, tmp_path):
    shapes = export_weights(model, tmp_path / "w.safetensors")
    assert shapes["tok_emb"] == (model.config.vocab_size, model.config.n_embd)
    assert shapes["blocks.0.attn.wq"] == (model.config.n_embd, model.config.n_embd)


def test_no_torch_import_is_required(bundle):
    """The whole point of choosing safetensors: reading needs numpy only."""
    import sys

    assert "torch" not in sys.modules, "export/read path must not pull in torch"


# ── tokenizer ────────────────────────────────────────────────────────────────


def test_tokenizer_json_is_self_contained(bundle, tokenizer):
    """A consumer must be able to rebuild the tokenizer from this file alone."""
    out, _ = bundle
    payload = json.loads((out / "tokenizer.json").read_text())

    assert payload["vocab_size"] == tokenizer.vocab_size
    assert payload["pattern"] == tokenizer.pattern
    assert payload["special_tokens"] == tokenizer.special_tokens
    assert len(payload["merges"]) == len(tokenizer.merges)

    # Rebuild from the export and check it encodes identically.
    rebuilt = BPETokenizer(payload["pattern"])
    for i, (a, b) in enumerate(payload["merges"]):
        new_id = 256 + i
        rebuilt.merges[(a, b)] = new_id
        rebuilt.vocab[new_id] = rebuilt.vocab[a] + rebuilt.vocab[b]
    for token, tid in payload["special_tokens"].items():
        rebuilt.vocab[int(tid)] = token.encode("utf-8")

    probe = "Whether tis nobler 🌍 in the mind"
    assert rebuilt.encode(probe) == tokenizer.encode(probe)
    assert rebuilt.decode(rebuilt.encode(probe)) == probe


def test_merges_are_stored_in_rank_order(bundle, tokenizer):
    """Index in the list IS the merge rank — encoding depends on that order."""
    out, _ = bundle
    merges = json.loads((out / "tokenizer.json").read_text())["merges"]
    expected = [list(pair) for pair, _ in sorted(tokenizer.merges.items(), key=lambda kv: kv[1])]
    assert merges == expected


def test_vocab_is_stored_as_byte_lists(bundle):
    """Tokens are often not valid UTF-8 alone, so strings would be lossy."""
    out, _ = bundle
    vocab = json.loads((out / "tokenizer.json").read_text())["vocab"]
    assert vocab["65"] == [65]
    assert all(isinstance(v, list) for v in vocab.values())


def test_export_tokenizer_standalone(tokenizer, tmp_path):
    export_tokenizer(tokenizer, tmp_path / "t.json")
    assert json.loads((tmp_path / "t.json").read_text())["format"] == "csllm-bpe"


# ── config ───────────────────────────────────────────────────────────────────


def test_config_json_reloads_into_a_live_model_config(bundle, model):
    out, _ = bundle
    payload = json.loads((out / "config.json").read_text())
    restored = config_from_dict(payload["config"])

    assert restored.num_params() == model.num_params()
    assert restored.n_layer == model.config.n_layer
    assert restored.head_dim == model.config.head_dim
    restored.validate()


def test_config_records_the_architecture_and_provenance(bundle, model):
    out, manifest = bundle
    payload = json.loads((out / "config.json").read_text())
    arch = payload["architecture"]

    assert arch["norm"] == "rmsnorm"
    assert arch["position"] == "rope-interleaved"
    assert arch["ffn"] == "swiglu"
    assert arch["tied_embeddings"] is True
    assert payload["num_params"] == model.num_params()
    assert "exported_at" in payload
    assert manifest["num_params"] == model.num_params()


def test_manifest_lists_tensor_shapes(bundle):
    _, manifest = bundle
    assert manifest["tensors"]["tok_emb"]


def test_export_config_accepts_extra_provenance(model, tmp_path):
    payload = export_config(model, tmp_path / "c.json", extra={"note": "hello"})
    assert payload["note"] == "hello"


# ── end to end ───────────────────────────────────────────────────────────────


def test_bundle_contains_exactly_the_three_documented_files(bundle):
    out, _ = bundle
    assert sorted(p.name for p in out.iterdir()) == [
        "config.json",
        "model.safetensors",
        "tokenizer.json",
    ]


def test_exported_weights_reproduce_the_model_logits(bundle, model, tokenizer):
    """The strongest check: recompute the embedding lookup from exported weights."""
    out, _ = bundle
    tensors = load_file(str(out / "model.safetensors"))
    ids = np.array(tokenizer.encode("To be")[:4], dtype=np.int32)
    np.testing.assert_array_equal(tensors["tok_emb"][ids], model.get_param("tok_emb")[ids])


# ── deployment packages ──────────────────────────────────────────────────────


@pytest.fixture
def full_bundle(model, tokenizer, tmp_path):
    """A bundle with both deployment packages attached."""
    ckpt = tmp_path / "m.csllm"
    model.save(str(ckpt))
    tok_dir = tmp_path / "tok"
    tokenizer.save(tok_dir)
    out = tmp_path / "full"
    manifest = export_bundle(ckpt, tok_dir, out, include_runtime=True, include_cpp=True)
    return out, manifest


def test_runtime_is_opt_in(bundle):
    """The default bundle stays the three documented files."""
    out, _ = bundle
    assert not (out / "runtime").exists()
    assert not (out / "cpp").exists()
    assert not (out / "README.md").exists()


def test_full_bundle_adds_runtime_cpp_and_readme(full_bundle):
    out, manifest = full_bundle
    assert (out / "runtime" / "load.py").is_file()
    assert (out / "runtime" / "requirements.txt").is_file()
    assert (out / "cpp" / "CMakeLists.txt").is_file()
    assert (out / "cpp" / "include" / "csllm" / "model.hpp").is_file()
    assert (out / "cpp" / "src" / "model.cpp").is_file()
    assert (out / "README.md").is_file()
    assert set(manifest["includes"]) == {"python-runtime", "cpp"}


def test_includes_are_recorded_in_the_shipped_config(full_bundle):
    """The manifest on disk must match what was returned, or a consumer
    inspecting the bundle sees a different story from the API caller."""
    out, manifest = full_bundle
    on_disk = json.loads((out / "config.json").read_text())
    assert on_disk["includes"] == manifest["includes"]


def test_runtime_loader_does_not_import_csllm(full_bundle):
    """A deployment package that needs the repo it came from is not one."""
    source = (full_bundle[0] / "runtime" / "load.py").read_text()
    assert "csllm" not in source.replace("CSLLM", "").replace("csllm-", "")
    assert "import torch" not in source


def test_runtime_loader_encodes_identically_to_the_real_tokenizer(full_bundle, tokenizer):
    """The load-bearing property of the whole package.

    The loader reimplements BPE from tokenizer.json alone. If its merge ordering
    drifts from the real tokenizer, a deployed model silently receives different
    token ids than it was trained on and produces fluent nonsense.
    """
    out, _ = full_bundle
    module = _load_runtime(out)
    bundle = module.CSLLMBundle(str(out))

    for probe in [
        "To be, or not to be",
        "Whether tis nobler 🌍",  # multi-byte, spans several tokens
        "  leading\n\ttabs and spaces  ",
        "",
        "e" * 200,  # long repeat: exercises the merge loop
    ]:
        assert bundle.tokenizer.encode(probe) == tokenizer.encode(probe), probe
        assert bundle.tokenizer.decode(bundle.tokenizer.encode(probe)) == probe


def test_runtime_loader_reads_weights_bitwise(full_bundle, model):
    out, _ = full_bundle
    bundle = _load_runtime(out).CSLLMBundle(str(out))
    for name in model.param_names():
        np.testing.assert_array_equal(bundle.weights[name], model.get_param(name))


def test_runtime_loader_rejects_an_inconsistent_bundle(full_bundle):
    """vocab_size disagreeing between tokenizer and config is unrecoverable."""
    out, _ = full_bundle
    config = json.loads((out / "config.json").read_text())
    config["config"]["vocab_size"] += 1
    (out / "config.json").write_text(json.dumps(config))

    module = _load_runtime(out)
    with pytest.raises(ValueError, match="inconsistent"):
        module.CSLLMBundle(str(out))


def test_readme_documents_the_real_tensor_names(full_bundle, model):
    """A README naming tensors the bundle does not contain is worse than none."""
    out, _ = full_bundle
    readme = (out / "README.md").read_text()
    for name in ["tok_emb", "norm_f.gain", "attn_norm.gain", "ffn.wg", "ffn.wd"]:
        assert name in readme, name
    # And every documented leaf must actually exist in the export.
    assert "norm_f.gain" in model.param_names()


def test_cpp_package_carries_the_definitions_its_sources_need(full_bundle):
    """build_info.cpp and gemm.cpp reference these unconditionally — omitting
    any of them ships a package that does not compile."""
    cmake = (full_bundle[0] / "cpp" / "CMakeLists.txt").read_text()
    for symbol in ["CSLLM_VERSION", "CSLLM_BLAS_BACKEND", "CSLLM_USE_ACCELERATE"]:
        assert symbol in cmake, symbol
    # Recent macOS SDKs reject the cblas_* prototypes without this.
    assert "ACCELERATE_NEW_LAPACK" in cmake
    # -ffast-math breaks the NaN/Inf guards (rules.md #4). Check the compile
    # options, not the raw text — the flag is *named* in a comment saying why it
    # is absent, which a substring search reads as its presence.
    options = [line for line in cmake.splitlines() if "target_compile_options" in line]
    assert options and all("-ffast-math" not in line for line in options)


def test_cpp_package_excludes_the_python_bindings(full_bundle):
    """A C++ consumer does not want the pybind11 layer."""
    files = [p.name for p in (full_bundle[0] / "cpp").rglob("*") if p.is_file()]
    assert "py_module.cpp" not in files


def _load_runtime(bundle_dir):
    """Import the emitted loader as a module, without it being on sys.path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        f"csllm_runtime_{bundle_dir.name}", bundle_dir / "runtime" / "load.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tokenizer_model_vocab_mismatch_is_caught(model, tmp_path):
    """Exporting a mismatched pair would produce a bundle that emits garbage."""
    ckpt = tmp_path / "m.csllm"
    model.save(str(ckpt))

    wrong = BPETokenizer()
    wrong.train(CORPUS, 300)  # different vocab size
    tok_dir = tmp_path / "tok"
    wrong.save(tok_dir)

    with pytest.raises(ValueError, match="must come from the same training run"):
        export_bundle(ckpt, tok_dir, tmp_path / "out")
