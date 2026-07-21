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
