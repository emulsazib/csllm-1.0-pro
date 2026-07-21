"""Analytic parameter/memory accounting and host probing.

The load-bearing property: ``calculate_model_params`` must agree with the C++
``ModelConfig::num_params()`` for *every* config, not just the two in
``configs/``. The Python side exists only so the UI can cost an architecture
without allocating one — the moment the two disagree, the dashboard is lying
about the model the user is about to build.
"""

from __future__ import annotations

import itertools

import pytest

from csllm import _csllm_core as core
from csllm.config import config_from_dict, config_to_dict, load_config
from csllm.params import BYTES_PER_ELEM, calculate_model_params
from csllm.resources import current_rss, probe_device, total_memory

BASE = {
    "vocab_size": 512,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "block_size": 32,
    "ffn_hidden": 128,
}


# ── the invariant that matters ───────────────────────────────────────────────


@pytest.mark.parametrize("path", ["configs/debug.json", "configs/shakespeare.json"])
def test_matches_engine_for_shipped_configs(path):
    cfg = load_config(path)
    breakdown, _ = calculate_model_params(config_to_dict(cfg))
    assert breakdown.total == cfg.num_params()


def test_matches_engine_across_a_config_sweep():
    """A sweep, not two spot checks: the C++ layout must not drift from Python."""
    for n_layer, n_head, n_embd, ffn_hidden, vocab_size in itertools.product(
        (1, 3, 6), (1, 2, 4), (32, 64, 256), (64, 341), (256, 4096)
    ):
        if n_embd % n_head or (n_embd // n_head) % 2:
            continue  # invalid: the engine would reject it
        config = {
            **BASE,
            "n_layer": n_layer,
            "n_head": n_head,
            "n_embd": n_embd,
            "ffn_hidden": ffn_hidden,
            "vocab_size": vocab_size,
        }
        breakdown, _ = calculate_model_params(config)
        assert breakdown.total == config_from_dict(config).num_params(), config


def test_shakespeare_is_the_documented_12_19M():
    """README and the exported bundles quote this number; pin it."""
    breakdown, _ = calculate_model_params(config_to_dict(load_config("configs/shakespeare.json")))
    assert breakdown.total == 12_194_688
    assert round(breakdown.total / 1e6, 2) == 12.19


# ── decomposition ────────────────────────────────────────────────────────────


def test_parts_sum_to_total():
    breakdown, _ = calculate_model_params(BASE)
    assert breakdown.embedding + breakdown.attention + breakdown.ffn + breakdown.norms == (
        breakdown.total
    )


def test_embedding_counted_once_because_lm_head_is_tied():
    """Double-counting the tied embedding overstates the 12M config by 1.57M."""
    breakdown, _ = calculate_model_params(BASE)
    assert breakdown.embedding == BASE["vocab_size"] * BASE["n_embd"]


def test_ffn_has_three_matrices_for_swiglu():
    breakdown, _ = calculate_model_params(BASE)
    expected = BASE["n_layer"] * 3 * BASE["n_embd"] * BASE["ffn_hidden"]
    assert breakdown.ffn == expected


def test_attention_scales_quadratically_in_n_embd():
    small, _ = calculate_model_params({**BASE, "n_embd": 64})
    large, _ = calculate_model_params({**BASE, "n_embd": 128})
    assert large.attention == small.attention * 4


# ── memory ───────────────────────────────────────────────────────────────────


def test_memory_breakdown_is_consistent():
    breakdown, memory = calculate_model_params(BASE)
    assert memory.params == breakdown.total * BYTES_PER_ELEM
    assert memory.gradients == memory.params
    assert memory.optimizer == memory.params * 2  # AdamW: exp_avg + exp_avg_sq
    assert memory.total == memory.params * 4 + memory.activations


def test_activations_use_the_engines_own_estimate():
    cfg = config_from_dict(BASE)
    _, memory = calculate_model_params(BASE, batch_size=4)
    assert memory.activations == core.estimate_activation_bytes(cfg, 4, cfg.block_size)


def test_seq_len_defaults_to_block_size_and_shortens_activations():
    _, full = calculate_model_params(BASE, batch_size=2)
    _, short = calculate_model_params(BASE, batch_size=2, seq_len=BASE["block_size"] // 2)
    _, explicit = calculate_model_params(BASE, batch_size=2, seq_len=BASE["block_size"])
    assert full.activations == explicit.activations
    assert short.activations < full.activations


def test_activations_grow_with_batch_size():
    _, small = calculate_model_params(BASE, batch_size=1)
    _, large = calculate_model_params(BASE, batch_size=16)
    assert large.activations > small.activations


# ── validation is the engine's ───────────────────────────────────────────────


def test_rejects_n_embd_not_divisible_by_n_head():
    with pytest.raises(RuntimeError, match="divisible"):
        calculate_model_params({**BASE, "n_embd": 65, "n_head": 2})


def test_rejects_odd_head_dim_because_rope_pairs_channels():
    with pytest.raises(RuntimeError, match="even"):
        calculate_model_params({**BASE, "n_embd": 6, "n_head": 2})


def test_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown config keys"):
        calculate_model_params({**BASE, "d_model": 64})


def test_accepts_a_prebuilt_modelconfig():
    cfg = config_from_dict(BASE)
    breakdown, _ = calculate_model_params(cfg)
    assert breakdown.total == cfg.num_params()


# ── host probing ─────────────────────────────────────────────────────────────


def test_probe_reports_a_labelled_device():
    device = probe_device()
    assert device.kind in {"cuda", "apple-silicon", "cpu"}
    assert device.memory_label in {"VRAM", "Unified memory", "RAM"}
    assert device.total_bytes > 0  # every supported host can report total memory
    assert device.device and device.source


def test_current_rss_is_positive_for_this_process():
    assert current_rss() > 0


def test_current_rss_is_zero_for_a_missing_pid():
    """A dead trainer must yield 0, not an exception on the telemetry path."""
    assert current_rss(9_999_999) == 0


def test_total_memory_is_stable():
    assert total_memory() == total_memory() > 0
