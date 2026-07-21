"""Attention capture and sampler introspection.

Phase 4's visualization is only worth building if it shows what the model
*actually did*. These tests pin that: captured weights are recomputed
independently in NumPy from the exported parameters, so a capture that silently
recorded the wrong buffer, the wrong layer, or stale data would fail here.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core


def make_model(seed=5):
    cfg = core.ModelConfig()
    cfg.vocab_size = 64
    cfg.n_layer = 3
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.block_size = 16
    cfg.ffn_hidden = 64
    return core.Model(cfg, seed)


@pytest.fixture(scope="module")
def model():
    return make_model()


@pytest.fixture
def session(model):
    s = core.GenerationSession(model, 0)
    s.set_capture_attention(True)
    return s


# ── basic contract ───────────────────────────────────────────────────────────


def test_capture_is_off_by_default(model):
    assert core.GenerationSession(model, 0).capture_attention is False


def test_enabling_capture_is_reflected(session):
    assert session.capture_attention is True
    session.set_capture_attention(False)
    assert session.capture_attention is False


def test_last_attention_requires_capture_enabled(model):
    s = core.GenerationSession(model, 0)
    s.prefill(np.array([1, 2, 3], dtype=np.int32), model.config.vocab_size)
    with pytest.raises(RuntimeError, match="capture is disabled"):
        s.last_attention()


def test_last_attention_requires_a_forward_pass(session):
    with pytest.raises(RuntimeError, match="no attention captured yet"):
        session.last_attention()


def test_shape_is_layers_heads_keys(session, model):
    ids = np.array([3, 9, 14, 2], dtype=np.int32)
    session.prefill(ids, model.config.vocab_size)
    a = session.last_attention()
    assert a.shape == (model.config.n_layer, model.config.n_head, len(ids))
    assert a.dtype == np.float32


# ── mathematical invariants ──────────────────────────────────────────────────


def test_rows_are_probability_distributions(session, model):
    session.prefill(np.array([1, 5, 9, 12, 3], dtype=np.int32), model.config.vocab_size)
    a = session.last_attention()
    np.testing.assert_allclose(a.sum(axis=-1), 1.0, atol=1e-5)
    assert (a >= 0).all()


def test_first_token_attends_only_to_itself(session, model):
    session.prefill(np.array([7], dtype=np.int32), model.config.vocab_size)
    a = session.last_attention()
    assert a.shape[-1] == 1
    assert np.all(a == 1.0)


def test_attention_width_grows_with_position(session, model):
    session.prefill(np.array([1, 2], dtype=np.int32), model.config.vocab_size)
    assert session.last_attention().shape[-1] == 2
    session.decode(4, model.config.vocab_size)
    assert session.last_attention().shape[-1] == 3
    session.decode(5, model.config.vocab_size)
    assert session.last_attention().shape[-1] == 4


def test_capture_does_not_change_the_logits(model):
    """Capture must be observation only — identical output with it on and off."""
    ids = np.array([2, 8, 15], dtype=np.int32)

    plain = core.GenerationSession(model, 0)
    a = plain.prefill(ids, model.config.vocab_size)

    watched = core.GenerationSession(model, 0)
    watched.set_capture_attention(True)
    b = watched.prefill(ids, model.config.vocab_size)

    np.testing.assert_array_equal(a, b)


def test_layer_zero_matches_an_independent_numpy_recomputation(session, model):
    """The decisive check.

    Layer 0's attention depends only on the embedding, the first RMSNorm, wq/wk
    and RoPE — no prior layers — so it can be rederived from the parameters alone.
    A capture that recorded a different layer, a pre-softmax score, or a stale
    buffer would not match.
    """
    cfg = model.config
    H, dh = cfg.n_head, cfg.head_dim
    ids = np.array([4, 11, 2, 9, 6], dtype=np.int32)
    t_len = len(ids)

    session.prefill(ids, cfg.vocab_size)
    captured = session.last_attention()[0]  # layer 0 -> [H, T]

    emb = np.asarray(model.get_param("tok_emb"), dtype=np.float64)
    gain = np.asarray(model.get_param("blocks.0.attn_norm.gain"), dtype=np.float64)
    wq = np.asarray(model.get_param("blocks.0.attn.wq"), dtype=np.float64)
    wk = np.asarray(model.get_param("blocks.0.attn.wk"), dtype=np.float64)

    h = emb[ids]
    hn = gain * h / np.sqrt((h**2).mean(axis=-1, keepdims=True) + cfg.norm_eps)
    q = (hn @ wq).reshape(t_len, H, dh).transpose(1, 0, 2)
    k = (hn @ wk).reshape(t_len, H, dh).transpose(1, 0, 2)

    def rope(x):
        pairs = np.arange(dh // 2)
        inv_freq = cfg.rope_theta ** (-2.0 * pairs / dh)
        angle = np.arange(x.shape[1])[:, None] * inv_freq[None, :]
        cos, sin = np.cos(angle), np.sin(angle)
        out = np.empty_like(x)
        out[..., 0::2] = x[..., 0::2] * cos - x[..., 1::2] * sin
        out[..., 1::2] = x[..., 0::2] * sin + x[..., 1::2] * cos
        return out

    q, k = rope(q), rope(k)
    scores = np.einsum("htd,hsd->hts", q, k) / np.sqrt(dh)
    last = scores[:, t_len - 1, :t_len]
    last = last - last.max(axis=-1, keepdims=True)
    expected = np.exp(last)
    expected /= expected.sum(axis=-1, keepdims=True)

    np.testing.assert_allclose(captured, expected, atol=1e-5)


def test_reset_clears_captured_state(session, model):
    session.prefill(np.array([1, 2, 3], dtype=np.int32), model.config.vocab_size)
    assert session.last_attention().shape[-1] == 3
    session.reset()
    session.prefill(np.array([4], dtype=np.int32), model.config.vocab_size)
    assert session.last_attention().shape[-1] == 1


# ── decode() binding ─────────────────────────────────────────────────────────


def test_decode_returns_logits_without_sampling(model):
    """Needed by the probability UI: logits at a step, no RNG consumed."""
    s = core.GenerationSession(model, 0)
    s.prefill(np.array([1, 2], dtype=np.int32), model.config.vocab_size)
    logits = s.decode(5, model.config.vocab_size)
    assert logits.shape == (model.config.vocab_size,)
    assert np.isfinite(logits).all()
    assert s.position == 3


def test_decode_then_sample_last_agrees_with_step(model):
    """decode()+sample_last() must equal step() — same math, split in two."""
    ids = np.array([2, 7], dtype=np.int32)
    greedy = core.SamplingParams(temperature=0.0)

    a = core.GenerationSession(model, 0)
    a.prefill(ids, model.config.vocab_size)
    via_step = a.step(9, greedy)

    b = core.GenerationSession(model, 0)
    b.prefill(ids, model.config.vocab_size)
    logits = b.decode(9, model.config.vocab_size)
    assert int(np.argmax(logits)) == via_step
    assert b.sample_last(greedy) == via_step


# ── sampler introspection ────────────────────────────────────────────────────


def test_filtered_distribution_is_a_distribution():
    rng = np.random.default_rng(0)
    logits = rng.standard_normal(50).astype(np.float32)
    dist = core.filtered_distribution(logits, core.SamplingParams(temperature=1.0))
    assert dist.shape == (50,)
    assert dist.sum() == pytest.approx(1.0, rel=1e-5)
    assert (dist >= 0).all()


def test_filtered_distribution_zeroes_excluded_tokens():
    rng = np.random.default_rng(1)
    logits = rng.standard_normal(40).astype(np.float32)
    dist = core.filtered_distribution(logits, core.SamplingParams(temperature=1.0, top_k=5))
    assert int((dist > 0).sum()) == 5
    kept = set(np.argsort(logits)[-5:].tolist())
    assert set(np.nonzero(dist)[0].tolist()) == kept


def test_greedy_distribution_is_one_hot():
    logits = np.array([1.0, 5.0, 2.0], dtype=np.float32)
    dist = core.filtered_distribution(logits, core.SamplingParams(temperature=0.0))
    np.testing.assert_array_equal(dist, [0.0, 1.0, 0.0])


def test_temperature_sharpens_the_distribution():
    logits = np.array([2.0, 1.0, 0.0], dtype=np.float32)
    cold = core.filtered_distribution(logits, core.SamplingParams(temperature=0.25))
    warm = core.filtered_distribution(logits, core.SamplingParams(temperature=2.0))
    assert cold[0] > warm[0]
    assert cold.max() > warm.max()


def test_distribution_matches_what_sample_actually_draws():
    """Parity is the whole point: sample() is built on distribution(), so what a
    UI displays cannot drift from what the model draws from."""
    rng = np.random.default_rng(3)
    logits = (rng.standard_normal(12) * 2).astype(np.float32)
    params = core.SamplingParams(temperature=0.9, top_k=6, top_p=0.9)

    expected = core.filtered_distribution(logits, params)
    draws = core.sample_logits(logits, params, 7, 40000)
    observed = np.bincount(draws, minlength=12) / len(draws)

    np.testing.assert_allclose(observed, expected, atol=0.01)
    # Every filtered-out token must never be drawn.
    assert not np.any(draws[:, None] == np.nonzero(expected == 0)[0][None, :])
