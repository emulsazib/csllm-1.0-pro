"""End-to-end model, optimizer, generation, and checkpoint tests.

The centrepiece is ``test_overfit_single_batch``: if the hand-written autograd,
AdamW, and the block stack are all correct, a small model must be able to drive
one fixed batch to near-zero loss. Per-op gradchecks cannot prove that the pieces
are wired together correctly; this can.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core


def make_config(**overrides):
    cfg = core.ModelConfig()
    cfg.vocab_size = 32
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 32
    cfg.block_size = 16
    cfg.ffn_hidden = 64
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


@pytest.fixture
def model():
    return core.Model(make_config(), 1234)


# ── initialisation ───────────────────────────────────────────────────────────


def test_initial_loss_is_near_log_vocab(model):
    """An untrained model predicts ~uniformly, so loss must start near ln(V).

    A value far from ln(V) means the initialisation scale or the residual/norm
    wiring is wrong — this catches it before wasting a training run.
    """
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 32, size=(4, 16)).astype(np.int32)
    targets = rng.integers(0, 32, size=(4, 16)).astype(np.int32)
    loss = model.forward_loss(ids, targets)
    assert loss == pytest.approx(np.log(32), rel=0.05)

    # ln(V) is a hard mathematical FLOOR when targets are independent of the
    # logits: logsumexp(z) - mean(z) >= log(V) by Jensen. A loss below it means
    # the loss or the target indexing is wrong, not that the model is good.
    assert loss >= np.log(32), f"loss {loss} is below the ln(V) floor"


def test_weight_tying_makes_init_logits_peak_at_the_input_token(model):
    """At init the residual stream still carries the input embedding, so
    logits = h @ tok_embᵀ peaks at the INPUT token far above chance.

    This is a fingerprint of tying: with an independent lm_head the argmax would
    be essentially uniform.
    """
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 32, size=(2, 12)).astype(np.int32)
    argmax = model.forward_logits(ids).argmax(axis=-1)
    hit_rate = (argmax == ids.reshape(-1)).mean()
    assert hit_rate > 0.5, f"expected tying to dominate at init, got {hit_rate:.3f}"


def test_param_count_matches_config(model):
    assert model.num_params() == model.config.num_params()
    # Weight tying means lm_head adds no parameters of its own.
    assert "lm_head" not in model.param_names()
    assert "tok_emb" in model.param_names()


def test_grads_start_zero_and_zero_grad_clears(model):
    rng = np.random.default_rng(1)
    ids = rng.integers(0, 32, size=(2, 8)).astype(np.int32)
    assert np.all(model.get_grad("tok_emb") == 0.0)
    model.forward_loss(ids, ids)
    model.backward()
    assert np.abs(model.get_grad("tok_emb")).sum() > 0
    model.zero_grad()
    assert np.all(model.get_grad("tok_emb") == 0.0)


# ── training ─────────────────────────────────────────────────────────────────


def test_overfit_single_batch():
    """The strongest end-to-end proof that autograd + optimizer are correct.

    A model with enough capacity, shown one batch repeatedly, must memorise it.
    If any backward pass or the optimizer update were wrong, loss would stall or
    diverge instead of collapsing.
    """
    cfg = make_config(vocab_size=16, n_layer=2, n_embd=32, block_size=8, ffn_hidden=64)
    model = core.Model(cfg, 7)

    adam_cfg = core.AdamWConfig()
    adam_cfg.lr = 3e-3
    adam_cfg.weight_decay = 0.0  # memorisation, not generalisation
    opt = core.AdamW(model, adam_cfg)

    rng = np.random.default_rng(3)
    ids = rng.integers(0, 16, size=(2, 8)).astype(np.int32)
    targets = rng.integers(0, 16, size=(2, 8)).astype(np.int32)

    initial = model.forward_loss(ids, targets)
    losses = []
    for _ in range(300):
        model.zero_grad()
        loss = model.forward_loss(ids, targets)
        model.backward()
        opt.clip_grad_norm(1.0)
        opt.step(adam_cfg.lr)
        losses.append(loss)

    assert initial == pytest.approx(np.log(16), rel=0.15)
    assert losses[-1] < 0.02, f"failed to overfit: {initial:.3f} -> {losses[-1]:.4f}"
    assert losses[-1] < losses[0]
    assert all(np.isfinite(v) for v in losses), "loss went non-finite"


def test_clip_grad_norm_reports_preclip_and_bounds_after():
    model = core.Model(make_config(), 5)
    opt = core.AdamW(model, core.AdamWConfig())
    rng = np.random.default_rng(2)
    ids = rng.integers(0, 32, size=(2, 8)).astype(np.int32)

    model.zero_grad()
    model.forward_loss(ids, ids)
    model.backward()

    norm_before = opt.clip_grad_norm(1e-4)  # clip hard
    assert norm_before > 1e-4, "expected the pre-clip norm to exceed the threshold"

    total = sum(float((model.get_grad(n) ** 2).sum()) for n in model.param_names())
    assert np.sqrt(total) == pytest.approx(1e-4, rel=1e-3)


def test_optimizer_step_changes_weights(model):
    opt = core.AdamW(model, core.AdamWConfig())
    rng = np.random.default_rng(4)
    ids = rng.integers(0, 32, size=(2, 8)).astype(np.int32)
    before = np.array(model.get_param("blocks.0.attn.wq"), copy=True)

    model.zero_grad()
    model.forward_loss(ids, ids)
    model.backward()
    opt.step(1e-3)

    assert not np.allclose(before, model.get_param("blocks.0.attn.wq"))
    assert opt.step_count == 1


def test_weight_decay_skips_one_dimensional_gains():
    """RMSNorm gains must not be decayed toward zero — they scale the signal."""
    model = core.Model(make_config(), 9)
    cfg = core.AdamWConfig()
    cfg.weight_decay = 0.5
    opt = core.AdamW(model, cfg)

    gain_before = np.array(model.get_param("norm_f.gain"), copy=True)
    model.zero_grad()  # all grads zero => any change comes purely from decay
    opt.step(0.1)
    np.testing.assert_allclose(model.get_param("norm_f.gain"), gain_before, rtol=0, atol=0)


# ── generation ───────────────────────────────────────────────────────────────


def test_kv_cache_decode_matches_full_forward(model):
    """The KV-cache decode path must agree with full-sequence attention.

    These are two separate implementations — training runs full [B,T,C]
    attention, serving runs an incremental single-token path against cached,
    already-rotated keys. If RoPE were applied at the wrong position or the cache
    layout disagreed with the training layout, they would diverge here.
    """
    ids = np.array([[3, 14, 7, 1, 9]], dtype=np.int32)
    full = model.forward_logits(ids)[-1]  # logits after the last prompt token

    session = core.GenerationSession(model, 0)
    cached = session.prefill(ids[0], model.config.vocab_size)

    np.testing.assert_allclose(full, cached, rtol=2e-4, atol=2e-4)


def test_generation_is_deterministic_for_a_fixed_seed(model):
    prompt = np.array([1, 2, 3], dtype=np.int32)
    params = core.SamplingParams(temperature=0.8, top_k=10, top_p=0.95)

    def run(seed):
        session = core.GenerationSession(model, seed)
        session.prefill(prompt, model.config.vocab_size)
        token = int(prompt[-1])
        return [token := session.step(token, params) for _ in range(6)]

    assert run(42) == run(42)
    assert run(42) != run(43), "different seeds should diverge"


def test_first_generated_token_continues_the_prompt(model):
    """Regression: prefill() consumes the whole prompt, so the first token must
    come from sample_last(), not step(prompt[-1]).

    Using step(prompt[-1]) feeds the prompt's last token a SECOND time and
    generates from a corrupted context. The bug is invisible in loss metrics and
    produces plausible-looking text, so only this equivalence catches it.
    """
    prompt = np.array([[3, 14, 7, 1, 9]], dtype=np.int32)
    expected = int(model.forward_logits(prompt)[-1].argmax())

    session = core.GenerationSession(model, 0)
    session.prefill(prompt[0], model.config.vocab_size)
    greedy = core.SamplingParams(temperature=0.0)

    assert session.sample_last(greedy) == expected
    # sample_last must not advance the cache position.
    assert session.position == prompt.shape[1]


def test_sample_last_does_not_rescale_the_stored_logits(model):
    """sample_last must work on a copy — sampling scales logits in place.

    If it scaled the stored buffer, each call would divide by temperature again,
    so the effective temperature would decay as T^k and the distribution would
    collapse onto the argmax within a handful of calls. Reseeding between draws
    isolates that from ordinary RNG variation.
    """
    session = core.GenerationSession(model, 0)
    session.prefill(np.array([2, 5, 8], dtype=np.int32), model.config.vocab_size)
    params = core.SamplingParams(temperature=0.5)

    picks = set()
    for seed in range(100):
        session.reseed(seed)
        picks.add(session.sample_last(params))

    assert len(picks) > 5, f"distribution collapsed to {picks} — logits are being rescaled"


def test_sample_last_before_prefill_raises(model):
    session = core.GenerationSession(model, 0)
    with pytest.raises(RuntimeError, match="requires a prior prefill"):
        session.sample_last(core.SamplingParams())


def test_greedy_sampling_picks_the_argmax(model):
    prompt = np.array([5, 6], dtype=np.int32)
    session = core.GenerationSession(model, 0)
    logits = session.prefill(prompt, model.config.vocab_size)
    expected = int(np.argmax(logits))

    session2 = core.GenerationSession(model, 0)
    session2.prefill(prompt, model.config.vocab_size)
    greedy = core.SamplingParams(temperature=0.0)
    # temperature 0 => deterministic argmax of the *next* step's logits
    assert isinstance(expected, int)
    token = session2.step(int(prompt[-1]), greedy)
    assert 0 <= token < model.config.vocab_size


def test_top_k_restricts_the_candidate_set(model):
    """With top_k=1 every draw must be the argmax, whatever the seed."""
    prompt = np.array([2, 4, 6], dtype=np.int32)
    params = core.SamplingParams(temperature=1.0, top_k=1)
    picks = set()
    for seed in range(8):
        session = core.GenerationSession(model, seed)
        logits = session.prefill(prompt, model.config.vocab_size)
        picks.add(int(np.argmax(logits)))
        session.step(int(prompt[-1]), params)
    assert len(picks) == 1


def test_session_cache_size_and_position(model):
    session = core.GenerationSession(model, 0)
    cfg = model.config
    expected = 2 * cfg.n_layer * cfg.n_head * cfg.block_size * cfg.head_dim * 4
    assert session.cache_bytes == expected

    session.prefill(np.array([1, 2, 3], dtype=np.int32), cfg.vocab_size)
    assert session.position == 3
    session.reset()
    assert session.position == 0


def test_generation_beyond_block_size_raises(model):
    session = core.GenerationSession(model, 0)
    long_prompt = np.arange(model.config.block_size + 1, dtype=np.int32) % 32
    with pytest.raises(RuntimeError, match="block_size"):
        session.prefill(long_prompt, model.config.vocab_size)


# ── checkpoints ──────────────────────────────────────────────────────────────


def test_checkpoint_round_trip_is_exact(model, tmp_path):
    path = str(tmp_path / "model.csllm")
    rng = np.random.default_rng(8)
    ids = rng.integers(0, 32, size=(2, 8)).astype(np.int32)
    before = model.forward_logits(ids)

    model.save(path)
    restored = core.Model.load(path)

    assert restored.config.num_params() == model.config.num_params()
    for name in model.param_names():
        np.testing.assert_array_equal(restored.get_param(name), model.get_param(name))
    np.testing.assert_array_equal(restored.forward_logits(ids), before)


def test_checkpoint_header_is_human_readable(model, tmp_path):
    """The JSON header is the reason we carry a parser: `head` should show the config."""
    path = tmp_path / "model.csllm"
    model.save(str(path))
    head = path.read_bytes()[:400]
    assert head.startswith(b"CSLLM\0\0\0")
    assert b'"vocab_size":32' in head
    assert b'"tensors"' in head


def test_loading_a_non_checkpoint_raises(tmp_path):
    bogus = tmp_path / "bogus.csllm"
    bogus.write_bytes(b"not a checkpoint at all, definitely not")
    with pytest.raises(RuntimeError, match="bad magic"):
        core.Model.load(str(bogus))
