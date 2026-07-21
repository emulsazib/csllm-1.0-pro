"""Integration: tokenizer -> binarized data -> training loop -> checkpoint -> generation.

The unit tests cover each stage in isolation; this covers the seams between
them — vocab agreement, uint16 round-tripping through disk, target shifting, and
the fact that the whole thing actually reduces loss on real text.
"""

from __future__ import annotations

import numpy as np
import pytest

from csllm import _csllm_core as core
from csllm.data import binarize, get_batch, load_split
from csllm.tokenizer import BPETokenizer

TEXT = (
    "First Citizen:\nBefore we proceed any further, hear me speak.\n"
    "All:\nSpeak, speak.\n"
    "First Citizen:\nYou are all resolved rather to die than to famish?\n"
    "All:\nResolved. resolved.\n"
    "First Citizen:\nFirst, you know Caius Marcius is chief enemy to the people.\n"
) * 20


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory):
    """Train a small tokenizer and binarize, once for the whole module."""
    out = tmp_path_factory.mktemp("pipeline")
    tok = BPETokenizer()
    tok.train(TEXT, 320)
    tok.save(out / "tokenizer")
    train_path, val_path = binarize(TEXT, tok, out, val_fraction=0.1)
    return tok, load_split(train_path), load_split(val_path)


def make_model(vocab_size, seed=17):
    cfg = core.ModelConfig()
    cfg.vocab_size = vocab_size
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 64
    cfg.block_size = 32
    cfg.ffn_hidden = 128
    return core.Model(cfg, seed)


def test_binarized_data_round_trips_through_disk(pipeline):
    """uint16 on disk must decode back to the original text."""
    tok, train_data, val_data = pipeline
    combined = np.concatenate([np.asarray(train_data), np.asarray(val_data)])
    assert combined.dtype == np.uint16
    assert tok.decode(combined.tolist()) == TEXT


def test_split_sizes_respect_val_fraction(pipeline):
    _, train_data, val_data = pipeline
    total = len(train_data) + len(val_data)
    assert len(val_data) / total == pytest.approx(0.1, abs=0.01)


def test_token_ids_stay_inside_the_vocab(pipeline):
    tok, train_data, val_data = pipeline
    for split in (train_data, val_data):
        assert np.asarray(split).max() < tok.vocab_size


def test_training_reduces_loss_on_real_text(pipeline):
    """The end-to-end proof that the pieces fit: real text, real tokenizer, loss falls."""
    tok, train_data, _ = pipeline
    model = make_model(tok.vocab_size)
    cfg = core.AdamWConfig()
    cfg.lr = 3e-3
    opt = core.AdamW(model, cfg)
    rng = np.random.default_rng(0)

    x, y = get_batch(train_data, 8, 32, rng)
    initial = model.forward_loss(x, y)
    # An untrained model is ~uniform over the vocabulary.
    assert initial == pytest.approx(np.log(tok.vocab_size), rel=0.1)

    for _ in range(150):
        x, y = get_batch(train_data, 8, 32, rng)
        model.zero_grad()
        loss = model.forward_loss(x, y)
        model.backward()
        opt.clip_grad_norm(1.0)
        opt.step(cfg.lr)
        assert np.isfinite(loss)

    x, y = get_batch(train_data, 8, 32, np.random.default_rng(0))
    final = model.forward_loss(x, y)
    assert final < initial * 0.7, f"loss barely moved: {initial:.3f} -> {final:.3f}"


def test_checkpoint_survives_a_training_run(pipeline, tmp_path):
    """Weights saved mid-training must reload to identical logits."""
    tok, train_data, _ = pipeline
    model = make_model(tok.vocab_size)
    opt = core.AdamW(model, core.AdamWConfig())
    rng = np.random.default_rng(1)

    for _ in range(10):
        x, y = get_batch(train_data, 4, 32, rng)
        model.zero_grad()
        model.forward_loss(x, y)
        model.backward()
        opt.step(1e-3)

    path = str(tmp_path / "ckpt.csllm")
    model.save(path)
    probe, _ = get_batch(train_data, 2, 32, np.random.default_rng(2))
    np.testing.assert_array_equal(
        core.Model.load(path).forward_logits(probe), model.forward_logits(probe)
    )


def test_generation_produces_decodable_text(pipeline):
    """Sampled ids must always decode — the tokenizer covers the whole vocab."""
    tok, train_data, _ = pipeline
    model = make_model(tok.vocab_size)

    prompt_ids = tok.encode("First Citizen:")[:8]
    session = core.GenerationSession(model, 0)
    session.prefill(np.asarray(prompt_ids, dtype=np.int32), model.config.vocab_size)

    params = core.SamplingParams(temperature=0.9, top_k=20, top_p=0.95)
    token = int(prompt_ids[-1])
    generated = []
    for _ in range(16):
        token = session.step(token, params)
        generated.append(token)

    assert all(0 <= t < tok.vocab_size for t in generated)
    assert isinstance(tok.decode(generated), str)


def test_cosine_schedule_drives_the_loop_shape():
    """Warmup rises to lr_max, then decay lands on lr_min at the final step."""
    warmup, total, hi, lo = 100, 3000, 1e-3, 1e-4
    values = [core.cosine_lr(s, warmup, total, hi, lo) for s in range(total + 1)]
    assert values[0] == pytest.approx(hi / warmup)
    assert max(values) == pytest.approx(hi)
    assert values[warmup - 1] == pytest.approx(hi)
    assert values[total] == pytest.approx(lo)
    # Monotonic decay after warmup.
    tail = values[warmup:]
    assert all(a >= b - 1e-12 for a, b in zip(tail, tail[1:], strict=False))
