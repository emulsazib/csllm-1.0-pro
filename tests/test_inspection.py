"""Inspection endpoints backing the diagnostics UI.

These exist to make the model's internals *observable*, so the tests focus on
whether what they report is actually true — byte spans that line up, embeddings
that come from the live weights, and a filtered distribution that matches what
the sampler would really draw from.
"""

from __future__ import annotations

import httpx
import numpy as np
import pytest

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer
from gateway.engine import Engine
from gateway.main import app
from gateway.settings import Settings

CORPUS = "To be, or not to be, that is the question 🌍\nWhether tis nobler in the mind\n" * 30


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, 400)
    return tok


@pytest.fixture(scope="module")
def model(tokenizer):
    cfg = core.ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.n_layer, cfg.n_head, cfg.n_embd = 2, 2, 32
    cfg.block_size, cfg.ffn_hidden = 32, 64
    return core.Model(cfg, 7)


@pytest.fixture
def client(model, tokenizer):
    app.state.engine = Engine(model, tokenizer, Settings())
    app.state.settings = Settings()
    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ── /tokenize ────────────────────────────────────────────────────────────────


async def test_tokenize_returns_tokens_with_spans(client, tokenizer):
    async with client as c:
        body = (await c.post("/tokenize", json={"text": "To be or not"})).json()

    assert body["count"] == len(tokenizer.encode("To be or not"))
    assert body["num_chars"] == 12
    assert [t["index"] for t in body["tokens"]] == list(range(body["count"]))


async def test_byte_spans_are_contiguous_and_reconstruct_the_input(client):
    """Spans must tile the input exactly — a gap or overlap would mislead the UI."""
    text = "Whether tis nobler"
    async with client as c:
        body = (await c.post("/tokenize", json={"text": text})).json()

    tokens = body["tokens"]
    assert tokens[0]["start"] == 0
    assert tokens[-1]["end"] == body["num_bytes"]
    for prev, nxt in zip(tokens, tokens[1:], strict=False):
        assert prev["end"] == nxt["start"], "spans must be contiguous"

    rebuilt = b"".join(bytes(t["bytes"]) for t in tokens)
    assert rebuilt.decode("utf-8") == text


async def test_partial_utf8_tokens_are_flagged(client):
    """A byte-level token can end mid-code-point; the UI must know not to split there."""
    async with client as c:
        body = (await c.post("/tokenize", json={"text": "🌍🦀"})).json()

    assert any(t["partial_utf8"] for t in body["tokens"]), "expected a fragment token"
    rebuilt = b"".join(bytes(t["bytes"]) for t in body["tokens"])
    assert rebuilt.decode("utf-8") == "🌍🦀"


async def test_compression_ratio_is_reported(client):
    async with client as c:
        body = (await c.post("/tokenize", json={"text": CORPUS[:400]})).json()
    assert body["compression"] > 1.0  # merges must shorten the sequence
    assert body["compression"] == pytest.approx(body["num_bytes"] / body["count"])


async def test_empty_text_is_rejected(client):
    async with client as c:
        assert (await c.post("/tokenize", json={"text": "x" * 9000})).status_code == 422
        assert (await c.post("/tokenize", json={"nope": 1})).status_code == 422


# ── /embeddings ──────────────────────────────────────────────────────────────


async def test_embeddings_come_from_the_live_weights(client, model, tokenizer):
    """Values must match get_param exactly — not a stale or rescaled copy."""
    text = "To be"
    ids = tokenizer.encode(text)
    async with client as c:
        body = (await c.post("/embeddings", json={"text": text})).json()

    assert body["ids"] == ids
    assert body["n_embd"] == model.config.n_embd
    np.testing.assert_allclose(
        np.array(body["vectors"], dtype=np.float32), model.get_param("tok_emb")[ids], rtol=0, atol=0
    )


async def test_embeddings_accept_explicit_ids(client):
    async with client as c:
        body = (await c.post("/embeddings", json={"ids": [1, 2, 3]})).json()
    assert body["ids"] == [1, 2, 3]
    assert len(body["vectors"]) == 3


async def test_embeddings_require_exactly_one_input(client):
    async with client as c:
        both = await c.post("/embeddings", json={"text": "hi", "ids": [1]})
        neither = await c.post("/embeddings", json={})
    assert both.status_code == 422
    assert neither.status_code == 422


async def test_out_of_range_ids_are_rejected(client, model):
    async with client as c:
        response = await c.post("/embeddings", json={"ids": [0, model.config.vocab_size + 5]})
    assert response.status_code == 422
    assert "out of range" in response.json()["detail"]


async def test_pca_projection_shape_and_variance(client):
    async with client as c:
        body = (await c.post("/embeddings", json={"text": "To be or not to be"})).json()

    assert len(body["projection"]) == len(body["ids"])
    assert all(len(p) == 3 for p in body["projection"])
    variance = body["explained_variance"]
    assert len(variance) == 3
    assert variance == sorted(variance, reverse=True), "components must be ordered"
    assert 0 <= sum(variance) <= 1.0 + 1e-6


async def test_projection_is_skipped_for_too_few_tokens(client):
    async with client as c:
        body = (await c.post("/embeddings", json={"ids": [5, 6]})).json()
    assert body["projection"] == []


async def test_projection_can_be_disabled(client):
    async with client as c:
        body = (await c.post("/embeddings", json={"text": "To be or not", "project": False})).json()
    assert body["projection"] == []


# ── /inspect/next_token ──────────────────────────────────────────────────────


async def test_inspect_returns_ranked_candidates(client):
    async with client as c:
        body = (await c.post("/inspect/next_token", json={"prompt": "To be", "top_n": 10})).json()

    assert body["prompt_tokens"] > 0
    assert len(body["candidates"]) == 10
    ranking = [max(t["raw_prob"], t["prob"]) for t in body["candidates"]]
    assert ranking == sorted(ranking, reverse=True)


async def test_raw_probabilities_are_a_distribution(client):
    """raw_prob is the model's own belief, independent of the sampling knobs."""
    async with client as c:
        a = (await c.post("/inspect/next_token",
                          json={"prompt": "To be", "temperature": 0.2})).json()
        b = (await c.post("/inspect/next_token",
                          json={"prompt": "To be", "temperature": 1.9})).json()

    raw_a = {t["id"]: t["raw_prob"] for t in a["candidates"]}
    raw_b = {t["id"]: t["raw_prob"] for t in b["candidates"]}
    shared = set(raw_a) & set(raw_b)
    for token_id in shared:
        assert raw_a[token_id] == pytest.approx(raw_b[token_id], rel=1e-6)


async def test_top_k_is_reflected_in_kept_count(client):
    async with client as c:
        body = (await c.post("/inspect/next_token",
                             json={"prompt": "To be", "top_k": 5, "top_p": 1.0})).json()
    assert body["kept_count"] == 5
    assert sum(1 for t in body["candidates"] if t["kept"]) == 5


async def test_filtered_probabilities_match_the_c_plus_plus_sampler(client, model, tokenizer):
    """The chart must show what the model would ACTUALLY draw from."""
    params = {"prompt": "To be", "temperature": 0.7, "top_k": 8, "top_p": 0.9, "top_n": 50}
    async with client as c:
        body = (await c.post("/inspect/next_token", json=params)).json()

    ids = tokenizer.encode("To be")
    session = core.GenerationSession(model, 0)
    logits = session.prefill(np.asarray(ids, dtype=np.int32), model.config.vocab_size)
    expected = core.filtered_distribution(
        logits, core.SamplingParams(temperature=0.7, top_k=8, top_p=0.9)
    )

    for candidate in body["candidates"]:
        assert candidate["prob"] == pytest.approx(float(expected[candidate["id"]]), abs=1e-6)


async def test_filtered_probabilities_sum_to_one(client):
    async with client as c:
        body = (await c.post("/inspect/next_token",
                             json={"prompt": "To be", "top_k": 5, "top_n": 50})).json()
    assert sum(t["prob"] for t in body["candidates"]) == pytest.approx(1.0, abs=1e-5)


async def test_greedy_collapses_onto_one_candidate(client):
    async with client as c:
        body = (await c.post("/inspect/next_token",
                             json={"prompt": "To be", "temperature": 0.0})).json()
    assert body["kept_count"] == 1
    assert body["filtered_entropy"] == pytest.approx(0.0, abs=1e-6)
    assert body["candidates"][0]["prob"] == pytest.approx(1.0)


async def test_entropy_falls_as_filters_tighten(client):
    async with client as c:
        loose = (await c.post("/inspect/next_token",
                              json={"prompt": "To be", "temperature": 1.5, "top_p": 1.0})).json()
        tight = (await c.post("/inspect/next_token",
                              json={"prompt": "To be", "temperature": 0.3, "top_k": 3})).json()

    assert tight["filtered_entropy"] < loose["filtered_entropy"]
    # The raw distribution is unaffected by the knobs.
    assert tight["raw_entropy"] == pytest.approx(loose["raw_entropy"], rel=1e-6)


async def test_filtered_out_but_likely_tokens_stay_visible(client):
    """Ranking by max(raw, filtered) keeps the tokens the user is judging on screen.

    Ranking by filtered alone would hide exactly the candidates a low top_k just
    excluded — the ones you need to see to decide whether the setting is right.
    """
    async with client as c:
        body = (await c.post("/inspect/next_token",
                             json={"prompt": "To be", "top_k": 2, "top_n": 10})).json()
    assert any(not t["kept"] for t in body["candidates"])
    assert all(t["prob"] == 0.0 for t in body["candidates"] if not t["kept"])


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hi", "temperature": 3.0},
        {"prompt": "hi", "top_p": 0.0},
        {"prompt": "hi", "top_n": 0},
        {"prompt": "hi", "typo": 1},
    ],
)
async def test_inspect_validation(client, payload):
    async with client as c:
        assert (await c.post("/inspect/next_token", json=payload)).status_code == 422
