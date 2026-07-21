"""WS /ws/inspect: the two-frame protocol carrying real attention.

The invariant that matters is that the binary frame a client reads back is
byte-for-byte the attention the model used — right shape, right order, rows
still summing to 1 after slicing.
"""

from __future__ import annotations

import numpy as np
import pytest
from fastapi.testclient import TestClient

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer
from gateway.engine import Engine
from gateway.main import app
from gateway.settings import Settings

CORPUS = "To be, or not to be, that is the question\nWhether tis nobler in the mind\n" * 30


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, 400)
    return tok


@pytest.fixture(scope="module")
def model(tokenizer):
    cfg = core.ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.n_layer, cfg.n_head, cfg.n_embd = 3, 2, 32
    cfg.block_size, cfg.ffn_hidden = 32, 64
    return core.Model(cfg, 5)


@pytest.fixture
def client(model, tokenizer):
    app.state.engine = Engine(model, tokenizer, Settings())
    app.state.settings = Settings()
    app.state.supervisor = None
    # TestClient (not httpx directly): it is the one that speaks WebSocket.
    return TestClient(app)


def collect(ws, expect_attention=True):
    """Read frames until `done`, pairing each token frame with its binary block."""
    start = ws.receive_json()
    tokens: list[dict] = []
    while True:
        message = ws.receive_json()
        if message["type"] in ("done", "error"):
            return start, tokens, message
        if expect_attention and "attn" in message:
            raw = ws.receive_bytes()
            shape = tuple(message["attn"]["shape"])
            message["attention"] = np.frombuffer(raw, dtype=np.float32).reshape(shape)
        tokens.append(message)


# ── protocol ─────────────────────────────────────────────────────────────────


def test_start_frame_describes_the_stream(client, model):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 3})
        start, tokens, done = collect(ws)

    assert start["type"] == "start"
    assert start["config"]["n_layer"] == model.config.n_layer
    assert start["layers"] == [0, 1, 2]
    assert start["heads"] == [0, 1]
    assert [t["id"] for t in start["prompt"]]
    assert done["type"] == "done"
    assert done["tokens"] == 3
    assert len(tokens) == 3


def test_token_frames_are_ordered_and_advance_position(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 5, "seed": 1})
        _, tokens, _ = collect(ws)

    assert [t["index"] for t in tokens] == [0, 1, 2, 3, 4]
    positions = [t["position"] for t in tokens]
    assert positions == sorted(positions)
    assert all(b - a == 1 for a, b in zip(positions, positions[1:], strict=False))


def test_binary_frame_matches_the_declared_shape(client, model):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 3, "seed": 2})
        _, tokens, _ = collect(ws)

    for frame in tokens:
        layers, heads, keys = frame["attn"]["shape"]
        assert layers == model.config.n_layer
        assert heads == model.config.n_head
        assert frame["attention"].shape == (layers, heads, keys)
        assert frame["attn"]["bytes"] == layers * heads * keys * 4


def test_attention_rows_are_still_distributions_after_slicing(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be or not", "max_tokens": 4, "seed": 3})
        _, tokens, _ = collect(ws)

    for frame in tokens:
        rows = frame["attention"].sum(axis=-1)
        np.testing.assert_allclose(rows, 1.0, atol=1e-5)
        assert (frame["attention"] >= 0).all()


def test_attention_width_grows_by_one_per_token(client):
    """Each generated token can attend to one more key than the last."""
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 4, "seed": 4})
        _, tokens, _ = collect(ws)

    widths = [t["attn"]["shape"][2] for t in tokens]
    assert all(b - a == 1 for a, b in zip(widths, widths[1:], strict=False))


def test_layer_and_head_selection_shrinks_the_payload(client):
    """The main lever on bandwidth: narrow before serialising."""
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 2, "layers": [0, 2], "heads": [1]})
        start, tokens, _ = collect(ws)

    assert start["layers"] == [0, 2]
    assert start["heads"] == [1]
    for frame in tokens:
        assert frame["attn"]["shape"][:2] == [2, 1]
        np.testing.assert_allclose(frame["attention"].sum(axis=-1), 1.0, atol=1e-5)


def test_selected_layers_match_the_full_stream(client):
    """Slicing must select, not reorder or resample."""
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 2, "seed": 9})
        _, full, _ = collect(ws)
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 2, "seed": 9, "layers": [2], "heads": [0]})
        _, sliced, _ = collect(ws)

    for whole, part in zip(full, sliced, strict=False):
        np.testing.assert_allclose(part["attention"][0, 0], whole["attention"][2, 0], atol=1e-6)


def test_out_of_range_selection_falls_back_to_all(client, model):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 1, "layers": [99], "heads": [42]})
        start, _, _ = collect(ws)
    assert start["layers"] == list(range(model.config.n_layer))
    assert start["heads"] == list(range(model.config.n_head))


# ── candidates ───────────────────────────────────────────────────────────────


def test_each_frame_carries_the_candidates_it_was_drawn_from(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 3, "top_n": 5, "seed": 6})
        _, tokens, _ = collect(ws)

    for frame in tokens:
        assert len(frame["top"]) == 5
        assert sum(c["prob"] for c in frame["top"]) <= 1.0 + 1e-6
        # The emitted token must be one the sampler could actually draw.
        kept = {c["id"] for c in frame["top"] if c["kept"]}
        assert frame["id"] in kept or frame["kept_count"] > len(frame["top"])
        assert frame["raw_entropy"] >= 0
        assert frame["filtered_entropy"] >= 0


def test_greedy_stream_picks_the_top_candidate(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 3, "temperature": 0.0, "top_n": 5})
        _, tokens, _ = collect(ws)

    for frame in tokens:
        assert frame["id"] == frame["top"][0]["id"]
        assert frame["kept_count"] == 1


def test_candidates_can_be_disabled(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 2, "top_n": 0})
        _, tokens, _ = collect(ws)
    assert all("top" not in t for t in tokens)


def test_attention_can_be_disabled(client):
    """Without attention there is no binary frame at all."""
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 3, "attention": False})
        start, tokens, done = collect(ws, expect_attention=False)

    assert start["attention"] is False
    assert all("attn" not in t for t in tokens)
    assert done["tokens"] == 3


# ── determinism & validation ─────────────────────────────────────────────────


def test_same_seed_reproduces_the_stream(client):
    def run(seed):
        with client.websocket_connect("/ws/inspect") as ws:
            ws.send_json({"prompt": "To be", "max_tokens": 4, "seed": seed})
            _, tokens, _ = collect(ws)
        return [t["id"] for t in tokens]

    assert run(11) == run(11)
    assert run(11) != run(12)


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},
        {"prompt": "hi", "max_tokens": 0},
        {"prompt": "hi", "temperature": 9},
        {"prompt": "hi", "top_p": 0},
        {"prompt": "hi", "typo": 1},
    ],
)
def test_invalid_subscribe_is_reported_then_closed(client, payload):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json(payload)
        message = ws.receive_json()
    assert message["type"] == "error"


def test_budget_is_capped_by_the_context_window(client, model):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be or not to be", "max_tokens": 250})
        start, _, done = collect(ws)
    assert start["max_tokens"] < 250
    assert done["tokens"] <= model.config.block_size


def test_session_slot_is_released_after_the_stream(client):
    with client.websocket_connect("/ws/inspect") as ws:
        ws.send_json({"prompt": "To be", "max_tokens": 2})
        collect(ws)
    assert client.app.state.engine.sessions_in_flight == 0
