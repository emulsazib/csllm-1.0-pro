"""Gateway: validation, SSE framing, concurrency, disconnect, and UTF-8 buffering.

Uses httpx's ASGI transport, so the whole stack runs in-process — no server, no
ports, no flakiness. A tiny untrained model is injected into ``app.state`` so the
tests never touch ``data/``.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer
from gateway.engine import Engine, flush_utf8
from gateway.main import app
from gateway.schemas import GenerateRequest
from gateway.settings import Settings

CORPUS = "To be, or not to be, that is the question. Whether tis nobler 🌍\n" * 30


@pytest.fixture(scope="module")
def tokenizer():
    tok = BPETokenizer()
    tok.train(CORPUS, 320)
    return tok


@pytest.fixture(scope="module")
def engine(tokenizer):
    cfg = core.ModelConfig()
    cfg.vocab_size = tokenizer.vocab_size
    cfg.n_layer = 2
    cfg.n_head = 2
    cfg.n_embd = 64
    cfg.block_size = 32
    cfg.ffn_hidden = 128
    model = core.Model(cfg, 3)
    return Engine(model, tokenizer, Settings(max_concurrent_sessions=2, acquire_timeout_s=5.0))


@pytest.fixture
async def client(engine):
    app.state.engine = engine
    app.state.settings = engine.settings
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def parse_sse(body: str) -> list[str]:
    """Extract the `data:` payloads from an SSE body, in order."""
    return [
        line[len("data:") :].strip()
        for line in body.splitlines()
        if line.startswith("data:")
    ]


# ── health ───────────────────────────────────────────────────────────────────


async def test_health_reports_model_metadata(client, engine):
    response = await client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["num_params"] == engine.model.num_params()
    assert body["vocab_size"] == engine.model.config.vocab_size
    assert body["blas_backend"] == "Accelerate"
    assert body["sessions_in_flight"] == 0
    assert body["kv_cache_bytes_per_session"] > 0


# ── validation ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "payload",
    [
        {"prompt": ""},                              # min_length
        {"prompt": "hi", "max_tokens": 0},           # ge=1
        {"prompt": "hi", "max_tokens": 99999},       # le=1024
        {"prompt": "hi", "temperature": -0.5},       # ge=0
        {"prompt": "hi", "temperature": 5.0},        # le=2
        {"prompt": "hi", "top_k": -1},               # ge=0
        {"prompt": "hi", "top_p": 0.0},              # gt=0
        {"prompt": "hi", "top_p": 1.5},              # le=1
        {"prompt": "hi", "seed": -1},                # ge=0
        {"prompt": "hi", "temperatur": 0.5},         # typo -> extra=forbid
        {},                                          # missing prompt
    ],
)
async def test_invalid_requests_are_rejected(client, payload):
    response = await client.post("/generate", json=payload)
    assert response.status_code == 422, payload


async def test_bounds_are_enforced_before_reaching_the_engine(client):
    """A rejected request must not consume a session slot."""
    await client.post("/generate", json={"prompt": "hi", "temperature": 99})
    assert (await client.get("/health")).json()["sessions_in_flight"] == 0


# ── non-streaming ────────────────────────────────────────────────────────────


async def test_non_streaming_returns_a_complete_body(client):
    response = await client.post(
        "/generate",
        json={"prompt": "To be", "max_tokens": 8, "stream": False, "seed": 1},
    )
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["text"], str)
    assert body["completion_tokens"] == 8
    assert body["prompt_tokens"] > 0
    assert body["finish_reason"] == "length"


async def test_seed_makes_generation_reproducible(client):
    payload = {"prompt": "To be", "max_tokens": 10, "stream": False, "seed": 42}
    first = (await client.post("/generate", json=payload)).json()["text"]
    second = (await client.post("/generate", json=payload)).json()["text"]
    assert first == second

    payload["seed"] = 43
    other = (await client.post("/generate", json=payload)).json()["text"]
    assert other != first, "different seeds should produce different text"


async def test_greedy_temperature_zero_is_deterministic_without_a_seed(client):
    payload = {"prompt": "To be", "max_tokens": 8, "stream": False, "temperature": 0.0}
    first = (await client.post("/generate", json=payload)).json()["text"]
    second = (await client.post("/generate", json=payload)).json()["text"]
    assert first == second


async def test_context_full_is_reported(client, engine):
    """Asking for more tokens than the context allows stops with context_full."""
    block = engine.model.config.block_size
    response = await client.post(
        "/generate",
        json={"prompt": "To be or not to be that is", "max_tokens": block * 4, "stream": False},
    )
    body = response.json()
    assert body["finish_reason"] == "context_full"
    assert body["completion_tokens"] < block * 4


# ── streaming ────────────────────────────────────────────────────────────────


async def test_sse_stream_is_ordered_and_terminated(client):
    response = await client.post(
        "/generate", json={"prompt": "To be", "max_tokens": 12, "seed": 7}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

    payloads = parse_sse(response.text)
    assert payloads[-1] == "[DONE]"

    chunks = [json.loads(p) for p in payloads[:-1]]
    assert [c["index"] for c in chunks] == sorted(c["index"] for c in chunks)
    assert chunks[-1]["finish_reason"] == "length"
    assert all(c["finish_reason"] is None for c in chunks[:-1])


async def test_streaming_and_non_streaming_agree_for_a_fixed_seed(client):
    payload = {"prompt": "To be", "max_tokens": 10, "seed": 99}
    whole = (await client.post("/generate", json={**payload, "stream": False})).json()["text"]

    streamed = "".join(
        json.loads(p)["text"] for p in parse_sse(
            (await client.post("/generate", json={**payload, "stream": True})).text
        )[:-1]
    )
    assert streamed == whole


# ── concurrency ──────────────────────────────────────────────────────────────


async def test_concurrent_streams_all_succeed(client):
    """Four simultaneous requests against a 2-slot engine must all complete.

    They queue on the semaphore rather than failing, and because the bindings
    release the GIL the event loop keeps serving throughout.
    """
    payloads = [
        {"prompt": "To be", "max_tokens": 6, "stream": False, "seed": i} for i in range(4)
    ]
    responses = await asyncio.gather(
        *(client.post("/generate", json=p) for p in payloads)
    )
    assert all(r.status_code == 200 for r in responses)
    assert all(r.json()["completion_tokens"] == 6 for r in responses)
    # Slots are returned afterwards.
    assert (await client.get("/health")).json()["sessions_in_flight"] == 0


async def test_capacity_exhaustion_raises_at_capacity(engine, tokenizer):
    """When every slot is busy past the timeout, the engine refuses rather than queueing forever."""
    tight = Engine(
        engine.model, tokenizer, Settings(max_concurrent_sessions=1, acquire_timeout_s=0.05)
    )
    req = GenerateRequest(prompt="To be", max_tokens=4)

    from gateway.engine import AtCapacity

    async def drain():
        return [c async for c in tight.stream(req)]

    # Hold the only slot while a second request tries to acquire it.
    async with tight._slot():
        with pytest.raises(AtCapacity):
            await drain()


# ── disconnect ───────────────────────────────────────────────────────────────


async def test_generation_aborts_when_the_client_disconnects(engine):
    """An abandoned request must stop early, not run to max_tokens.

    Without this an abandoned browser tab would keep a CPU core busy for the
    full budget.
    """
    req = GenerateRequest(prompt="To be", max_tokens=25, seed=5)
    calls = {"n": 0}

    async def disconnected_after_three() -> bool:
        calls["n"] += 1
        return calls["n"] >= 3

    chunks = [c async for c in engine.stream(req, disconnected_after_three)]
    assert chunks[-1].finish_reason == "disconnected"
    assert chunks[-1].index < 25, "should have stopped well before the budget"


async def test_disconnect_releases_the_session_slot(engine):
    req = GenerateRequest(prompt="To be", max_tokens=20)

    async def immediately_disconnected() -> bool:
        return True

    _ = [c async for c in engine.stream(req, immediately_disconnected)]
    assert engine.sessions_in_flight == 0


# ── UTF-8 streaming ──────────────────────────────────────────────────────────


def test_flush_utf8_holds_back_incomplete_sequences():
    globe = "🌍".encode()  # 4 bytes
    text, pending = flush_utf8(globe[:2])
    assert text == "" and pending == globe[:2]

    text, pending = flush_utf8(globe[:3])
    assert text == "" and pending == globe[:3]

    text, pending = flush_utf8(globe)
    assert text == "🌍" and pending == b""


def test_flush_utf8_emits_the_complete_prefix():
    data = "ab🌍".encode()
    text, pending = flush_utf8(data[:-1])
    assert text == "ab"
    assert pending == "🌍".encode()[:-1]


def test_flush_utf8_does_not_buffer_invalid_bytes_forever():
    """Genuinely invalid bytes must be replaced, not held indefinitely."""
    text, pending = flush_utf8(b"\xff\xfe\xfd\xfc\xfb")
    assert pending == b""
    assert "�" in text


async def test_multibyte_characters_survive_streaming(engine, tokenizer):
    """Reassembled stream text must equal the non-streamed text exactly.

    A single BPE token can end mid-code-point, so this is the property that
    proves the gateway never emits mojibake.
    """
    req = GenerateRequest(prompt="🌍", max_tokens=20, seed=3)
    streamed = "".join([c.text async for c in engine.stream(req)])
    whole = (await engine.complete(GenerateRequest(**{**req.model_dump(), "seed": 3}))).text
    assert streamed == whole
    assert "�" not in streamed or "�" in whole
