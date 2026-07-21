"""Configuration versioning, the /configure_model endpoint, and dataset routes.

The load-bearing property is that **C++ owns the invariants**. Python bounds are
sanity limits; the rules that actually matter (n_embd divisible by n_head, EVEN
head_dim for RoPE) live in ModelConfig::validate(), so a config the engine would
refuse comes back as a 422 rather than crashing at build time.
"""

from __future__ import annotations

import json

import httpx
import pytest

from csllm import _csllm_core as core
from csllm.tokenizer import BPETokenizer
from gateway.engine import Engine
from gateway.main import app
from gateway.settings import Settings
from gateway.versioning import ConfigStore

BASE = {
    "vocab_size": 512,
    "n_layer": 2,
    "n_head": 2,
    "n_embd": 64,
    "block_size": 32,
    "ffn_hidden": 128,
}


@pytest.fixture
def store(tmp_path):
    return ConfigStore(tmp_path / "versions")


@pytest.fixture
def client(tmp_path):
    tok = BPETokenizer()
    tok.train("hello world this is a small corpus for testing " * 40, 320)
    cfg = core.ModelConfig()
    cfg.vocab_size = tok.vocab_size
    cfg.n_layer, cfg.n_head, cfg.n_embd = 2, 2, 32
    cfg.block_size, cfg.ffn_hidden = 16, 64
    model = core.Model(cfg, 3)

    app.state.engine = Engine(model, tok, Settings())
    app.state.settings = Settings()
    app.state.config_store = ConfigStore(tmp_path / "versions")
    app.state.supervisor = None

    transport = httpx.ASGITransport(app=app)
    return httpx.AsyncClient(transport=transport, base_url="http://test")


# ── store ────────────────────────────────────────────────────────────────────


def test_create_writes_a_version_file(store):
    version, created = store.create(BASE, note="first")
    assert created is True
    assert version.index == 1
    assert version.version_id.startswith("v1-")
    assert version.path.exists()
    assert version.note == "first"
    assert version.num_params > 0
    assert version.activation_bytes > 0


def test_identical_config_is_idempotent(store):
    first, created_a = store.create(BASE)
    second, created_b = store.create(BASE)
    assert created_a is True and created_b is False
    assert first.version_id == second.version_id
    assert len(store.list()) == 1


def test_different_config_makes_a_new_version(store):
    store.create(BASE)
    other, created = store.create({**BASE, "n_layer": 4})
    assert created is True
    assert other.index == 2
    assert len(store.list()) == 2


def test_num_params_matches_the_engine(store):
    version, _ = store.create(BASE)
    cfg = core.ModelConfig()
    for key, value in BASE.items():
        setattr(cfg, key, value)
    assert version.num_params == cfg.num_params()


def test_get_and_list_round_trip(store):
    created, _ = store.create(BASE, note="hello")
    fetched = store.get(created.version_id)
    assert fetched.config == created.config
    assert fetched.note == "hello"
    assert [v.version_id for v in store.list()] == [created.version_id]


def test_missing_version_raises(store):
    with pytest.raises(FileNotFoundError, match="no such config version"):
        store.get("v99-deadbeef")


def test_corrupt_version_file_does_not_break_listing(store):
    store.create(BASE)
    (store.directory / "v9-bad.json").write_text("{not json")
    assert len(store.list()) == 1  # the good one still lists


def test_build_model_writes_a_loadable_checkpoint(store, tmp_path):
    version, _ = store.create(BASE)
    path = store.build_model(version, tmp_path / "fresh.csllm")
    model = core.Model.load(str(path))
    assert model.num_params() == version.num_params


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"n_embd": 64, "n_head": 5}, "divisible"),      # 64 % 5 != 0
        ({"n_embd": 6, "n_head": 2}, "even for RoPE"),   # head_dim = 3, odd
    ],
)
def test_cpp_invariants_are_enforced(store, override, message):
    with pytest.raises(RuntimeError, match=message):
        store.create({**BASE, **override})


# ── endpoint ─────────────────────────────────────────────────────────────────


async def test_configure_model_creates_a_version(client):
    async with client as c:
        response = await c.post("/configure_model", json={**BASE, "note": "via api"})
    assert response.status_code == 200
    body = response.json()
    assert body["created"] is True
    assert body["version_id"].startswith("v1-")
    assert body["num_params"] > 0
    assert body["activation_bytes"] > 0
    assert body["note"] == "via api"
    assert body["checkpoint"] is None


async def test_configure_model_can_initialize_a_checkpoint(client, tmp_path):
    out = tmp_path / "built.csllm"
    async with client as c:
        response = await c.post(
            "/configure_model", json={**BASE, "initialize": True, "out": str(out)}
        )
    body = response.json()
    assert body["checkpoint"] == str(out)
    assert out.exists()
    assert core.Model.load(str(out)).num_params() == body["num_params"]


async def test_resubmitting_returns_the_same_version(client):
    async with client as c:
        first = (await c.post("/configure_model", json=BASE)).json()
        second = (await c.post("/configure_model", json=BASE)).json()
    assert first["version_id"] == second["version_id"]
    assert second["created"] is False


async def test_odd_head_dim_is_a_422_not_a_crash(client):
    """RoPE rotates channel pairs, so head_dim must be even — enforced in C++."""
    async with client as c:
        response = await c.post("/configure_model", json={**BASE, "n_embd": 10, "n_head": 2})
    assert response.status_code == 422
    assert "even for RoPE" in response.json()["detail"]


async def test_indivisible_embedding_is_a_422(client):
    async with client as c:
        response = await c.post("/configure_model", json={**BASE, "n_embd": 64, "n_head": 5})
    assert response.status_code == 422
    assert "divisible" in response.json()["detail"]


@pytest.mark.parametrize(
    "payload",
    [
        {**BASE, "n_layer": 0},          # ge=1
        {**BASE, "vocab_size": 10},      # ge=256
        {**BASE, "rope_theta": 0},       # gt=0
        {**BASE, "typo": 1},             # extra=forbid
    ],
)
async def test_schema_bounds_are_enforced(client, payload):
    async with client as c:
        assert (await c.post("/configure_model", json=payload)).status_code == 422


async def test_configs_can_be_listed_and_fetched(client):
    async with client as c:
        created = (await c.post("/configure_model", json=BASE)).json()
        listing = (await c.get("/configs")).json()
        fetched = (await c.get(f"/configs/{created['version_id']}")).json()
    assert [v["version_id"] for v in listing] == [created["version_id"]]
    assert fetched["config"] == created["config"]


async def test_unknown_config_version_is_404(client):
    async with client as c:
        assert (await c.get("/configs/v42-nope")).status_code == 404


# ── datasets routes ──────────────────────────────────────────────────────────


async def test_datasets_endpoint_lists_supported_extensions(client):
    async with client as c:
        body = (await c.get("/datasets")).json()
    assert ".jsonl" in body["supported_extensions"]
    assert isinstance(body["datasets"], list)


async def test_dataset_path_traversal_is_rejected(client):
    """`name` comes straight from the URL, so traversal must not escape raw/."""
    async with client as c:
        response = await c.get("/datasets/..%2F..%2Fpyproject.toml")
    assert response.status_code == 404


# ── training routes without a supervisor ─────────────────────────────────────


async def test_training_routes_report_unavailable_without_a_supervisor(client):
    async with client as c:
        assert (await c.get("/train/status")).status_code == 503


# ── export endpoint ──────────────────────────────────────────────────────────


async def test_export_endpoint_writes_a_bundle(client, tmp_path):
    tok = BPETokenizer()
    tok.train("hello world this is a small corpus for testing " * 40, 320)
    cfg = core.ModelConfig()
    cfg.vocab_size = tok.vocab_size
    cfg.n_layer, cfg.n_head, cfg.n_embd = 2, 2, 32
    cfg.block_size, cfg.ffn_hidden = 16, 64
    core.Model(cfg, 3).save(str(tmp_path / "m.csllm"))
    tok.save(tmp_path / "tok")

    async with client as c:
        response = await c.post(
            "/export",
            json={
                "checkpoint": str(tmp_path / "m.csllm"),
                "tokenizer_dir": str(tmp_path / "tok"),
                "out": str(tmp_path / "bundle"),
            },
        )
    body = response.json()
    assert response.status_code == 200
    assert set(body["files"]) == {"config.json", "model.safetensors", "tokenizer.json"}
    assert body["num_params"] == cfg.num_params()
    assert json.loads((tmp_path / "bundle" / "config.json").read_text())["format"] == "csllm"
