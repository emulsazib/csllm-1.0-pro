"""Configuration versioning, the /configure_model endpoint, and dataset routes.

The load-bearing property is that **C++ owns the invariants**. Python bounds are
sanity limits; the rules that actually matter (n_embd divisible by n_head, EVEN
head_dim for RoPE) live in ModelConfig::validate(), so a config the engine would
refuse comes back as a 422 rather than crashing at build time.
"""

from __future__ import annotations

import json
from pathlib import Path

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


# ── estimate endpoint ────────────────────────────────────────────────────────


async def test_estimate_matches_the_engine(client):
    async with client as c:
        body = (await c.post("/configure_model/estimate", json=BASE)).json()
    assert body["num_params"] == core_num_params(BASE)
    params = body["params"]
    assert params["embedding"] + params["attention"] + params["ffn"] + params["norms"] == (
        body["num_params"]
    )
    assert body["memory"]["total"] > body["memory"]["params"] > 0
    assert body["device"]["memory_label"] in {"VRAM", "Unified memory", "RAM"}
    assert body["fits"] is True


async def test_estimate_writes_no_version(client):
    """The sliders call this on every movement; it must not persist anything."""
    async with client as c:
        for n_layer in range(1, 6):
            await c.post("/configure_model/estimate", json={**BASE, "n_layer": n_layer})
        assert (await c.get("/configs")).json() == []


async def test_estimate_reports_a_config_that_does_not_fit(client):
    """A 64-layer 4096-wide model at batch 512 exceeds any single host."""
    huge = {
        **BASE,
        "n_layer": 64,
        "n_embd": 4096,
        "n_head": 32,
        "ffn_hidden": 16384,
        "block_size": 4096,
        "vocab_size": 65536,
    }
    async with client as c:
        body = (await c.post("/configure_model/estimate", json={**huge, "batch_size": 512})).json()
    assert body["fits"] is False


async def test_estimate_rejects_invalid_architecture(client):
    async with client as c:
        odd = await c.post("/configure_model/estimate", json={**BASE, "n_embd": 10, "n_head": 2})
        indivisible = await c.post(
            "/configure_model/estimate", json={**BASE, "n_embd": 64, "n_head": 5}
        )
    assert odd.status_code == 422 and "even for RoPE" in odd.json()["detail"]
    assert indivisible.status_code == 422 and "divisible" in indivisible.json()["detail"]


async def test_estimate_seq_len_shortens_activations(client):
    async with client as c:
        full = (await c.post("/configure_model/estimate", json=BASE)).json()
        short = (
            await c.post(
                "/configure_model/estimate",
                json={**BASE, "seq_len": BASE["block_size"] // 2},
            )
        ).json()
    assert short["memory"]["activations"] < full["memory"]["activations"]
    assert short["num_params"] == full["num_params"]  # seq_len is not architecture


def core_num_params(config: dict) -> int:
    from csllm.config import config_from_dict

    return config_from_dict(config).num_params()


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


async def test_prepare_on_a_missing_dataset_is_404(client):
    """Path validation runs before the supervisor lookup, so this is 404 not 503."""
    async with client as c:
        assert (await c.post("/datasets/nope.txt/prepare", json={})).status_code == 404


@pytest.mark.parametrize("name", ["../pyproject.toml", "../../etc/passwd", "..", "sub/dir.txt"])
def test_resolve_refuses_to_escape_the_raw_directory(name):
    """Tested directly, not through the router.

    An HTTP client normalises `..` out of the path before the request is sent,
    so a route-level traversal test never reaches this guard and passes for the
    wrong reason.
    """
    from fastapi import HTTPException

    from gateway.routes.datasets import _resolve

    with pytest.raises(HTTPException) as excinfo:
        _resolve(name)
    assert excinfo.value.status_code == 404


def test_resolve_refuses_a_symlink_pointing_outside(tmp_path, monkeypatch):
    """A link's parent is raw/ wherever it points — resolve the file itself."""
    from fastapi import HTTPException

    import datasets as ds_module
    from gateway.routes import datasets as routes

    raw = tmp_path / "raw"
    raw.mkdir()
    secret = tmp_path / "secret.txt"
    secret.write_text("not yours")
    (raw / "innocent.txt").symlink_to(secret)

    monkeypatch.setattr(ds_module, "RAW_DIR", raw)
    monkeypatch.setattr(routes.ds, "RAW_DIR", raw)

    with pytest.raises(HTTPException) as excinfo:
        routes._resolve("innocent.txt")
    assert excinfo.value.status_code == 404


def test_resolve_accepts_a_real_dataset(tmp_path, monkeypatch):
    from gateway.routes import datasets as routes

    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "corpus.txt").write_text("hello")
    monkeypatch.setattr(routes.ds, "RAW_DIR", raw)

    assert routes._resolve("corpus.txt").name == "corpus.txt"


# ── prepare must not clobber the checked-in corpora ──────────────────────────


def test_prepared_output_is_isolated_per_dataset():
    """Regression: this destroyed data/debug/*.bin during Phase 2 verification.

    Deriving the output directory from the CONFIG put `configs/debug.json` at
    `data/debug`, which is exactly where the shipped debug corpus lives. Keying
    on the dataset instead keeps prepares out of `data/` entirely.
    """
    from gateway.routes.datasets import PREPARED_DIR, prepared_dir_for

    for dataset in ["speeches.jsonl", "shakespeare-sample.txt", "notes.csv"]:
        target = prepared_dir_for(dataset)
        assert PREPARED_DIR in target.parents
        # The two directories holding checked-in splits.
        assert target != Path("data")
        assert target != Path("data/debug")

    assert prepared_dir_for("a.txt") != prepared_dir_for("b.txt")


def test_prepare_defaults_never_point_at_the_shipped_corpus():
    """An omitted out/data_dir must resolve under data/prepared/, not data/."""
    from gateway.schemas import PrepareRequest

    req = PrepareRequest()
    assert req.out is None and req.data_dir is None


async def test_prepared_listing_reports_real_token_counts(client, tmp_path, monkeypatch):
    """vocab_size comes from the field in vocab.json, not from counting keys.

    The file is {vocab_size, pattern, special_tokens, tokens} — len() of it is 4
    for every tokenizer ever written.
    """
    from gateway.routes import datasets as routes

    prepared = tmp_path / "prepared"
    directory = prepared / "speeches"
    (directory / "tokenizer").mkdir(parents=True)
    # uint16 on disk: 10 tokens train, 4 val.
    (directory / "train.bin").write_bytes(b"\x00\x01" * 10)
    (directory / "val.bin").write_bytes(b"\x00\x01" * 4)
    (directory / "tokenizer" / "vocab.json").write_text(
        json.dumps({"vocab_size": 512, "pattern": "x", "special_tokens": 0, "tokens": {}})
    )
    monkeypatch.setattr(routes, "PREPARED_DIR", prepared)

    async with client as c:
        body = (await c.get("/prepared")).json()

    assert len(body["prepared"]) == 1
    entry = body["prepared"][0]
    assert entry["name"] == "speeches"
    assert entry["train_tokens"] == 10
    assert entry["val_tokens"] == 4
    assert entry["vocab_size"] == 512


async def test_prepared_listing_skips_an_interrupted_prepare(client, tmp_path, monkeypatch):
    """A directory without both splits and a tokenizer is not trainable."""
    from gateway.routes import datasets as routes

    prepared = tmp_path / "prepared"
    (prepared / "halfway").mkdir(parents=True)
    (prepared / "halfway" / "train.bin").write_bytes(b"\x00\x01")  # no val.bin, no tokenizer
    monkeypatch.setattr(routes, "PREPARED_DIR", prepared)

    async with client as c:
        assert (await c.get("/prepared")).json()["prepared"] == []


async def test_prepare_request_rejects_unknown_fields():
    from pydantic import ValidationError

    from gateway.schemas import PrepareRequest

    with pytest.raises(ValidationError):
        PrepareRequest(dataset="oops.txt")  # `dataset` comes from the URL path


# ── training routes without a supervisor ─────────────────────────────────────


@pytest.mark.parametrize(
    "method,path",
    [
        ("get", "/train/status"),
        ("post", "/train/start"),
        ("post", "/train/stop"),
        ("post", "/train/pause"),
        ("post", "/train/resume"),
    ],
)
async def test_training_routes_report_unavailable_without_a_supervisor(client, method, path):
    async with client as c:
        call = getattr(c, method)
        response = await (call(path, json={}) if method == "post" else call(path))
    assert response.status_code == 503


# ── export endpoint ──────────────────────────────────────────────────────────


async def test_exports_listing_and_zip_download(client, tmp_path, monkeypatch):
    """The zip is what a user actually receives, so round-trip it."""
    import io
    import zipfile

    from gateway.routes import config as routes

    exports = tmp_path / "exports"
    bundle = exports / "v9"
    (bundle / "runtime").mkdir(parents=True)
    (bundle / "config.json").write_text(
        json.dumps({"num_params": 1234, "exported_at": "2026-01-01T00:00:00+00:00",
                    "includes": ["python-runtime"]})
    )
    (bundle / "model.safetensors").write_bytes(b"\x00" * 64)
    (bundle / "runtime" / "load.py").write_text("# loader\n")
    monkeypatch.setattr(routes, "EXPORTS_DIR", exports)

    async with client as c:
        listing = (await c.get("/exports")).json()
        response = await c.get("/exports/v9/download".replace("/exports/", "/export/"))

    assert len(listing) == 1
    entry = listing[0]
    assert entry["name"] == "v9"
    assert entry["num_params"] == 1234
    assert entry["includes"] == ["python-runtime"]
    # Nested files count: a deployment package is not three files.
    assert entry["file_count"] == 3

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    assert 'filename="v9.zip"' in response.headers["content-disposition"]

    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = sorted(archive.namelist())
        assert names == ["v9/config.json", "v9/model.safetensors", "v9/runtime/load.py"]
        # Contents survive the round trip, not just the names.
        assert archive.read("v9/model.safetensors") == b"\x00" * 64
        assert archive.read("v9/runtime/load.py") == b"# loader\n"


async def test_exports_listing_skips_non_bundles(client, tmp_path, monkeypatch):
    from gateway.routes import config as routes

    exports = tmp_path / "exports"
    (exports / "not-a-bundle").mkdir(parents=True)  # no config.json
    (exports / "stray.txt").write_text("x")
    monkeypatch.setattr(routes, "EXPORTS_DIR", exports)

    async with client as c:
        assert (await c.get("/exports")).json() == []


async def test_download_of_an_unknown_export_is_404(client, tmp_path, monkeypatch):
    from gateway.routes import config as routes

    monkeypatch.setattr(routes, "EXPORTS_DIR", tmp_path / "exports")
    async with client as c:
        assert (await c.get("/export/nope/download")).status_code == 404


def test_resolve_export_refuses_to_escape(tmp_path, monkeypatch):
    """Tested directly: an HTTP client normalises `..` out before sending."""
    from fastapi import HTTPException

    from gateway.routes import config as routes

    exports = tmp_path / "exports"
    (exports / "real").mkdir(parents=True)
    (exports / "real" / "config.json").write_text("{}")
    outside = tmp_path / "secret"
    outside.mkdir()
    (exports / "link").symlink_to(outside)
    monkeypatch.setattr(routes, "EXPORTS_DIR", exports)

    assert routes._resolve_export("real").name == "real"
    for name in ["../secret", "..", "link", "nope"]:
        with pytest.raises(HTTPException) as excinfo:
            routes._resolve_export(name)
        assert excinfo.value.status_code == 404, name


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
