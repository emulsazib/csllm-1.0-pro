"""Pydantic v2 request/response models.

Sampling bounds are enforced here so invalid parameters can never reach the C++
sampler — the edge is the right place for it, and it turns a would-be crash into
a 422 with a readable message.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "ArchitectureParams",
    "CandidateToken",
    "ConfigVersionResponse",
    "ConfigureModelRequest",
    "DeviceResponse",
    "EmbeddingsRequest",
    "EmbeddingsResponse",
    "ErrorResponse",
    "EstimateRequest",
    "EstimateResponse",
    "ExportSummary",
    "ExportRequest",
    "ExportResponse",
    "GenerateRequest",
    "GenerateResponse",
    "HealthResponse",
    "InspectRequest",
    "InspectResponse",
    "InspectStreamRequest",
    "PrepareRequest",
    "StreamChunk",
    "TokenInfo",
    "TokenizeRequest",
    "TokenizeResponse",
    "TrainStartRequest",
    "TrainStatusResponse",
]

FinishReason = Literal["length", "context_full", "disconnected"]


class GenerateRequest(BaseModel):
    model_config = {"extra": "forbid"}  # typo in a field name should 422, not be ignored

    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(128, ge=1, le=1024)
    # 0.0 means greedy argmax — the C++ sampler short-circuits rather than
    # dividing by zero.
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0, description="0 disables top-k")
    top_p: float = Field(0.95, gt=0.0, le=1.0, description="1.0 disables top-p")
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    stream: bool = True


class StreamChunk(BaseModel):
    """One SSE `data:` payload."""

    text: str
    index: int
    finish_reason: FinishReason | None = None


class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason


class HealthResponse(BaseModel):
    status: Literal["ok"]
    version: str
    num_params: int
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    blas_backend: str
    max_concurrent_sessions: int
    sessions_in_flight: int
    kv_cache_bytes_per_session: int


class ErrorResponse(BaseModel):
    detail: str


# ── configuration ────────────────────────────────────────────────────────────


class ArchitectureParams(BaseModel):
    """The hyperparameters that define a model architecture.

    Bounds here are sanity limits only. The real invariants — n_embd divisible by
    n_head, and an EVEN head_dim for RoPE's pairwise rotation — are enforced by
    the C++ ``ModelConfig.validate()`` so Python and C++ cannot disagree.

    Shared by ``/configure_model`` and ``/configure_model/estimate`` so the form
    the UI submits and the one it previews cannot drift apart.
    """

    model_config = {"extra": "forbid"}

    vocab_size: int = Field(4096, ge=256, le=65536)
    n_layer: int = Field(6, ge=1, le=64)
    n_head: int = Field(6, ge=1, le=64)
    n_embd: int = Field(384, ge=8, le=8192)
    block_size: int = Field(256, ge=8, le=8192)
    ffn_hidden: int = Field(1024, ge=8, le=32768)
    rope_theta: float = Field(10000.0, gt=0.0)
    norm_eps: float = Field(1e-5, gt=0.0, le=1.0)


class ConfigureModelRequest(ArchitectureParams):
    """Hyperparameters for a new model version."""

    note: str = Field("", max_length=500)
    #  Only affects the reported activation estimate, not the stored config.
    batch_size: int = Field(8, ge=1, le=512)
    #  When set, initialize a fresh model at this config and save a checkpoint.
    initialize: bool = False
    out: str | None = None


class EstimateRequest(ArchitectureParams):
    """Dry run: cost this architecture without persisting anything.

    Called on every slider movement, so it must stay a pure calculation — no
    version file, no model allocation.
    """

    batch_size: int = Field(8, ge=1, le=512)
    #: Training sequence length. Defaults to block_size (the worst case).
    seq_len: int | None = Field(None, ge=1, le=8192)


class DeviceResponse(BaseModel):
    kind: str
    device: str
    #: "VRAM" | "Unified memory" | "RAM" — depends on the host, not hard-coded.
    memory_label: str
    total_bytes: int
    used_bytes: int
    source: str


class EstimateResponse(BaseModel):
    config: dict
    num_params: int
    #: {embedding, attention, ffn, norms, total}
    params: dict[str, int]
    #: {params, gradients, optimizer, activations, total} in bytes.
    memory: dict[str, int]
    device: DeviceResponse
    #: False when the training footprint exceeds this host's memory.
    fits: bool


class ConfigVersionResponse(BaseModel):
    version_id: str
    index: int
    config: dict
    num_params: int
    activation_bytes: int
    created_at: float
    note: str = ""
    created: bool = True
    checkpoint: str | None = None


# ── training ─────────────────────────────────────────────────────────────────


class TrainStartRequest(BaseModel):
    model_config = {"extra": "forbid"}

    config: str = "configs/shakespeare.json"
    steps: int = Field(1000, ge=1, le=1_000_000)
    batch_size: int = Field(8, ge=1, le=512)
    lr: float = Field(1e-3, gt=0.0, le=1.0)
    min_lr: float = Field(5e-5, ge=0.0, le=1.0)
    warmup: int = Field(100, ge=0)
    weight_decay: float = Field(0.1, ge=0.0, le=1.0)
    grad_clip: float = Field(1.0, gt=0.0)
    eval_every: int = Field(100, ge=1)
    eval_iters: int = Field(20, ge=1)
    sample_every: int = Field(0, ge=0)
    seed: int = Field(1337, ge=0)
    out: str = "data/model.csllm"
    data_dir: str = "data"
    tokenizer_dir: str = "data/tokenizer"
    resume: bool = False


class PrepareRequest(BaseModel):
    """Train a tokenizer on a dataset and binarize it.

    ``vocab_size`` is taken from the config rather than passed here — the model's
    embedding table is sized from the same file, and letting the two disagree
    produces a tokenizer the model cannot load.
    """

    model_config = {"extra": "forbid"}

    config: str = "configs/shakespeare.json"
    #: Both default to ``data/prepared/<dataset>/`` when omitted. They are NOT
    #: defaulted to ``data/`` — that holds the checked-in corpus, and a prepare
    #: pointed there would overwrite it.
    out: str | None = None
    data_dir: str | None = None
    val_fraction: float = Field(0.1, gt=0.0, lt=1.0)


class TrainStatusResponse(BaseModel):
    run_id: str | None = None
    #: "train" | "prepare" | None when idle.
    kind: str | None = None
    running: bool = False
    #: SIGSTOPped: still resident, not consuming CPU.
    paused: bool = False
    step: int = 0
    total_steps: int = 0
    progress: float = 0.0
    last_loss: float | None = None
    best_val: float | None = None
    elapsed_s: float = 0.0
    returncode: int | None = None
    command: str = ""
    subscribers: int = 0
    pid: int | None = None


# ── export ───────────────────────────────────────────────────────────────────


class ExportRequest(BaseModel):
    model_config = {"extra": "forbid"}

    checkpoint: str = "data/model.csllm"
    tokenizer_dir: str = "data/tokenizer"
    out: str = "exports/latest"
    #: Add a torch-free Python loader that works without this repository.
    include_runtime: bool = False
    #: Add the C++20 engine sources and a standalone CMakeLists.
    include_cpp: bool = False


class ExportResponse(BaseModel):
    out_dir: str
    #: Bundle name — the path segment `/export/{name}/download` takes.
    name: str
    num_params: int
    #: Flat file -> size in bytes, including anything under runtime/ or cpp/.
    files: dict[str, int]
    total_bytes: int
    includes: list[str] = []


class ExportSummary(BaseModel):
    name: str
    path: str
    num_params: int | None = None
    total_bytes: int
    file_count: int
    exported_at: str | None = None
    includes: list[str] = []


# ── inspection (diagnostics UI) ───────────────────────────────────────────────


class TokenizeRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str = Field(..., max_length=8192)


class TokenInfo(BaseModel):
    index: int
    id: int
    text: str
    #: Raw bytes, because a token need not be valid UTF-8 on its own.
    bytes: list[int]
    #: Byte offsets into the UTF-8 encoding of the input.
    start: int
    end: int
    #: True when this token alone is not decodable — the UI must not split here.
    partial_utf8: bool


class TokenizeResponse(BaseModel):
    tokens: list[TokenInfo]
    count: int
    num_chars: int
    num_bytes: int
    compression: float
    vocab_size: int


class EmbeddingsRequest(BaseModel):
    model_config = {"extra": "forbid"}

    text: str | None = Field(None, max_length=8192)
    ids: list[int] | None = None
    #: Project to 3D with PCA for the graph view.
    project: bool = True


class EmbeddingsResponse(BaseModel):
    ids: list[int]
    labels: list[str]
    n_embd: int
    #: [n_tokens, n_embd] — the raw embedding rows, for the heatmap.
    vectors: list[list[float]]
    vmin: float
    vmax: float
    #: [n_tokens, 3] PCA projection, or empty when project=False or n<3.
    projection: list[list[float]] = []
    explained_variance: list[float] = []


class InspectRequest(BaseModel):
    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1, max_length=8192)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    top_n: int = Field(20, ge=1, le=200)


class CandidateToken(BaseModel):
    id: int
    text: str
    logit: float
    #: Unfiltered softmax at temperature 1 — the model's own belief.
    raw_prob: float
    #: What the sampler would actually draw from; 0 when filtered out.
    prob: float
    kept: bool


class InspectResponse(BaseModel):
    prompt_tokens: int
    candidates: list[CandidateToken]
    kept_count: int
    #: Entropy (nats) of the raw and filtered distributions.
    raw_entropy: float
    filtered_entropy: float
    vocab_size: int


class InspectStreamRequest(BaseModel):
    """Subscribe message for WS /ws/inspect.

    `layers` / `heads` narrow the attention block BEFORE it is serialised, which
    is the main lever on payload size: the full block is n_layer x n_head x keys
    float32 (~36 KB per token at the 12M config with a full context).
    """

    model_config = {"extra": "forbid"}

    prompt: str = Field(..., min_length=1, max_length=8192)
    max_tokens: int = Field(24, ge=1, le=256)
    temperature: float = Field(0.8, ge=0.0, le=2.0)
    top_k: int = Field(40, ge=0)
    top_p: float = Field(0.95, gt=0.0, le=1.0)
    seed: int | None = Field(None, ge=0, le=2**63 - 1)
    top_n: int = Field(8, ge=0, le=50)
    attention: bool = True
    layers: list[int] | None = None
    heads: list[int] | None = None
