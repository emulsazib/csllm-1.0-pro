# Phases & Roadmap

## Current Phase

**CSLLM 3.0 — enterprise control dashboard. Phases 1-2 of 4 complete, awaiting approval.**
377 Python tests + 60 frontend tests pass, 0 compiler warnings, ruff clean, `tsc -b` clean.
Every UI phase is verified by driving the real app in headless Chrome, not by inspection.

The 3.0 brief (fourth section of `doc/prompt.md`) asks for a PyTorch core engine; that was
**declined and confirmed with the user** — rule #1 stands, the C++ engine already exposes
everything the dashboard needs. Tailwind/Recharts likewise declined in favour of the existing
`styles.css` token system and Chart.js. "GPU VRAM" is reported as whatever the host actually
has, via `csllm/resources.py`.

## Completed

**CSLLM 1.0**
- [x] C++20 engine, hand-derived autograd, all gradients verified in float64.
- [x] Byte-level BPE + training loop. Trained model: 11.9 min, 1.41 nats/char.
- [x] FastAPI SSE gateway: 19 ms TTFT, 4 concurrent streams at 1.95x.

**CSLLM 2.0 Phase 1 — Datasets & export**
- [x] Plugin registry (`.txt`/`.md`/`.jsonl`/`.ndjson`/`.csv`/`.tsv`) with auto-registration.
- [x] Optional tokenizer special tokens; safetensors bundle bitwise-identical to the engine.

**CSLLM 2.0 Phase 2 — Telemetry, config API, attention capture**
- [x] C++ attention capture (NumPy-verified to 2.5e-07), bound `decode()`,
      `Sampler::distribution()` with `sample()` built on it.
- [x] `TrainingSupervisor` + `WS /ws/train`; `ConfigStore` + `/configure_model`; router split.

**CSLLM 2.0 Phase 3 — Diagnostics UI**
- [x] `/tokenize`, `/embeddings`, `/inspect/next_token`.
- [x] `web/` Vite + React + TS: tokenizer panel, embedding heatmap, sampling playground.

**CSLLM 2.0 Phase 4 — Attention animation & training dashboard**
- [x] `WS /ws/inspect`: JSON frame + **binary Float32Array** attention frame per token, with
      layer/head filtering applied before serialisation (~36 KB/token at full context otherwise).
- [x] `TransformerGraph` (React Three Fiber, lazy-loaded, adaptive camera): layer stack with
      height AND colour encoding attention, per-head arcs, sequential one-hue ramp.
- [x] `AttentionPanel`: layer/head selection, clickable token history, per-step candidate table,
      per-head weight table.
- [x] `TrainingDashboard`: loss curves (train + val on ONE axis), LR and grad-norm sparklines in
      their own charts, virtualized log console, preset selection, start/stop.
- [x] **Verified live over CDP:** 16 tokens / 25.9 KB binary attention; a 300-step training run
      started from the browser (loss 6.24 → 3.84, best val 3.8373, exit 0, ~102k tok/s).

**CSLLM 3.0 Phase 1 — Model configurator & parameter calculator**

- [x] `csllm/params.py`: `calculate_model_params` → param breakdown (embedding/attention/ffn/
      norms) + train-memory breakdown (weights/grads/AdamW/activations), swept against the C++
      `ModelConfig::num_params()`.
- [x] `csllm/resources.py`: `probe_device` — NVIDIA VRAM via `nvidia-smi`, else unified
      memory / RAM via sysconf + `/proc` or `ps`. No new dependencies.
- [x] Side-effect-free `POST /configure_model/estimate`; `ArchitectureParams` shared by the
      estimate and configure requests so preview and submission cannot drift.
- [x] `ConfiguratorPanel`: sliders, live count, stacked memory bar, headroom vs. this host,
      version list, create / create+initialize.
- [x] **Verified live over CDP:** 12,194,688 at the shakespeare preset; 6→12 layers updates to
      22,816,128; only legal head counts offered; version creation round-trips.

**CSLLM 3.0 Phase 2 — Training manager & WebSocket telemetry**

- [x] `TrainingSupervisor` supervises a job *kind* (`JOBS`: `train` | `prepare`); the pump,
      broadcast, history and back-pressure logic is shared rather than duplicated.
- [x] `train/train_tokenizer.py --dataset` reads through the plugin registry (so `.jsonl`/`.csv`
      work) and emits staged progress; `train/emit.py` holds the shared emitter.
- [x] `POST /datasets/{name}/prepare`, `GET /prepared`, `POST /train/{pause,resume}`.
- [x] Trainer emits `epoch` + `rss_bytes`; host memory facts ride the `start` row.
- [x] `DatasetBrowser` tab; `TrainingDashboard` steps slider, prepared-data selector,
      pause/resume, epoch + memory readouts and sparkline.
- [x] **Verified live over CDP:** prepared `speeches.jsonl` (901 docs) → trained on it → paused
      (step frozen at 600 across 2 s) → resumed (600 → 994) → stopped (SIGTERM, ~2 s).

## In Progress

**Awaiting approval to start 3.0 Phase 3** (unified explainable playground + attention
matrix heat-map).

## Next Up

1. **3.0 Phase 3** — unified explainable playground + attention matrix heat-map.
2. **3.0 Phase 4** — zip download, standalone deployment package, export modal.

Carried over from 2.0:

- **Regularization (dropout)** — the train/val gap is +1.26; the largest quality win available.
- **Continuous/batched decoding** — would lift concurrency from ~1.95x toward linear.
- **Gradient checkpointing** — the ~1.5 GB activation arena at B=8/T=256 caps batch size.

## Backlog

- Sanitize C++ assertion paths out of API error responses before any non-localhost exposure.
- Whole-stack flow animation in the 3D view (arcs are per-layer today).
- Persist Adam moments so `--resume` is exact; multiple concurrent training runs.
- Metal / NEON kernels; bf16 checkpoints; KV-cache quantization.
- Larger corpora; retrain with an EOT token now that multi-document datasets are supported.
- Gateway auth / rate limiting.
- Frontend component tests (only pure logic is covered today; the panels are verified visually).
