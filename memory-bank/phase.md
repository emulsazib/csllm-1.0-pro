# Phases & Roadmap

## Current Phase

**COMPLETE — CSLLM 1.0 and 2.0, all eight phases delivered.**
325 Python tests + 39 frontend tests pass, 0 compiler warnings, ruff clean, `tsc -b` clean.
Every UI phase was verified by driving the real app in headless Chrome, not by inspection.

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

## In Progress

Nothing. The project meets every goal in `prd.md` and every deliverable in `doc/prompt.md`.

## Next Up

No committed work. Highest-value items if it is picked up again:

1. **Regularization (dropout)** — the train/val gap is +1.26; the largest quality win available.
2. **Continuous/batched decoding** — would lift concurrency from ~1.95x toward linear.
3. **Gradient checkpointing** — the ~1.5 GB activation arena at B=8/T=256 caps batch size.

## Backlog

- Sanitize C++ assertion paths out of API error responses before any non-localhost exposure.
- Whole-stack flow animation in the 3D view (arcs are per-layer today).
- Persist Adam moments so `--resume` is exact; multiple concurrent training runs.
- Metal / NEON kernels; bf16 checkpoints; KV-cache quantization.
- Larger corpora; retrain with an EOT token now that multi-document datasets are supported.
- Gateway auth / rate limiting.
- Frontend component tests (only pure logic is covered today; the panels are verified visually).
