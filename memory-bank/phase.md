# Phases & Roadmap

## Current Phase

**CSLLM 2.0 Phase 1 — Dataset Plugins & Export: COMPLETE.**
224 tests pass, 0 compiler warnings, ruff clean. The real 12M model exports to a portable bundle
that is bitwise identical to the live engine and readable without torch.
**Awaiting explicit user approval to start 2.0 Phase 2.**

## Completed

**CSLLM 1.0 (all four phases)**
- [x] C++20 engine: arena, `Tensor<T>`, tape autograd, ten ops with hand-derived backwards, fused
      causal MHA with RoPE, weight-tied block stack, AdamW, sampler, `.csllm` checkpoints,
      KV-cache decode. All gradients verified in float64.
- [x] Byte-level BPE, TinyShakespeare pipeline, training loop. **Trained model: 11.9 min,
      val 4.39/token = 1.41 nats/char.**
- [x] FastAPI SSE gateway: TTFT **19 ms**, 4 concurrent streams at 1.95x, disconnect abort.

**CSLLM 2.0 Phase 1 — Datasets & export**
- [x] `datasets/`: `DatasetPlugin` ABC with `__init_subclass__` auto-registration, extension-keyed
      registry, `discover()` / `describe()` / `iter_documents()`.
- [x] Built-ins: `.txt`/`.md` (whole or blank-line), `.jsonl`/`.ndjson`, `.csv`/`.tsv`.
      All malformed input fails with the file name and line number.
- [x] Tokenizer **optional** special tokens — inert unless `allow_special=True`, ids persisted
      explicitly, single-document behaviour byte-identical to before.
- [x] `encode_documents()` + `prepare_dataset()` — separator BETWEEN documents only.
- [x] `csllm/export.py` → `model.safetensors` + `tokenizer.json` + `config.json`; `make export`,
      `make datasets`.
- [x] **Verified:** 56 tensors / 12,194,688 params, **max abs difference 0.0** vs the live engine,
      bundle consumable with numpy+safetensors alone, `torch` never imported.

## In Progress

Nothing — Phase 1 is closed and the project is paused at the approval gate.

## Next Up

1. **2.0 Phase 2 — Telemetry & configuration API.**
   - **C++ first (one rebuild for Phases 2-4):** `set_capture_attention()` / `last_attention()` on
     `GenerationSession` (copy `sc[t]*inv` at [model.cpp:437](core/src/model.cpp)); bind the
     currently-unbound `decode()`; add `filtered_distribution()` so the UI's probability chart
     cannot drift from the real sampler.
   - `train/train.py --emit-jsonl --run-id`; `gateway/telemetry.py` `TrainingSupervisor`
     (subprocess + per-client queue fan-out); `POST /train/{start,stop}`, `GET /train/status`,
     `WS /ws/train`.
   - `POST /configure_model` validating through the existing C++ `ModelConfig.validate()`, writing
     `configs/versions/`, returning `num_params` and `estimate_activation_bytes`.
2. **2.0 Phase 3 — Tokenize/embed/probability UI** (Vite + React + TS in `web/`).
3. **2.0 Phase 4 — R3F attention animation + training dashboard.**

*Each phase stops for explicit user approval before the next begins.*

## Backlog

- Regularization (dropout) — train/val gap is +1.26.
- Continuous/batched decoding across concurrent sessions (would lift 1.95x toward linear).
- Gradient checkpointing — the ~1.5 GB arena at B=8/T=256 caps batch size.
- Persist Adam moments so `--resume` is exact.
- Benchmark thread-pool granularity against Accelerate's internal threading.
- Metal / NEON kernels; bf16 checkpoints; KV-cache quantization.
- Larger corpora and a bigger vocab; retrain with an EOT token now that multi-document
  datasets are supported.
- Gateway auth / rate limiting (explicit non-goal in `prd.md`).
