# Project Memory

## Summary

**CSLLM is complete.** A ~12M-parameter autoregressive Transformer built entirely from scratch:
a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch), a **Python**
byte-level BPE tokenizer and training loop, a **FastAPI** gateway with SSE and WebSocket
telemetry, and a **React + TypeScript** diagnostics UI with a 3D attention view.
Llama-style: RoPE + RMSNorm + SwiGLU, pre-norm residuals, weight-tied `lm_head`.

**325 Python tests + 39 frontend tests pass, 0 compiler warnings, ruff clean, tsc clean.**

## Recent Changes

*(reverse-chronological)*

- **2026-07-22 — 2.0 Phase 4: attention animation & training dashboard.**
  `WS /ws/inspect` streams a JSON frame then a **binary Float32Array** attention frame per token,
  with layer/head filtering applied before serialisation. `TransformerGraph` (React Three Fiber,
  lazy-loaded) renders the layer stack with height AND colour encoding attention;
  `AttentionPanel` adds layer/head selection and a per-head weight table; `TrainingDashboard`
  subscribes to `/ws/train` with loss curves, LR/grad-norm sparklines, a log console and
  start/stop. 20 backend + 9 frontend tests added.
  **Verified by driving the real app over CDP:** 16 tokens / 25.9 KB of binary attention,
  and a full 300-step training run started from the browser (loss 6.2 → 3.8, exit 0).
- **2026-07-21 — 2.0 Phase 3: diagnostics UI.** `/tokenize`, `/embeddings`, `/inspect/next_token`;
  `web/` with Vite + React + TS, tokenizer panel, embedding heatmap, sampling playground.
- **2026-07-21 — 2.0 Phase 2: telemetry, config API, attention capture.**
- **2026-07-21 — 2.0 Phase 1: dataset plugins & safetensors export.**
- **2026-07-21 — 1.0 Phases 1-4.** Engine, trained model (1.41 nats/char), SSE gateway.

## Learnings & Gotchas

**Environment**

- `cmake` **4.4.0**; **pybind11 must be >= 3.0** (2.13.x predates Python 3.14's C-API).
- Homebrew **Python 3.14.6** for `.venv`; **Node v26.3.0 / npm 11.16.0**.
- arm64 + Apple clang 21; Accelerate gives multithreaded BLAS free. 8 hardware threads.
- Starlette 1.3 deprecated `HTTP_422_UNPROCESSABLE_ENTITY` → `HTTP_422_UNPROCESSABLE_CONTENT`.
- **Chrome is installed and drivable over CDP** — the highest-value UI check available:
  launch `--headless --remote-debugging-port=9222 <url>`, attach to the page target that
  already has the URL (navigating an `about:blank` target silently screenshots blank), then
  `Runtime.evaluate` to click and `Page.captureScreenshot` with `captureBeyondViewport`.

**Engine**

- The tape needs no topological sort (nodes are appended in forward order).
- **Weight tying**: `matmul_bt` shares storage with `tok_emb`; the gradient accumulates from the
  embedding scatter-add *and* `matmul_bt`'s dW.
- **ln(V) is a hard floor** for initial loss when targets are independent of the logits.
- **RoPE uses INTERLEAVED pairs**; backward is free (orthogonal ⇒ negated angles). Attention
  backward must re-apply the causal mask. Cached keys are stored already-rotated.
- **fp32 finite differences are useless for gradchecking** — hence the f64 instantiation.
- **Attention capture** is opt-in and observation-only; layer 0 is rederivable from `tok_emb` +
  first RMSNorm + wq/wk + RoPE, which is how the test verifies it (matches to 2.5e-07).
- **`Sampler::sample()` is built on `distribution()`** — any filtering change must go there.
- **`prefill()` consumes the ENTIRE prompt**; the first token must come from `sample_last()`.
  In the WS loop use `decode()` rather than `step()`, since only `decode()` hands back the
  logits the UI needs to chart alongside the token.

**Training / data**

- **BPE merge count is bounded by corpus DIVERSITY, not length** — bit me twice in tests.
- **Match the cosine schedule to the steps you will actually run.**
- A 20-batch eval is optimistically biased (4.2996 vs 4.3877 over 200 batches).
- **Per-token loss is not comparable across vocabularies** — divide by chars/token.
- Multi-document corpora need an EOT separator BETWEEN documents only.

**Serving / async**

- Concurrent `GenerationSession`s are safe; `Model::forward_logits` is **not** (shared arena).
- **`@dataclass` kills hashability** (generated `__eq__` sets `__hash__ = None`) — use `eq=False`.
- **`stop()` must wait for the output pump**, not just the process; unit tests passed this by
  timing luck and only the live run exposed it.
- Bounded subscriber queues drop the OLDEST and always insert the newest, so terminal events are
  never lost. The full record is in `runs/<id>/metrics.jsonl`.
- **Restart uvicorn after editing gateway code** unless `--reload` is on.

**Frontend / dataviz**

- **`/api` exists only in DEV.** Vite's proxy strips it; a production build is served by FastAPI at
  the root. `API_BASE = import.meta.env.DEV ? "/api" : ""`. **No httpx test caught this** — only
  screenshotting the built app did.
- `import.meta.env` needs `"types": ["vite/client"]` or `tsc -b` fails while `vitest` still passes,
  so `npm test` alone is not enough — run the build.
- **vitest 2.x bundles its own Vite** and collides with Vite 6 types; vitest 3 is the pair. Use
  `defineConfig` from `vitest/config` when the config has a `test` key.
- **A diverging scale needs a neutral BAND, not a neutral point** — with continuous data an exact
  zero never occurs, so a point-neutral gives every cell a hue.
- **Scale a heatmap to a robust percentile, not max()** — scaling to max put 86% of real embedding
  cells into two near-identical steps.
- **Dark-mode diverging ramps invert**: the midpoint is the DARKEST point and arms lighten outward.
- **384 dims at panel width is ~2.6px per cell** — moiré, not data. Window the dimensions.
- **Frame a 3D scene from the data's extent.** A fixed camera left the short wide attention grid
  stranded in the middle of the canvas; distance derived from keys × layers fixed it.
- **Encode magnitude twice (height AND colour)** in the 3D view so the reading survives greyscale.
- Attention is magnitude without polarity → **sequential** one-hue ramp, not diverging.
- **Lazy-load three.js** (~830 KB) so the other tabs never download it.
- Run `scripts/validate_palette.js` from the dataviz skill rather than eyeballing CVD safety.

**Tooling**

- **`make datasets` / `make export` must be `.PHONY`** — a `datasets/` directory exists.
- The local `datasets/` package **shadows HuggingFace `datasets`** if that is ever installed.
- `safetensors` uses its numpy backend — it does not pull in torch.

## Known Issues

- C++ assertion messages leak absolute source paths into API error responses. Fine for a localhost
  tool with no auth; sanitize before real exposure.
- One training run at a time by design.
- Adam moments are not persisted; `--resume` reloads weights and the step counter only.
- No dropout; no gradient checkpointing (~1.5 GB arena at B=8/T=256).
- Checkpoints fp32 only; one sequence per session (4 concurrent streams ≈ 1.95x, not 4x).
- The 3D graph draws arcs for one layer at a time; a whole-stack flow animation is backlog.
- No auth or rate limiting on the gateway (explicit non-goal in `prd.md`).
