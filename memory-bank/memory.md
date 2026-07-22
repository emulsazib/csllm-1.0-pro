# Project Memory

## Summary

**CSLLM is complete.** A ~12M-parameter autoregressive Transformer built entirely from scratch:
a pure **C++20 engine** with hand-derived reverse-mode autograd (no PyTorch), a **Python**
byte-level BPE tokenizer and training loop, a **FastAPI** gateway with SSE and WebSocket
telemetry, and a **React + TypeScript** diagnostics UI with a 3D attention view.
Llama-style: RoPE + RMSNorm + SwiGLU, pre-norm residuals, weight-tied `lm_head`.

**391 Python tests + 82 frontend tests pass, 0 compiler warnings, ruff clean, tsc clean.**

## Recent Changes

*(reverse-chronological)*

- **2026-07-22 — 3.0 Phase 4: export manager & packaging. CSLLM 3.0 COMPLETE.**
  `export_bundle` gained `include_runtime` / `include_cpp` (both default off, so the plain
  bundle stays the three documented files). `runtime/` ships a torch-free loader emitted from
  `csllm/runtime_template.py` that rebuilds the BPE tokenizer from `tokenizer.json` alone;
  `cpp/` ships the C++20 engine plus a standalone CMakeLists; `README.md` documents the real
  tensor names. New `GET /exports` and `GET /export/{name}/download` (spooled ZIP_STORED zip,
  built on a worker thread). New `ExportModal` reachable from the app header. 14 tests added.
  **Verified live over CDP and then for real:** built a bundle from the modal, downloaded the
  zip (49,210,819 bytes, 36 entries, integrity OK), extracted it to a clean directory, ran
  `runtime/load.py` there, and built `libcsllm_engine.a` from `cpp/` with 0 warnings.

- **2026-07-22 — 3.0 Phase 3: explainable inference playground.**
  New `PlaygroundPanel` tab: one prompt, ONE `/ws/inspect` subscription, and a per-token
  breakdown (tokenization chips → attention → probability distribution) for whichever token you
  click. New `AttentionHeatmap` renders the query x key matrix on a canvas — the causal
  staircase, with cells past a row's end drawn as surface rather than zero. `attentionVector` /
  `attentionMatrix` added to `api/ws.ts`; the sequential ramp was promoted from
  `TransformerGraph` into `theme.ts` (`sequentialColor`/`sequentialLegend`) so the 2D and 3D
  views cannot drift; token-chip markup extracted to `TokenText.tsx`. **No backend changes** —
  the stream already carried everything. 22 frontend tests added.
  **Verified live over CDP:** 24 tokens generated; heat-map rows match the token chips exactly;
  clicking token #2 repointed the breakdown; head and layer switches both repaint; zero console
  errors.

- **2026-07-22 — 3.0 Phase 2: training manager & dataset wiring.**
  `TrainingSupervisor` now supervises a *job kind* (`JOBS`: `train` | `prepare`) rather than
  assuming the trainer; `train/train_tokenizer.py` gained `--dataset` (read via the plugin
  registry, so `.jsonl`/`.csv` work) plus `--emit-jsonl`, and `train/emit.py` holds the emitter
  both entrypoints share. New `POST /datasets/{name}/prepare`, `GET /prepared`,
  `POST /train/{pause,resume}` (SIGSTOP/SIGCONT). Trainer emits `epoch` + `rss_bytes` per
  throughput row and host memory facts in `start`. New `DatasetBrowser` tab; `TrainingDashboard`
  gained a steps slider, a prepared-data selector fed by `GET /prepared`, pause/resume, and an
  epoch + memory readout. `applyTrainEvent` extracted from `useTrainingStream` so the reducer is
  testable. 27 backend + 11 frontend tests added.
  **Verified live over CDP:** prepared `speeches.jsonl` (901 docs) → trained on it → paused
  (step frozen at 600 across 2 s) → resumed (600 → 994) → stopped (SIGTERM, ~2 s).

- **2026-07-22 — 3.0 Phase 1: model configurator & parameter calculator.**
  New `csllm/params.py` (`calculate_model_params` → param breakdown + train-memory breakdown)
  and `csllm/resources.py` (`probe_device` — reports VRAM on an NVIDIA host, unified memory on
  Apple Silicon, RAM elsewhere; no new dependencies). New side-effect-free
  `POST /configure_model/estimate` behind the sliders; `ArchitectureParams` extracted as the
  shared base of `ConfigureModelRequest` and `EstimateRequest`. New `ConfiguratorPanel` tab with
  live parameter count, stacked memory bar, headroom check, and version creation.
  25 backend + 10 frontend tests added.
  **Verified by driving the real app over CDP:** shakespeare preset reads 12,194,688; dragging
  layers 6→12 updates live to 22,816,128; head options offered for d_model=384 are
  1,2,3,4,6,8,12,16,24,32,48,64 (every invalid divisor excluded); version creation round-trips.
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
  **After `make web-build`, reload with `Page.reload(ignoreCache=True)`** — a plain
  `Page.navigate` serves the cached bundle and you verify the code you just replaced.
  To move a React-controlled slider, call the native `HTMLInputElement.value` setter and then
  dispatch `new Event('input', {bubbles:true})`; assigning `.value` alone does not notify React.
- **Port 8000 is not always free on this machine** — use another port (8010) for verification
  runs rather than assuming `make serve` bound successfully.

**Dashboard (3.0)**

- **`csllm/params.py` must agree with `ModelConfig::num_params()`** (`core/src/model.cpp`).
  Two traps: `lm_head` is **weight-tied** to `tok_emb` so the embedding counts ONCE (double
  counting overstates the 12M config by 1.57M), and SwiGLU's FFN has **three** matrices
  (gate, up, down), not two. `tests/test_params.py` sweeps configs rather than spot-checking.
- **Activations dominate**: at B=8/T=256 the 12M config needs ~1.5 GB of arena against 49 MB of
  weights — ~88% of the training footprint. Any memory bar must render that segment in a visible
  colour; `--gridline` is a hairline token and made the largest slice look like empty space.
- **This dev machine has 8 GB**, so the headroom readout is meaningful, not decorative.
- **Memory probing lives in `csllm/`, not `gateway/`** — `train/train.py` consumes it and the
  trainer must never import the gateway (the gateway reads its stdout, not the reverse).
- **Head-count options are derived from `n_embd`** rather than validated after the fact: a head
  count needs `n_embd % n_head == 0` AND an even `head_dim`, so e.g. 128 is a clean divisor of
  384 that the engine still rejects. Offering only legal values removes the confusing failure.
- **UI presets must mirror `configs/*.json`** — `ffn_hidden` drifted (192 vs 128) during Phase 1
  and quoted a parameter count no config file could produce; a vitest now imports both JSONs.
- **Prepared datasets go to `data/prepared/<dataset>/`, keyed by DATASET not config.** Deriving
  the path from the config put `configs/debug.json` at `data/debug` — exactly where the shipped
  debug corpus lives — and a prepare destroyed `data/debug/*.bin` during verification. They were
  restored by re-binarizing `data/tinyshakespeare.txt` with the untouched `data/tokenizer-debug`
  (517,810 / 57,535 tokens, matching the original run). `PrepareRequest.out`/`data_dir` now
  default to None and are filled in server-side.
- **`SIGSTOP` then `SIGTERM` hangs.** A stopped process never reaches its signal handler, so
  `stop()` must `SIGCONT` first or it waits out the full 10 s timeout and needs `SIGKILL` — the
  UI's Stop button appears dead for ten seconds.
- **A failed measurement must not be plotted as 0.** `current_rss()` shells out to `ps` on macOS
  and returned 0 once under a saturated training loop; charted, it drew a cliff to zero. The
  trainer now omits `rss_bytes` when the probe fails, and the reducer drops non-positive samples.
- **`vocab.json` is `{vocab_size, pattern, special_tokens, tokens}`** — read the `vocab_size`
  field; `len()` of the file reports 4 for every tokenizer ever written.
- **A CDP UI check must outlast the thing it is checking.** The 300-step debug run finishes in
  ~4 s at ~100k tok/s, so the first pause/resume verification found the buttons already disabled
  and reported a false failure. Drive a run long enough to still be alive when you click.
- **The attention row for generated token `i` has `4 + i` keys, not `4 + i + 1`.** The capture is
  from the decode step that PRODUCED that token: the query is the previous position, so the last
  key label is unused for the final row. Getting this off by one shifts every column label by one
  token and still looks entirely plausible.
- **A ragged matrix must stay ragged.** Padding short rows to the widest draws "could not attend"
  (the key was not in context yet) identically to "chose not to attend" — different claims.
- **`attentionVector` clamps layer/head.** Panels hold them in state across generations, so a
  stale index on a shallower model reads past the buffer and renders adjacent memory as attention.
- **Two `tokenLabel`-ish forms exist on purpose**: `tokenLabel` (ProbabilityChart) returns a plain
  string for tables/axes; `TokenText` (TokenText.tsx) returns markup so each whitespace glyph can
  be styled. Do not merge them — the first is already covered by tests asserting `\n` not `↵`.
- **`document.querySelector('.panel canvas')` finds the Chart.js chart, not the heat-map.** The
  probability chart renders first in the breakdown, so a CDP pixel check must target
  `.scroll-x canvas` — targeting the wrong one reported a false "canvas did not repaint" bug.
**Export (3.0 Phase 4)**

- **The exported C++ package needs the definitions the project's CMakeLists supplies.**
  `build_info.cpp` and `gemm.cpp` reference `CSLLM_VERSION` / `CSLLM_BLAS_BACKEND` /
  `CSLLM_USE_ACCELERATE` unconditionally, and recent macOS SDKs reject the `cblas_*` prototypes
  without `ACCELERATE_NEW_LAPACK=1`. The first shipped CMakeLists omitted all four and did not
  compile — always *build* the emitted package, never just eyeball it.
- **The bundled loader must not import `csllm`.** It is emitted from a string in
  `csllm/runtime_template.py`, so ruff and tsc cannot see it; `tests/test_export.py` imports the
  emitted file and asserts it encodes identically to the real tokenizer, which is the only thing
  standing between a drifted merge order and a deployed model that reads different token ids
  than it was trained on.
- **`exports/` is excluded from ruff** — bundles are generated artifacts, and a bundle
  containing Python otherwise fails the project lint.
- **Zip with ZIP_STORED, not DEFLATE.** safetensors is raw float32; deflating a 49 MB bundle
  burns CPU for ~1%. Build it in a `SpooledTemporaryFile` on a worker thread, and close the
  buffer in the generator's `finally` or an aborted download leaks a temp file per attempt.
- **`button.action` / `button.ghost` are element-qualified selectors.** A download has to be a
  real `<a>` to get a browser download rather than a fetch-into-blob, and an `<a class="action">`
  matched none of the button styling until `a.action` was added.
- **`.controls` bottom-aligns, so every control in a row needs the same number of note lines.**
  A missing note sits a control lower than its peers; a wrapping one lifts it higher. This has
  now caused a visible misalignment in three separate panels.
- **The Bash tool's cwd persists between calls.** A `cd web` in one command left a later
  `rm -rf exports/...` running in `web/`, which then reported `exports/` as missing and looked
  briefly like data loss. Prefer absolute paths for anything destructive.
- **`datasets/raw/` ships empty** (`.gitkeep` only). `shakespeare-sample.txt` and
  `speeches.jsonl` were added as fixtures so the browser and prepare path have something to
  read; they are slices of `data/tinyshakespeare.txt` and safe to delete.

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
