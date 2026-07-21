# Rules & Constraints

## Must Do

1. **Follow the knbase workflow.** `start_session` → `begin_task` → work → update `memory.md` →
   `complete_task`. `complete_task` refuses without a memory update.
2. **Respect the phase gates.** The project is delivered in four phases (scaffolding → C++ core →
   training loop → gateway). **Stop and wait for explicit user approval between phases.**
   Do not start Phase N+1 because Phase N went well.
3. **Every op ships with two proofs**, written *before* the op is considered done:
   a NumPy forward oracle in `tests/reference.py`, and a **double-precision** finite-difference
   gradient check. An op without both is unfinished.
4. **Build bottom-up.** Tensor → autograd → GEMM → elementwise ops → attention → block → model.
   Verify each layer before building on it; a wrong gradient deep in the stack is very hard to
   localize after the fact.
5. **Derive gradients on paper first**, then implement. Record the formula in `design.md`.
6. **Keep `configs/debug.json` working.** Every change must still run end-to-end in seconds on the
   tiny config. It is the fastest possible signal that something broke.
7. **Update `memory.md`** with learnings and gotchas after each task — future agents depend on it.

## Must Not Do

1. **Never add PyTorch, TensorFlow, JAX, or any autograd/tensor library** (including Eigen for the
   model path). This is the project's defining constraint. If something seems to require one,
   stop and ask.
2. **Never use NumPy in the model's forward/backward path.** NumPy is for tests and data
   binarization only. The C++ engine does the math.
3. **Never claim a gradient is correct without a passing double-precision gradcheck.** fp32 finite
   differences are too noisy to be evidence.
4. **Never `-ffast-math`.** It licenses reassociation and breaks NaN/Inf guards and reproducibility.
5. **Never block the FastAPI event loop.** Model compute goes through `asyncio.to_thread` with the
   GIL released in C++; a synchronous call in a route handler stalls every concurrent stream.
6. **Never share a KV cache between requests.** Each `GenerationSession` owns its own.
7. **Never hand-edit `.knbase/`** (index, mind map, activity log) — knbase manages those.
8. **Never commit** trained weights, `.venv/`, `build/`, or downloaded corpora.

## Coding Standards

- **C++20.** Headers in `core/include/csllm/`, one `.cpp` per header in `core/src/`.
  `snake_case` for functions/variables, `PascalCase` for types, trailing `_` for in-place ops.
- Compile with `-O3 -funroll-loops -fno-math-errno -Wall -Wextra`. Warnings are bugs.
- Core ops are `template<typename T>` with explicit `float`/`double` instantiation, so
  gradchecks can run in double. No exceptions thrown on the hot path.
- **Python**: PEP 8, `ruff`-clean, type hints on all public functions, dataclasses for config.
- Bindings wrap every compute call in `py::gil_scoped_release`.
- Prefer zero-copy: expose C++ buffers to Python as NumPy views rather than copying.

## Guardrails

- **Requires explicit approval:** installing system packages (`brew install …`), any `git push`,
  and advancing to the next project phase.
- **Off-limits:** `.knbase/*` (generated), `doc/prompt.md` (the immutable project brief).
- **Destructive ops to avoid:** deleting checkpoints or trained tokenizer artifacts; overwriting
  `data/` without checking; `rm -rf` on anything outside `build/`.
- **Long-running work:** full training runs take tens of minutes to hours. Run them in the
  background with checkpointing enabled; never start one inside a blocking foreground call.
- **Resource limits:** the gateway caps concurrent `GenerationSession`s with a semaphore.
  Each costs ~4.5 MiB of KV cache plus a CPU core while decoding.
