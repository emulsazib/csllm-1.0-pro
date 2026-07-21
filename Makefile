# `datasets` and `export` must be PHONY: a datasets/ directory exists, so make
# would otherwise consider the target already satisfied and do nothing.
.PHONY: help setup build rebuild test lint clean tokenizer train sample debug \
        serve curl smoke export datasets web-install web-dev web-build web-test dev

PORT    ?= 8000

PY      ?= /opt/homebrew/bin/python3.14
VENV    := .venv
VPY     := $(VENV)/bin/python
VPIP    := $(VENV)/bin/pip
CONFIG  ?= configs/shakespeare.json
STEPS   ?= 3000
BATCH   ?= 8
LR      ?= 1e-3
PROMPT  ?= KING RICHARD:
CKPT       ?= data/model.csllm
EXPORT_DIR ?= exports/v1

help:
	@echo "CSLLM — build targets"
	@echo "  make setup      Create .venv and install the project (compiles the C++)"
	@echo "  make build      Recompile the C++ extension"
	@echo "  make rebuild    Clean build dir, then recompile from scratch"
	@echo "  make smoke      Print build info (BLAS backend, threads, params)"
	@echo "  make test       Run the pytest suite"
	@echo "  make lint       ruff check + format --check"
	@echo "  make tokenizer  Train the BPE tokenizer + binarize  CONFIG=$(CONFIG)"
	@echo "  make train      Train the model    STEPS=$(STEPS) BATCH=$(BATCH) LR=$(LR)"
	@echo "  make sample     Generate from data/model.csllm      PROMPT='$(PROMPT)'"
	@echo "  make export     Export a portable bundle -> $(EXPORT_DIR)"
	@echo "  make datasets   List datasets in datasets/raw/"
	@echo "  make debug      Full tokenizer+train cycle on configs/debug.json (seconds)"
	@echo "  make serve      Run the FastAPI gateway on PORT=$(PORT)"
	@echo "  make web-build  Build the React app (FastAPI then serves it at /)"
	@echo "  make web-dev    Vite dev server with HMR, proxying to the gateway"
	@echo "  make web-test   Frontend unit tests (vitest)"
	@echo "  make dev        Gateway + Vite together for development"
	@echo "  make curl       Stream a completion from a running gateway"
	@echo "  make clean      Remove build artifacts"

$(VENV):
	$(PY) -m venv $(VENV)
	$(VPIP) install --upgrade pip

setup: $(VENV)
	$(VPIP) install -e ".[dev]" -v

build:
	$(VPIP) install -e . --no-build-isolation

rebuild: clean build

smoke:
	@$(VPY) -c "import csllm; b = csllm.build_info(); \
	print(f'CSLLM {b.version}'); \
	print(f'  BLAS backend : {b.blas_backend}'); \
	print(f'  fast-math    : {b.fast_math}  (must be False)'); \
	print(f'  hw threads   : {b.hardware_threads}'); \
	print(f'  pool size    : {csllm.thread_pool_size()}'); \
	print(f'  C++ standard : {b.cxx_standard}'); \
	c = csllm.load_config('configs/shakespeare.json'); \
	print(f'  shakespeare  : {c.num_params():,} params, head_dim={c.head_dim}')"

test:
	$(VPY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/ruff format --check .

tokenizer:
	$(VPY) -m train.train_tokenizer --config $(CONFIG)

train:
	$(VPY) -u -m train.train --config $(CONFIG) --steps $(STEPS) --batch-size $(BATCH) --lr $(LR)

sample:
	$(VPY) -m train.sample --prompt "$(PROMPT)" --stream

# Export a portable bundle (safetensors + tokenizer.json + config.json).
export:
	$(VPY) -m csllm.export --checkpoint $(CKPT) --tokenizer-dir data/tokenizer --out $(EXPORT_DIR)

# List datasets dropped into datasets/raw/.
datasets:
	@$(VPY) -c "import datasets; \
	files = datasets.discover(); \
	print('supported:', ' '.join(datasets.supported_extensions())); \
	print(f'{len(files)} file(s) in datasets/raw/') if True else None; \
	[print(f'  {i.name:<24} {i.plugin:<6} {i.num_documents:>7,} docs {i.num_chars:>10,} chars') \
	 for i in (datasets.describe(f) for f in files)]"

# The fast feedback loop: whole pipeline end-to-end in a few seconds.
debug:
	$(VPY) -m train.train_tokenizer --config configs/debug.json \
	    --out data/tokenizer-debug --data-dir data/debug
	$(VPY) -u -m train.train --config configs/debug.json --steps 300 --batch-size 16 \
	    --lr 3e-3 --min-lr 3e-4 --warmup 30 --eval-every 100 --eval-iters 10 \
	    --tokenizer-dir data/tokenizer-debug --data-dir data/debug \
	    --out data/debug/model.csllm

serve:
	$(VENV)/bin/uvicorn gateway.main:app --reload --port $(PORT)

# ── frontend ────────────────────────────────────────────────────────────────

web-install:
	cd web && npm install

web-build: web-install
	cd web && npm run build

web-test: web-install
	cd web && npm run test && npm run typecheck

# Two processes in development: Vite serves the UI with HMR and proxies /api and
# /ws to uvicorn. A production build needs only `make web-build && make serve`.
web-dev:
	cd web && npm run dev

dev:
	@echo "gateway :$(PORT)  +  vite :5173  (Ctrl-C stops both)"
	@$(VENV)/bin/uvicorn gateway.main:app --reload --port $(PORT) & \
	 cd web && npm run dev; kill %1 2>/dev/null

# End-to-end smoke test against a running gateway (make serve in another shell).
curl:
	curl -sN -X POST http://127.0.0.1:$(PORT)/generate \
	    -H 'Content-Type: application/json' \
	    -d '{"prompt":"$(PROMPT)\n","max_tokens":120,"temperature":0.8,"top_k":40}'

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find csllm -name '_csllm_core*.so' -delete
