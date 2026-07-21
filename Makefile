.PHONY: help setup build rebuild test lint clean tokenizer train serve smoke

PY      ?= /opt/homebrew/bin/python3.14
VENV    := .venv
VPY     := $(VENV)/bin/python
VPIP    := $(VENV)/bin/pip
CONFIG  ?= configs/debug.json

help:
	@echo "CSLLM — build targets"
	@echo "  make setup      Create .venv and install the project (compiles the C++)"
	@echo "  make build      Recompile the C++ extension"
	@echo "  make rebuild    Clean build dir, then recompile from scratch"
	@echo "  make smoke      Print build info (BLAS backend, threads, params)"
	@echo "  make test       Run the pytest suite"
	@echo "  make lint       ruff check + format --check"
	@echo "  make tokenizer  Train the BPE tokenizer      [Phase 3]"
	@echo "  make train      Train the model  CONFIG=$(CONFIG)   [Phase 3]"
	@echo "  make serve      Run the FastAPI gateway      [Phase 4]"
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
	$(VPY) -m train.train --config $(CONFIG)

serve:
	$(VENV)/bin/uvicorn gateway.main:app --reload --port 8000

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find csllm -name '_csllm_core*.so' -delete
