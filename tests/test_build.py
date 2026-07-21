"""Phase 1 exit check: the C++/Python bridge is real and correctly configured.

These tests deliberately verify the *build*, not the model. They exist to catch
the failure modes that are expensive to discover later: a BLAS backend that was
detected but does not actually work, -ffast-math sneaking back in, or config
invariants (RoPE's even head_dim) going unenforced.
"""

from __future__ import annotations

import numpy as np
import pytest

import csllm


def test_extension_imports_and_reports_version():
    assert csllm.__version__ == "1.0.0"
    assert csllm.build_info().version == "1.0.0"


def test_accelerate_is_the_active_blas_backend():
    info = csllm.build_info()
    # On this project's target (arm64 macOS) Accelerate must be found. If this
    # fails we silently fell back to the naive triple loop and training would
    # be orders of magnitude slower.
    assert info.blas_backend == "Accelerate", f"unexpected BLAS backend: {info.blas_backend}"
    assert info.accelerate_enabled is True


def test_fast_math_is_disabled():
    # -ffast-math would license reassociation and break NaN/Inf guards and
    # reproducibility, which the hand-written autograd depends on.
    assert csllm.build_info().fast_math is False


def test_cxx20_and_threading():
    info = csllm.build_info()
    assert info.cxx_standard >= 202002, f"expected C++20, got __cplusplus={info.cxx_standard}"
    assert info.hardware_threads >= 1
    assert csllm.thread_pool_size() >= 1


@pytest.mark.parametrize(
    ("fn", "dtype", "rtol"),
    [(csllm.matmul_f32, np.float32, 1e-4), (csllm.matmul_f64, np.float64, 1e-12)],
)
def test_gemm_matches_numpy(fn, dtype, rtol):
    """Proves the Accelerate link actually computes, not just that CMake found it."""
    rng = np.random.default_rng(0)
    for m, k, n in [(1, 1, 1), (3, 5, 7), (64, 128, 32), (128, 384, 256)]:
        a = rng.standard_normal((m, k)).astype(dtype)
        b = rng.standard_normal((k, n)).astype(dtype)
        np.testing.assert_allclose(fn(a, b), a @ b, rtol=rtol, atol=rtol)


def test_gemm_rejects_shape_mismatch():
    a = np.zeros((3, 4), dtype=np.float32)
    b = np.zeros((5, 6), dtype=np.float32)
    with pytest.raises(RuntimeError, match="inner dimensions"):
        csllm.matmul_f32(a, b)


def test_shakespeare_config_shape_and_size():
    cfg = csllm.load_config("configs/shakespeare.json")
    assert cfg.head_dim == 64
    assert cfg.head_dim % 2 == 0  # required for RoPE pair rotation
    assert cfg.num_params() == 12_194_688


def test_debug_config_is_small_enough_to_be_fast():
    cfg = csllm.load_config("configs/debug.json")
    assert cfg.num_params() < 200_000
    assert cfg.head_dim % 2 == 0


def test_config_validation_rejects_odd_head_dim():
    cfg = csllm.ModelConfig()
    cfg.n_embd = 6
    cfg.n_head = 2  # head_dim = 3, odd -> RoPE cannot pair channels
    with pytest.raises(RuntimeError, match="even for RoPE"):
        cfg.validate()


def test_config_validation_rejects_indivisible_embd():
    cfg = csllm.ModelConfig()
    cfg.n_embd = 100
    cfg.n_head = 7
    with pytest.raises(RuntimeError, match="divisible"):
        cfg.validate()


def test_arena_exhaustion_raises_rather_than_falling_back():
    arena = csllm.Arena(1024)
    arena.allocate(512)
    assert arena.used >= 512
    with pytest.raises(RuntimeError, match="arena exhausted"):
        arena.allocate(4096)
    arena.reset()
    assert arena.used == 0
    assert arena.high_water >= 512


def test_cosine_lr_schedule_shape():
    warmup, total, lr_max, lr_min = 10, 100, 1e-3, 1e-5
    assert csllm.cosine_lr(0, warmup, total, lr_max, lr_min) == pytest.approx(lr_max / warmup)
    assert csllm.cosine_lr(warmup - 1, warmup, total, lr_max, lr_min) == pytest.approx(lr_max)
    assert csllm.cosine_lr(total, warmup, total, lr_max, lr_min) == pytest.approx(lr_min)
    mid = csllm.cosine_lr(total // 2, warmup, total, lr_max, lr_min)
    assert lr_min < mid < lr_max
