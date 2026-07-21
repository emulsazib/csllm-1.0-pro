"""Host memory probing, adapted to whatever machine is actually running.

The dashboard wants a "how much memory will this cost / is this using" readout.
The honest answer depends on the host: this project is CPU-only (Accelerate BLAS
on Apple Silicon, naive GEMM elsewhere), so on the development machine the answer
is unified memory, not VRAM. On a host with an NVIDIA GPU we report real VRAM.
Nothing here hard-codes "VRAM" onto a machine that has none.

Lives in ``csllm/`` rather than ``gateway/`` deliberately: ``train/train.py``
emits memory telemetry and must not import the gateway. The trainer stays
standalone (see ``gateway/telemetry.py``) — the gateway reads its stdout, not the
other way round.

No new dependencies. ``psutil`` would be the obvious tool and is deliberately not
used; everything here is stdlib plus, where present, the ``nvidia-smi`` binary.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

__all__ = ["DeviceInfo", "current_rss", "probe_device", "total_memory"]

#: nvidia-smi is a subprocess; never let a hung driver stall a request.
_SMI_TIMEOUT_S = 1.0

#: GPU memory is re-read at most this often — the estimate endpoint is called on
#: every slider movement and must not fork a process per keystroke.
_GPU_POLL_INTERVAL_S = 1.0

_gpu_cache: tuple[float, tuple[int, int] | None] = (0.0, None)


@dataclass(frozen=True)
class DeviceInfo:
    """What memory this host has, and what to call it in the UI."""

    #: "cuda" | "apple-silicon" | "cpu"
    kind: str
    #: Human label for the device itself, e.g. "Apple M2 Pro" or "NVIDIA RTX 4090".
    device: str
    #: What the memory pool is called here: "VRAM" | "Unified memory" | "RAM".
    memory_label: str
    total_bytes: int
    used_bytes: int
    #: Where the numbers came from, so a surprising readout is traceable.
    source: str

    def to_dict(self) -> dict:
        return asdict(self)


# ── raw probes ───────────────────────────────────────────────────────────────


def _run(cmd: list[str]) -> str | None:
    """Run a probe command, returning None on any failure. Never raises."""
    try:
        out = subprocess.run(
            cmd, capture_output=True, text=True, timeout=_SMI_TIMEOUT_S, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _nvidia_memory() -> tuple[int, int] | None:
    """(total, used) VRAM bytes for GPU 0, or None when there is no NVIDIA GPU."""
    global _gpu_cache
    now = time.monotonic()
    cached_at, cached = _gpu_cache
    if cached is not None and now - cached_at < _GPU_POLL_INTERVAL_S:
        return cached

    if shutil.which("nvidia-smi") is None:
        return None
    raw = _run(
        [
            "nvidia-smi",
            "--query-gpu=memory.total,memory.used",
            "--format=csv,noheader,nounits",
            "--id=0",
        ]
    )
    if not raw:
        return None
    try:
        total_mib, used_mib = (int(part.strip()) for part in raw.splitlines()[0].split(","))
    except (ValueError, IndexError):
        return None
    result = (total_mib * 1024 * 1024, used_mib * 1024 * 1024)
    _gpu_cache = (now, result)
    return result


@lru_cache(maxsize=1)
def _nvidia_name() -> str:
    raw = _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"])
    return raw.splitlines()[0].strip() if raw else "NVIDIA GPU"


@lru_cache(maxsize=1)
def total_memory() -> int:
    """Total physical RAM in bytes. 0 when it cannot be determined."""
    # sysconf is the portable path and works on both macOS and Linux.
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        pass
    if sys.platform == "darwin":
        raw = _run(["sysctl", "-n", "hw.memsize"])
        if raw and raw.isdigit():
            return int(raw)
    return 0


@lru_cache(maxsize=1)
def _cpu_name() -> str:
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or "Apple Silicon"
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return "CPU"


def current_rss(pid: int | None = None) -> int:
    """Current resident set size in bytes for ``pid`` (default: this process).

    Deliberately *current* rather than peak. ``resource.getrusage`` reports
    ``ru_maxrss``, a high-water mark that only ever climbs — plotted over a
    training run it draws a monotonic staircase that hides the real allocation
    pattern. It also silently changes units by platform (bytes on macOS, KiB on
    Linux), which is its own bug factory.
    """
    target = os.getpid() if pid is None else pid

    # Linux: statm field 2 is resident pages. No subprocess, safe to call often.
    statm = Path(f"/proc/{target}/statm")
    try:
        resident_pages = int(statm.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE")
    except (OSError, IndexError, ValueError):
        pass

    # macOS has no /proc; ps reports RSS in KiB.
    raw = _run(["ps", "-o", "rss=", "-p", str(target)])
    if raw and raw.strip().isdigit():
        return int(raw.strip()) * 1024
    return 0


# ── public ───────────────────────────────────────────────────────────────────


def probe_device(pid: int | None = None) -> DeviceInfo:
    """Describe this host's memory, labelled for what it actually is.

    ``pid`` selects whose RSS to report on CPU hosts — the gateway passes the
    training subprocess's pid so the dashboard graphs the trainer, not itself.
    """
    gpu = _nvidia_memory()
    if gpu is not None:
        total, used = gpu
        return DeviceInfo(
            kind="cuda",
            device=_nvidia_name(),
            memory_label="VRAM",
            total_bytes=total,
            used_bytes=used,
            source="nvidia-smi",
        )

    total = total_memory()
    # Apple Silicon has no discrete VRAM: the GPU and CPU share one pool, so
    # "unified memory" is the accurate name for what a model occupies here.
    apple = sys.platform == "darwin" and os.uname().machine == "arm64"
    return DeviceInfo(
        kind="apple-silicon" if apple else "cpu",
        device=_cpu_name(),
        memory_label="Unified memory" if apple else "RAM",
        total_bytes=total,
        used_bytes=current_rss(pid),
        source="sysconf + ps" if sys.platform == "darwin" else "sysconf + /proc",
    )
