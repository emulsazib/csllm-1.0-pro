"""Supervise a long-running job subprocess and fan its output out to WebSocket
clients.

Two job kinds share this machinery (see ``JOBS``): ``train`` runs the training
loop, ``prepare`` trains a BPE tokenizer and binarizes a dataset. They differ
only in which module is spawned and which flags it takes — the pump, broadcast,
history and back-pressure logic is identical, and duplicating it for the sake of
a second entrypoint would mean two places to get back-pressure wrong.

Design notes, in the order they matter:

* **The subprocess stays standalone.** It gains only ``--emit-jsonl``; the
  gateway reads its stdout. Nothing in the training loop knows the gateway
  exists, so training still works with the gateway down.
* **A slow client must never stall training.** Each subscriber gets a bounded
  ``asyncio.Queue``; when it fills, that subscriber's oldest events are dropped
  and a ``dropped`` counter is surfaced. Back-pressuring the pipe instead would
  block the trainer on a browser tab that stopped reading.
* **History is replayed on connect.** A dashboard opened mid-run needs the loss
  curve so far, not just the next point, so a bounded ring buffer of recent
  events is replayed before live streaming begins.
* **One run at a time.** The engine holds a single model and the machine has one
  set of cores; a second concurrent run would just make both slower.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import shlex
import signal
import sys
import time
import uuid
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["JOBS", "RunAlreadyActive", "RunHandle", "TrainingSupervisor"]

RUNS_DIR = Path("runs")

#: Job kind -> (module to run, option name -> CLI flag).
#:
#: Both entrypoints accept ``--emit-jsonl``/``--run-id``, which is the entire
#: contract the supervisor depends on. Adding a kind means adding a row here,
#: not another supervisor.
JOBS: dict[str, tuple[str, dict[str, str]]] = {
    "train": (
        "train.train",
        {
            "config": "--config",
            "steps": "--steps",
            "batch_size": "--batch-size",
            "lr": "--lr",
            "min_lr": "--min-lr",
            "warmup": "--warmup",
            "weight_decay": "--weight-decay",
            "grad_clip": "--grad-clip",
            "eval_every": "--eval-every",
            "eval_iters": "--eval-iters",
            "sample_every": "--sample-every",
            "seed": "--seed",
            "out": "--out",
            "data_dir": "--data-dir",
            "tokenizer_dir": "--tokenizer-dir",
        },
    ),
    "prepare": (
        "train.train_tokenizer",
        {
            "config": "--config",
            "dataset": "--dataset",
            "corpus": "--corpus",
            "out": "--out",
            "data_dir": "--data-dir",
            "val_fraction": "--val-fraction",
        },
    ),
}

#: Per-subscriber queue depth. Deep enough to absorb a burst of step events,
#: shallow enough that a dead client cannot pin much memory.
SUBSCRIBER_QUEUE_SIZE = 512

#: Events replayed to a client that connects mid-run.
HISTORY_SIZE = 2000


class RunAlreadyActive(RuntimeError):
    """A training run is already in flight."""


@dataclass
class RunHandle:
    run_id: str
    command: list[str]
    started_at: float
    kind: str = "train"
    process: asyncio.subprocess.Process | None = None
    returncode: int | None = None
    finished_at: float | None = None
    last_step: int = 0
    total_steps: int = 0
    last_loss: float | None = None
    best_val: float | None = None
    stopping: bool = False
    #: SIGSTOPped. Still `running` — the process exists and holds its memory.
    paused: bool = False

    @property
    def running(self) -> bool:
        return self.returncode is None and self.process is not None

    def status(self) -> dict[str, Any]:
        elapsed = (self.finished_at or time.time()) - self.started_at
        progress = (self.last_step / self.total_steps) if self.total_steps else 0.0
        return {
            "run_id": self.run_id,
            "kind": self.kind,
            "running": self.running,
            "paused": self.paused,
            "command": " ".join(shlex.quote(c) for c in self.command),
            "started_at": self.started_at,
            "elapsed_s": elapsed,
            "returncode": self.returncode,
            "step": self.last_step,
            "total_steps": self.total_steps,
            "progress": round(progress, 4),
            "last_loss": self.last_loss,
            "best_val": self.best_val,
            "pid": self.process.pid if self.process is not None else None,
        }


# eq=False keeps identity hashing, so instances can live in a set. The generated
# __eq__ would set __hash__ = None and make them unhashable.
@dataclass(eq=False)
class _Subscriber:
    queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(SUBSCRIBER_QUEUE_SIZE))
    dropped: int = 0


class TrainingSupervisor:
    """Owns at most one training subprocess and broadcasts its events."""

    def __init__(self, runs_dir: str | Path = RUNS_DIR, python: str | None = None) -> None:
        self.runs_dir = Path(runs_dir)
        self.python = python or sys.executable
        self._run: RunHandle | None = None
        self._subscribers: set[_Subscriber] = set()
        self._history: deque[dict] = deque(maxlen=HISTORY_SIZE)
        self._reader: asyncio.Task | None = None
        self._log_file = None

    # ── state ────────────────────────────────────────────────────────────────

    @property
    def run(self) -> RunHandle | None:
        return self._run

    def status(self) -> dict[str, Any]:
        if self._run is None:
            return {
                "run_id": None,
                "kind": None,
                "running": False,
                "paused": False,
                "subscribers": len(self._subscribers),
            }
        return {**self._run.status(), "subscribers": len(self._subscribers)}

    # ── lifecycle ────────────────────────────────────────────────────────────

    async def start(self, options: dict[str, Any], run_id: str | None = None) -> RunHandle:
        if self._run is not None and self._run.running:
            raise RunAlreadyActive(
                f"run {self._run.run_id} is still active; stop it before starting another"
            )

        kind = options.get("kind", "train")
        if kind not in JOBS:
            raise ValueError(f"unknown job kind {kind!r}; expected one of {sorted(JOBS)}")

        run_id = run_id or f"{kind}-{uuid.uuid4().hex[:8]}"
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(options, run_id)
        self._history.clear()

        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,  # interleave, so ordering is preserved
        )
        handle = RunHandle(
            run_id=run_id,
            kind=kind,
            command=command,
            started_at=time.time(),
            process=process,
            total_steps=int(options.get("steps", 0) or 0),
        )
        self._run = handle
        self._log_file = (run_dir / "stdout.log").open("w", encoding="utf-8")
        self._reader = asyncio.create_task(self._pump(handle, run_dir))
        return handle

    def _build_command(self, options: dict[str, Any], run_id: str) -> list[str]:
        module, flags = JOBS[options.get("kind", "train")]
        cmd = [self.python, "-u", "-m", module, "--emit-jsonl", "--run-id", run_id]
        for key, flag in flags.items():
            value = options.get(key)
            if value is not None:
                cmd += [flag, str(value)]
        if options.get("resume") and "resume" not in flags:
            cmd.append("--resume")
        return cmd

    # ── pause / resume ───────────────────────────────────────────────────────

    async def pause(self) -> bool:
        """SIGSTOP the job. Returns False if there is nothing running to pause.

        SIGSTOP rather than checkpoint-and-restart: the run keeps its arena, its
        Adam moments and its step counter, so resuming is exact and free. The
        cost is that the process still holds its memory while paused, which is
        the right trade for "let me look at the loss curve for a second".
        """
        run = self._run
        if run is None or not run.running or run.process is None or run.paused:
            return False
        run.process.send_signal(signal.SIGSTOP)
        run.paused = True
        self._broadcast({"type": "paused", "run_id": run.run_id, "step": run.last_step})
        return True

    async def resume(self) -> bool:
        run = self._run
        if run is None or not run.running or run.process is None or not run.paused:
            return False
        run.process.send_signal(signal.SIGCONT)
        run.paused = False
        self._broadcast({"type": "resumed", "run_id": run.run_id, "step": run.last_step})
        return True

    async def stop(self) -> bool:
        """Terminate the active run. Returns False if there was nothing to stop."""
        run = self._run
        if run is None or not run.running or run.process is None:
            return False
        run.stopping = True
        # A SIGSTOPped process never reaches its SIGTERM handler, so terminating
        # a paused run would hang until the 10s timeout and then need SIGKILL.
        if run.paused:
            run.process.send_signal(signal.SIGCONT)
            run.paused = False
        run.process.terminate()
        try:
            await asyncio.wait_for(run.process.wait(), timeout=10)
        except (TimeoutError, asyncio.TimeoutError):
            run.process.kill()  # ignored SIGTERM; do not leave an orphan burning cores
            await run.process.wait()

        # The pump task finalises returncode/finished_at in its `finally`, and it
        # has not run yet at this point. Without waiting, status() immediately
        # after stop() still reports running=True — a race the UI would render as
        # a run that refuses to die.
        await self._await_reader()
        return True

    async def _await_reader(self, timeout: float = 5.0) -> None:
        if self._reader is None or self._reader.done():
            return
        with contextlib.suppress(asyncio.TimeoutError, TimeoutError, asyncio.CancelledError):
            await asyncio.wait_for(asyncio.shield(self._reader), timeout=timeout)

    async def shutdown(self) -> None:
        await self.stop()
        if self._reader is not None:
            self._reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._reader

    # ── output pump ──────────────────────────────────────────────────────────

    async def _pump(self, handle: RunHandle, run_dir: Path) -> None:
        metrics_path = run_dir / "metrics.jsonl"
        assert handle.process is not None and handle.process.stdout is not None
        try:
            with metrics_path.open("w", encoding="utf-8") as metrics:
                async for raw in handle.process.stdout:
                    line = raw.decode("utf-8", errors="replace").rstrip("\n")
                    if self._log_file is not None:
                        self._log_file.write(line + "\n")
                        self._log_file.flush()

                    event = self._parse(line)
                    if event["type"] != "log":
                        metrics.write(json.dumps(event) + "\n")
                        metrics.flush()
                        self._apply(handle, event)
                    self._broadcast(event)

            await handle.process.wait()
            handle.returncode = handle.process.returncode
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # pragma: no cover - defensive
            self._broadcast({"type": "error", "message": f"supervisor failure: {exc}"})
            handle.returncode = -1
        finally:
            handle.finished_at = time.time()
            if handle.returncode is None:
                handle.returncode = handle.process.returncode if handle.process else -1
            if self._log_file is not None:
                self._log_file.close()
                self._log_file = None
            self._broadcast(
                {
                    "type": "exit",
                    "run_id": handle.run_id,
                    "returncode": handle.returncode,
                    "stopped": handle.stopping,
                    "elapsed_s": handle.finished_at - handle.started_at,
                }
            )

    @staticmethod
    def _parse(line: str) -> dict:
        """A parseable JSON object is a metric row; anything else is log text."""
        stripped = line.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                event = json.loads(stripped)
                if isinstance(event, dict) and "type" in event:
                    return event
            except json.JSONDecodeError:
                pass
        return {"type": "log", "message": line}

    @staticmethod
    def _apply(handle: RunHandle, event: dict) -> None:
        kind = event.get("type")
        if kind == "start":
            handle.total_steps = int(event.get("steps", handle.total_steps))
        elif kind == "step":
            handle.last_step = int(event.get("step", handle.last_step))
            handle.last_loss = event.get("loss")
        elif kind == "eval":
            handle.best_val = event.get("best_val", handle.best_val)

    def _broadcast(self, event: dict) -> None:
        self._history.append(event)
        for sub in self._subscribers:
            try:
                sub.queue.put_nowait(event)
            except asyncio.QueueFull:
                # Drop the oldest so a stalled client loses history, not liveness.
                with contextlib.suppress(asyncio.QueueEmpty):
                    sub.queue.get_nowait()
                sub.dropped += 1
                with contextlib.suppress(asyncio.QueueFull):
                    sub.queue.put_nowait(event)

    # ── subscription ─────────────────────────────────────────────────────────

    async def subscribe(self, replay: bool = True) -> AsyncIterator[dict]:
        """Yield events until the subscriber disconnects.

        Replays recent history first so a dashboard opened mid-run still draws
        the whole curve.
        """
        sub = _Subscriber()
        self._subscribers.add(sub)
        try:
            if replay:
                for event in list(self._history):
                    yield event
            while True:
                event = await sub.queue.get()
                if sub.dropped:
                    yield {"type": "dropped", "count": sub.dropped}
                    sub.dropped = 0
                yield event
        finally:
            self._subscribers.discard(sub)
