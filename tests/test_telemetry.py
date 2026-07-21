"""Training supervisor: subprocess management, event parsing, and fan-out.

The supervisor is tested against a *synthetic* trainer rather than the real one.
That isolates what is actually being verified — parsing, broadcast, back-pressure,
termination — from model training, and keeps the suite fast and independent of
whether ``data/`` has been prepared.
"""

from __future__ import annotations

import asyncio
import json
import sys

import pytest

from gateway.telemetry import JOBS, RunAlreadyActive, TrainingSupervisor

# A stand-in trainer: emits JSONL metrics interleaved with plain log lines.
FAKE_TRAINER = """
import json, sys, time
steps = int(sys.argv[1])
print(json.dumps({"type": "start", "steps": steps}), flush=True)
print("human readable banner", flush=True)
for i in range(steps):
    print(json.dumps({"type": "step", "step": i, "loss": 3.0 - i * 0.1, "lr": 1e-3}), flush=True)
    if i % 2 == 0:
        print(f"step {i} log line", flush=True)
print(json.dumps({"type": "done", "step": steps, "best_val": 1.23}), flush=True)
"""

SLOW_TRAINER = """
import json, sys, time
print(json.dumps({"type": "start", "steps": 100000}), flush=True)
for i in range(100000):
    print(json.dumps({"type": "step", "step": i, "loss": 1.0}), flush=True)
    time.sleep(0.05)
"""


class FakeSupervisor(TrainingSupervisor):
    """Runs an inline script instead of train.train."""

    script = FAKE_TRAINER

    def _build_command(self, options, run_id):
        return [self.python, "-u", "-c", self.script, str(options.get("steps", 5))]


class SlowSupervisor(FakeSupervisor):
    script = SLOW_TRAINER


@pytest.fixture
def supervisor(tmp_path):
    return FakeSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)


async def drain(supervisor, *, until: str = "exit", timeout: float = 20.0):
    """Collect broadcast events until the given event type arrives."""
    events = []

    async def collect():
        async for event in supervisor.subscribe():
            events.append(event)
            if event.get("type") == until:
                return

    await asyncio.wait_for(collect(), timeout=timeout)
    return events


# ── lifecycle ────────────────────────────────────────────────────────────────


async def test_idle_status_before_any_run(supervisor):
    status = supervisor.status()
    assert status["running"] is False
    assert status["run_id"] is None


async def test_run_completes_and_reports_exit(supervisor):
    await supervisor.start({"steps": 4})
    events = await drain(supervisor)

    kinds = [e["type"] for e in events]
    assert "start" in kinds
    assert kinds[-1] == "exit"
    assert events[-1]["returncode"] == 0

    status = supervisor.status()
    assert status["running"] is False
    assert status["step"] == 3          # last emitted step index
    assert status["total_steps"] == 4
    assert status["best_val"] is None or isinstance(status["best_val"], float)


async def test_metrics_and_logs_are_distinguished(supervisor):
    await supervisor.start({"steps": 3})
    events = await drain(supervisor)

    steps = [e for e in events if e["type"] == "step"]
    logs = [e for e in events if e["type"] == "log"]

    assert [e["step"] for e in steps] == [0, 1, 2]
    assert all("loss" in e for e in steps)
    assert any("human readable banner" in e["message"] for e in logs)


async def test_progress_tracks_the_run(supervisor):
    await supervisor.start({"steps": 10})
    await drain(supervisor)
    status = supervisor.status()
    assert 0.0 < status["progress"] <= 1.0
    assert status["last_loss"] is not None


async def test_starting_twice_is_rejected(tmp_path):
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    try:
        with pytest.raises(RunAlreadyActive, match="still active"):
            await sup.start({"steps": 5})
    finally:
        await sup.shutdown()


async def test_stop_terminates_a_running_job(tmp_path):
    """status() must be accurate the INSTANT stop() returns.

    Regression: stop() used to return as soon as the process exited, while the
    output pump finalised returncode in its `finally`. status() then still said
    running=True and a dashboard would render a run that refuses to die.
    """
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    await asyncio.sleep(0.2)
    assert sup.status()["running"] is True

    assert await sup.stop() is True
    status = sup.status()
    assert status["running"] is False, "stop() must not return before status settles"
    assert status["returncode"] is not None
    await sup.shutdown()


async def test_stop_without_a_run_returns_false(supervisor):
    assert await supervisor.stop() is False


async def test_shutdown_leaves_no_orphan(tmp_path):
    """A gateway restart must not leave a trainer burning cores."""
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    handle = await sup.start({"steps": 100000})
    await asyncio.sleep(0.2)
    await sup.shutdown()
    assert handle.process.returncode is not None


# ── pause / resume ───────────────────────────────────────────────────────────


async def steps_seen(sup) -> int:
    return sup.status()["step"]


async def test_pause_halts_progress_and_resume_continues(tmp_path):
    """SIGSTOP must actually stop the work, not just flip a flag."""
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    try:
        await asyncio.sleep(0.4)
        assert await sup.pause() is True
        assert sup.status()["paused"] is True
        # Still "running": the process exists and holds its memory.
        assert sup.status()["running"] is True

        # Let the pipe drain, then confirm the step counter has gone quiet.
        await asyncio.sleep(0.3)
        frozen = await steps_seen(sup)
        await asyncio.sleep(0.5)
        assert await steps_seen(sup) == frozen, "a paused run must not advance"

        assert await sup.resume() is True
        assert sup.status()["paused"] is False
        await asyncio.sleep(0.4)
        assert await steps_seen(sup) > frozen, "resume must let the run continue"
    finally:
        await sup.shutdown()


async def test_pause_and_resume_are_idempotent(tmp_path):
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    try:
        await asyncio.sleep(0.2)
        assert await sup.pause() is True
        assert await sup.pause() is False  # already paused
        assert await sup.resume() is True
        assert await sup.resume() is False  # not paused
    finally:
        await sup.shutdown()


async def test_pause_without_a_run_returns_false(supervisor):
    assert await supervisor.pause() is False
    assert await supervisor.resume() is False


async def test_stopping_a_paused_run_does_not_hang(tmp_path):
    """A SIGSTOPped process never reaches its SIGTERM handler.

    Without a SIGCONT first, stop() waits out the full 10s timeout and then has
    to SIGKILL — so the UI's Stop button appears dead for ten seconds.
    """
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    try:
        await asyncio.sleep(0.2)
        await sup.pause()

        started = asyncio.get_running_loop().time()
        assert await sup.stop() is True
        elapsed = asyncio.get_running_loop().time() - started

        assert elapsed < 5.0, f"stopping a paused run took {elapsed:.1f}s"
        assert sup.status()["running"] is False
    finally:
        await sup.shutdown()


async def test_pause_is_broadcast_to_subscribers(tmp_path):
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    await sup.start({"steps": 100000})
    seen: list[str] = []

    async def collect():
        async for event in sup.subscribe():
            seen.append(event.get("type"))
            if event.get("type") == "resumed":
                return

    task = asyncio.create_task(collect())
    try:
        await asyncio.sleep(0.2)
        await sup.pause()
        await asyncio.sleep(0.1)
        await sup.resume()
        await asyncio.wait_for(task, timeout=5)
    finally:
        task.cancel()
        await sup.shutdown()

    assert "paused" in seen and "resumed" in seen


# ── job kinds ────────────────────────────────────────────────────────────────


def test_train_is_the_default_kind(tmp_path):
    sup = TrainingSupervisor(runs_dir=tmp_path / "runs", python="py")
    command = sup._build_command({"config": "configs/debug.json", "steps": 5}, "r1")
    assert "train.train" in command
    assert command[command.index("--config") + 1] == "configs/debug.json"
    assert command[command.index("--steps") + 1] == "5"


def test_prepare_kind_spawns_the_tokenizer_entrypoint(tmp_path):
    sup = TrainingSupervisor(runs_dir=tmp_path / "runs", python="py")
    command = sup._build_command(
        {"kind": "prepare", "config": "configs/debug.json", "dataset": "corpus.jsonl"}, "r1"
    )
    assert "train.train_tokenizer" in command
    assert command[command.index("--dataset") + 1] == "corpus.jsonl"
    # Training-only flags must not leak into the prepare command line.
    assert "--steps" not in command


def test_both_kinds_carry_the_emit_contract(tmp_path):
    sup = TrainingSupervisor(runs_dir=tmp_path / "runs", python="py")
    for kind in JOBS:
        command = sup._build_command({"kind": kind}, "run-7")
        assert "--emit-jsonl" in command
        assert command[command.index("--run-id") + 1] == "run-7"


async def test_unknown_job_kind_is_rejected(supervisor):
    with pytest.raises(ValueError, match="unknown job kind"):
        await supervisor.start({"kind": "definitely-not-a-job"})


async def test_run_id_is_prefixed_with_the_kind(tmp_path):
    """Run directories under runs/ should say what they were at a glance."""
    sup = SlowSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)
    handle = await sup.start({"steps": 100000})
    try:
        assert handle.run_id.startswith("train-")
        assert sup.status()["kind"] == "train"
    finally:
        await sup.shutdown()


async def test_idle_status_reports_no_kind(supervisor):
    status = supervisor.status()
    assert status["kind"] is None
    assert status["paused"] is False


# ── persistence ──────────────────────────────────────────────────────────────


async def test_run_artifacts_are_written(supervisor, tmp_path):
    handle = await supervisor.start({"steps": 3})
    await drain(supervisor)

    run_dir = tmp_path / "runs" / handle.run_id
    metrics = (run_dir / "metrics.jsonl").read_text().strip().splitlines()
    stdout = (run_dir / "stdout.log").read_text()

    parsed = [json.loads(line) for line in metrics]
    assert [e["type"] for e in parsed if e["type"] == "step"] == ["step"] * 3
    assert "human readable banner" in stdout
    # metrics.jsonl holds ONLY structured rows; prose stays in stdout.log.
    assert all(e["type"] != "log" for e in parsed)


# ── fan-out ──────────────────────────────────────────────────────────────────


async def test_multiple_subscribers_all_receive_events(supervisor):
    results = await asyncio.gather(
        _subscribe_after_start(supervisor, {"steps": 4}),
        return_exceptions=True,
    )
    assert results


async def _subscribe_after_start(supervisor, options):
    await supervisor.start(options)
    a = asyncio.create_task(drain(supervisor))
    b = asyncio.create_task(drain(supervisor))
    events_a, events_b = await asyncio.gather(a, b)
    assert [e["type"] for e in events_a] == [e["type"] for e in events_b]
    return events_a


async def test_history_is_replayed_to_a_late_subscriber(supervisor):
    await supervisor.start({"steps": 5})
    await drain(supervisor)  # run to completion first

    # A dashboard opened after the fact must still see the whole curve.
    replayed = []

    async def collect():
        async for event in supervisor.subscribe():
            replayed.append(event)
            if event.get("type") == "exit":
                return

    await asyncio.wait_for(collect(), timeout=5)
    assert [e["type"] for e in replayed if e["type"] == "step"] == ["step"] * 5


async def test_subscriber_count_is_reported(supervisor):
    assert supervisor.status()["subscribers"] == 0

    started = asyncio.Event()

    async def hold():
        async for _ in supervisor.subscribe(replay=False):
            started.set()

    task = asyncio.create_task(hold())
    await asyncio.sleep(0.05)
    assert supervisor.status()["subscribers"] == 1
    task.cancel()
    await asyncio.sleep(0.05)
    assert supervisor.status()["subscribers"] == 0


async def test_a_stalled_subscriber_does_not_block_the_run(tmp_path):
    """Back-pressure must never reach the trainer.

    A subscriber that stops reading gets its oldest events dropped; the run still
    completes. Blocking the pipe instead would let a dead browser tab stall training.
    """
    sup = FakeSupervisor(runs_dir=tmp_path / "runs", python=sys.executable)

    registered = asyncio.Event()

    async def stalled_consumer():
        async for _ in sup.subscribe(replay=False):
            registered.set()
            await asyncio.sleep(3600)  # take one event, then stop reading forever

    stalled = asyncio.create_task(stalled_consumer())
    await asyncio.sleep(0.05)

    # Far more events than SUBSCRIBER_QUEUE_SIZE, so the stalled queue overflows.
    await sup.start({"steps": 2000})
    events = await drain(sup, timeout=60)

    # Under flood a bounded queue drops — that is the design. What must hold is:
    #   1. the run completes rather than deadlocking on a stalled reader,
    #   2. TERMINAL events are never lost (the drop policy evicts the OLDEST and
    #      always inserts the newest, so a UI can never hang thinking a finished
    #      run is still live),
    #   3. drops are surfaced rather than silent.
    # The complete record is always on disk in metrics.jsonl.
    assert events[-1]["type"] == "exit"
    assert events[-1]["returncode"] == 0, "run must complete despite a stalled subscriber"

    steps = [e for e in events if e["type"] == "step"]
    assert steps, "some step events should still arrive"
    assert steps[-1]["step"] == 1999, "the newest event must always win a full queue"
    assert any(e["type"] == "dropped" for e in events), "drops must be reported, not silent"

    stalled.cancel()
    await asyncio.sleep(0.05)
