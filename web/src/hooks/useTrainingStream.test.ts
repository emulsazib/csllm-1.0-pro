/** The /ws/train reducer.
 *
 *  Two job kinds share this socket now, and pause is a state the UI has to get
 *  right in both directions — those are state transitions, so they are testable
 *  without rendering anything.
 */

import { describe, expect, it } from "vitest";
import type { TrainEvent } from "../api/ws";
import { applyTrainEvent, EMPTY_TRAINING_STATE, type TrainingState } from "./useTrainingStream";

function fold(events: TrainEvent[], from: TrainingState = EMPTY_TRAINING_STATE): TrainingState {
  return events.reduce(applyTrainEvent, from);
}

describe("run lifecycle", () => {
  it("records host memory facts from the start row", () => {
    const state = fold([
      {
        type: "start",
        steps: 300,
        device: "Apple M1",
        memory_label: "Unified memory",
        memory_total_bytes: 8_589_934_592,
      },
    ]);
    expect(state.kind).toBe("train");
    expect(state.running).toBe(true);
    expect(state.memoryLabel).toBe("Unified memory");
    expect(state.memoryTotal).toBe(8_589_934_592);
    expect(state.device).toBe("Apple M1");
  });

  it("clears the previous curve when a new run starts but keeps the logs", () => {
    const first = fold([
      { type: "start", steps: 10 },
      { type: "step", step: 0, loss: 3 },
      { type: "log", message: "banner" },
    ]);
    expect(first.train).toHaveLength(1);

    const second = applyTrainEvent(first, { type: "start", steps: 10 });
    expect(second.train).toHaveLength(0);
    expect(second.logs).toEqual(["banner"]);
  });

  it("collects epoch and memory from throughput rows", () => {
    const state = fold([
      { type: "start", steps: 100 },
      { type: "throughput", step: 0, tokens_per_s: 21_000, epoch: 0.004, rss_bytes: 54_575_104 },
      { type: "throughput", step: 50, tokens_per_s: 101_000, epoch: 0.212, rss_bytes: 54_738_944 },
    ]);
    expect(state.epoch).toBeCloseTo(0.212);
    expect(state.memory).toEqual([
      { step: 0, loss: 54_575_104 },
      { step: 50, loss: 54_738_944 },
    ]);
  });

  it("ignores a throughput row with no memory sample", () => {
    const state = fold([{ type: "start" }, { type: "throughput", step: 0, tokens_per_s: 5 }]);
    expect(state.memory).toEqual([]);
  });

  it("drops a zero memory sample rather than charting a cliff", () => {
    // Observed live: current_rss() shells out to `ps` on macOS and returned 0
    // once under a saturated training loop. Zero is "not measured".
    const state = fold([
      { type: "start" },
      { type: "throughput", step: 0, rss_bytes: 63_209_472 },
      { type: "throughput", step: 50, rss_bytes: 0 },
      { type: "throughput", step: 100, rss_bytes: 63_242_240 },
    ]);
    expect(state.memory.map((p) => p.step)).toEqual([0, 100]);
  });
});

describe("pause and resume", () => {
  it("round-trips the paused flag", () => {
    const running = fold([{ type: "start", steps: 10 }, { type: "step", step: 0, loss: 3 }]);
    expect(running.paused).toBe(false);

    const paused = applyTrainEvent(running, { type: "paused", step: 0 });
    expect(paused.paused).toBe(true);
    expect(paused.running).toBe(true); // still resident, just stopped

    expect(applyTrainEvent(paused, { type: "resumed", step: 0 }).paused).toBe(false);
  });

  it("clears paused when the run exits", () => {
    const paused = fold([{ type: "start" }, { type: "paused" }]);
    const exited = applyTrainEvent(paused, { type: "exit", returncode: 0 });
    expect(exited.paused).toBe(false);
    expect(exited.running).toBe(false);
    expect(exited.exitCode).toBe(0);
  });

  it("takes paused from a status frame on reconnect", () => {
    const state = applyTrainEvent(EMPTY_TRAINING_STATE, {
      type: "status",
      running: true,
      paused: true,
      kind: "train",
      run_id: "train-abc",
      step: 120,
      total_steps: 300,
    });
    expect(state).toMatchObject({ running: true, paused: true, kind: "train", step: 120 });
  });
});

describe("prepare jobs share the socket", () => {
  it("is identified by the stage field on start, not by steps", () => {
    const state = fold([
      { type: "start", stage: "tokenizer", source: "speeches.jsonl", num_chars: 139_100 },
    ]);
    expect(state.kind).toBe("prepare");
    expect(state.stage).toBe("tokenizer");
    expect(state.totalSteps).toBe(0);
  });

  it("advances through stages", () => {
    const state = fold([
      { type: "start", stage: "tokenizer" },
      { type: "stage", stage: "bpe" },
      { type: "stage", stage: "roundtrip" },
      { type: "stage", stage: "binarize" },
      { type: "done", stage: "complete" },
    ]);
    expect(state.stage).toBe("complete");
    expect(state.running).toBe(false);
  });

  it("produces no loss curve", () => {
    const state = fold([
      { type: "start", stage: "tokenizer" },
      { type: "stage", stage: "bpe" },
      { type: "log", message: "merge 500" },
    ]);
    expect(state.train).toEqual([]);
    expect(state.logs).toEqual(["merge 500"]);
  });
});
