import { useCallback, useEffect, useRef, useState } from "react";
import { type TrainEvent, WS_BASE } from "../api/ws";

export interface LossPoint {
  step: number;
  loss: number;
}
export interface EvalPoint {
  step: number;
  val: number;
}

export interface TrainingState {
  connected: boolean;
  running: boolean;
  /** SIGSTOPped: resident and resumable, not consuming CPU. */
  paused: boolean;
  /** "train" | "prepare" — a prepare job has no loss curve, only stages. */
  kind: string | null;
  runId: string | null;
  step: number;
  totalSteps: number;
  train: LossPoint[];
  val: EvalPoint[];
  lr: LossPoint[];
  gradNorm: LossPoint[];
  /** Trainer RSS over time, sampled with each throughput row. */
  memory: LossPoint[];
  logs: string[];
  bestVal: number | null;
  tokensPerSecond: number | null;
  epoch: number | null;
  /** Host memory facts from the run's `start` row. */
  memoryLabel: string | null;
  memoryTotal: number | null;
  device: string | null;
  /** Current stage of a prepare job. */
  stage: string | null;
  dropped: number;
  exitCode: number | null;
}

export const EMPTY_TRAINING_STATE: TrainingState = {
  connected: false,
  running: false,
  paused: false,
  kind: null,
  runId: null,
  step: 0,
  totalSteps: 0,
  train: [],
  val: [],
  lr: [],
  gradNorm: [],
  memory: [],
  logs: [],
  bestVal: null,
  tokensPerSecond: null,
  epoch: null,
  memoryLabel: null,
  memoryTotal: null,
  device: null,
  stage: null,
  dropped: 0,
  exitCode: null,
};

/** Keep the log console bounded — a long run emits tens of thousands of lines. */
const MAX_LOGS = 400;

const EMPTY = EMPTY_TRAINING_STATE;

/**
 * Fold one event into the dashboard state.
 *
 * Pure and exported so the reducer can be tested directly — the interesting
 * cases (a prepare job sharing the socket, pause/resume, a new run clearing the
 * previous curve) are all state transitions, not rendering.
 */
export function applyTrainEvent(
  prev: TrainingState,
  event: TrainEvent,
): TrainingState {
  switch (event.type) {
    case "status":
      return {
        ...prev,
        running: Boolean(event.running),
        paused: Boolean(event.paused),
        kind: (event.kind as string) ?? prev.kind,
        runId: (event.run_id as string) ?? null,
        step: (event.step as number) ?? prev.step,
        totalSteps: (event.total_steps as number) ?? prev.totalSteps,
      };
    case "start":
      // A new run clears the previous curve rather than appending to it.
      return {
        ...EMPTY,
        connected: true,
        running: true,
        // A prepare job emits `stage` instead of `steps`; treat the presence
        // of a stage as the marker so the UI can drop the loss chart.
        kind: event.stage ? "prepare" : "train",
        stage: (event.stage as string) ?? null,
        runId: (event.run_id as string) ?? null,
        totalSteps: (event.steps as number) ?? 0,
        memoryLabel: (event.memory_label as string) ?? null,
        memoryTotal: (event.memory_total_bytes as number) ?? null,
        device: (event.device as string) ?? null,
        logs: prev.logs,
      };
    case "stage":
      return {
        ...prev,
        running: true,
        stage: (event.stage as string) ?? prev.stage,
      };
    case "paused":
      return { ...prev, paused: true };
    case "resumed":
      return { ...prev, paused: false };
    case "step": {
      const step = event.step ?? 0;
      return {
        ...prev,
        running: true,
        step,
        train: [...prev.train, { step, loss: event.loss ?? 0 }],
        lr:
          event.lr === undefined
            ? prev.lr
            : [...prev.lr, { step, loss: event.lr }],
        gradNorm:
          event.grad_norm === undefined
            ? prev.gradNorm
            : [...prev.gradNorm, { step, loss: event.grad_norm }],
      };
    }
    case "throughput":
      return {
        ...prev,
        tokensPerSecond: (event.tokens_per_s as number) ?? null,
        epoch: (event.epoch as number) ?? prev.epoch,
        // A non-positive sample means the probe failed, not that the trainer
        // freed everything — plotting it draws a cliff to zero.
        memory:
          typeof event.rss_bytes !== "number" || event.rss_bytes <= 0
            ? prev.memory
            : [
                ...prev.memory,
                { step: event.step ?? 0, loss: event.rss_bytes },
              ],
      };
    case "eval":
      return {
        ...prev,
        val: [...prev.val, { step: event.step ?? 0, val: event.val_loss ?? 0 }],
        bestVal: event.best_val ?? prev.bestVal,
      };
    case "log":
      return {
        ...prev,
        logs: [...prev.logs, String(event.message ?? "")].slice(-MAX_LOGS),
      };
    case "dropped":
      return {
        ...prev,
        dropped: prev.dropped + ((event.count as number) ?? 0),
      };
    case "exit":
      return {
        ...prev,
        running: false,
        paused: false,
        exitCode: (event.returncode as number) ?? null,
        logs: [
          ...prev.logs,
          `— run finished (exit ${event.returncode}) —`,
        ].slice(-MAX_LOGS),
      };
    case "done":
      return {
        ...prev,
        running: false,
        bestVal: event.best_val ?? prev.bestVal,
        stage: (event.stage as string) ?? prev.stage,
      };
    case "error":
      return {
        ...prev,
        logs: [...prev.logs, `error: ${event.message}`].slice(-MAX_LOGS),
      };
    default:
      return prev;
  }
}

/**
 * Subscribe to WS /ws/train.
 *
 * The server replays recent history on connect, so a dashboard opened mid-run
 * still draws the whole curve rather than starting from the next point.
 */
export function useTrainingStream(enabled = true) {
  const [state, setState] = useState<TrainingState>(EMPTY);
  const socketRef = useRef<WebSocket | null>(null);

  const apply = useCallback((event: TrainEvent) => {
    setState((prev) => applyTrainEvent(prev, event));
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const socket = new WebSocket(`${WS_BASE}/ws/train`);
    socketRef.current = socket;

    socket.onopen = () => setState((prev) => ({ ...prev, connected: true }));
    socket.onmessage = (event) =>
      apply(JSON.parse(event.data as string) as TrainEvent);
    socket.onclose = () => setState((prev) => ({ ...prev, connected: false }));
    socket.onerror = () => setState((prev) => ({ ...prev, connected: false }));

    return () => socket.close();
  }, [enabled, apply]);

  return state;
}
