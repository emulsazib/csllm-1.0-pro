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
  runId: string | null;
  step: number;
  totalSteps: number;
  train: LossPoint[];
  val: EvalPoint[];
  lr: LossPoint[];
  gradNorm: LossPoint[];
  logs: string[];
  bestVal: number | null;
  tokensPerSecond: number | null;
  dropped: number;
  exitCode: number | null;
}

const EMPTY: TrainingState = {
  connected: false,
  running: false,
  runId: null,
  step: 0,
  totalSteps: 0,
  train: [],
  val: [],
  lr: [],
  gradNorm: [],
  logs: [],
  bestVal: null,
  tokensPerSecond: null,
  dropped: 0,
  exitCode: null,
};

/** Keep the log console bounded — a long run emits tens of thousands of lines. */
const MAX_LOGS = 400;

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
    setState((prev) => {
      switch (event.type) {
        case "status":
          return {
            ...prev,
            running: Boolean(event.running),
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
            runId: (event.run_id as string) ?? null,
            totalSteps: (event.steps as number) ?? 0,
            logs: prev.logs,
          };
        case "step": {
          const step = event.step ?? 0;
          return {
            ...prev,
            running: true,
            step,
            train: [...prev.train, { step, loss: event.loss ?? 0 }],
            lr: event.lr === undefined ? prev.lr : [...prev.lr, { step, loss: event.lr }],
            gradNorm:
              event.grad_norm === undefined
                ? prev.gradNorm
                : [...prev.gradNorm, { step, loss: event.grad_norm }],
          };
        }
        case "throughput":
          return { ...prev, tokensPerSecond: (event.tokens_per_s as number) ?? null };
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
          return { ...prev, dropped: prev.dropped + ((event.count as number) ?? 0) };
        case "exit":
          return {
            ...prev,
            running: false,
            exitCode: (event.returncode as number) ?? null,
            logs: [
              ...prev.logs,
              `— run finished (exit ${event.returncode}) —`,
            ].slice(-MAX_LOGS),
          };
        case "done":
          return { ...prev, running: false, bestVal: event.best_val ?? prev.bestVal };
        case "error":
          return {
            ...prev,
            logs: [...prev.logs, `error: ${event.message}`].slice(-MAX_LOGS),
          };
        default:
          return prev;
      }
    });
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const socket = new WebSocket(`${WS_BASE}/ws/train`);
    socketRef.current = socket;

    socket.onopen = () => setState((prev) => ({ ...prev, connected: true }));
    socket.onmessage = (event) => apply(JSON.parse(event.data as string) as TrainEvent);
    socket.onclose = () => setState((prev) => ({ ...prev, connected: false }));
    socket.onerror = () => setState((prev) => ({ ...prev, connected: false }));

    return () => socket.close();
  }, [enabled, apply]);

  return state;
}
