/** Live training: loss curves, throughput, logs, and run control.
 *
 *  The gateway supervises the trainer as a subprocess and fans its stdout out
 *  over WS /ws/train, replaying recent history on connect — so opening this tab
 *  mid-run still draws the whole curve.
 */

import { useEffect, useRef, useState } from "react";
import { API_BASE, ApiError } from "../api/client";
import { useTrainingStream } from "../hooks/useTrainingStream";
import { chartTokens } from "../theme";
import { LossChart, Sparkline } from "./LossChart";

interface StartOptions {
  config: string;
  steps: number;
  batch_size: number;
  lr: number;
  eval_every: number;
  data_dir: string;
  tokenizer_dir: string;
  out: string;
}

const PRESETS: Record<string, StartOptions> = {
  debug: {
    config: "configs/debug.json",
    steps: 300,
    batch_size: 16,
    lr: 3e-3,
    eval_every: 50,
    data_dir: "data/debug",
    tokenizer_dir: "data/tokenizer-debug",
    out: "data/debug/model.csllm",
  },
  shakespeare: {
    config: "configs/shakespeare.json",
    steps: 1500,
    batch_size: 8,
    lr: 1e-3,
    eval_every: 100,
    data_dir: "data",
    tokenizer_dir: "data/tokenizer",
    out: "data/model.csllm",
  },
};

export function TrainingDashboard() {
  const state = useTrainingStream(true);
  const [preset, setPreset] = useState<keyof typeof PRESETS>("debug");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const tokens = chartTokens();

  // Follow the tail, but only while the user is already at the bottom —
  // otherwise scrolling back to read something would keep yanking them away.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [state.logs]);

  async function control(path: string, body?: unknown) {
    setBusy(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: body ? JSON.stringify(body) : undefined,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new ApiError(payload?.detail ?? `${response.status}`, response.status);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const progress = state.totalSteps ? state.step / state.totalSteps : 0;
  const lastLoss = state.train.at(-1)?.loss;

  return (
    <>
      <section className="panel">
        <h2>Training</h2>
        <p className="hint">
          The gateway runs the trainer as a subprocess and streams its metrics and logs.
          Training is independent of the model being served — a crashing run cannot take
          inference down.
        </p>

        <div className="controls">
          <div className="control" style={{ flex: "none" }}>
            <label>Preset</label>
            <div style={{ display: "flex", gap: 4 }}>
              {Object.keys(PRESETS).map((name) => (
                <button
                  key={name}
                  className="ghost"
                  aria-pressed={preset === name}
                  disabled={state.running}
                  onClick={() => setPreset(name as keyof typeof PRESETS)}
                >
                  {name}
                </button>
              ))}
            </div>
            <span className="note">
              {PRESETS[preset].steps} steps · batch {PRESETS[preset].batch_size} · lr{" "}
              {PRESETS[preset].lr}
            </span>
          </div>

          <div className="control" style={{ flex: "none" }}>
            <label>&nbsp;</label>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="action"
                disabled={busy || state.running}
                onClick={() => control("/train/start", PRESETS[preset])}
              >
                Start run
              </button>
              <button
                className="ghost"
                disabled={busy || !state.running}
                onClick={() => control("/train/stop")}
              >
                Stop
              </button>
            </div>
          </div>

          <div className="control">
            <label>
              Connection{" "}
              <span className="value">{state.connected ? "live" : "disconnected"}</span>
            </label>
            <span className="note">
              {state.runId ?? "no run"}
              {state.dropped > 0 && ` · ${state.dropped} events dropped`}
            </span>
          </div>
        </div>

        {error && <div className="error">{error}</div>}

        <div className="stats">
          <div className="stat">
            <div className="label">Step</div>
            <div className="value">
              {state.step.toLocaleString()}
              {state.totalSteps > 0 && (
                <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
                  {" "}
                  / {state.totalSteps.toLocaleString()}
                </span>
              )}
            </div>
            <div className="sub">{(progress * 100).toFixed(1)}% complete</div>
          </div>
          <div className="stat">
            <div className="label">Train loss</div>
            <div className="value">{lastLoss !== undefined ? lastLoss.toFixed(4) : "—"}</div>
          </div>
          <div className="stat">
            <div className="label">Best validation</div>
            <div className="value">{state.bestVal !== null ? state.bestVal.toFixed(4) : "—"}</div>
            <div className="sub">checkpoint saved on improvement</div>
          </div>
          <div className="stat">
            <div className="label">Throughput</div>
            <div className="value">
              {state.tokensPerSecond ? Math.round(state.tokensPerSecond).toLocaleString() : "—"}
            </div>
            <div className="sub">tokens / second</div>
          </div>
          {state.exitCode !== null && (
            <div className="stat">
              <div className="label">Exit</div>
              <div className="value">{state.exitCode}</div>
              <div className="sub">{state.exitCode === 0 ? "completed" : "stopped"}</div>
            </div>
          )}
        </div>

        {state.train.length > 0 ? (
          <LossChart train={state.train} val={state.val} />
        ) : (
          <div className="empty">
            No metrics yet. Start a run, or open this tab during one to replay its history.
          </div>
        )}
      </section>

      {state.train.length > 0 && (
        <section className="panel">
          <h2>Schedule &amp; gradients</h2>
          <p className="hint">
            Separate charts rather than extra axes on the loss plot — these carry different
            units, and a shared axis would imply a relationship the scale choice invented.
          </p>
          <div className="grid-2">
            <div>
              <div className="stat" style={{ marginBottom: 4 }}>
                <div className="label">Learning rate</div>
              </div>
              <Sparkline points={state.lr} label="lr" color={tokens.series1} />
            </div>
            <div>
              <div className="stat" style={{ marginBottom: 4 }}>
                <div className="label">Gradient norm (pre-clip)</div>
              </div>
              <Sparkline points={state.gradNorm} label="|g|" color={tokens.series2} />
            </div>
          </div>
        </section>
      )}

      <section className="panel">
        <h2>Logs</h2>
        <p className="hint">
          Trainer stdout. Structured metric rows go to the charts above; everything else
          appears here. Capped at the most recent 400 lines.
        </p>
        <div
          ref={logRef}
          style={{
            height: 260,
            overflowY: "auto",
            background: "var(--surface-0)",
            border: "1px solid var(--border)",
            borderRadius: 8,
            padding: 10,
            fontFamily: "var(--font-mono)",
            fontSize: 12,
            lineHeight: 1.6,
            whiteSpace: "pre-wrap",
          }}
        >
          {state.logs.length === 0 ? (
            <span style={{ color: "var(--text-muted)" }}>No output yet.</span>
          ) : (
            state.logs.map((line, i) => <div key={i}>{line}</div>)
          )}
        </div>
      </section>
    </>
  );
}
