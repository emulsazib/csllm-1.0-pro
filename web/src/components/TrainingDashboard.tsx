/** Live training: loss curves, throughput, logs, and run control.
 *
 *  The gateway supervises the trainer as a subprocess and fans its stdout out
 *  over WS /ws/train, replaying recent history on connect — so opening this tab
 *  mid-run still draws the whole curve.
 */

import { useEffect, useRef, useState } from "react";
import { api } from "../api/client";
import type { PreparedDataset } from "../api/types";
import { useTrainingStream } from "../hooks/useTrainingStream";
import { chartTokens } from "../theme";
import { formatBytes } from "./ConfiguratorPanel";
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

/** Presets name a config AND the artifacts prepared for it. The pairing matters:
 *  a tokenizer's vocab_size must match the model's, so mixing a debug tokenizer
 *  with the shakespeare config fails at load rather than training badly. */
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
  const [dataChoice, setDataChoice] = useState("");
  const [prepared, setPrepared] = useState<PreparedDataset[]>([]);
  const [steps, setSteps] = useState<number | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const logRef = useRef<HTMLDivElement>(null);
  const tokens = chartTokens();

  // Refresh after every run: a prepare job that just finished should show up
  // in the data selector without a page reload.
  useEffect(() => {
    if (state.running) return;
    api
      .prepared()
      .then((body) => setPrepared(body.prepared))
      .catch(() => {});
  }, [state.running]);

  // Follow the tail, but only while the user is already at the bottom —
  // otherwise scrolling back to read something would keep yanking them away.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 60;
    if (atBottom) el.scrollTop = el.scrollHeight;
  }, [state.logs]);

  async function run(action: () => Promise<unknown>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function startOptions(): StartOptions {
    const base = { ...PRESETS[preset], steps: steps ?? PRESETS[preset].steps };
    const chosen = prepared.find((p) => p.data_dir === dataChoice);
    if (!chosen) return base;
    return {
      ...base,
      data_dir: chosen.data_dir,
      tokenizer_dir: chosen.tokenizer_dir,
      out: `${chosen.data_dir}/model.csllm`,
    };
  }

  const progress = state.totalSteps ? state.step / state.totalSteps : 0;
  const lastLoss = state.train.at(-1)?.loss;
  const lastRss = state.memory.at(-1)?.loss ?? null;
  // A prepare job shares this socket but has no loss curve — only stages.
  const preparing = state.running && state.kind === "prepare";

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
            <label>Config</label>
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
              batch {PRESETS[preset].batch_size} · lr {PRESETS[preset].lr}
            </span>
          </div>

          <div className="control" style={{ maxWidth: 190 }}>
            <label htmlFor="steps">
              Steps <span className="value">{steps ?? PRESETS[preset].steps}</span>
            </label>
            <input
              id="steps"
              type="range"
              min={50}
              max={5000}
              step={50}
              disabled={state.running}
              value={steps ?? PRESETS[preset].steps}
              onChange={(e) => setSteps(Number(e.target.value))}
            />
            <span className="note">
              <button className="ghost" onClick={() => setSteps(null)} disabled={state.running}>
                reset to preset
              </button>
            </span>
          </div>

          <div className="control" style={{ maxWidth: 240 }}>
            <label htmlFor="prepared">Data</label>
            <select
              id="prepared"
              value={dataChoice}
              disabled={state.running}
              onChange={(e) => setDataChoice(e.target.value)}
            >
              <option value="">{PRESETS[preset].data_dir} (checked in)</option>
              {prepared.map((p) => (
                <option key={p.data_dir} value={p.data_dir}>
                  {p.name} — {(p.train_tokens / 1000).toFixed(0)}k tokens
                </option>
              ))}
            </select>
            <span className="note">
              {dataChoice
                ? "prepared from the Datasets tab"
                : prepared.length === 0
                  ? "prepare a dataset to add options"
                  : `${prepared.length} prepared dataset(s) available`}
            </span>
          </div>

          <div className="control" style={{ flex: "none" }}>
            <label>&nbsp;</label>
            <div style={{ display: "flex", gap: 8 }}>
              <button
                className="action"
                disabled={busy || state.running}
                onClick={() => run(() => api.startTraining(startOptions()))}
              >
                Start run
              </button>
              <button
                className="ghost"
                disabled={busy || !state.running}
                onClick={() =>
                  run(() => api.trainControl(state.paused ? "resume" : "pause"))
                }
              >
                {state.paused ? "Resume" : "Pause"}
              </button>
              <button
                className="ghost"
                disabled={busy || !state.running}
                onClick={() => run(() => api.trainControl("stop"))}
              >
                Stop
              </button>
            </div>
          </div>

          <div className="control">
            <label>
              Connection{" "}
              <span className="value">
                {state.paused ? "paused" : state.connected ? "live" : "disconnected"}
              </span>
            </label>
            <span className="note">
              {state.runId ?? "no run"}
              {preparing && ` · preparing (${state.stage ?? "…"})`}
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
          <div className="stat">
            <div className="label">Epoch</div>
            <div className="value">{state.epoch !== null ? state.epoch.toFixed(2) : "—"}</div>
            <div className="sub">passes over the corpus</div>
          </div>
          <div className="stat">
            <div className="label">{state.memoryLabel ?? "Memory"}</div>
            <div className="value">{lastRss !== null ? formatBytes(lastRss) : "—"}</div>
            <div className="sub">
              {state.memoryTotal
                ? `of ${formatBytes(state.memoryTotal)} · ${state.device ?? ""}`
                : "trainer resident set"}
            </div>
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
            {preparing
              ? `Preparing a dataset (${state.stage ?? "…"}) — logs below.`
              : "No metrics yet. Start a run, or open this tab during one to replay its history."}
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

          {state.memory.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div className="stat" style={{ marginBottom: 4 }}>
                <div className="label">
                  {state.memoryLabel ?? "Memory"} — trainer resident set
                </div>
                <div className="sub">
                  sampled every 50 steps; the activation arena is allocated up front, so this
                  should be flat rather than climbing
                </div>
              </div>
              <Sparkline points={state.memory} label="bytes" color={tokens.series1} />
            </div>
          )}
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
