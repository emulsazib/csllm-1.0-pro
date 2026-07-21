/** Architecture configurator: size the model before it exists.
 *
 *  Every number shown here comes from `POST /configure_model/estimate`, which is
 *  the same `calculate_model_params` the version store records — so the count on
 *  the slider is the count in the bundle. Nothing is computed in JS, because a
 *  second implementation of the parameter formula is a second thing to be wrong.
 *
 *  Two invariants make head selection awkward: `n_embd` must divide by `n_head`,
 *  and `head_dim` must be EVEN (RoPE rotates channel pairs). Rather than let the
 *  user drag into an invalid pair and read an error, the head options are derived
 *  from the current `n_embd` — invalid combinations are not offered. The server
 *  still validates; this only removes the most confusing way to fail.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api/client";
import type { ArchitectureConfig, ConfigVersion, EstimateResponse } from "../api/types";
import { useAsync, useDebounced } from "../hooks/useDebounced";

/** Mirrors the shipped `configs/*.json` exactly — a preset that quietly differs
 *  from the file of the same name would report a parameter count for a model
 *  nobody can train. */
export const PRESETS: Record<string, ArchitectureConfig> = {
  debug: {
    vocab_size: 512,
    n_layer: 2,
    n_head: 2,
    n_embd: 64,
    block_size: 32,
    ffn_hidden: 192,
    rope_theta: 10000,
    norm_eps: 1e-5,
  },
  shakespeare: {
    vocab_size: 4096,
    n_layer: 6,
    n_head: 6,
    n_embd: 384,
    block_size: 256,
    ffn_hidden: 1024,
    rope_theta: 10000,
    norm_eps: 1e-5,
  },
};

/** Head counts that divide `nEmbd` into an EVEN head_dim, capped at the API bound. */
export function validHeadCounts(nEmbd: number): number[] {
  const heads: number[] = [];
  for (let h = 1; h <= Math.min(64, nEmbd); h++) {
    if (nEmbd % h === 0 && (nEmbd / h) % 2 === 0) heads.push(h);
  }
  return heads;
}

export function formatParams(n: number): string {
  if (n >= 1e9) return `${(n / 1e9).toFixed(2)} B`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(2)} M`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)} K`;
  return String(n);
}

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(2)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  if (n >= 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${n} B`;
}

/* Activations is usually ~90% of the bar, so it needs a fill that is actually
   visible in both themes — `--gridline` is a hairline token and rendered the
   dominant segment as empty space. */
const MEMORY_PARTS = [
  { key: "params", label: "Weights", color: "var(--series-1)" },
  { key: "gradients", label: "Gradients", color: "var(--series-2)" },
  { key: "optimizer", label: "AdamW moments", color: "var(--baseline)" },
  { key: "activations", label: "Activations", color: "var(--text-muted)" },
] as const;

export function ConfiguratorPanel() {
  const [config, setConfig] = useState<ArchitectureConfig>(PRESETS.shakespeare);
  const [batchSize, setBatchSize] = useState(8);
  const [note, setNote] = useState("");

  const debounced = useDebounced(config, 180);
  const debouncedBatch = useDebounced(batchSize, 180);

  const estimate = useAsync(
    (signal) => api.estimate(debounced, debouncedBatch, undefined, signal),
    [debounced, debouncedBatch],
  );

  // useAsync nulls `data` on error. Dragging n_head through a value the engine
  // rejects would otherwise blank the whole readout mid-drag, so the last good
  // estimate stays on screen underneath the error.
  const lastGood = useRef<EstimateResponse | null>(null);
  if (estimate.data) lastGood.current = estimate.data;
  const shown = estimate.data ?? lastGood.current;

  const heads = useMemo(() => validHeadCounts(config.n_embd), [config.n_embd]);

  // Keep n_head legal when n_embd moves: snap to the nearest offered value.
  useEffect(() => {
    if (heads.length && !heads.includes(config.n_head)) {
      const nearest = heads.reduce((a, b) =>
        Math.abs(b - config.n_head) < Math.abs(a - config.n_head) ? b : a,
      );
      setConfig((prev) => ({ ...prev, n_head: nearest }));
    }
  }, [heads, config.n_head]);

  function set<K extends keyof ArchitectureConfig>(key: K, value: ArchitectureConfig[K]) {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  // ── version creation ──────────────────────────────────────────────────────

  const [versions, setVersions] = useState<ConfigVersion[]>([]);
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<string | null>(null);
  const [createError, setCreateError] = useState<string | null>(null);

  async function refresh() {
    try {
      setVersions(await api.listConfigs());
    } catch {
      /* the panel is still usable without the history */
    }
  }

  useEffect(() => {
    void refresh();
  }, []);

  async function create(initialize: boolean) {
    setBusy(true);
    setCreateError(null);
    setResult(null);
    try {
      const version = await api.configure(config, { note, batch_size: batchSize, initialize });
      setResult(
        version.created
          ? `Created ${version.version_id}${version.checkpoint ? ` → ${version.checkpoint}` : ""}`
          : `${version.version_id} already exists — identical hyperparameters`,
      );
      await refresh();
    } catch (err: unknown) {
      setCreateError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <section className="panel">
      <h2>Architecture configurator</h2>
      <p className="hint">
        Size the model before building it. Parameter counts come from the engine's own formula,
        so what you see here is what the checkpoint will contain.
      </p>

      <div className="controls">
        <div className="control">
          <label htmlFor="n_layer">
            Layers <span className="value">{config.n_layer}</span>
          </label>
          <input
            id="n_layer"
            type="range"
            min={1}
            max={32}
            step={1}
            value={config.n_layer}
            onChange={(e) => set("n_layer", Number(e.target.value))}
          />
          <span className="note">transformer blocks</span>
        </div>

        <div className="control">
          <label htmlFor="n_embd">
            d_model <span className="value">{config.n_embd}</span>
          </label>
          <input
            id="n_embd"
            type="range"
            min={32}
            max={2048}
            step={32}
            value={config.n_embd}
            onChange={(e) => set("n_embd", Number(e.target.value))}
          />
          <span className="note">hidden dimension</span>
        </div>

        <div className="control">
          <label htmlFor="n_head">
            Heads <span className="value">{config.n_head}</span>
          </label>
          <select
            id="n_head"
            value={config.n_head}
            onChange={(e) => set("n_head", Number(e.target.value))}
          >
            {heads.map((h) => (
              <option key={h} value={h}>
                {h} — head_dim {config.n_embd / h}
              </option>
            ))}
          </select>
          <span className="note">head_dim must be even</span>
        </div>

        <div className="control">
          <label htmlFor="ffn_hidden">
            d_ff <span className="value">{config.ffn_hidden}</span>
          </label>
          <input
            id="ffn_hidden"
            type="range"
            min={32}
            max={8192}
            step={32}
            value={config.ffn_hidden}
            onChange={(e) => set("ffn_hidden", Number(e.target.value))}
          />
          <span className="note">{(config.ffn_hidden / config.n_embd).toFixed(2)}x d_model</span>
        </div>

        <div className="control">
          <label htmlFor="block_size">
            Context <span className="value">{config.block_size}</span>
          </label>
          <input
            id="block_size"
            type="range"
            min={16}
            max={2048}
            step={16}
            value={config.block_size}
            onChange={(e) => set("block_size", Number(e.target.value))}
          />
          <span className="note">tokens of context</span>
        </div>

        <div className="control">
          <label htmlFor="vocab_size">
            Vocabulary <span className="value">{config.vocab_size.toLocaleString()}</span>
          </label>
          <input
            id="vocab_size"
            type="range"
            min={256}
            max={32768}
            step={256}
            value={config.vocab_size}
            onChange={(e) => set("vocab_size", Number(e.target.value))}
          />
          <span className="note">BPE merges + 256 byte tokens</span>
        </div>

        {/* Capped: as the last control it wraps to its own row and would
            otherwise flex to the full panel width. */}
        <div className="control" style={{ maxWidth: 200 }}>
          <label htmlFor="batch_size">
            Batch <span className="value">{batchSize}</span>
          </label>
          <input
            id="batch_size"
            type="range"
            min={1}
            max={64}
            step={1}
            value={batchSize}
            onChange={(e) => setBatchSize(Number(e.target.value))}
          />
          <span className="note">affects memory only, not parameters</span>
        </div>
      </div>

      <div className="legend" style={{ marginBottom: 14 }}>
        <span>Presets:</span>
        {Object.entries(PRESETS).map(([name, preset]) => (
          <button key={name} className="ghost" onClick={() => setConfig(preset)}>
            {name}
          </button>
        ))}
      </div>

      {estimate.error && <div className="error">{estimate.error}</div>}

      {shown && (
        <>
          <div className="stats">
            <div className="stat">
              <div className="label">Parameters</div>
              <div className="value">{formatParams(shown.num_params)}</div>
              <div className="sub">{shown.num_params.toLocaleString()} trainable</div>
            </div>
            <div className="stat">
              <div className="label">Training footprint</div>
              <div className="value">{formatBytes(shown.memory.total)}</div>
              <div className="sub">
                batch {debouncedBatch} x {shown.config.block_size} tokens
              </div>
            </div>
            <div className="stat">
              <div className="label">{shown.device.memory_label}</div>
              <div className="value">{formatBytes(shown.device.total_bytes)}</div>
              <div className="sub">{shown.device.device}</div>
            </div>
            <div className="stat">
              <div className="label">Headroom</div>
              <div
                className="value"
                style={{ color: shown.fits ? "var(--status-good)" : "var(--status-critical)" }}
              >
                {shown.fits
                  ? `${((1 - shown.memory.total / shown.device.total_bytes) * 100).toFixed(0)}%`
                  : "exceeds"}
              </div>
              <div className="sub">
                {shown.fits ? "fits on this host" : "will not fit in memory"}
              </div>
            </div>
          </div>

          {/* Stacked bar: which part of the footprint actually costs the memory.
              At realistic batch sizes activations dominate the weights ~30x. */}
          <div className="mem-bar" role="img" aria-label="Memory breakdown">
            {MEMORY_PARTS.map((part) => {
              const bytes = shown.memory[part.key];
              const pct = (bytes / shown.memory.total) * 100;
              return (
                <div
                  key={part.key}
                  className="seg"
                  style={{ width: `${pct}%`, background: part.color }}
                  title={`${part.label}: ${formatBytes(bytes)} (${pct.toFixed(1)}%)`}
                />
              );
            })}
          </div>
          <div className="legend">
            {MEMORY_PARTS.map((part) => (
              <span className="item" key={part.key}>
                <span className="swatch" style={{ background: part.color }} />
                {part.label} {formatBytes(shown.memory[part.key])}
              </span>
            ))}
          </div>

          <details style={{ marginTop: 14 }}>
            <summary
              style={{ cursor: "pointer", fontSize: 12.5, color: "var(--text-secondary)" }}
            >
              Parameter breakdown
            </summary>
            <div className="scroll-x" style={{ marginTop: 10 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>Component</th>
                    <th style={{ textAlign: "right" }}>Parameters</th>
                    <th style={{ textAlign: "right" }}>Share</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Embedding (tied with lm_head)</td>
                    <td className="num">{shown.params.embedding.toLocaleString()}</td>
                    <td className="num">
                      {((shown.params.embedding / shown.num_params) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td>Attention (wq, wk, wv, wo)</td>
                    <td className="num">{shown.params.attention.toLocaleString()}</td>
                    <td className="num">
                      {((shown.params.attention / shown.num_params) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td>Feed-forward (SwiGLU: gate, up, down)</td>
                    <td className="num">{shown.params.ffn.toLocaleString()}</td>
                    <td className="num">
                      {((shown.params.ffn / shown.num_params) * 100).toFixed(1)}%
                    </td>
                  </tr>
                  <tr>
                    <td>RMSNorm gains</td>
                    <td className="num">{shown.params.norms.toLocaleString()}</td>
                    <td className="num">
                      {((shown.params.norms / shown.num_params) * 100).toFixed(1)}%
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}

      {/* ── version creation ──────────────────────────────────────────────── */}

      <div className="controls" style={{ marginTop: 22, alignItems: "flex-end" }}>
        <div className="control" style={{ flex: 2 }}>
          <label htmlFor="note">Note</label>
          <input
            id="note"
            type="text"
            value={note}
            placeholder="what makes this version different"
            onChange={(e) => setNote(e.target.value)}
          />
        </div>
        <button className="action" disabled={busy || !shown} onClick={() => create(false)}>
          Create version
        </button>
        <button className="ghost" disabled={busy || !shown} onClick={() => create(true)}>
          Create + initialize weights
        </button>
      </div>

      {createError && <div className="error">{createError}</div>}
      {result && <div className="hint">{result}</div>}

      {versions.length > 0 && (
        <div className="scroll-x" style={{ marginTop: 10 }}>
          <table className="data">
            <thead>
              <tr>
                <th>Version</th>
                <th>Shape</th>
                <th style={{ textAlign: "right" }}>Parameters</th>
                <th>Note</th>
              </tr>
            </thead>
            <tbody>
              {versions.map((v) => (
                <tr key={v.version_id}>
                  <td className="mono">{v.version_id}</td>
                  <td className="mono">
                    {v.config.n_layer}L x {v.config.n_head}H x {v.config.n_embd}d
                  </td>
                  <td className="num">{formatParams(v.num_params)}</td>
                  <td>{v.note || <span style={{ color: "var(--text-muted)" }}>—</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}
