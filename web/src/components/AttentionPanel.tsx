/** Generate a continuation and watch the attention the model actually used. */

import { Suspense, lazy, useMemo, useState } from "react";
import { attentionRow } from "../api/ws";
import { useInspectStream } from "../hooks/useInspectStream";
// three.js is ~700 KB; code-split so the other tabs never download it.
const TransformerGraph = lazy(() =>
  import("./TransformerGraph").then((m) => ({ default: m.TransformerGraph })),
);
import { tokenLabel } from "./ProbabilityChart";

const DEFAULT_PROMPT = "KING RICHARD:\n";

export function AttentionPanel() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(16);
  const [layerIndex, setLayerIndex] = useState(0);
  const [headIndex, setHeadIndex] = useState<number | "all">("all");
  const [selected, setSelected] = useState<number | null>(null);

  const stream = useInspectStream();
  const config = stream.start?.config;

  // Default to the newest token so the view follows generation live.
  const activeIndex = selected ?? stream.tokens.length - 1;
  const active = stream.tokens[activeIndex] ?? null;
  const block = active?.attention ?? null;

  const keyLabels = useMemo(() => {
    const promptLabels = stream.start?.prompt.map((t) => t.text) ?? [];
    return [...promptLabels, ...stream.tokens.map((t) => t.text)];
  }, [stream.start, stream.tokens]);

  const headWeights = useMemo(() => {
    if (!block || headIndex === "all") return null;
    return Array.from(attentionRow(block, layerIndex, headIndex));
  }, [block, layerIndex, headIndex]);

  function start() {
    setSelected(null);
    stream.run({
      prompt,
      max_tokens: maxTokens,
      temperature: 0.8,
      top_k: 40,
      top_p: 0.95,
      seed: 7,
      top_n: 6,
    });
  }

  return (
    <>
      <section className="panel">
        <h2>Attention flow</h2>
        <p className="hint">
          Real per-head weights captured from the decode loop — not a reconstruction. Height
          and colour both encode how much a layer attends to each key, so the reading survives
          greyscale.
        </p>

        <textarea
          rows={2}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          aria-label="Prompt"
          spellCheck={false}
        />

        <div className="controls" style={{ marginTop: 14 }}>
          <div className="control" style={{ maxWidth: 190 }}>
            <label htmlFor="max-tokens">
              Tokens <span className="value">{maxTokens}</span>
            </label>
            <input
              id="max-tokens"
              type="range"
              min={4}
              max={48}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </div>

          {config && (
            <>
              <div className="control" style={{ maxWidth: 190 }}>
                <label htmlFor="layer">
                  Layer <span className="value">{layerIndex}</span>
                </label>
                <input
                  id="layer"
                  type="range"
                  min={0}
                  max={config.n_layer - 1}
                  value={layerIndex}
                  onChange={(e) => setLayerIndex(Number(e.target.value))}
                />
                <span className="note">arcs show this layer</span>
              </div>

              <div className="control" style={{ flex: "none" }}>
                <label>Head</label>
                <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                  <button
                    className="ghost"
                    aria-pressed={headIndex === "all"}
                    onClick={() => setHeadIndex("all")}
                  >
                    all
                  </button>
                  {Array.from({ length: config.n_head }, (_, h) => (
                    <button
                      key={h}
                      className="ghost"
                      aria-pressed={headIndex === h}
                      onClick={() => setHeadIndex(h)}
                    >
                      {h}
                    </button>
                  ))}
                </div>
              </div>
            </>
          )}

          <div className="control" style={{ flex: "none" }}>
            <label>&nbsp;</label>
            <button className="action" onClick={start} disabled={stream.running}>
              {stream.running ? "Generating…" : "Generate"}
            </button>
          </div>
        </div>

        {stream.error && <div className="error">{stream.error}</div>}

        <Suspense fallback={<div className="empty">Loading 3D view…</div>}>
          <TransformerGraph
            block={block}
            labels={keyLabels}
            layerIndex={layerIndex}
            headIndex={headIndex}
          />
        </Suspense>

        {stream.start && (
          <div className="stats" style={{ marginTop: 12 }}>
            <div className="stat">
              <div className="label">Tokens generated</div>
              <div className="value">{stream.tokens.length}</div>
            </div>
            <div className="stat">
              <div className="label">Attention received</div>
              <div className="value">{(stream.bytesReceived / 1024).toFixed(1)} KB</div>
              <div className="sub">binary float32 frames</div>
            </div>
            {block && (
              <div className="stat">
                <div className="label">Block</div>
                <div className="value">
                  {block.layers}×{block.heads}×{block.keys}
                </div>
                <div className="sub">layers × heads × keys</div>
              </div>
            )}
          </div>
        )}
      </section>

      {stream.tokens.length > 0 && (
        <section className="panel">
          <h2>Generated sequence</h2>
          <p className="hint">
            Click a token to inspect the attention it used. Showing{" "}
            {activeIndex >= 0 ? `#${activeIndex}` : "none"}.
          </p>

          <div className="token-stream">
            {stream.start?.prompt.map((t, i) => (
              <span key={`p${i}`} className="token" style={{ opacity: 0.55 }} title="prompt">
                {tokenLabel(t.text)}
              </span>
            ))}
            {stream.tokens.map((t, i) => (
              <span
                key={t.index}
                className="token"
                data-selected={i === activeIndex}
                style={{ cursor: "pointer", background: "rgba(42,120,214,0.13)" }}
                onClick={() => setSelected(i)}
                title={`id ${t.id} · position ${t.position}`}
              >
                {tokenLabel(t.text)}
              </span>
            ))}
          </div>

          {active?.top && (
            <div className="scroll-x" style={{ marginTop: 14 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>Candidate at this step</th>
                    <th style={{ textAlign: "right" }}>Model belief</th>
                    <th style={{ textAlign: "right" }}>After filters</th>
                  </tr>
                </thead>
                <tbody>
                  {active.top.map((c) => (
                    <tr key={c.id} data-kept={c.kept}>
                      <td className="mono">
                        {tokenLabel(c.text)}
                        {c.id === active.id && " ←"}
                      </td>
                      <td className="num">{(c.raw_prob * 100).toFixed(2)}%</td>
                      <td className="num">
                        {c.kept ? `${(c.prob * 100).toFixed(2)}%` : "excluded"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {headWeights && (
            <>
              <p className="hint" style={{ marginTop: 16, marginBottom: 6 }}>
                Layer {layerIndex}, head {headIndex} — where this token looked:
              </p>
              <div className="scroll-x">
                <table className="data">
                  <thead>
                    <tr>
                      <th>Key</th>
                      {headWeights.map((_, k) => (
                        <th key={k} style={{ textAlign: "right" }}>
                          {tokenLabel(keyLabels[k] ?? String(k))}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>weight</td>
                      {headWeights.map((w, k) => (
                        <td key={k} className="num">
                          {(w * 100).toFixed(1)}%
                        </td>
                      ))}
                    </tr>
                  </tbody>
                </table>
              </div>
            </>
          )}
        </section>
      )}
    </>
  );
}
