/** Prompt testing playground with a per-token inference breakdown.
 *
 *  One prompt, one `/ws/inspect` subscription, three views of the SAME token:
 *  what it was tokenized as, where its attention went, and which candidates the
 *  sampler was choosing between. Splitting these across tabs (as the diagnostics
 *  panels do) means three separate generations with three different seeds —
 *  nothing lines up, and "why did it pick that token" cannot be answered.
 *
 *  Everything here is captured from the decode loop itself: the candidates come
 *  from the C++ `filtered_distribution` that `Sampler::sample` draws from, and
 *  the attention is copied out of the softmax rather than recomputed.
 */

import { useEffect, useMemo, useState } from "react";
import { attentionVector } from "../api/ws";
import { useInspectStream } from "../hooks/useInspectStream";
import { AttentionHeatmap } from "./AttentionHeatmap";
import { ProbabilityChart, tokenLabel } from "./ProbabilityChart";
import { TOKEN_TINTS, TokenText } from "./TokenText";

const DEFAULT_PROMPT = "KING RICHARD:\n";

/** Top-N candidates to request per token. The server serialises this per step,
 *  so it is a payload knob, not just a display one. */
const TOP_N = 12;

export function PlaygroundPanel() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [maxTokens, setMaxTokens] = useState(24);
  const [temperature, setTemperature] = useState(0.8);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.95);
  const [seed, setSeed] = useState(7);
  const [layer, setLayer] = useState(0);
  const [head, setHead] = useState<number | "all">("all");
  const [selected, setSelected] = useState<number | null>(null);

  const stream = useInspectStream();
  const config = stream.start?.config;
  const promptTokens = stream.start?.prompt ?? [];

  // Follow generation live, but stop following once the user picks a token.
  const activeIndex = selected ?? stream.tokens.length - 1;
  const active = stream.tokens[activeIndex] ?? null;

  // A new run invalidates any selection: index 12 of the old sequence is a
  // different token in the new one.
  useEffect(() => {
    if (stream.running) setSelected(null);
  }, [stream.running]);

  const keyLabels = useMemo(
    () => [...promptTokens.map((t) => t.text), ...stream.tokens.map((t) => t.text)],
    [promptTokens, stream.tokens],
  );

  const activeWeights = useMemo(() => {
    if (!active?.attention) return null;
    return Array.from(attentionVector(active.attention, layer, head));
  }, [active, layer, head]);

  /** Which keys this token leaned on hardest — the readable form of one row. */
  const topKeys = useMemo(() => {
    if (!activeWeights) return [];
    return activeWeights
      .map((weight, key) => ({ weight, key }))
      .sort((a, b) => b.weight - a.weight)
      .slice(0, 6);
  }, [activeWeights]);

  const completion = stream.tokens.map((t) => t.text).join("");

  function generate() {
    setSelected(null);
    stream.run({
      prompt,
      max_tokens: maxTokens,
      temperature,
      top_k: topK,
      top_p: topP,
      seed,
      top_n: TOP_N,
    });
  }

  return (
    <>
      <section className="panel">
        <h2>Prompt playground</h2>
        <p className="hint">
          Generate a continuation, then click any token to see exactly why it was chosen —
          its attention over the context and the distribution it was sampled from. All three
          views describe the same generation step, not three separate runs.
        </p>

        <textarea
          rows={3}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          aria-label="Prompt"
          spellCheck={false}
        />

        <div className="controls" style={{ marginTop: 14 }}>
          <div className="control" style={{ maxWidth: 170 }}>
            <label htmlFor="pg-tokens">
              Tokens <span className="value">{maxTokens}</span>
            </label>
            <input
              id="pg-tokens"
              type="range"
              min={4}
              max={64}
              value={maxTokens}
              onChange={(e) => setMaxTokens(Number(e.target.value))}
            />
          </div>
          <div className="control" style={{ maxWidth: 170 }}>
            <label htmlFor="pg-temp">
              Temperature <span className="value">{temperature.toFixed(2)}</span>
            </label>
            <input
              id="pg-temp"
              type="range"
              min={0}
              max={2}
              step={0.05}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
            />
            <span className="note">{temperature === 0 ? "greedy" : "flattens or sharpens"}</span>
          </div>
          <div className="control" style={{ maxWidth: 170 }}>
            <label htmlFor="pg-topk">
              Top-K <span className="value">{topK === 0 ? "off" : topK}</span>
            </label>
            <input
              id="pg-topk"
              type="range"
              min={0}
              max={100}
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
            />
          </div>
          <div className="control" style={{ maxWidth: 170 }}>
            <label htmlFor="pg-topp">
              Top-P <span className="value">{topP.toFixed(2)}</span>
            </label>
            <input
              id="pg-topp"
              type="range"
              min={0.05}
              max={1}
              step={0.01}
              value={topP}
              onChange={(e) => setTopP(Number(e.target.value))}
            />
          </div>
          <div className="control" style={{ maxWidth: 130 }}>
            <label htmlFor="pg-seed">
              Seed <span className="value">{seed}</span>
            </label>
            <input
              id="pg-seed"
              type="range"
              min={0}
              max={99}
              value={seed}
              onChange={(e) => setSeed(Number(e.target.value))}
            />
            <span className="note">same seed → same tokens</span>
          </div>
          <div className="control" style={{ flex: "none" }}>
            <label>&nbsp;</label>
            <div style={{ display: "flex", gap: 8 }}>
              <button className="action" onClick={generate} disabled={stream.running}>
                {stream.running ? "Generating…" : "Generate"}
              </button>
              {stream.running && (
                <button className="ghost" onClick={stream.stop}>
                  Stop
                </button>
              )}
            </div>
          </div>
        </div>

        {stream.error && <div className="error">{stream.error}</div>}

        {/* The completion as prose, before it is broken into chips. */}
        {(completion || stream.running) && (
          <div
            style={{
              background: "var(--surface-0)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius)",
              padding: 12,
              fontFamily: "var(--font-mono)",
              fontSize: 13,
              lineHeight: 1.65,
              whiteSpace: "pre-wrap",
              marginBottom: 14,
            }}
          >
            <span style={{ color: "var(--text-muted)" }}>{prompt}</span>
            {completion}
            {stream.running && <span style={{ color: "var(--series-1)" }}>▌</span>}
          </div>
        )}

        {stream.tokens.length > 0 && (
          <>
            <div className="token-stream">
              {promptTokens.map((t, i) => (
                <span
                  key={`p${i}`}
                  className="token"
                  style={{ opacity: 0.5 }}
                  title={`prompt · id ${t.id}`}
                >
                  <TokenText text={t.text} />
                </span>
              ))}
              {stream.tokens.map((t, i) => (
                <span
                  key={t.index}
                  className="token"
                  data-selected={i === activeIndex}
                  style={{
                    cursor: "pointer",
                    background: TOKEN_TINTS[i % TOKEN_TINTS.length],
                  }}
                  title={`id ${t.id} · position ${t.position} — click to inspect`}
                  onClick={() => setSelected(i)}
                >
                  <TokenText text={t.text} />
                </span>
              ))}
            </div>

            <div className="legend">
              <span className="item" style={{ opacity: 0.6 }}>
                <span className="swatch" style={{ background: "transparent", border: "1px solid var(--border)" }} />
                prompt
              </span>
              <span className="item">
                <span className="swatch" style={{ background: TOKEN_TINTS[0] }} /> generated —
                click to inspect
              </span>
              <span className="item" style={{ color: "var(--text-muted)" }}>
                ␣ space · ↵ newline · ⇥ tab
              </span>
            </div>
          </>
        )}

        {!stream.start && !stream.running && (
          <div className="empty">Enter a prompt and press Generate.</div>
        )}
      </section>

      {/* ── per-token breakdown ─────────────────────────────────────────────── */}

      {active && (
        <section className="panel">
          <h2>
            Inference breakdown — token #{activeIndex} “{tokenLabel(active.text)}”
          </h2>
          <p className="hint">
            Position {active.position} in the sequence, token id {active.id}.
            {selected === null && " Following the newest token; click one to pin it."}
          </p>

          <div className="stats">
            <div className="stat">
              <div className="label">Chosen</div>
              <div className="value mono">{tokenLabel(active.text)}</div>
              <div className="sub">id {active.id}</div>
            </div>
            {active.kept_count !== undefined && (
              <div className="stat">
                <div className="label">Candidates kept</div>
                <div className="value">{active.kept_count.toLocaleString()}</div>
                <div className="sub">
                  of {config?.vocab_size.toLocaleString() ?? "?"} after top-k / top-p
                </div>
              </div>
            )}
            {active.raw_entropy !== undefined && (
              <div className="stat">
                <div className="label">Entropy</div>
                <div className="value">{active.filtered_entropy?.toFixed(3) ?? "—"}</div>
                <div className="sub">nats · model belief {active.raw_entropy.toFixed(3)}</div>
              </div>
            )}
            {active.attention && (
              <div className="stat">
                <div className="label">Attention block</div>
                <div className="value">
                  {active.attention.layers}×{active.attention.heads}×{active.attention.keys}
                </div>
                <div className="sub">layers × heads × keys</div>
              </div>
            )}
          </div>

          {active.top && active.top.length > 0 && (
            <>
              <p className="hint" style={{ marginTop: 18, marginBottom: 6 }}>
                <strong>Probability distribution.</strong> Orange is the model&apos;s own
                belief; blue is what the sampler actually drew from after your filters. A bar
                with only orange is a token these settings excluded.
              </p>
              <ProbabilityChart candidates={active.top} />
            </>
          )}

          {active.attention && (
            <>
              <p className="hint" style={{ marginTop: 22, marginBottom: 8 }}>
                <strong>Attention.</strong> Where each generated token looked. Weights are
                copied out of the decode loop&apos;s softmax — verified against an independent
                NumPy recomputation to 2.5e-07, not a plausible-looking reconstruction.
              </p>

              {config && (
                <div className="controls">
                  <div className="control" style={{ maxWidth: 200 }}>
                    <label htmlFor="pg-layer">
                      Layer <span className="value">{layer}</span>
                    </label>
                    <input
                      id="pg-layer"
                      type="range"
                      min={0}
                      max={Math.max(0, active.attention.layers - 1)}
                      value={layer}
                      onChange={(e) => setLayer(Number(e.target.value))}
                    />
                    <span className="note">of {active.attention.layers} captured</span>
                  </div>
                  <div className="control" style={{ flex: "none" }}>
                    <label>Head</label>
                    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
                      <button
                        className="ghost"
                        aria-pressed={head === "all"}
                        onClick={() => setHead("all")}
                      >
                        mean
                      </button>
                      {Array.from({ length: active.attention.heads }, (_, h) => (
                        <button
                          key={h}
                          className="ghost"
                          aria-pressed={head === h}
                          onClick={() => setHead(h)}
                        >
                          {h}
                        </button>
                      ))}
                    </div>
                    <span className="note">
                      {head === "all" ? "averaged across this layer's heads" : `head ${head}`}
                    </span>
                  </div>
                </div>
              )}

              <AttentionHeatmap
                tokens={stream.tokens}
                keyLabels={keyLabels}
                layer={layer}
                head={head}
              />

              {topKeys.length > 0 && (
                <>
                  <p className="hint" style={{ marginTop: 16, marginBottom: 6 }}>
                    Strongest keys for token #{activeIndex}:
                  </p>
                  <div className="token-stream">
                    {topKeys.map(({ key, weight }) => (
                      <span
                        key={key}
                        className="token"
                        title={`key ${key} · ${(weight * 100).toFixed(2)}%`}
                        style={{ background: TOKEN_TINTS[0] }}
                      >
                        <TokenText text={keyLabels[key] ?? String(key)} />
                        <span style={{ color: "var(--text-muted)" }}>
                          {" "}
                          {(weight * 100).toFixed(1)}%
                        </span>
                      </span>
                    ))}
                  </div>
                </>
              )}
            </>
          )}

          <details style={{ marginTop: 18 }}>
            <summary style={{ cursor: "pointer", fontSize: 12.5, color: "var(--text-secondary)" }}>
              Candidate table
            </summary>
            <div className="scroll-x" style={{ marginTop: 10 }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>Token</th>
                    <th>Id</th>
                    <th style={{ textAlign: "right" }}>Logit</th>
                    <th style={{ textAlign: "right" }}>Model belief</th>
                    <th style={{ textAlign: "right" }}>After filters</th>
                  </tr>
                </thead>
                <tbody>
                  {(active.top ?? []).map((c) => (
                    <tr key={c.id} data-kept={c.kept}>
                      <td className="mono">
                        {tokenLabel(c.text)}
                        {c.id === active.id && " ← chosen"}
                      </td>
                      <td className="num">{c.id}</td>
                      <td className="num">{c.logit.toFixed(3)}</td>
                      <td className="num">{(c.raw_prob * 100).toFixed(2)}%</td>
                      <td className="num">
                        {c.kept ? `${(c.prob * 100).toFixed(2)}%` : "excluded"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </section>
      )}
    </>
  );
}
