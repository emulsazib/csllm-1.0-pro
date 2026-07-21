/** Sampling playground: move the knobs, watch which tokens survive.
 *
 *  The filtered probabilities come from the C++ `filtered_distribution`, which
 *  `Sampler::sample` is itself built on — so this chart shows what the model
 *  would actually draw from, not a JS re-implementation that could drift.
 */

import { useState } from "react";
import { api } from "../api/client";
import type { SamplingSettings } from "../api/types";
import { useAsync, useDebounced } from "../hooks/useDebounced";
import { ProbabilityChart, tokenLabel } from "./ProbabilityChart";

const DEFAULT_PROMPT = "KING RICHARD:\n";

export function ProbabilityPanel() {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [settings, setSettings] = useState<SamplingSettings>({
    temperature: 0.8,
    top_k: 40,
    top_p: 0.95,
  });
  const [topN, setTopN] = useState(15);

  const debouncedPrompt = useDebounced(prompt, 300);
  const debouncedSettings = useDebounced(settings, 150);
  const enabled = debouncedPrompt.trim().length > 0;

  const inspect = useAsync(
    (signal) => api.inspect(debouncedPrompt, debouncedSettings, topN, signal),
    [debouncedPrompt, debouncedSettings, topN],
    enabled,
  );

  function update<K extends keyof SamplingSettings>(key: K, value: SamplingSettings[K]) {
    setSettings((prev) => ({ ...prev, [key]: value }));
  }

  const data = inspect.data;

  return (
    <section className="panel">
      <h2>Sampling playground</h2>
      <p className="hint">
        The next-token distribution for this prompt, before and after your filters. A bar
        with only the orange half is a token the model wanted but these settings excluded.
      </p>

      <textarea
        rows={3}
        value={prompt}
        onChange={(e) => setPrompt(e.target.value)}
        aria-label="Prompt"
        spellCheck={false}
      />

      {/* Filters in one row above the chart. */}
      <div className="controls" style={{ marginTop: 14 }}>
        <div className="control">
          <label htmlFor="temperature">
            Temperature <span className="value">{settings.temperature.toFixed(2)}</span>
          </label>
          <input
            id="temperature"
            type="range"
            min={0}
            max={2}
            step={0.05}
            value={settings.temperature}
            onChange={(e) => update("temperature", Number(e.target.value))}
          />
          <span className="note">
            {settings.temperature === 0 ? "greedy — always the argmax" : "flattens or sharpens"}
          </span>
        </div>

        <div className="control">
          <label htmlFor="top_k">
            Top-K <span className="value">{settings.top_k === 0 ? "off" : settings.top_k}</span>
          </label>
          <input
            id="top_k"
            type="range"
            min={0}
            max={100}
            step={1}
            value={settings.top_k}
            onChange={(e) => update("top_k", Number(e.target.value))}
          />
          <span className="note">keep the K largest logits</span>
        </div>

        <div className="control">
          <label htmlFor="top_p">
            Top-P <span className="value">{settings.top_p.toFixed(2)}</span>
          </label>
          <input
            id="top_p"
            type="range"
            min={0.05}
            max={1}
            step={0.01}
            value={settings.top_p}
            onChange={(e) => update("top_p", Number(e.target.value))}
          />
          <span className="note">
            {settings.top_p >= 1 ? "off — nothing truncated" : "smallest set reaching P"}
          </span>
        </div>

        <div className="control" style={{ maxWidth: 150 }}>
          <label htmlFor="top_n">
            Show <span className="value">{topN}</span>
          </label>
          <input
            id="top_n"
            type="range"
            min={5}
            max={40}
            step={1}
            value={topN}
            onChange={(e) => setTopN(Number(e.target.value))}
          />
          <span className="note">candidates charted</span>
        </div>
      </div>

      {inspect.error && <div className="error">{inspect.error}</div>}

      {data && (
        <>
          <div className="stats">
            <div className="stat">
              <div className="label">Prompt tokens</div>
              <div className="value">{data.prompt_tokens}</div>
            </div>
            <div className="stat">
              <div className="label">Candidates kept</div>
              <div className="value">{data.kept_count.toLocaleString()}</div>
              <div className="sub">of {data.vocab_size.toLocaleString()} vocabulary</div>
            </div>
            <div className="stat">
              <div className="label">Entropy after filters</div>
              <div className="value">{data.filtered_entropy.toFixed(3)}</div>
              <div className="sub">nats · model belief {data.raw_entropy.toFixed(3)}</div>
            </div>
          </div>

          <ProbabilityChart candidates={data.candidates} />

          {/* Table view: the accessible alternative to reading the bars, and the
              only place the exact logits are legible. */}
          <details style={{ marginTop: 14 }}>
            <summary style={{ cursor: "pointer", fontSize: 12.5, color: "var(--text-secondary)" }}>
              Table view
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
                  {data.candidates.map((c) => (
                    <tr key={c.id} data-kept={c.kept}>
                      <td className="mono">{tokenLabel(c.text)}</td>
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
        </>
      )}

      {!enabled && <div className="empty">Enter a prompt to inspect.</div>}
    </section>
  );
}
