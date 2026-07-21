/** Dataset browser: pick a corpus, inspect it, turn it into trainable data.
 *
 *  A raw `.txt`/`.jsonl`/`.csv` cannot be trained on directly — a BPE tokenizer
 *  has to be learned from it and the text binarized into `train.bin`/`val.bin`.
 *  That takes minutes on a real corpus, so "Prepare" starts a supervised
 *  subprocess and its progress arrives on the same WS /ws/train the training
 *  dashboard already listens to.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ConfigVersion, DatasetInfo } from "../api/types";
import { useTrainingStream } from "../hooks/useTrainingStream";
import { formatBytes } from "./ConfiguratorPanel";

/** The shipped configs, plus any version created in the configurator. */
const BUILT_IN_CONFIGS = ["configs/debug.json", "configs/shakespeare.json"];

export function DatasetBrowser() {
  const state = useTrainingStream(true);
  const [listing, setListing] = useState<DatasetInfo[]>([]);
  const [extensions, setExtensions] = useState<string[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [preview, setPreview] = useState<string[] | null>(null);
  const [config, setConfig] = useState(BUILT_IN_CONFIGS[0]);
  const [versions, setVersions] = useState<ConfigVersion[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    api
      .datasets()
      .then((body) => {
        if (!live) return;
        setListing(body.datasets);
        setExtensions(body.supported_extensions);
      })
      .catch((err: unknown) => live && setError(err instanceof Error ? err.message : String(err)));
    api.listConfigs().then((v) => live && setVersions(v)).catch(() => {});
    return () => {
      live = false;
    };
  }, []);

  useEffect(() => {
    if (!selected) {
      setPreview(null);
      return;
    }
    const controller = new AbortController();
    api
      .previewDataset(selected, 3, controller.signal)
      .then((body) => setPreview(body.documents))
      .catch(() => setPreview(null));
    return () => controller.abort();
  }, [selected]);

  async function prepare() {
    if (!selected) return;
    setBusy(true);
    setError(null);
    setNotice(null);
    try {
      // No paths passed: the server writes to data/prepared/<dataset>/, which
      // keeps prepares away from the checked-in corpora in data/ and data/debug.
      await api.prepareDataset(selected, { config });
      setNotice(`Preparing ${selected} — progress is streaming on the Training tab.`);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  const preparing = state.running && state.kind === "prepare";
  const chosen = listing.find((d) => d.name === selected);

  return (
    <section className="panel">
      <h2>Datasets</h2>
      <p className="hint">
        Files in <code>datasets/raw/</code>, read through the plugin registry — the extension
        picks the reader, so <code>.jsonl</code> and <code>.csv</code> work as well as plain
        text. Supported: {extensions.join(", ") || "—"}.
      </p>

      {error && <div className="error">{error}</div>}

      {listing.length === 0 ? (
        <div className="empty">
          No datasets found. Drop a {extensions.join(" / ") || "text"} file into{" "}
          <code>datasets/raw/</code>.
        </div>
      ) : (
        <div className="scroll-x">
          <table className="data">
            <thead>
              <tr>
                <th>Dataset</th>
                <th>Reader</th>
                <th style={{ textAlign: "right" }}>Documents</th>
                <th style={{ textAlign: "right" }}>Characters</th>
                <th style={{ textAlign: "right" }}>Size</th>
              </tr>
            </thead>
            <tbody>
              {listing.map((d) => (
                <tr
                  key={d.name}
                  data-kept={!d.error}
                  aria-selected={selected === d.name}
                  style={{
                    cursor: d.error ? "not-allowed" : "pointer",
                    outline: selected === d.name ? "2px solid var(--series-1)" : undefined,
                  }}
                  onClick={() => !d.error && setSelected(d.name)}
                >
                  <td className="mono">{d.name}</td>
                  <td>{d.error ? <span className="mono">unreadable</span> : d.plugin}</td>
                  <td className="num">{d.error ? "—" : d.num_documents.toLocaleString()}</td>
                  <td className="num">{d.error ? "—" : d.num_chars.toLocaleString()}</td>
                  <td className="num">{d.error ? "—" : formatBytes(d.num_bytes)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {chosen?.error && <div className="error">{chosen.error}</div>}

      {preview && preview.length > 0 && (
        <>
          <div className="legend" style={{ marginTop: 14 }}>
            First {preview.length} document{preview.length === 1 ? "" : "s"} of {selected}
          </div>
          <div
            style={{
              background: "var(--surface-0)",
              border: "1px solid var(--border)",
              borderRadius: 8,
              padding: 10,
              fontFamily: "var(--font-mono)",
              fontSize: 12,
              lineHeight: 1.6,
              maxHeight: 200,
              overflowY: "auto",
              whiteSpace: "pre-wrap",
            }}
          >
            {preview.map((doc, i) => (
              <div key={i} style={{ marginBottom: 8 }}>
                {doc.slice(0, 600)}
                {doc.length > 600 && <span style={{ color: "var(--text-muted)" }}>…</span>}
              </div>
            ))}
          </div>
        </>
      )}

      <div className="controls" style={{ marginTop: 18, alignItems: "flex-end" }}>
        <div className="control" style={{ maxWidth: 320 }}>
          <label htmlFor="prepare_config">Size the tokenizer from</label>
          <select
            id="prepare_config"
            value={config}
            onChange={(e) => setConfig(e.target.value)}
            disabled={state.running}
          >
            {BUILT_IN_CONFIGS.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
          <span className="note">
            vocab_size comes from this config — it must match the model's
            {versions.length > 0 && ` · ${versions.length} saved version(s)`}
          </span>
        </div>

        <button className="action" disabled={busy || !selected || state.running} onClick={prepare}>
          {preparing ? "Preparing…" : "Prepare dataset"}
        </button>
      </div>

      {notice && <div className="hint">{notice}</div>}
      {state.running && state.kind !== "prepare" && (
        <div className="hint">A training run is active — stop it before preparing a dataset.</div>
      )}

      {preparing && (
        <div className="stats">
          <div className="stat">
            <div className="label">Stage</div>
            <div className="value">{state.stage ?? "starting"}</div>
            <div className="sub">{state.runId}</div>
          </div>
        </div>
      )}
    </section>
  );
}
