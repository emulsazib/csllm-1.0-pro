/** Package a trained checkpoint for deployment elsewhere.
 *
 *  The bundle is always the three portable files (safetensors + tokenizer.json +
 *  config.json). The two toggles add deployment packages on top: a torch-free
 *  Python loader, and the C++20 engine sources with a CMakeLists that builds
 *  them standalone.
 *
 *  Download is a plain navigation rather than a fetch-into-blob: a 12M bundle is
 *  ~49 MB, and buffering that in JS to hand back to the same browser only adds a
 *  copy and loses the filename.
 */

import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ExportResponse, ExportSummary } from "../api/types";
import { formatBytes } from "./ConfiguratorPanel";

interface Props {
  onClose: () => void;
}

export function ExportModal({ onClose }: Props) {
  const [checkpoint, setCheckpoint] = useState("data/model.csllm");
  const [tokenizerDir, setTokenizerDir] = useState("data/tokenizer");
  const [name, setName] = useState("release");
  const [includeRuntime, setIncludeRuntime] = useState(true);
  const [includeCpp, setIncludeCpp] = useState(false);

  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<ExportResponse | null>(null);
  const [existing, setExisting] = useState<ExportSummary[]>([]);

  async function refresh() {
    try {
      setExisting(await api.exports());
    } catch {
      /* the modal still works without the history */
    }
  }

  useEffect(() => {
    void refresh();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function build() {
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const response = await api.exportBundle({
        checkpoint,
        tokenizer_dir: tokenizerDir,
        out: `exports/${name}`,
        include_runtime: includeRuntime,
        include_cpp: includeCpp,
      });
      setResult(response);
      await refresh();
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div
      className="modal-backdrop"
      role="dialog"
      aria-modal="true"
      aria-label="Export model"
      onClick={(e) => e.target === e.currentTarget && onClose()}
    >
      <div className="modal">
        <div className="modal-head">
          <h2>Export model</h2>
          <button className="ghost" onClick={onClose} aria-label="Close">
            ✕
          </button>
        </div>

        <p className="hint">
          Writes <code>model.safetensors</code>, <code>tokenizer.json</code> and{" "}
          <code>config.json</code> — readable by torch, JAX or numpy, none of which this
          project depends on.
        </p>

        <div className="controls" style={{ alignItems: "flex-end" }}>
          <div className="control">
            <label htmlFor="ex-checkpoint">Checkpoint</label>
            <input
              id="ex-checkpoint"
              type="text"
              value={checkpoint}
              onChange={(e) => setCheckpoint(e.target.value)}
            />
            {/* Every control in this row needs a one-line note: `.controls`
                bottom-aligns, so one without a note sits lower than its peers. */}
            <span className="note">the trained weights to package</span>
          </div>
          <div className="control">
            <label htmlFor="ex-tokenizer">Tokenizer directory</label>
            <input
              id="ex-tokenizer"
              type="text"
              value={tokenizerDir}
              onChange={(e) => setTokenizerDir(e.target.value)}
            />
            {/* One line: `.controls` bottom-aligns, so a wrapping note lifts this
                control's label out of the row. */}
            <span className="note">must match the checkpoint</span>
          </div>
          <div className="control" style={{ maxWidth: 200 }}>
            <label htmlFor="ex-name">Bundle name</label>
            <input
              id="ex-name"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value.replace(/[^\w.-]/g, "-"))}
            />
            <span className="note">exports/{name || "…"}</span>
          </div>
        </div>

        <div className="controls">
          <label className="toggle">
            <input
              type="checkbox"
              checked={includeRuntime}
              onChange={(e) => setIncludeRuntime(e.target.checked)}
            />
            <span>
              <strong>Python runtime</strong>
              <span className="note">
                standalone loader — no torch, no dependency on this repo
              </span>
            </span>
          </label>
          <label className="toggle">
            <input
              type="checkbox"
              checked={includeCpp}
              onChange={(e) => setIncludeCpp(e.target.checked)}
            />
            <span>
              <strong>C++ engine</strong>
              <span className="note">
                headers, sources and a CMakeLists that builds them standalone
              </span>
            </span>
          </label>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", marginTop: 4 }}>
          <button className="action" disabled={busy || !name} onClick={build}>
            {busy ? "Packaging…" : "Build bundle"}
          </button>
          {result && (
            <a className="action" href={api.downloadUrl(result.name)} download>
              Download {result.name}.zip
            </a>
          )}
        </div>

        {error && <div className="error" style={{ marginTop: 12 }}>{error}</div>}

        {result && (
          <>
            <div className="stats">
              <div className="stat">
                <div className="label">Parameters</div>
                <div className="value">{result.num_params.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="label">Bundle size</div>
                <div className="value">{formatBytes(result.total_bytes)}</div>
                <div className="sub">{Object.keys(result.files).length} files</div>
              </div>
              {result.includes.length > 0 && (
                <div className="stat">
                  <div className="label">Packages</div>
                  <div className="value" style={{ fontSize: 14 }}>
                    {result.includes.join(" · ")}
                  </div>
                </div>
              )}
            </div>

            <div className="scroll-x" style={{ marginTop: 8, maxHeight: 190, overflowY: "auto" }}>
              <table className="data">
                <thead>
                  <tr>
                    <th>File</th>
                    <th style={{ textAlign: "right" }}>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(result.files)
                    .sort(([, a], [, b]) => b - a)
                    .map(([file, size]) => (
                      <tr key={file}>
                        <td className="mono">{file}</td>
                        <td className="num">{formatBytes(size)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          </>
        )}

        {existing.length > 0 && (
          <>
            <p className="hint" style={{ marginTop: 18, marginBottom: 6 }}>
              Previously exported:
            </p>
            <div className="scroll-x">
              <table className="data">
                <thead>
                  <tr>
                    <th>Bundle</th>
                    <th style={{ textAlign: "right" }}>Parameters</th>
                    <th style={{ textAlign: "right" }}>Size</th>
                    <th>Packages</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {existing.map((bundle) => (
                    <tr key={bundle.name}>
                      <td className="mono">{bundle.name}</td>
                      <td className="num">
                        {bundle.num_params?.toLocaleString() ?? "—"}
                      </td>
                      <td className="num">{formatBytes(bundle.total_bytes)}</td>
                      <td>
                        {bundle.includes.length ? (
                          bundle.includes.join(" · ")
                        ) : (
                          <span style={{ color: "var(--text-muted)" }}>bundle only</span>
                        )}
                      </td>
                      <td>
                        <a className="ghost" href={api.downloadUrl(bundle.name)} download>
                          Download
                        </a>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
