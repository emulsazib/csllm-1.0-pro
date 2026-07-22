/** Tokenization + embedding inspection.
 *
 *  Shows how text is actually segmented (including tokens that are not valid
 *  UTF-8 on their own) and what the embedding rows for those tokens look like.
 */

import { useMemo, useState } from "react";
import { api } from "../api/client";
import { useAsync, useDebounced } from "../hooks/useDebounced";
import { EmbeddingHeatmap } from "./EmbeddingHeatmap";
import { TOKEN_TINTS, TokenText } from "./TokenText";

const DEFAULT_TEXT = "KING RICHARD:\nBut this is it, I have thought it with the world.";

export function TokenizerPanel() {
  const [text, setText] = useState(DEFAULT_TEXT);
  const [selected, setSelected] = useState<number | null>(null);
  const debounced = useDebounced(text, 250);
  const trimmed = debounced.trim();

  const tokenize = useAsync(
    (signal) => api.tokenize(debounced, signal),
    [debounced],
    trimmed.length > 0,
  );
  const embeddings = useAsync(
    (signal) => api.embeddings(debounced, true, signal),
    [debounced],
    trimmed.length > 0,
  );

  const partialCount = useMemo(
    () => tokenize.data?.tokens.filter((t) => t.partial_utf8).length ?? 0,
    [tokenize.data],
  );

  return (
    <>
      <section className="panel">
        <h2>Tokenization</h2>
        <p className="hint">
          Byte-level BPE. Hover a token for its id and byte span; a dashed border marks a
          token that is not valid UTF-8 on its own.
        </p>

        <textarea
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          aria-label="Text to tokenize"
          spellCheck={false}
        />

        {tokenize.error && <div className="error">{tokenize.error}</div>}

        {tokenize.data && (
          <>
            <div className="stats">
              <div className="stat">
                <div className="label">Tokens</div>
                <div className="value">{tokenize.data.count.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="label">Characters</div>
                <div className="value">{tokenize.data.num_chars.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="label">Bytes</div>
                <div className="value">{tokenize.data.num_bytes.toLocaleString()}</div>
              </div>
              <div className="stat">
                <div className="label">Bytes / token</div>
                <div className="value">{tokenize.data.compression.toFixed(2)}</div>
                <div className="sub">compression</div>
              </div>
              <div className="stat">
                <div className="label">Vocabulary</div>
                <div className="value">{tokenize.data.vocab_size.toLocaleString()}</div>
              </div>
            </div>

            <div className="token-stream" style={{ marginTop: 12 }}>
              {tokenize.data.tokens.map((token) => (
                <span
                  key={token.index}
                  className="token"
                  data-partial={token.partial_utf8}
                  data-selected={selected === token.index}
                  style={{ background: TOKEN_TINTS[token.index % TOKEN_TINTS.length] }}
                  title={`id ${token.id} · bytes ${token.start}–${token.end} · [${token.bytes.join(", ")}]`}
                  onMouseEnter={() => setSelected(token.index)}
                  onMouseLeave={() => setSelected(null)}
                >
                  <TokenText text={token.text} />
                </span>
              ))}
            </div>

            <div className="legend">
              <span className="item">
                <span className="swatch" style={{ background: TOKEN_TINTS[0] }} /> token boundary
              </span>
              <span className="item">
                <span
                  className="swatch"
                  style={{ border: "1px dashed var(--series-2)", background: "transparent" }}
                />
                partial UTF-8 ({partialCount})
              </span>
              <span className="item" style={{ color: "var(--text-muted)" }}>
                ␣ space · ↵ newline · ⇥ tab
              </span>
            </div>
          </>
        )}

        {!trimmed && <div className="empty">Enter some text to tokenize.</div>}
      </section>

      <section className="panel">
        <h2>Embeddings</h2>
        <p className="hint">
          Each row is one token&apos;s embedding vector, read live from the model&apos;s{" "}
          <code>tok_emb</code> table. Blue is negative, red positive, gray near zero.
        </p>

        {embeddings.error && <div className="error">{embeddings.error}</div>}

        {embeddings.data ? (
          <>
            <EmbeddingHeatmap data={embeddings.data} />
            {embeddings.data.explained_variance.length > 0 && (
              <p className="hint" style={{ marginTop: 14, marginBottom: 0 }}>
                PCA to 3D explains{" "}
                <strong>
                  {(
                    embeddings.data.explained_variance.reduce((a, b) => a + b, 0) * 100
                  ).toFixed(1)}
                  %
                </strong>{" "}
                of the variance across these tokens (
                {embeddings.data.explained_variance
                  .map((v) => `${(v * 100).toFixed(1)}%`)
                  .join(" · ")}
                ). Used by the attention graph.
              </p>
            )}
          </>
        ) : (
          !embeddings.loading && <div className="empty">No embeddings yet.</div>
        )}
      </section>
    </>
  );
}
