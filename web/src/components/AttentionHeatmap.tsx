/** Attention as a query x key matrix.
 *
 *  One row per generated token, one column per key it could look at. Decoding is
 *  causal, so row `i` is one cell shorter than row `i+1` and the matrix is a
 *  staircase — cells past a row's end are drawn as the surface, not as zero
 *  weight, because "could not attend" and "chose not to" are different claims.
 *
 *  Weights are magnitudes in [0,1] with no polarity, so the ramp is SEQUENTIAL
 *  (theme.ts, shared with the 3D view). A diverging scale would invent a
 *  midpoint that means nothing here.
 *
 *  Drawn on a canvas: 24 tokens against a 40-key context is ~1000 cells, and a
 *  canvas repaints in one pass instead of churning that many DOM nodes.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { AttentionBlock } from "../api/ws";
import { attentionMatrix } from "../api/ws";
import { currentMode, robustLimit, sequentialColor, sequentialLegend } from "../theme";
import { tokenLabel } from "./ProbabilityChart";

const ROW_HEIGHT = 16;
const LABEL_WIDTH = 96;
/** Below this the column labels are unreadable, so they are dropped entirely
 *  rather than drawn as overlapping smears. */
const MIN_LABELLED_CELL = 13;
const MIN_CELL = 5;
const MAX_CELL = 34;

interface Hover {
  row: number;
  key: number;
  value: number | null;
  x: number;
  y: number;
}

interface Props {
  tokens: { text: string; attention?: AttentionBlock }[];
  /** Prompt tokens followed by generated ones — the key axis labels. */
  keyLabels: string[];
  layer: number;
  head: number | "all";
  width?: number;
}

export function AttentionHeatmap({ tokens, keyLabels, layer, head, width = 860 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<Hover | null>(null);
  const mode = currentMode();

  const { rows, keys } = useMemo(
    () => attentionMatrix(tokens, layer, head),
    [tokens, layer, head],
  );

  // A robust extent, not the max: self-attention routinely takes a large share
  // of one row, and scaling to it flattens every other cell to the palest step.
  // Same reasoning as the embedding heatmap, which hit this with outliers.
  const { limit, clipped } = useMemo(() => {
    const flat: number[] = [];
    for (const row of rows) for (const v of row) flat.push(v);
    const l = robustLimit(flat, 0.98) || 1;
    return { limit: l, clipped: flat.filter((v) => v > l).length };
  }, [rows]);

  const cell = keys
    ? Math.max(MIN_CELL, Math.min(MAX_CELL, (width - LABEL_WIDTH) / keys))
    : MIN_CELL;
  const gridWidth = keys * cell;
  const showColumnLabels = cell >= MIN_LABELLED_CELL;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !rows.length || !keys) return;

    const height = rows.length * ROW_HEIGHT;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.floor(gridWidth * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = `${gridWidth}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, gridWidth, height);

    for (let r = 0; r < rows.length; r++) {
      const row = rows[r];
      for (let k = 0; k < row.length; k++) {
        ctx.fillStyle = sequentialColor(row[k] / limit, mode);
        // Math.ceil prevents sub-pixel seams between adjacent cells.
        ctx.fillRect(k * cell, r * ROW_HEIGHT, Math.ceil(cell), ROW_HEIGHT - 1);
      }
    }
  }, [rows, keys, limit, mode, cell, gridWidth]);

  function onMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const row = Math.floor(y / ROW_HEIGHT);
    const key = Math.floor(x / cell);
    if (row < 0 || row >= rows.length || key < 0 || key >= keys) {
      setHover(null);
      return;
    }
    const value = key < rows[row].length ? rows[row][key] : null;
    setHover({ row, key, value, x, y });
  }

  if (!rows.length) {
    return <div className="empty">No attention captured yet.</div>;
  }

  const legend = sequentialLegend(mode);

  return (
    <div>
      <div className="scroll-x">
        <div style={{ display: "flex", minWidth: LABEL_WIDTH + gridWidth }}>
          {/* Query labels: which generated token each row belongs to. */}
          <div style={{ width: LABEL_WIDTH, flex: "none" }}>
            {showColumnLabels && <div style={{ height: 18 }} />}
            {rows.map((_, r) => (
              <div
                key={r}
                title={tokens[r]?.text}
                style={{
                  height: ROW_HEIGHT,
                  lineHeight: `${ROW_HEIGHT - 1}px`,
                  fontFamily: "var(--font-mono)",
                  fontSize: 10.5,
                  textAlign: "right",
                  paddingRight: 6,
                  color: hover?.row === r ? "var(--text-primary)" : "var(--text-secondary)",
                  whiteSpace: "nowrap",
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
              >
                {tokenLabel(tokens[r]?.text ?? "")}
              </div>
            ))}
          </div>

          <div style={{ position: "relative" }}>
            {showColumnLabels && (
              <div style={{ display: "flex", height: 18 }}>
                {Array.from({ length: keys }, (_, k) => (
                  <div
                    key={k}
                    title={keyLabels[k]}
                    style={{
                      width: cell,
                      flex: "none",
                      fontFamily: "var(--font-mono)",
                      fontSize: 9.5,
                      color: hover?.key === k ? "var(--text-primary)" : "var(--text-muted)",
                      overflow: "hidden",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {tokenLabel(keyLabels[k] ?? "").slice(0, 3)}
                  </div>
                ))}
              </div>
            )}

            <canvas
              ref={canvasRef}
              onMouseMove={onMove}
              onMouseLeave={() => setHover(null)}
              style={{ display: "block", cursor: "crosshair" }}
            />

            {hover && (
              <div
                style={{
                  position: "absolute",
                  left: Math.min(hover.x + 12, gridWidth - 190),
                  top: hover.y + (showColumnLabels ? 26 : 8),
                  pointerEvents: "none",
                  background: "var(--surface-1)",
                  border: "1px solid var(--border)",
                  borderRadius: 6,
                  padding: "5px 8px",
                  fontSize: 11.5,
                  fontFamily: "var(--font-mono)",
                  whiteSpace: "nowrap",
                  zIndex: 2,
                }}
              >
                {tokenLabel(tokens[hover.row]?.text ?? "")} →{" "}
                {tokenLabel(keyLabels[hover.key] ?? String(hover.key))}
                {" · "}
                {hover.value === null ? (
                  <span style={{ color: "var(--text-muted)" }}>not yet in context</span>
                ) : (
                  `${(hover.value * 100).toFixed(1)}%`
                )}
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="legend" style={{ marginTop: 10 }}>
        <span className="item">
          rows = generated tokens · columns = keys attended to
        </span>
        <span className="item">
          0
          {legend.map((colour, i) => (
            <span
              key={i}
              className="swatch"
              style={{ background: colour, width: 12, height: 10, borderRadius: 0 }}
            />
          ))}
          {(limit * 100).toFixed(0)}%
        </span>
        {clipped > 0 && (
          <span className="item" style={{ color: "var(--text-muted)" }}>
            {clipped} cell{clipped === 1 ? "" : "s"} above the scale, drawn at the top step
          </span>
        )}
        {!showColumnLabels && (
          <span className="item" style={{ color: "var(--text-muted)" }}>
            column labels hidden at this width — hover a cell
          </span>
        )}
      </div>
    </div>
  );
}
