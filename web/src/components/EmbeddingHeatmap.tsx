/** Embedding rows as a heatmap: one row per token, one column per dimension.
 *
 *  Diverging blue↔red with a neutral gray midpoint, because embedding values are
 *  signed and centred near zero — the sign carries meaning, and a sequential ramp
 *  would flatten "strongly negative" and "strongly positive" into the same end.
 *
 *  Drawn on a canvas: a 12M model has n_embd=384, so even ten tokens is ~4k
 *  cells. That is far too many DOM nodes, and a canvas repaints in one pass.
 */

import { useEffect, useMemo, useRef, useState } from "react";
import type { EmbeddingsResponse } from "../api/types";
import { currentMode, divergingColor, divergingLegend, robustLimit } from "../theme";

const ROW_HEIGHT = 18;
const LABEL_WIDTH = 92;

/** How many dimensions to show at once.
 *
 *  n_embd is 384 on the 12M model, which at typical panel width is under 3px per
 *  cell — that renders as moiré rather than data. Windowing keeps cells wide
 *  enough to read and hover; "all" is kept for the whole-row fingerprint. */
const DIM_WINDOWS = [32, 64, 128, 0] as const;

interface HoverCell {
  row: number;
  dim: number;
  value: number;
  x: number;
  y: number;
}

export function EmbeddingHeatmap({ data }: { data: EmbeddingsResponse }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [hover, setHover] = useState<HoverCell | null>(null);
  const [windowSize, setWindowSize] = useState<number>(64);
  const [offset, setOffset] = useState(0);
  const mode = currentMode();

  const shown = windowSize === 0 ? data.n_embd : Math.min(windowSize, data.n_embd);
  const start = Math.min(offset, Math.max(0, data.n_embd - shown));

  // Symmetric extent so zero always sits on the neutral band. Using vmin/vmax
  // independently would shift zero off-centre and imply a polarity the data does
  // not have.
  //
  // The extent is the 98th percentile of |v|, not the max: embedding values are
  // sharply peaked near zero with a long tail, and scaling to the single largest
  // value pushed ~86% of cells into the two innermost steps — a block of near
  // uniform colour carrying almost no information.
  const { limit, clipped } = useMemo(() => {
    const flat = data.vectors.flat();
    const l = robustLimit(flat, 0.98) || 1;
    return { limit: l, clipped: flat.filter((v) => Math.abs(v) > l).length };
  }, [data]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rows = data.vectors.length;
    const dims = shown;
    if (!rows || !dims) return;

    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = rows * ROW_HEIGHT;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const cellWidth = width / dims;
    for (let row = 0; row < rows; row++) {
      const vector = data.vectors[row];
      for (let dim = 0; dim < dims; dim++) {
        ctx.fillStyle = divergingColor(vector[start + dim], limit, mode);
        // Math.ceil on the width prevents sub-pixel seams between cells.
        ctx.fillRect(dim * cellWidth, row * ROW_HEIGHT, Math.ceil(cellWidth), ROW_HEIGHT - 1);
      }
    }
  }, [data, limit, mode, shown, start]);

  function onMove(event: React.MouseEvent<HTMLCanvasElement>) {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const row = Math.floor(y / ROW_HEIGHT);
    const dim = start + Math.floor((x / rect.width) * shown);
    if (row < 0 || row >= data.vectors.length || dim < start || dim >= start + shown) {
      setHover(null);
      return;
    }
    setHover({ row, dim, value: data.vectors[row][dim], x, y });
  }

  const legend = divergingLegend(mode);
  const maxOffset = Math.max(0, data.n_embd - shown);

  return (
    <div>
      {/* Filters in one row above the chart. */}
      <div className="controls" style={{ marginBottom: 12 }}>
        <div className="control" style={{ flex: "none", minWidth: 0 }}>
          <label>Dimensions shown</label>
          <div style={{ display: "flex", gap: 4 }}>
            {DIM_WINDOWS.map((size) => (
              <button
                key={size}
                className="ghost"
                aria-pressed={windowSize === size}
                onClick={() => {
                  setWindowSize(size);
                  setOffset(0);
                }}
              >
                {size === 0 ? `all ${data.n_embd}` : size}
              </button>
            ))}
          </div>
        </div>
        {maxOffset > 0 && (
          <div className="control">
            <label htmlFor="dim-offset">
              Starting at dim <span className="value">{start}</span>
            </label>
            <input
              id="dim-offset"
              type="range"
              min={0}
              max={maxOffset}
              step={1}
              value={start}
              onChange={(e) => setOffset(Number(e.target.value))}
            />
            <span className="note">
              showing {start}–{start + shown - 1} of {data.n_embd}
            </span>
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <div style={{ width: LABEL_WIDTH, flex: "none" }}>
          {data.labels.map((label, i) => (
            <div
              key={i}
              title={`id ${data.ids[i]}`}
              style={{
                height: ROW_HEIGHT,
                lineHeight: `${ROW_HEIGHT - 1}px`,
                fontFamily: "var(--font-mono)",
                fontSize: 11,
                color: hover?.row === i ? "var(--text-primary)" : "var(--text-secondary)",
                overflow: "hidden",
                textOverflow: "ellipsis",
                whiteSpace: "nowrap",
                textAlign: "right",
                paddingRight: 6,
              }}
            >
              {label.replace(/\n/g, "\\n").replace(/ /g, "␣") || "∅"}
            </div>
          ))}
        </div>
        <div style={{ position: "relative", flex: 1, minWidth: 0 }}>
          <canvas
            ref={canvasRef}
            style={{ width: "100%", display: "block", cursor: "crosshair" }}
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
          {hover && (
            <div
              role="tooltip"
              style={{
                position: "absolute",
                left: Math.min(hover.x + 12, 240),
                top: hover.y + 14,
                background: mode === "dark" ? "#2c2c2a" : "#0b0b0b",
                color: "#fff",
                padding: "6px 9px",
                borderRadius: 6,
                fontSize: 11.5,
                pointerEvents: "none",
                whiteSpace: "nowrap",
                zIndex: 5,
              }}
            >
              {`dim ${hover.dim} · ${hover.value >= 0 ? "+" : ""}${hover.value.toFixed(4)}`}
            </div>
          )}
        </div>
      </div>

      <div className="legend" style={{ marginTop: 10 }}>
        <span>{(-limit).toFixed(3)}</span>
        <div style={{ display: "flex", gap: 0 }}>
          {legend.map((color, i) => (
            <span
              key={i}
              className="swatch"
              style={{ background: color, width: 16, height: 10, borderRadius: 0 }}
            />
          ))}
        </div>
        <span>+{limit.toFixed(3)}</span>
        <span style={{ color: "var(--text-muted)" }}>
          {data.vectors.length} tokens × {shown}
          {shown < data.n_embd ? ` of ${data.n_embd}` : ""} dims
          {clipped > 0 && ` · ${clipped} values beyond ±${limit.toFixed(3)} clipped`}
        </span>
      </div>
    </div>
  );
}
