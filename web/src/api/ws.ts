/** WebSocket helpers.
 *
 *  The inspect stream interleaves two frame kinds: a JSON token frame, then —
 *  when it declares an `attn` block — the raw float32 attention. `parseAttention`
 *  is the pairing rule, kept pure so it can be tested without a socket.
 */

export const WS_BASE = (() => {
  if (typeof window === "undefined") return "";
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${protocol}//${window.location.host}`;
})();

export interface AttentionBlock {
  /** [layer, head, key] weights, sliced to the layers/heads requested. */
  data: Float32Array;
  layers: number;
  heads: number;
  keys: number;
}

/** Read one [layer][head] row out of the flat block. */
export function attentionRow(block: AttentionBlock, layer: number, head: number): Float32Array {
  const offset = (layer * block.heads + head) * block.keys;
  return block.data.subarray(offset, offset + block.keys);
}

/** Mean attention each key receives across every selected layer and head. */
export function attentionByKey(block: AttentionBlock): Float32Array {
  const out = new Float32Array(block.keys);
  const rows = block.layers * block.heads;
  for (let r = 0; r < rows; r++) {
    const base = r * block.keys;
    for (let k = 0; k < block.keys; k++) out[k] += block.data[base + k];
  }
  if (rows > 0) for (let k = 0; k < block.keys; k++) out[k] /= rows;
  return out;
}

/** Mean attention a single layer gives each key, averaged over its heads. */
export function attentionByLayerKey(block: AttentionBlock, layer: number): Float32Array {
  const out = new Float32Array(block.keys);
  for (let h = 0; h < block.heads; h++) {
    const row = attentionRow(block, layer, h);
    for (let k = 0; k < block.keys; k++) out[k] += row[k];
  }
  if (block.heads > 0) for (let k = 0; k < block.keys; k++) out[k] /= block.heads;
  return out;
}

/**
 * The weights one query used, for a chosen layer and head.
 *
 * `"all"` averages the layer's heads rather than picking one — the same
 * reduction the 3D view uses, so the two panels agree on what "all heads" means.
 *
 * Indices are CLAMPED to the block. Panels hold layer/head in component state
 * that outlives a generation, so switching to a shallower model would otherwise
 * read past the end of the buffer and render whatever followed it as attention.
 */
export function attentionVector(
  block: AttentionBlock,
  layer: number,
  head: number | "all",
): Float32Array {
  const l = Math.max(0, Math.min(block.layers - 1, layer));
  if (head === "all") return attentionByLayerKey(block, l);
  return attentionRow(block, l, Math.max(0, Math.min(block.heads - 1, head)));
}

/**
 * Stack per-token attention into a query x key matrix.
 *
 * Each generated token is one query row. Decoding is causal and the context
 * grows by one per step, so row `i` is SHORTER than row `i+1` — the matrix is
 * ragged, and callers must treat missing cells as "not attended to", not zero
 * weight. `keys` is the widest row.
 */
export function attentionMatrix(
  tokens: { attention?: AttentionBlock }[],
  layer: number,
  head: number | "all",
): { rows: Float32Array[]; keys: number } {
  const rows: Float32Array[] = [];
  let keys = 0;
  for (const token of tokens) {
    if (!token.attention) continue;
    const row = attentionVector(token.attention, layer, head);
    rows.push(row);
    keys = Math.max(keys, row.length);
  }
  return { rows, keys };
}

export function parseAttention(shape: number[], buffer: ArrayBuffer): AttentionBlock {
  const [layers, heads, keys] = shape;
  const expected = layers * heads * keys * 4;
  if (buffer.byteLength !== expected) {
    throw new Error(
      `attention frame is ${buffer.byteLength} bytes, expected ${expected} for [${shape}]`,
    );
  }
  return { data: new Float32Array(buffer), layers, heads, keys };
}

// ── message shapes ───────────────────────────────────────────────────────────

export interface InspectCandidate {
  id: number;
  text: string;
  logit: number;
  raw_prob: number;
  prob: number;
  kept: boolean;
}

export interface InspectStart {
  type: "start";
  prompt: { id: number; text: string }[];
  config: {
    n_layer: number;
    n_head: number;
    n_embd: number;
    block_size: number;
    vocab_size: number;
  };
  layers: number[];
  heads: number[];
  max_tokens: number;
  attention: boolean;
}

export interface InspectToken {
  type: "token";
  index: number;
  id: number;
  text: string;
  position: number;
  top?: InspectCandidate[];
  kept_count?: number;
  raw_entropy?: number;
  filtered_entropy?: number;
  attn?: { shape: number[]; bytes: number };
  /** Attached client-side from the binary frame that follows. */
  attention?: AttentionBlock;
}

export type InspectMessage =
  | InspectStart
  | InspectToken
  | { type: "done"; reason: string; tokens: number }
  | { type: "error"; message: string };

export interface TrainEvent {
  type: string;
  step?: number;
  loss?: number;
  lr?: number;
  grad_norm?: number;
  val_loss?: number;
  best_val?: number;
  message?: string;
  steps?: number;
  running?: boolean;
  returncode?: number;
  run_id?: string | null;
  total_steps?: number;
  tokens_per_s?: number;
  ms_per_step?: number;
  count?: number;
  /** "train" | "prepare" — which job produced this event. */
  kind?: string | null;
  paused?: boolean;
  /** Fractional passes over the corpus, from the trainer's `throughput` rows. */
  epoch?: number;
  /** Trainer RSS. Labelled by `memory_label` from the `start` row. */
  rss_bytes?: number;
  memory_label?: string;
  memory_total_bytes?: number;
  device?: string;
  /** Prepare-job progress: "bpe" | "roundtrip" | "binarize" | "complete". */
  stage?: string;
  [key: string]: unknown;
}
