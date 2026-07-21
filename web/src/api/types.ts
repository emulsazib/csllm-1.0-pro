/** Mirrors gateway/schemas.py. Kept hand-written rather than generated so the
 *  shapes the UI depends on are visible in one place. */

export interface Health {
  status: "ok";
  version: string;
  num_params: number;
  vocab_size: number;
  block_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  blas_backend: string;
  max_concurrent_sessions: number;
  sessions_in_flight: number;
  kv_cache_bytes_per_session: number;
}

export interface TokenInfo {
  index: number;
  id: number;
  text: string;
  /** Raw bytes — a token need not be valid UTF-8 on its own. */
  bytes: number[];
  /** Byte offsets into the UTF-8 encoding of the input. */
  start: number;
  end: number;
  /** True when this token alone is not decodable. */
  partial_utf8: boolean;
}

export interface TokenizeResponse {
  tokens: TokenInfo[];
  count: number;
  num_chars: number;
  num_bytes: number;
  compression: number;
  vocab_size: number;
}

export interface EmbeddingsResponse {
  ids: number[];
  labels: string[];
  n_embd: number;
  /** [n_tokens, n_embd] */
  vectors: number[][];
  vmin: number;
  vmax: number;
  /** [n_tokens, 3] PCA projection; empty when fewer than 3 tokens. */
  projection: number[][];
  explained_variance: number[];
}

export interface CandidateToken {
  id: number;
  text: string;
  logit: number;
  /** Unfiltered softmax at temperature 1 — the model's own belief. */
  raw_prob: number;
  /** What the sampler would actually draw from; 0 when filtered out. */
  prob: number;
  kept: boolean;
}

export interface InspectResponse {
  prompt_tokens: number;
  candidates: CandidateToken[];
  kept_count: number;
  raw_entropy: number;
  filtered_entropy: number;
  vocab_size: number;
}

export interface SamplingSettings {
  temperature: number;
  top_k: number;
  top_p: number;
}
