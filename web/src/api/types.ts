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

/** The hyperparameters that define an architecture — mirrors ArchitectureParams. */
export interface ArchitectureConfig {
  vocab_size: number;
  n_layer: number;
  n_head: number;
  n_embd: number;
  block_size: number;
  /** d_ff. Named for the C++ field so the wire shape stays one vocabulary. */
  ffn_hidden: number;
  rope_theta: number;
  norm_eps: number;
}

export interface ParamBreakdown {
  embedding: number;
  attention: number;
  ffn: number;
  norms: number;
  total: number;
}

/** Bytes required to *train* at the requested batch/sequence shape. */
export interface MemoryEstimate {
  params: number;
  gradients: number;
  optimizer: number;
  activations: number;
  total: number;
}

export interface DeviceInfo {
  kind: "cuda" | "apple-silicon" | "cpu";
  device: string;
  /** "VRAM" | "Unified memory" | "RAM" — whatever this host actually has. */
  memory_label: string;
  total_bytes: number;
  used_bytes: number;
  source: string;
}

export interface EstimateResponse {
  config: ArchitectureConfig;
  num_params: number;
  params: ParamBreakdown;
  memory: MemoryEstimate;
  device: DeviceInfo;
  /** False when the training footprint exceeds this host's memory. */
  fits: boolean;
}

export interface DatasetInfo {
  name: string;
  path: string;
  /** Which registered plugin reads this extension. */
  plugin: string;
  num_documents: number;
  num_chars: number;
  num_bytes: number;
  sample: string;
  /** Present instead of the counts when the file could not be read. */
  error?: string;
}

export interface DatasetListing {
  supported_extensions: string[];
  datasets: DatasetInfo[];
}

export interface ExportRequest {
  checkpoint: string;
  tokenizer_dir: string;
  out: string;
  include_runtime: boolean;
  include_cpp: boolean;
}

export interface ExportResponse {
  out_dir: string;
  name: string;
  num_params: number;
  /** Relative path -> bytes, including anything under runtime/ or cpp/. */
  files: Record<string, number>;
  total_bytes: number;
  includes: string[];
}

export interface ExportSummary {
  name: string;
  path: string;
  num_params: number | null;
  total_bytes: number;
  file_count: number;
  exported_at: string | null;
  includes: string[];
}

/** A dataset that has been tokenized and binarized — ready to train on. */
export interface PreparedDataset {
  name: string;
  data_dir: string;
  tokenizer_dir: string;
  train_tokens: number;
  val_tokens: number;
  vocab_size: number | null;
  prepared_at: number;
}

export interface TrainStatus {
  run_id: string | null;
  /** "train" | "prepare", or null when idle. */
  kind: string | null;
  running: boolean;
  /** SIGSTOPped: still resident, not consuming CPU. */
  paused: boolean;
  step: number;
  total_steps: number;
  progress: number;
  last_loss: number | null;
  best_val: number | null;
  elapsed_s: number;
  returncode: number | null;
  command: string;
  subscribers: number;
  pid: number | null;
}

export interface ConfigVersion {
  version_id: string;
  index: number;
  config: ArchitectureConfig;
  num_params: number;
  activation_bytes: number;
  created_at: number;
  note: string;
  /** False when the config already existed — submission is idempotent. */
  created: boolean;
  checkpoint: string | null;
}
