# release — CSLLM export bundle

12,194,688 parameters · 6 layers × 6 heads × 384 d_model ·
context 256 · vocab 4096

Exported 2026-07-22T08:24:36+00:00 by CSLLM engine 1.0.0.

## Files

| File | What it is |
| --- | --- |
| `model.safetensors` | Weights, fp32, flat dotted names. JSON header + raw little-endian bytes. |
| `tokenizer.json` | Byte-level BPE: pattern, merges in rank order, vocab as byte lists. |
| `config.json` | Hyperparameters, architecture flags, and provenance. |
| `runtime/load.py` | Standalone loader — no torch, no dependency on the source repo. |

## Architecture

Llama-style decoder-only Transformer:

- **Normalisation** — RMSNorm, pre-norm residuals (`norm_eps = 9.999999747378752e-06`)
- **Positions** — RoPE, **interleaved** channel pairs (`rope_theta = 10000.0`).
  Pairs are `(0,1), (2,3), …`, not the split-half convention some implementations use.
- **Feed-forward** — SwiGLU, three matrices per block: `wg` (gate), `wu` (up), `wd` (down)
- **Embeddings** — `lm_head` is **weight-tied** to `tok_emb`; there is no separate output
  matrix, and the parameter count includes the embedding exactly once
- **Attention** — causal multi-head, `head_dim = n_embd / n_head = 64` (even, as RoPE
  rotates adjacent pairs)

## Tensor names

```
tok_emb                              [vocab_size, n_embd]  also the lm_head (tied)
blocks.{i}.attn_norm.gain            [n_embd]
blocks.{i}.attn.wq / wk / wv / wo    [n_embd, n_embd]
blocks.{i}.ffn_norm.gain             [n_embd]
blocks.{i}.ffn.wg / wu               [n_embd, ffn_hidden]
blocks.{i}.ffn.wd                    [ffn_hidden, n_embd]
norm_f.gain                          [n_embd]
```

`config.json` carries the exact shapes under `tensors` — read them from there rather than
deriving them, so a future layout change surfaces as a mismatch instead of silent garbage.

## Loading it

Anything that reads safetensors works — torch, JAX, numpy — none of which this bundle
depends on. Weights were written with the **numpy** backend precisely so reading them
requires no framework.

```python
from safetensors.numpy import load_file
weights = load_file("model.safetensors")
```

### Bundled loader

```bash
pip install -r runtime/requirements.txt
python runtime/load.py          # self-check: loads, encodes, round-trips
```

```python
from runtime.load import CSLLMBundle
bundle = CSLLMBundle('.')
ids = bundle.tokenizer.encode('KING RICHARD:')
```
