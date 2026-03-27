# Autoresearch Program: English vs. Loga LLM Efficiency Experiment

## Experiment Objective

We are testing two independent efficiency hypotheses simultaneously via a 2×2
factorial design:

**H1 (corpus)**: A model trained on Loga (a purpose-built conlang) achieves
better val_bpb than an identical model trained on English, because Loga's
regular 95-char ASCII morphology aligns naturally with BPE tokenization.

**H2 (quantization)**: Ternary-weight training (weights ∈ {-1, 0, +1},
BitNet b1.58 style) achieves comparable val_bpb to float16 training while
using dramatically less memory and compute.

**H3 (interaction)**: H1 and H2 compound — a Loga model may tolerate ternary
weight quantization *better* than an English model, because the lower
information complexity of the Loga training distribution requires less
numerical precision to represent.

**Primary metric**: `val_bpb` (validation bits-per-byte). Lower is better.

---

## Experimental Setup

Four training runs in a 2×2 factorial design:

| Run | Corpus | Weights | Priority |
|-----|--------|---------|----------|
| `A-english-fp16`    | `data/prepared/english/` | float16                | 1st — establishes BPB floor |
| `B-loga-fp16`       | `data/prepared/loga/`    | float16                | 2nd — tests H1 |
| `C-english-ternary` | `data/prepared/english/` | ternary {-1, 0, +1}   | 3rd — tests H2 |
| `D-loga-ternary`    | `data/prepared/loga/`    | ternary {-1, 0, +1}   | 4th — tests H3 (novel) |

Run in order A → B → C → D. All four use identical architecture. The only
differences are data directory, tokenizer, and weight quantization mode.

### Ternary Weight Implementation (Runs C and D)

Use BitNet b1.58-style quantization (Ma et al., arXiv:2402.17764):
- Replace `Linear` layers with `BitLinear`: weights quantized via `absmean`
  rounding to {-1, 0, +1} on each forward pass
- Activations remain at float16 — fully ternary activations are not
  production-ready for LLMs and will be skipped
- Straight-through estimator passes gradients through the quantization step
- Per-layer float scaling factor recovers weight magnitude

Reference implementation for MLX: https://github.com/exo-explore/mlx-bitnet
Microsoft reference kernels (bitnet.cpp): https://github.com/microsoft/BitNet

---

## What autoresearch Should Explore

Work exclusively on `train.py`. Do NOT modify `prepare.py`.

Explore the following axes (in priority order):

### 1. Architecture hyperparameters
- `n_layer`: try 4, 6, 8, 12
- `n_head`: try 4, 6, 8
- `n_embd`: try 128, 256, 384, 512
- `block_size` (context length): try 256, 512, 1024
- Goal: find smallest model that achieves val_bpb < 1.5

### 2. Optimizer and learning rate schedule
- `learning_rate`: sweep 1e-4, 3e-4, 6e-4, 1e-3
- `lr_decay`: cosine vs. linear
- `warmup_iters`: try 50, 100, 200
- `weight_decay`: try 0.0, 0.01, 0.1

### 3. Regularization
- `dropout`: try 0.0, 0.1, 0.2
- Gradient clipping: try max_norm 0.5, 1.0, 5.0

### 4. Batch size
- Try `batch_size` values that saturate M4 Max unified memory without OOM:
  start at 32, increment by 32 until memory is full
- Prefer larger batch sizes; they generally improve BPB at fixed step budget

### 5. Ternary-specific hyperparameters (Runs C and D only)
- `ternary_threshold`: the absmean multiplier for the quantization boundary.
  Default is 1.0 (threshold = mean(|W|)). Try 0.5, 1.0, 1.5.
- `ternary_warmup`: number of steps before quantization is applied. Try 0,
  100, 500. Starting with full precision and gradually introducing quantization
  often stabilises training.
- Expect ~30-50% of weights to be zero (natural sparsity from ternary
  quantization). If sparsity is <20% or >70%, adjust threshold.

### 6. Positional encoding variants (advanced)
- Default: learned absolute PE
- Try: RoPE (rotary position embedding) if implementable in MLX
- Try: no positional encoding (pure attention on short sequences)

---

## Experiment Protocol

Run each variation for exactly **5 minutes** of wall-clock training time.
Record `val_bpb` at the end of the 5-minute window.

Accept a change if `val_bpb` improves by ≥ 0.01 (significant improvement).
Revert if `val_bpb` worsens by any amount or shows no meaningful change.

**Four sequential tracks** (run in order, sharing hyperparameter discoveries):
1. Run A (English fp16) — establishes best architecture config and BPB floor.
2. Run B (Loga fp16) — use same best architecture from A; only vary corpus.
3. Run C (English ternary) — apply best architecture from A + ternary weights.
4. Run D (Loga ternary) — use best config from B + ternary weights.

When beginning Runs C and D, start autoresearch from the best `train.py`
config found in the corresponding fp16 run, then add ternary-specific
hyperparameter axes (§5 above).

---

## Stopping Condition

Stop when any of:
- 100 experiments have been run on one track
- `val_bpb` has not improved in 20 consecutive experiments
- `val_bpb` reaches below 1.1 (excellent result)

---

## Hardware Notes

**Target hardware**: Apple MacBook Pro M4 Max, 64GB unified memory
**Framework**: MLX (Apple Silicon native)
**autoresearch fork**: https://github.com/trevin-creator/autoresearch-mlx

MLX-specific guidance:
- Use `mlx.core` for tensor ops; avoid PyTorch-specific APIs
- M4 Max can handle larger batch sizes than typical cloud GPUs due to unified memory
- MLX lazy evaluation means `.eval()` is required before timing measurements
- Memory ceiling: ~50GB usable for model + activations; leave 14GB for OS

---

## Key Files

```
data/prepared/
  english/train.bin    — tokenized English corpus (memmap)
  english/val.bin      — English validation split
  english/meta.pkl     — tokenizer metadata
  loga/train.bin       — tokenized Loga corpus (memmap)
  loga/val.bin         — Loga validation split
  loga/meta.pkl        — tokenizer metadata

tokenizer/
  english-bpe/tokenizer.json
  loga-bpe/tokenizer.json

train/
  train.py             — model + training loop (MUTABLE by autoresearch)
  prepare.py           — data pipeline (IMMUTABLE)
  program.md           — this file
  results.tsv          — experiment log (auto-updated)
```

---

## Interpreting Results

`val_bpb` = bits per byte = log2(perplexity) / avg_bytes_per_token

Lower is better. A lower val_bpb means the model more efficiently compresses
the validation text — i.e., it has learned more structure per parameter.

### 2×2 Result Matrix

```
                    English corpus     Loga corpus
                  ┌────────────────┬──────────────────┐
float16 weights   │ A: ~1.8–2.2    │ B: < A?          │
                  │ (reference)    │ (tests H1)       │
                  ├────────────────┼──────────────────┤
ternary weights   │ C: ≈ A?        │ D: < C and < B?  │
                  │ (tests H2)     │ (tests H3)       │
                  └────────────────┴──────────────────┘
```

**Expected range for 10-50M param models on ~160MB corpus:**
- Run A (English fp16): 1.8–2.2 bpb (autoresearch-mlx reference: 1.294)
- Run B (Loga fp16): target < Run A bpb if H1 holds
- Run C (English ternary): target within 0.1 bpb of Run A if H2 holds
- Run D (Loga ternary): target < both B and C if H3 (interaction) holds

**Secondary metrics for ternary runs (C and D):**
- Weight sparsity: % of weights that are zero (expect 30–50%)
- Inference throughput: tokens/sec vs. fp16 equivalent (expect 2–5x on M4 Max)
- Model size on disk: bytes of packed ternary weights (expect ~8–10x reduction)

**If H1 fails** (B ≥ A): check translation quality and corpus byte parity.
**If H2 fails** (C >> A): check ternary warmup schedule and threshold tuning.
**If H3 holds** (D < both B and C): this is the novel finding — report it.
