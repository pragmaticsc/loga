# loga

**Can a purpose-built constructed language make LLMs more efficient?**

This repository accompanies the preprint:

> Shaun Russell. *The Language We Train On: A Case for Purpose-Built Substrate Languages in Large Language Model Pre-Training.* 2026. [[arXiv]](https://arxiv.org/abs/XXXX.XXXXX) [[PDF]](docs/perspective-arxiv.md)
>
> https://github.com/pragmaticsc/loga

---

## The Idea

Large language models are trained on text written for humans. Human language evolved under biological and social constraints that have nothing to do with machine learning. We argue this is a contingent choice, not a necessary one — and that the training substrate is a design variable worth optimising.

We propose three conjectures, each independently testable:

**Conjecture 1 — Efficiency**: A transformer trained on *Loga* (a conlang designed around BPE tokenisation) achieves lower bits-per-byte than an identical model trained on English at equivalent compute.

**Conjecture 2 — Quantisation tolerance**: A Loga-trained model suffers a smaller accuracy penalty from ternary weight quantisation ({−1, 0, +1}) than an English-trained model, because the lower information complexity of the training distribution reduces the capacity cost of discretisation.

**Conjecture 3 — Structured sparsity**: The zeros produced by ternary quantisation cluster at the attention head level — not randomly across weights — in Loga-trained models more than English-trained models, enabling post-training structured pruning with less degradation.

Together these form a compression cascade: **conlang training → ternary QAT → structured pruning**, each step made cheaper by the one before.

---

## Status

| Phase | Status |
|-------|--------|
| Conlang design (Loga spec) | ✅ Complete |
| Ternary weight implementation (MLX) | ✅ Complete |
| Sparsity analysis tooling | ✅ Complete |
| Corpus download + parsing | ✅ Complete |
| Translation pipeline | ✅ Complete (needs API key + runtime) |
| Tokenizer training | ✅ Complete (needs translated corpus) |
| autoresearch training runs | ⏳ In progress |
| Experimental results | ⏳ Pending |

---

## Experiment Design

A 2×2 factorial experiment on Simple English Wikipedia (~160MB, ~250K articles):

```
                    English corpus     Loga corpus
                  ┌────────────────┬──────────────────┐
float16 weights   │ A  (reference) │ B  (Conjecture 1)│
                  ├────────────────┼──────────────────┤
ternary weights   │ C  (baseline)  │ D  (novel)       │
                  └────────────────┴──────────────────┘
```

All four cells: identical nanochat-scale architecture (10–50M params), identical compute budget, identical BPE vocab size (8,192). Automated hyperparameter search via [autoresearch-mlx](https://github.com/trevin-creator/autoresearch-mlx) on Apple M4 Max. Primary metric: `val_bpb`.

---

## Loga: The Constructed Language

Loga is designed around one measurable goal: maximise the conditional predictability of the token stream. Key properties:

- **95 printable ASCII characters** — 6.57 bits/byte, the information-theoretic optimum within UTF-8's 1-byte range
- **2-character roots** — 9,025 possible roots in 2 bytes (vs. 2,500 in 4 bytes for a typical CVCV design)
- **Character class encodes syntactic role** — lowercase = noun root, uppercase = verb root, `!`–`/` = case suffixes, `:`–`@` = tense markers. Syntactic function is readable from the first byte.
- **Agglutinative, invariant roots** — no irregular alternations (go/went, is/was/be)
- **Strict SOV, context-free grammar** — EBNF fits on one page, no heuristics

See [`conlang-spec.md`](conlang-spec.md) for the full grammar, sample sentences, and tokenisation analysis.

---

## Repository Structure

```
loga/
├── conlang-spec.md          # Full Loga grammar specification
├── pyproject.toml           # Python dependencies
│
├── data/
│   └── download.sh          # Download + parse Simple English Wikipedia
│
├── translator/
│   └── translate.py         # English → Loga via Claude API (async, resumable)
│
├── tokenizer/
│   └── tokenizer_train.py   # BPE tokenizer training + efficiency comparison
│
├── train/
│   ├── bitlinear.py         # MLX BitLinear: ternary weight layer (BitNet b1.58)
│   └── program.md           # autoresearch objective file (2×2 experiment)
│
├── eval/
│   ├── benchmark.py         # val_bpb comparison + learning curve plots
│   └── sparsity.py          # Head-level zero variance (Conjecture 3)
│
└── docs/
    └── perspective-arxiv.md # Preprint (also on arXiv)
```

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/pragmaticsc/loga
cd loga
pip install -e ".[train]"

# 2. Set your Anthropic API key
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your_key_here

# 3. Download corpus (~1.5GB compressed)
bash data/download.sh

# 4. Translate to Loga (~$20 at claude-sonnet-4-6 rates)
python -m translator.translate --workers 8

# 5. Train tokenizers and compare efficiency
python -m tokenizer.tokenizer_train train

# 6. Clone autoresearch-mlx and run experiments
git clone https://github.com/trevin-creator/autoresearch-mlx train/autoresearch
# Follow train/program.md for the 2×2 run sequence
```

---

## Key Implementation: BitLinear (Ternary Weights)

`train/bitlinear.py` implements the ternary weight layer for MLX following [BitNet b1.58](https://arxiv.org/abs/2402.17764). Drop-in replacement for `nn.Linear`:

```python
from train.bitlinear import BitLinear, replace_linear_with_bitlinear

# Option 1: use directly
self.proj = BitLinear(d_model, d_model, bias=False, warmup_steps=100)

# Option 2: convert an existing model
model = replace_linear_with_bitlinear(model, warmup_steps=100, skip_modules=["lm_head"])

# Measure sparsity after training
from train.bitlinear import model_sparsity
print(model_sparsity(model))
# → {'transformer.h.0.attn.c_attn': 0.42, ..., 'overall': 0.39}
```

---

## Key Implementation: Sparsity Analysis (Conjecture 3)

`eval/sparsity.py` measures the *structure* of ternary sparsity — whether zeros cluster at the head level (Loga prediction) or distribute uniformly (English prediction):

```bash
# Compare English vs. Loga ternary models
python -m eval.sparsity compare \
    --english train/checkpoints/english-ternary/best.npz \
    --loga    train/checkpoints/loga-ternary/best.npz \
    --plot    eval/head_zero_distribution.png
```

Outputs:
- Per-head zero-weight fractions for each model
- **Head-level zero variance** — the key metric for Conjecture 3
- Side-by-side histogram plot

---

## Conjectures and Predictions

| | Prediction | Measurement |
|--|--|--|
| **C1** | Loga val_bpb < English val_bpb | autoresearch runs B vs. A |
| **C2** | Loga ternary penalty < English ternary penalty | (D−B) < (C−A) in val_bpb |
| **C3** | Loga head-zero variance > English head-zero variance | `eval/sparsity.py compare` |

---

## References

The conjectures are grounded in these key papers:

- Galke, Ram & Raviv (2024). *What Makes a Language Easy to Deep-Learn?* Nature Communications. [arXiv:2302.12239](https://arxiv.org/abs/2302.12239)
- Ma, Wang et al. (2024). *The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits.* [arXiv:2402.17764](https://arxiv.org/abs/2402.17764)
- Schut, Gal et al. (2025). *Do Multilingual LLMs Think In English?* ICLR 2025. [arXiv:2502.15603](https://arxiv.org/abs/2502.15603)
- Voita et al. (2019). *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned.* ACL 2019.
- Full reference list in the [preprint](docs/perspective-arxiv.md).

---

## Citation

```bibtex
@misc{russell2026language,
  title   = {The Language We Train On: A Case for Purpose-Built Substrate
             Languages in Large Language Model Pre-Training},
  author  = {Russell, Shaun},
  year    = {2026},
  note    = {Preprint},
  url     = {https://arxiv.org/abs/XXXX.XXXXX},
  code    = {https://github.com/pragmaticsc/loga}
}
```

---

## Contributing

Experiments not yet run. If you have GPU/Apple Silicon compute and want to
run one of the four cells, open an issue and we can coordinate to avoid
duplication. The most valuable contribution right now is running **cell A**
(English float16 baseline) to establish the val_bpb floor.
