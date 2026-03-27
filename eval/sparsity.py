"""
eval/sparsity.py
================
Head-level sparsity analysis for ternary-weight transformer models.
Tests Conjecture 3: Loga-trained models exhibit more *structured* sparsity
(zeros clustered at the attention head level) than English-trained models.

The key measurement:
  - For each attention head, compute the fraction of zero weights.
  - Compare the *variance* of this distribution across heads.
  - High variance → structured sparsity (some heads mostly zero, others dense).
  - Low variance → unstructured sparsity (zeros distributed uniformly).

Also measures pruning headroom: how much val_bpb degrades as heads are pruned
in order of increasing zero-weight fraction.

Usage:
    # Analyse a single model
    python -m eval.sparsity analyse \
        --checkpoint train/checkpoints/loga-ternary/best.npz \
        --model-config train/config.json \
        --output eval/sparsity_loga.json

    # Compare English vs. Loga ternary models
    python -m eval.sparsity compare \
        --english train/checkpoints/english-ternary/best.npz \
        --loga    train/checkpoints/loga-ternary/best.npz \
        --model-config train/config.json \
        --output eval/sparsity_comparison.json

    # Plot pruning curves
    python -m eval.sparsity pruning-curve \
        --english train/checkpoints/english-ternary/best.npz \
        --loga    train/checkpoints/loga-ternary/best.npz \
        --val-data data/prepared/english/val.bin \
        --output eval/pruning_curve.png
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import click
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class HeadSparsity:
    """Sparsity metrics for a single attention head."""
    layer: int
    head: int
    zero_fraction: float       # fraction of weights that are zero
    weight_shape: tuple        # (d_head, d_model) or similar


@dataclass
class ModelSparsityReport:
    """Complete sparsity report for one model."""
    name: str
    total_params: int
    total_zero: int
    overall_sparsity: float
    heads: list[HeadSparsity] = field(default_factory=list)

    # Derived statistics (computed from heads)
    head_zero_variance: float = 0.0       # KEY METRIC for Conjecture 3
    head_zero_mean: float = 0.0
    head_zero_std: float = 0.0
    n_heads_above_50pct_zero: int = 0     # heads that are majority zero
    n_heads_above_80pct_zero: int = 0     # heads that are nearly fully zero


# ---------------------------------------------------------------------------
# Weight extraction helpers
# ---------------------------------------------------------------------------

def extract_attention_head_weights(
    weights: dict,
    n_layers: int,
    n_heads: int,
    d_model: int,
) -> list[HeadSparsity]:
    """
    Extract per-head weight matrices from a model checkpoint and compute
    zero fractions for each head.

    Expects weight keys following the nanochat naming convention:
        transformer.h.{layer}.attn.c_attn.weight  — combined QKV projection
        transformer.h.{layer}.attn.c_proj.weight  — output projection

    Returns a list of HeadSparsity objects, one per (layer, head).
    """
    d_head = d_model // n_heads
    head_metrics = []

    for layer_idx in range(n_layers):
        # Combined QKV weight: shape (3 * d_model, d_model)
        qkv_key = f"transformer.h.{layer_idx}.attn.c_attn.weight"
        if qkv_key not in weights:
            log.warning("Key not found: %s — trying alternate naming", qkv_key)
            # Try alternate key patterns
            for alt in [
                f"layers.{layer_idx}.attention.wqkv.weight",
                f"model.layers.{layer_idx}.self_attn.q_proj.weight",
            ]:
                if alt in weights:
                    qkv_key = alt
                    break
            else:
                log.warning("Could not find attention weights for layer %d", layer_idx)
                continue

        w = weights[qkv_key]  # numpy array after loading

        # Split into Q, K, V if combined
        if w.shape[0] == 3 * d_model:
            q_w = w[:d_model]
            k_w = w[d_model:2 * d_model]
            v_w = w[2 * d_model:]
        else:
            # Assume it's just Q (handle Q/K/V separately)
            q_w = w
            k_w = weights.get(f"transformer.h.{layer_idx}.attn.k.weight", w)
            v_w = weights.get(f"transformer.h.{layer_idx}.attn.v.weight", w)

        # Split each of Q, K, V into per-head slices
        for head_idx in range(n_heads):
            start = head_idx * d_head
            end = (head_idx + 1) * d_head

            # Concatenate Q, K, V slices for this head
            head_weights = np.concatenate([
                q_w[start:end].flatten(),
                k_w[start:end].flatten(),
                v_w[start:end].flatten(),
            ])

            zero_count = np.sum(np.abs(head_weights) < 0.5)  # ternary: |w| < 0.5 → zero
            zero_fraction = float(zero_count) / len(head_weights)

            head_metrics.append(HeadSparsity(
                layer=layer_idx,
                head=head_idx,
                zero_fraction=zero_fraction,
                weight_shape=(d_head, d_model),
            ))

    return head_metrics


def compute_report(
    name: str,
    checkpoint_path: Path,
    n_layers: int,
    n_heads: int,
    d_model: int,
) -> ModelSparsityReport:
    """Load a checkpoint and compute the full sparsity report."""
    import numpy as np

    log.info("Loading checkpoint: %s", checkpoint_path)
    weights = dict(np.load(checkpoint_path, allow_pickle=False))

    total_params = sum(v.size for v in weights.values())
    total_zero = sum(int(np.sum(np.abs(v) < 0.5)) for v in weights.values())
    overall_sparsity = total_zero / total_params if total_params > 0 else 0.0

    heads = extract_attention_head_weights(weights, n_layers, n_heads, d_model)

    report = ModelSparsityReport(
        name=name,
        total_params=total_params,
        total_zero=total_zero,
        overall_sparsity=round(overall_sparsity, 4),
        heads=heads,
    )

    if heads:
        zero_fractions = np.array([h.zero_fraction for h in heads])
        report.head_zero_variance = round(float(np.var(zero_fractions)), 6)
        report.head_zero_mean = round(float(np.mean(zero_fractions)), 4)
        report.head_zero_std = round(float(np.std(zero_fractions)), 4)
        report.n_heads_above_50pct_zero = int(np.sum(zero_fractions > 0.5))
        report.n_heads_above_80pct_zero = int(np.sum(zero_fractions > 0.8))

    return report


# ---------------------------------------------------------------------------
# Comparison and plotting
# ---------------------------------------------------------------------------

def compare_reports(
    english: ModelSparsityReport,
    loga: ModelSparsityReport,
) -> dict:
    """
    Compare sparsity structure between English and Loga models.
    Returns a dict of metrics relevant to Conjecture 3.
    """
    return {
        "overall_sparsity": {
            "english": english.overall_sparsity,
            "loga": loga.overall_sparsity,
            "delta": round(loga.overall_sparsity - english.overall_sparsity, 4),
        },
        "head_zero_variance": {
            "english": english.head_zero_variance,
            "loga": loga.head_zero_variance,
            "ratio": round(
                loga.head_zero_variance / english.head_zero_variance, 4
            ) if english.head_zero_variance > 0 else None,
            "interpretation": (
                "Loga shows more STRUCTURED sparsity (higher head-level variance)"
                if loga.head_zero_variance > english.head_zero_variance
                else "English shows more structured sparsity — Conjecture 3 not supported"
            ),
        },
        "head_zero_std": {
            "english": english.head_zero_std,
            "loga": loga.head_zero_std,
        },
        "n_heads_above_50pct_zero": {
            "english": english.n_heads_above_50pct_zero,
            "loga": loga.n_heads_above_50pct_zero,
        },
        "n_heads_above_80pct_zero": {
            "english": english.n_heads_above_80pct_zero,
            "loga": loga.n_heads_above_80pct_zero,
        },
        "conjecture_3_supported": loga.head_zero_variance > english.head_zero_variance,
    }


def plot_head_zero_distributions(
    english: ModelSparsityReport,
    loga: ModelSparsityReport,
    output: Path,
) -> None:
    """
    Side-by-side histogram of per-head zero fractions for English and Loga.
    The key visual: Loga should show a wider, more bimodal distribution
    (some heads near 0% zero, others near 100% zero).
    English should show a tighter, more unimodal distribution around the mean.
    """
    eng_fractions = [h.zero_fraction for h in english.heads]
    loga_fractions = [h.zero_fraction for h in loga.heads]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    bins = np.linspace(0, 1, 21)

    axes[0].hist(eng_fractions, bins=bins, color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].axvline(np.mean(eng_fractions), color="navy", linestyle="--", linewidth=1.5,
                    label=f"mean={np.mean(eng_fractions):.2f}")
    axes[0].set_title(f"English ternary\nvar={english.head_zero_variance:.4f}", fontsize=13)
    axes[0].set_xlabel("Zero-weight fraction per head", fontsize=11)
    axes[0].set_ylabel("Number of heads", fontsize=11)
    axes[0].legend(fontsize=10)

    axes[1].hist(loga_fractions, bins=bins, color="firebrick", edgecolor="white", alpha=0.85)
    axes[1].axvline(np.mean(loga_fractions), color="darkred", linestyle="--", linewidth=1.5,
                    label=f"mean={np.mean(loga_fractions):.2f}")
    axes[1].set_title(f"Loga ternary\nvar={loga.head_zero_variance:.4f}", fontsize=13)
    axes[1].set_xlabel("Zero-weight fraction per head", fontsize=11)
    axes[1].legend(fontsize=10)

    verdict = (
        "Conjecture 3 SUPPORTED: Loga shows higher head-level variance"
        if loga.head_zero_variance > english.head_zero_variance
        else "Conjecture 3 NOT SUPPORTED: English shows equal or higher variance"
    )
    fig.suptitle(
        f"Per-Head Zero-Weight Distribution: English vs. Loga Ternary\n{verdict}",
        fontsize=12,
    )
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    log.info("Distribution plot saved → %s", output)
    plt.close(fig)


def plot_pruning_curves(
    english_fractions: list[float],
    loga_fractions: list[float],
    english_bpbs: list[float],
    loga_bpbs: list[float],
    output: Path,
) -> None:
    """
    Plot val_bpb vs. fraction of heads pruned for English and Loga.
    Loga should show a flatter curve (more pruning headroom).
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    ax.plot(english_fractions, english_bpbs, "b-o", markersize=5,
            label=f"English ternary (baseline bpb={english_bpbs[0]:.4f})")
    ax.plot(loga_fractions, loga_bpbs, "r-o", markersize=5,
            label=f"Loga ternary (baseline bpb={loga_bpbs[0]:.4f})")

    ax.set_xlabel("Fraction of heads pruned", fontsize=12)
    ax.set_ylabel("val_bpb (lower = better)", fontsize=12)
    ax.set_title("Pruning Headroom: English vs. Loga Ternary Models\n"
                 "Flatter curve = more structured sparsity (Conjecture 3)", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    log.info("Pruning curve saved → %s", output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Head-level sparsity analysis for ternary LLM models (Conjecture 3)."""
    pass


@cli.command("analyse")
@click.option("--checkpoint", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--name", default="model", show_default=True)
@click.option("--n-layers", default=6, show_default=True)
@click.option("--n-heads", default=6, show_default=True)
@click.option("--d-model", default=384, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default="eval/sparsity.json")
def analyse_cmd(checkpoint, name, n_layers, n_heads, d_model, output):
    """Compute per-head sparsity report for a single checkpoint."""
    report = compute_report(name, checkpoint, n_layers, n_heads, d_model)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    # Convert to JSON-serialisable dict
    data = asdict(report)
    with open(output, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"Overall sparsity:     {report.overall_sparsity:.1%}")
    print(f"Head zero-frac mean:  {report.head_zero_mean:.3f}")
    print(f"Head zero-frac std:   {report.head_zero_std:.3f}")
    print(f"Head zero-frac var:   {report.head_zero_variance:.6f}  ← KEY METRIC")
    print(f"Heads >50% zero:      {report.n_heads_above_50pct_zero}")
    print(f"Heads >80% zero:      {report.n_heads_above_80pct_zero}")
    print(f"{'='*50}")
    print(f"Full report → {output}")


@cli.command("compare")
@click.option("--english", "english_ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--loga", "loga_ckpt", type=click.Path(exists=True, path_type=Path), required=True)
@click.option("--n-layers", default=6, show_default=True)
@click.option("--n-heads", default=6, show_default=True)
@click.option("--d-model", default=384, show_default=True)
@click.option("--output", type=click.Path(path_type=Path), default="eval/sparsity_comparison.json")
@click.option("--plot", type=click.Path(path_type=Path), default="eval/head_zero_distribution.png")
def compare_cmd(english_ckpt, loga_ckpt, n_layers, n_heads, d_model, output, plot):
    """Compare structured sparsity between English and Loga ternary models."""
    english = compute_report("english-ternary", english_ckpt, n_layers, n_heads, d_model)
    loga = compute_report("loga-ternary", loga_ckpt, n_layers, n_heads, d_model)

    comparison = compare_reports(english, loga)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({
            "english": asdict(english),
            "loga": asdict(loga),
            "comparison": comparison,
        }, f, indent=2)

    plot_head_zero_distributions(english, loga, Path(plot))

    print(f"\n{'='*60}")
    print("CONJECTURE 3: STRUCTURED SPARSITY COMPARISON")
    print('='*60)
    print(f"  Overall sparsity   English: {english.overall_sparsity:.1%}  "
          f"Loga: {loga.overall_sparsity:.1%}")
    print(f"  Head zero variance English: {english.head_zero_variance:.6f}  "
          f"Loga: {loga.head_zero_variance:.6f}")
    if english.head_zero_variance > 0:
        ratio = loga.head_zero_variance / english.head_zero_variance
        print(f"  Variance ratio (Loga/English): {ratio:.3f}x")
    print(f"\n  → {comparison['head_zero_variance']['interpretation']}")
    print(f"\n  Conjecture 3 supported: {comparison['conjecture_3_supported']}")
    print('='*60)


if __name__ == "__main__":
    cli()
