"""
eval/benchmark.py
=================
Evaluation harness for the English vs. Loga experiment.

Modes:
  similarity   — measure back-translation semantic similarity on validation set
  tokenizer    — compare tokens-per-article between English and Loga tokenizers
  learning-curve — plot val_bpb vs. training steps from results.tsv
  summary      — run all modes and write a markdown report

Usage:
    python -m eval.benchmark similarity \
        --loga-articles data/translated/loga-articles.jsonl \
        --english-articles data/raw/simplewiki-articles.jsonl

    python -m eval.benchmark tokenizer \
        --english-tokenizer tokenizer/english-bpe/tokenizer.json \
        --loga-tokenizer tokenizer/loga-bpe/tokenizer.json \
        --english-corpus data/raw/simplewiki-sentences.txt \
        --loga-corpus data/translated/loga-sentences.txt

    python -m eval.benchmark learning-curve \
        --results-english train/results_english.tsv \
        --results-loga train/results_loga.tsv \
        --output eval/learning_curve.png

    python -m eval.benchmark summary --output eval/report.md
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Semantic similarity evaluation
# ---------------------------------------------------------------------------

def load_similarity_scores(loga_articles_path: Path) -> list[float]:
    """Load pre-computed back-translation similarity scores from translated JSONL."""
    scores = []
    with open(loga_articles_path) as f:
        for line in f:
            try:
                record = json.loads(line)
                score = record.get("similarity_score", -1.0)
                if score >= 0:
                    scores.append(score)
            except (json.JSONDecodeError, KeyError):
                continue
    return scores


def report_similarity(scores: list[float]) -> dict:
    """Compute summary statistics for similarity scores."""
    if not scores:
        return {"error": "no similarity scores found — run translate.py with --validate-ratio > 0"}

    arr = np.array(scores)
    below_threshold = float(np.mean(arr < 0.75))
    return {
        "n_validated": len(scores),
        "mean_similarity": round(float(arr.mean()), 4),
        "median_similarity": round(float(np.median(arr)), 4),
        "std_similarity": round(float(arr.std()), 4),
        "min_similarity": round(float(arr.min()), 4),
        "max_similarity": round(float(arr.max()), 4),
        "pct_below_0_75": round(100 * below_threshold, 1),
        "quality_judgment": (
            "GOOD — translation fidelity is high"
            if arr.mean() >= 0.80 else
            "MARGINAL — consider re-translating low-similarity articles"
            if arr.mean() >= 0.70 else
            "POOR — translation quality may confound experiment results"
        ),
    }


# ---------------------------------------------------------------------------
# Tokenizer efficiency
# ---------------------------------------------------------------------------

def compare_tokenizers(
    english_tokenizer_path: Path,
    loga_tokenizer_path: Path,
    english_corpus: Path,
    loga_corpus: Path,
    sample_lines: int = 5000,
) -> dict:
    """Compare tokens-per-line and chars-per-token for English vs. Loga."""
    from tokenizers import Tokenizer

    eng_tok = Tokenizer.from_file(str(english_tokenizer_path))
    loga_tok = Tokenizer.from_file(str(loga_tokenizer_path))

    def measure(tok: object, corpus: Path, n: int) -> dict:
        total_tokens, total_chars, lines = 0, 0, 0
        with open(corpus) as f:
            for i, line in enumerate(f):
                if i >= n:
                    break
                line = line.strip()
                if not line:
                    continue
                enc = tok.encode(line)
                total_tokens += len(enc.ids)
                total_chars += len(line)
                lines += 1
        if total_tokens == 0:
            return {"error": "no tokens"}
        return {
            "chars_per_token": round(total_chars / total_tokens, 4),
            "tokens_per_line": round(total_tokens / lines, 2),
            "total_tokens": total_tokens,
            "total_chars": total_chars,
        }

    eng_m = measure(eng_tok, english_corpus, sample_lines)
    loga_m = measure(loga_tok, loga_corpus, sample_lines)

    result = {"english": eng_m, "loga": loga_m}

    if "chars_per_token" in eng_m and "chars_per_token" in loga_m:
        delta = loga_m["chars_per_token"] - eng_m["chars_per_token"]
        pct = 100 * delta / eng_m["chars_per_token"]
        result["comparison"] = {
            "chars_per_token_delta": round(delta, 4),
            "efficiency_gain_pct": round(pct, 2),
            "hypothesis_support": delta > 0,
            "interpretation": (
                f"Loga is {pct:.1f}% more efficient (more chars/token → fewer tokens for same content)"
                if delta > 0 else
                f"English is {abs(pct):.1f}% more efficient — Loga tokenizer needs improvement"
            ),
        }

    return result


# ---------------------------------------------------------------------------
# Learning curve plotting
# ---------------------------------------------------------------------------

def parse_results_tsv(path: Path) -> tuple[list[int], list[float]]:
    """
    Parse autoresearch results.tsv into (steps, val_bpb) lists.
    Expected TSV columns: experiment_id, val_bpb, description, ...
    """
    steps, bpbs = [], []
    if not path.exists():
        log.warning("Results file not found: %s", path)
        return steps, bpbs

    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                steps.append(i)
                bpbs.append(float(parts[1]))
            except (ValueError, IndexError):
                continue

    return steps, bpbs


def plot_learning_curves(
    results_english: Path,
    results_loga: Path,
    output: Path,
) -> None:
    """Plot val_bpb vs. experiment number for English and Loga."""
    eng_steps, eng_bpbs = parse_results_tsv(results_english)
    loga_steps, loga_bpbs = parse_results_tsv(results_loga)

    if not eng_bpbs and not loga_bpbs:
        log.error("No data found in either results file.")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    if eng_bpbs:
        # Plot running minimum (best so far)
        eng_best = [min(eng_bpbs[: i + 1]) for i in range(len(eng_bpbs))]
        ax.plot(eng_steps, eng_bpbs, "b.", alpha=0.3, markersize=4, label="English (each exp.)")
        ax.plot(eng_steps, eng_best, "b-", linewidth=2, label=f"English best ({min(eng_bpbs):.4f})")

    if loga_bpbs:
        loga_best = [min(loga_bpbs[: i + 1]) for i in range(len(loga_bpbs))]
        ax.plot(loga_steps, loga_bpbs, "r.", alpha=0.3, markersize=4, label="Loga (each exp.)")
        ax.plot(loga_steps, loga_best, "r-", linewidth=2, label=f"Loga best ({min(loga_bpbs):.4f})")

    ax.set_xlabel("Experiment Number", fontsize=12)
    ax.set_ylabel("Validation bits-per-byte (lower = better)", fontsize=12)
    ax.set_title("English vs. Loga: val_bpb Learning Curves", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate delta at final best
    if eng_bpbs and loga_bpbs:
        eng_final = min(eng_bpbs)
        loga_final = min(loga_bpbs)
        delta = eng_final - loga_final
        color = "green" if delta > 0 else "red"
        ax.annotate(
            f"Δbpb = {delta:+.4f}\n({'Loga wins' if delta > 0 else 'English wins'})",
            xy=(0.98, 0.05),
            xycoords="axes fraction",
            ha="right",
            va="bottom",
            fontsize=12,
            color=color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor=color),
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    log.info("Learning curve saved → %s", output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary report
# ---------------------------------------------------------------------------

def write_summary_report(
    similarity_scores: list[float] | None,
    tokenizer_comparison: dict | None,
    results_english: Path | None,
    results_loga: Path | None,
    output: Path,
) -> None:
    """Write a markdown summary of all evaluation results."""
    lines = [
        "# Conlang LLM Experiment: Evaluation Report",
        "",
        "## 1. Translation Quality (Back-Translation Similarity)",
        "",
    ]

    if similarity_scores:
        sim = report_similarity(similarity_scores)
        lines += [
            f"- Validated articles: {sim['n_validated']}",
            f"- Mean cosine similarity: **{sim['mean_similarity']}**",
            f"- Median: {sim['median_similarity']}, Std: {sim['std_similarity']}",
            f"- % below 0.75 threshold: {sim['pct_below_0_75']}%",
            f"- Judgment: **{sim['quality_judgment']}**",
            "",
        ]
    else:
        lines += ["_No similarity scores available. Run `translate.py` with `--validate-ratio > 0`._", ""]

    lines += ["## 2. Tokenizer Efficiency", ""]

    if tokenizer_comparison and "comparison" in tokenizer_comparison:
        eng = tokenizer_comparison["english"]
        loga = tokenizer_comparison["loga"]
        cmp = tokenizer_comparison["comparison"]
        lines += [
            f"| Metric | English | Loga |",
            f"|--------|---------|------|",
            f"| chars/token | {eng.get('chars_per_token', 'N/A')} | {loga.get('chars_per_token', 'N/A')} |",
            f"| tokens/line | {eng.get('tokens_per_line', 'N/A')} | {loga.get('tokens_per_line', 'N/A')} |",
            "",
            f"**Efficiency gain**: {cmp['efficiency_gain_pct']:+.1f}%",
            "",
            f"**Interpretation**: {cmp['interpretation']}",
            "",
        ]
    else:
        lines += ["_Tokenizer comparison not available._", ""]

    lines += ["## 3. Training Results (val_bpb)", ""]

    eng_steps, eng_bpbs = ([], [])
    loga_steps, loga_bpbs = ([], [])
    if results_english and results_english.exists():
        eng_steps, eng_bpbs = parse_results_tsv(results_english)
    if results_loga and results_loga.exists():
        loga_steps, loga_bpbs = parse_results_tsv(results_loga)

    if eng_bpbs or loga_bpbs:
        lines += [
            f"| | English | Loga |",
            f"|--|---------|------|",
            f"| Experiments run | {len(eng_bpbs)} | {len(loga_bpbs)} |",
            f"| Best val_bpb | {min(eng_bpbs):.6f} | {min(loga_bpbs):.6f} |" if eng_bpbs and loga_bpbs else "| Best val_bpb | N/A | N/A |",
        ]
        if eng_bpbs and loga_bpbs:
            delta = min(eng_bpbs) - min(loga_bpbs)
            winner = "Loga" if delta > 0 else "English"
            lines += [
                "",
                f"**Result**: {winner} wins with Δbpb = {abs(delta):.6f}",
                "",
                (
                    "**Hypothesis H1 SUPPORTED**: Loga model achieves better compression efficiency "
                    "per parameter than the English baseline."
                    if delta > 0 else
                    "**Hypothesis H0 holds**: English model achieved equal or better bits-per-byte. "
                    "See Open Questions section for next steps."
                ),
            ]
    else:
        lines += ["_No training results yet. Run autoresearch on both corpora first._"]

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines))
    log.info("Report written → %s", output)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """Evaluation harness for the English vs. Loga LLM experiment."""
    pass


@cli.command("similarity")
@click.option("--loga-articles", type=click.Path(exists=True, path_type=Path),
              default="data/translated/loga-articles.jsonl")
def similarity_cmd(loga_articles):
    """Report back-translation semantic similarity statistics."""
    scores = load_similarity_scores(loga_articles)
    result = report_similarity(scores)
    print(json.dumps(result, indent=2))


@cli.command("tokenizer")
@click.option("--english-tokenizer", type=click.Path(path_type=Path),
              default="tokenizer/english-bpe/tokenizer.json")
@click.option("--loga-tokenizer", type=click.Path(path_type=Path),
              default="tokenizer/loga-bpe/tokenizer.json")
@click.option("--english-corpus", type=click.Path(path_type=Path),
              default="data/raw/simplewiki-sentences.txt")
@click.option("--loga-corpus", type=click.Path(path_type=Path),
              default="data/translated/loga-sentences.txt")
@click.option("--sample-lines", default=5000, show_default=True)
def tokenizer_cmd(english_tokenizer, loga_tokenizer, english_corpus, loga_corpus, sample_lines):
    """Compare tokenizer efficiency between English and Loga."""
    result = compare_tokenizers(
        english_tokenizer, loga_tokenizer,
        english_corpus, loga_corpus,
        sample_lines,
    )
    print(json.dumps(result, indent=2))


@cli.command("learning-curve")
@click.option("--results-english", type=click.Path(path_type=Path),
              default="train/results_english.tsv")
@click.option("--results-loga", type=click.Path(path_type=Path),
              default="train/results_loga.tsv")
@click.option("--output", type=click.Path(path_type=Path),
              default="eval/learning_curve.png")
def learning_curve_cmd(results_english, results_loga, output):
    """Plot val_bpb vs. experiment number for both models."""
    plot_learning_curves(results_english, results_loga, output)


@cli.command("summary")
@click.option("--loga-articles", type=click.Path(path_type=Path),
              default="data/translated/loga-articles.jsonl")
@click.option("--english-tokenizer", type=click.Path(path_type=Path),
              default="tokenizer/english-bpe/tokenizer.json")
@click.option("--loga-tokenizer", type=click.Path(path_type=Path),
              default="tokenizer/loga-bpe/tokenizer.json")
@click.option("--english-corpus", type=click.Path(path_type=Path),
              default="data/raw/simplewiki-sentences.txt")
@click.option("--loga-corpus", type=click.Path(path_type=Path),
              default="data/translated/loga-sentences.txt")
@click.option("--results-english", type=click.Path(path_type=Path),
              default="train/results_english.tsv")
@click.option("--results-loga", type=click.Path(path_type=Path),
              default="train/results_loga.tsv")
@click.option("--output", type=click.Path(path_type=Path),
              default="eval/report.md")
def summary_cmd(loga_articles, english_tokenizer, loga_tokenizer, english_corpus,
                loga_corpus, results_english, results_loga, output):
    """Generate full evaluation report in Markdown."""
    scores = None
    if Path(loga_articles).exists():
        scores = load_similarity_scores(Path(loga_articles))

    tok_comparison = None
    if Path(english_tokenizer).exists() and Path(loga_tokenizer).exists():
        tok_comparison = compare_tokenizers(
            Path(english_tokenizer), Path(loga_tokenizer),
            Path(english_corpus), Path(loga_corpus),
        )

    write_summary_report(scores, tok_comparison, results_english, results_loga, output)
    print(f"Report written to {output}")


if __name__ == "__main__":
    cli()
