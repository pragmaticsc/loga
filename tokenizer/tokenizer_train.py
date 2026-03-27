"""
tokenizer/tokenizer_train.py
============================
Train BPE tokenizers for both English and Loga corpora, then compare
vocabulary efficiency (tokens per article / chars per token).

Outputs:
  tokenizer/english-bpe/     — HuggingFace tokenizers format
  tokenizer/loga-bpe/        — HuggingFace tokenizers format
  tokenizer/efficiency_report.json — comparison metrics

Usage:
    python -m tokenizer.tokenizer_train \
        --english-corpus data/raw/simplewiki-sentences.txt \
        --loga-corpus data/translated/loga-sentences.txt \
        --vocab-size 8192

The Loga sentence file is produced by:
    python -m tokenizer.tokenizer_train extract-sentences \
        --input data/translated/loga-articles.jsonl \
        --output data/translated/loga-sentences.txt
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import click
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[BOS]", "[EOS]"]


def train_bpe(
    corpus_path: Path,
    output_dir: Path,
    vocab_size: int,
    name: str,
) -> Tokenizer:
    """Train a BPE tokenizer on a text file (one sentence/paragraph per line)."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        min_frequency=2,
    )

    log.info("Training %s BPE tokenizer (vocab=%d) on %s", name, vocab_size, corpus_path)
    tokenizer.train(files=[str(corpus_path)], trainer=trainer)

    save_path = output_dir / "tokenizer.json"
    tokenizer.save(str(save_path))
    log.info("Saved %s tokenizer → %s", name, save_path)
    return tokenizer


def measure_efficiency(
    tokenizer: Tokenizer,
    sample_file: Path,
    max_lines: int = 10000,
) -> dict:
    """
    Measure tokenizer efficiency on a corpus sample.

    Returns:
        chars_per_token: average characters per token (higher = more efficient)
        tokens_per_line: average tokens per input line
        total_tokens: total token count
        total_chars: total character count
    """
    total_tokens = 0
    total_chars = 0
    lines_processed = 0

    with open(sample_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            line = line.strip()
            if not line:
                continue
            encoding = tokenizer.encode(line)
            total_tokens += len(encoding.ids)
            total_chars += len(line)
            lines_processed += 1

    if total_tokens == 0:
        return {"error": "no tokens produced"}

    return {
        "chars_per_token": round(total_chars / total_tokens, 4),
        "tokens_per_line": round(total_tokens / lines_processed, 2),
        "total_tokens": total_tokens,
        "total_chars": total_chars,
        "lines_processed": lines_processed,
    }


def build_efficiency_report(
    english_metrics: dict,
    loga_metrics: dict,
    english_vocab_size: int,
    loga_vocab_size: int,
) -> dict:
    """Compare efficiency metrics and compute relative improvement."""
    report = {
        "english": {**english_metrics, "vocab_size": english_vocab_size},
        "loga": {**loga_metrics, "vocab_size": loga_vocab_size},
    }

    if "chars_per_token" in english_metrics and "chars_per_token" in loga_metrics:
        e_cpt = english_metrics["chars_per_token"]
        l_cpt = loga_metrics["chars_per_token"]
        report["comparison"] = {
            "chars_per_token_delta": round(l_cpt - e_cpt, 4),
            "efficiency_gain_pct": round(100 * (l_cpt - e_cpt) / e_cpt, 2),
            "interpretation": (
                "Loga tokenizer is more efficient (more meaning per token)"
                if l_cpt > e_cpt else
                "English tokenizer is more efficient"
            ),
        }

    return report


# ---------------------------------------------------------------------------
# Sentence extraction helper (from translated JSONL)
# ---------------------------------------------------------------------------

def extract_loga_sentences(input_path: Path, output_path: Path) -> None:
    """Extract Loga text from translated JSONL and write one paragraph per line."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with open(input_path, encoding="utf-8") as inf, \
         open(output_path, "w", encoding="utf-8") as outf:
        for line in inf:
            try:
                record = json.loads(line)
                text = record.get("text_loga", "")
                for para in text.split("\n\n"):
                    para = para.strip()
                    if len(para) > 10:
                        outf.write(para + "\n")
                        count += 1
            except (json.JSONDecodeError, KeyError):
                continue
    log.info("Wrote %d paragraphs → %s", count, output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """BPE tokenizer training and efficiency analysis for loga."""
    pass


@cli.command("train")
@click.option("--english-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/raw/simplewiki-sentences.txt")
@click.option("--loga-corpus", type=click.Path(exists=True, path_type=Path),
              default="data/translated/loga-sentences.txt")
@click.option("--output-dir", type=click.Path(path_type=Path),
              default="tokenizer")
@click.option("--vocab-size", default=8192, show_default=True)
@click.option("--sample-lines", default=10000, show_default=True,
              help="Lines to sample for efficiency measurement")
def train_cmd(english_corpus, loga_corpus, output_dir, vocab_size, sample_lines):
    """Train BPE tokenizers for English and Loga and report efficiency."""
    eng_dir = output_dir / "english-bpe"
    loga_dir = output_dir / "loga-bpe"

    eng_tok = train_bpe(english_corpus, eng_dir, vocab_size, "English")
    loga_tok = train_bpe(loga_corpus, loga_dir, vocab_size, "Loga")

    log.info("Measuring efficiency...")
    eng_metrics = measure_efficiency(eng_tok, english_corpus, sample_lines)
    loga_metrics = measure_efficiency(loga_tok, loga_corpus, sample_lines)

    report = build_efficiency_report(
        eng_metrics, loga_metrics,
        english_vocab_size=vocab_size,
        loga_vocab_size=vocab_size,
    )

    report_path = output_dir / "efficiency_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    log.info("Efficiency report → %s", report_path)

    # Print summary
    print("\n" + "=" * 60)
    print("TOKENIZER EFFICIENCY COMPARISON")
    print("=" * 60)
    print(f"  English  chars/token: {eng_metrics.get('chars_per_token', 'N/A')}")
    print(f"  Loga     chars/token: {loga_metrics.get('chars_per_token', 'N/A')}")
    if "comparison" in report:
        cmp = report["comparison"]
        print(f"  Delta: {cmp['chars_per_token_delta']:+.4f} ({cmp['efficiency_gain_pct']:+.1f}%)")
        print(f"  → {cmp['interpretation']}")
    print("=" * 60)


@cli.command("extract-sentences")
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path),
              default="data/translated/loga-articles.jsonl")
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              default="data/translated/loga-sentences.txt")
def extract_sentences_cmd(input_path, output_path):
    """Extract Loga sentences from translated JSONL for tokenizer training."""
    extract_loga_sentences(input_path, output_path)


if __name__ == "__main__":
    cli()
