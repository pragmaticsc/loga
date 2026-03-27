"""
translator/translate.py
=======================
English → Loga translation pipeline.

Reads Simple English Wikipedia articles from data/raw/simplewiki-articles.jsonl,
translates each article using the Claude API (claude-sonnet-4-6), and writes
translated articles to data/translated/loga-articles.jsonl.

Also runs back-translation validation: a sample of translated articles are
translated back to English and semantic similarity is measured using
sentence-transformers. Articles below a configurable similarity threshold are
flagged for review.

Usage:
    python -m translator.translate --input data/raw/simplewiki-articles.jsonl \
        --output data/translated/loga-articles.jsonl \
        --workers 8 \
        --validate-ratio 0.01

Requirements:
    ANTHROPIC_API_KEY environment variable (or .env file)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator

import anthropic
import click
from dotenv import load_dotenv
from tqdm.asyncio import tqdm as atqdm

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loga grammar reference injected into every translation prompt
# ---------------------------------------------------------------------------

GRAMMAR_PREAMBLE = """\
You are a translator from English to Loga, a constructed language designed for
LLM tokenization efficiency. Follow these rules exactly:

PHONEME INVENTORY
- Consonants: p t k b d g s n m l  (10 total)
- Vowels: a e i o u  (5 total)
- All syllables are CV (consonant-vowel). No consonant clusters.

MORPHOLOGY
- Noun roots are CVCV (4 chars). Append case suffix directly:
    -a  = subject (nominative)
    -e  = direct object (accusative)
    -i  = genitive (of / possessive)
    -o  = dative (to / for)
    -un = locative (at / on / in)
    -ul = lative (toward / to)
    -uk = ablative (from)
- Verb roots are CVCV. Append tense suffix:
    -pa = past,  -ta = present,  -ka = future
  Optionally add aspect: -s (perfective), -l (imperfective)
- Adjectives: root + -no (follow the noun they modify)
- Negation: prefix su- on verb or noun root
- Questions: prefix sentence with particle "ki"

SYNTAX (strict SOV)
  [SUBJ-case] [OBJ-case] [ADJ-no] [VERB-tense]

CORE VOCABULARY (partial)
  mia=I/me, tua=you, sia=he/she/it
  sita=sit, dela=be, pabo=know, telu=say, nago=move, mako=make
  pena=see, dako=give, libo=take, muto=eat, somi=sleep, naki=think
  kuda=water, gelo=fire, tupa=time, sela=world/place, molu=person
  belu=beautiful, bipu=big, temo=small, ganu=good, beko=bad
  lipe=life, tobu=city, nola=many (plural marker)
  Numbers: zopa=0, poka=1, toka=2, soka=3, noka=4, loka=5,
           moka=6, goka=7, boka=8, doka=9

PROPER NOUNS: transliterate phonemically into CVCV patterns (capitalize first letter).
  Example: "England" → "Egalande", "Paris" → "Palise"

COMPOUNDS: concatenate two roots for new concepts.
  Example: water-make = kuda-mako = irrigate; fire-place = gelo-sela = volcano

NUMBERS: compose root words. "23" → "toka-soka", "100" → "poka-zopa-zopa"

TRANSLATION RULES:
1. Translate the MEANING, not word-for-word. Choose the Loga root that best matches.
2. If no root exists for a concept, create a compound or use phonemic transliteration.
3. Keep sentences short. Split long English sentences into multiple Loga sentences.
4. Preserve paragraph structure (blank lines between paragraphs).
5. Do NOT include English text in your output. Output ONLY Loga.
6. Do NOT add explanation or commentary. Output ONLY the translated text.
"""

BACK_TRANSLATE_PREAMBLE = """\
You are a translator from Loga (a constructed language) back to English.
Loga rules:
- SOV word order; noun cases: -a(subj), -e(obj), -i(gen), -o(dat), -un(loc), -ul(lative), -uk(ablative)
- Verb tenses: -pa(past), -ta(pres), -ka(fut); aspect: -s(done), -l(ongoing)
- Adjectives: root-no; negation: su- prefix; question: ki prefix
- Core vocab: mia=I, tua=you, sia=he/she/it, sita=sit, dela=be, pabo=know,
  telu=say, nago=move, pena=see, kuda=water, molu=person, belu=beautiful,
  bipu=big, temo=small, ganu=good, beko=bad, lipe=life, tobu=city

Translate the following Loga text to natural English. Output ONLY English.
"""


# ---------------------------------------------------------------------------
# Article data classes
# ---------------------------------------------------------------------------

@dataclass
class Article:
    id: str
    title: str
    text: str

@dataclass
class TranslatedArticle:
    id: str
    title_en: str
    title_loga: str
    text_loga: str
    back_translated_text: str = ""
    similarity_score: float = -1.0


# ---------------------------------------------------------------------------
# Claude API helpers
# ---------------------------------------------------------------------------

async def translate_text(client: anthropic.AsyncAnthropic, text: str, model: str) -> str:
    """Translate English text to Loga using Claude."""
    # Chunk long articles to stay within context limits
    chunks = _chunk_text(text, max_chars=6000)
    translated_chunks = []
    for chunk in chunks:
        message = await client.messages.create(
            model=model,
            max_tokens=8192,
            messages=[
                {
                    "role": "user",
                    "content": f"{GRAMMAR_PREAMBLE}\n\nTranslate this English text to Loga:\n\n{chunk}"
                }
            ]
        )
        translated_chunks.append(message.content[0].text.strip())
    return "\n\n".join(translated_chunks)


async def back_translate_text(client: anthropic.AsyncAnthropic, loga_text: str, model: str) -> str:
    """Translate Loga text back to English for validation."""
    message = await client.messages.create(
        model=model,
        max_tokens=8192,
        messages=[
            {
                "role": "user",
                "content": f"{BACK_TRANSLATE_PREAMBLE}\n\n{loga_text}"
            }
        ]
    )
    return message.content[0].text.strip()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text on paragraph boundaries, keeping chunks under max_chars."""
    paragraphs = text.split("\n\n")
    chunks, current = [], []
    current_len = 0
    for para in paragraphs:
        if current_len + len(para) > max_chars and current:
            chunks.append("\n\n".join(current))
            current, current_len = [], 0
        current.append(para)
        current_len += len(para)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Validation: semantic similarity via sentence-transformers
# ---------------------------------------------------------------------------

_embed_model = None

def compute_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between two texts using sentence-transformers."""
    global _embed_model
    if _embed_model is None:
        from sentence_transformers import SentenceTransformer
        _embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    import numpy as np
    embeddings = _embed_model.encode([text_a, text_b], normalize_embeddings=True)
    return float(np.dot(embeddings[0], embeddings[1]))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

async def process_article(
    client: anthropic.AsyncAnthropic,
    article: Article,
    model: str,
    validate: bool,
    semaphore: asyncio.Semaphore,
) -> TranslatedArticle:
    async with semaphore:
        title_loga = await translate_text(client, article.title, model)
        text_loga = await translate_text(client, article.text, model)

        result = TranslatedArticle(
            id=article.id,
            title_en=article.title,
            title_loga=title_loga,
            text_loga=text_loga,
        )

        if validate:
            back = await back_translate_text(client, text_loga[:2000], model)
            result.back_translated_text = back
            # Use first 2000 chars of original for comparison
            result.similarity_score = compute_similarity(article.text[:2000], back)

        return result


async def run_translation_pipeline(
    input_path: Path,
    output_path: Path,
    model: str,
    workers: int,
    validate_ratio: float,
    max_articles: int,
    resume: bool,
) -> None:
    client = anthropic.AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load already-translated IDs if resuming
    done_ids: set[str] = set()
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["id"])
                except (json.JSONDecodeError, KeyError):
                    pass
        log.info("Resuming: %d articles already translated", len(done_ids))

    # Stream input articles
    articles: list[Article] = []
    with open(input_path) as f:
        for i, line in enumerate(f):
            if max_articles > 0 and i >= max_articles:
                break
            try:
                d = json.loads(line)
                art = Article(id=d["id"], title=d["title"], text=d["text"])
                if art.id not in done_ids:
                    articles.append(art)
            except (json.JSONDecodeError, KeyError):
                continue

    log.info("Articles to translate: %d (workers=%d)", len(articles), workers)

    # Decide which articles to validate (random sample)
    validate_ids = set(
        a.id for a in random.sample(articles, max(1, int(len(articles) * validate_ratio)))
    )

    semaphore = asyncio.Semaphore(workers)
    low_similarity_log = output_path.parent / "low_similarity.jsonl"

    with open(output_path, "a") as out_f, \
         open(low_similarity_log, "a") as low_f:

        tasks = [
            process_article(
                client, art, model,
                validate=(art.id in validate_ids),
                semaphore=semaphore
            )
            for art in articles
        ]

        async for result in atqdm(
            _as_completed_iter(tasks),
            total=len(tasks),
            desc="Translating"
        ):
            record = {
                "id": result.id,
                "title_en": result.title_en,
                "title_loga": result.title_loga,
                "text_loga": result.text_loga,
            }
            if result.similarity_score >= 0:
                record["back_translated_text"] = result.back_translated_text
                record["similarity_score"] = result.similarity_score
                if result.similarity_score < 0.75:
                    low_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    low_f.flush()

            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()

    log.info("Done. Output: %s", output_path)
    if validate_ratio > 0:
        log.info("Low-similarity articles logged to: %s", low_similarity_log)


async def _as_completed_iter(tasks):
    """Yield task results as they complete."""
    futures = [asyncio.ensure_future(t) for t in tasks]
    pending = set(futures)
    while pending:
        done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
        for fut in done:
            yield await fut


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option("--input", "input_path", type=click.Path(exists=True, path_type=Path),
              default="data/raw/simplewiki-articles.jsonl", show_default=True)
@click.option("--output", "output_path", type=click.Path(path_type=Path),
              default="data/translated/loga-articles.jsonl", show_default=True)
@click.option("--model", default="claude-sonnet-4-6", show_default=True)
@click.option("--workers", default=8, show_default=True,
              help="Concurrent API requests")
@click.option("--validate-ratio", default=0.01, show_default=True,
              help="Fraction of articles to back-translate for validation")
@click.option("--max-articles", default=0, show_default=True,
              help="Maximum articles to process (0 = all)")
@click.option("--resume/--no-resume", default=True, show_default=True,
              help="Skip already-translated articles")
def cli(input_path, output_path, model, workers, validate_ratio, max_articles, resume):
    """Translate Simple English Wikipedia articles to Loga."""
    asyncio.run(run_translation_pipeline(
        input_path=input_path,
        output_path=output_path,
        model=model,
        workers=workers,
        validate_ratio=validate_ratio,
        max_articles=max_articles,
        resume=resume,
    ))


if __name__ == "__main__":
    cli()
