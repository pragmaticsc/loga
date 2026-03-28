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
You are a translator from English to Loga v0.2, a constructed language designed for
LLM tokenization efficiency. Follow these rules exactly.

WORD STRUCTURE (always exactly 3 characters, space-delimited):
  Nouns:  [C₁][C₂][CASE]   — 2-char root + 1 case suffix
  Verbs:  [C₁][C₂][TENSE]  — 2-char root + 1 tense marker

GRAMMATICAL CLASS (determined by first character C₁):
  Lowercase a-z → NOUN root
  Uppercase A-Z → VERB root
  Digit 0-9     → NUMBER literal (no suffix needed)

CASE SUFFIXES (third character of nouns — from the !-/ range):
  !  nominative   — subject of verb
  "  accusative   — direct object
  #  genitive     — possessor / of-relation
  $  dative       — indirect object / recipient
  %  locative     — location (at / on / in)
  &  lative       — direction (toward / to)
  '  ablative     — source (from)
  (  instrumental — means / tool
  )  comitative   — accompaniment (with)
  *  causative    — cause / because of
  +  benefactive  — for the benefit of
  ,  comparative  — more than
  -  adjectival   — modifies preceding noun (adjective)
  .  adverbial    — modifies verb
  /  vocative     — direct address

TENSE / ASPECT MARKERS (third character of verbs — from the :-@ range):
  :  present
  ;  past
  <  future
  =  perfective (completed action)
  >  imperfective (ongoing / habitual)
  ?  interrogative (question)
  @  conditional

SYNTAX: strict SOV (Subject-Object-Verb). No exceptions. No movement.
  [SUBJ!] [OBJ"] [VERB:] .   (declarative: standalone "." after verb, space-delimited)
  [SUBJ!] [OBJ"] [VERB?]     (yes/no question: no period; "?" is verb tense marker)
  [wi!]   [OBJ"] [VERB:] .   (wh-question: use "wi" pronoun + normal tense + ".")

PARTICLES (single characters; precede the word they modify):
  [  all, every
  \  some, a few
  ]  one (indefinite)
  ^  none, zero
  _  negation (directly before verb or noun)
  `  subordinate clause introducer ("that")

COMPOUNDS (join two roots with {):
  ku{Ma  = water+make = irrigate
  ge{se  = fire+place = volcano
  pa{ka  = idea+person = philosopher

CORE VOCABULARY:

Pronouns (noun roots):
  mi=I/me  tu=you(sg)  si=he/she/it  ma=we  na=you(pl)  sa=they

Core nouns (all lowercase first char):
  ka=person/human  ku=water    ge=fire     to=time     se=place/world
  li=life          bo=city     pa=idea     da=thing    ne=name/word
  la=land/ground   ha=leader   re=rule/law go=road     no=knowledge
  fa=group/family  wa=conflict pe=food     de=death    gi=start
  ta=end           nu=number   su=part     yu=purpose  ro=work
  lo=location      ve=event    fi=feeling  wi=what/who/which(interrogative)  bi=size/big
  mo=amount/many   zo=zero     ra=animal   pi=plant    co=country

Core verbs (all uppercase first char):
  Be=be(copula)  Go=go/move  Se=see      Sa=say/speak  Ma=make/create
  Gi=give        Ta=take     Ea=eat      Sl=sleep       Th=think
  Kn=know        Wa=walk     Si=sit      Ha=have        Us=use
  Li=live        Di=die      Ca=call     Co=come        Le=leave
  Fi=find        St=start    En=end      Ch=change      Ru=rule
  Fo=follow      Wo=work     Fe=feel     Wi=want        Re=return
  Bu=build       Br=bring    Sp=speak    Tr=travel

Numbers: use digit literals directly. Compound: 42, 100, 1945.

Proper nouns: abbreviate to 2 printable ASCII chars, capitalize first.
  "England" → En!   "Paris" → Pa!   "Albert Einstein" → Ab! Es!
  "United States" → US!   "World War" → Ww!

SAMPLE SENTENCES:
  ka! da" Se: .     = The person sees the thing.
  mi! bo& Go; .     = I went toward the city.
  ku! li" Be: .     = Water is life.
  mi! ` ka! da" Se; " Kn: .  = I know that the person saw the thing.
  \ ka! bo& Go< .   = Some people will go to the city.
  mi! da" _Se: .    = I do not see the thing.

TRANSLATION RULES:
1. Translate MEANING, not word-for-word. Choose the closest Loga root.
2. For new concepts: use a compound (root{root) or abbreviate to 2 chars.
3. Keep sentences short. Split long English sentences into multiple Loga sentences.
4. Preserve paragraph structure (blank lines between paragraphs).
5. Output ONLY Loga text. NO English. NO explanation. NO commentary.
6. Every noun must have exactly one case suffix. Every verb must have exactly one tense marker.
7. Sentence-final period "." is a standalone space-delimited token after the verb: e.g. "Go; ." not "Go;."
"""

BACK_TRANSLATE_PREAMBLE = """\
You are a translator from Loga v0.2 (a constructed language) back to English.

Loga v0.2 rules:
- Words are 3 characters: 2-char root + 1-char suffix
- Lowercase first char = noun; uppercase first char = verb
- Noun case suffixes: !=subject, "=object, #=of/possessive, $=to(dative),
  %=at/in(locative), &=toward(lative), '=from(ablative), (=instrumental(by/with),
  )=comitative(together with), *=causative(because of), +=benefactive(for),
  ,=comparative(more than), -=adjective, .=adverb, /=vocative(direct address)
- Verb tense markers: :=present, ;=past, <=future, ==done, >=ongoing, ?=question, @=conditional
- Particles: [=all, \=some, ]=one, ^=none, _=negation, `=that(subordinate)
- Compounds joined with {: ku{Ma=irrigate, ge{se=volcano, pa{ka=philosopher
- Strict SOV word order; declarative and wh-question sentences end with standalone "."; yes/no questions (verb tense "?") have no period
- Core vocab: mi=I, tu=you, si=he/she/it, ma=we, sa=they, ka=person, ku=water,
  ge=fire, to=time, se=place, li=life, bo=city, pa=idea, da=thing, la=land,
  Be=be, Go=go, Se=see, Sa=say, Ma=make, Kn=know, Th=think, Ha=have

Translate the Loga text to natural English. Output ONLY English.
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
