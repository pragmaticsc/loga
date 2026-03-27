#!/usr/bin/env bash
# Download and prepare Simple English Wikipedia corpus for translation.
#
# Usage:
#   ./data/download.sh [--output-dir data/raw] [--max-articles 50000]
#
# Output:
#   data/raw/simplewiki-articles.jsonl  — one JSON object per article:
#     {"id": "...", "title": "...", "text": "..."}
#   data/raw/simplewiki-sentences.txt   — one sentence per line (for tokenizer training)

set -euo pipefail

OUTPUT_DIR="${1:-data/raw}"
MAX_ARTICLES="${2:-0}"  # 0 = all articles

DUMP_URL="https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
DUMP_FILE="$OUTPUT_DIR/simplewiki-latest.xml.bz2"

mkdir -p "$OUTPUT_DIR"

echo "==> Downloading Simple English Wikipedia dump..."
if [ ! -f "$DUMP_FILE" ]; then
    curl -L --progress-bar -o "$DUMP_FILE" "$DUMP_URL"
    echo "    Saved to $DUMP_FILE"
else
    echo "    Already downloaded: $DUMP_FILE"
fi

echo "==> Extracting and parsing articles..."
python3 - <<'PYEOF'
import sys
import os
import json
import bz2
import re
import xml.etree.ElementTree as ET

output_dir = os.environ.get("OUTPUT_DIR", "data/raw")
max_articles = int(os.environ.get("MAX_ARTICLES", "0"))
dump_file = os.path.join(output_dir, "simplewiki-latest.xml.bz2")
out_jsonl = os.path.join(output_dir, "simplewiki-articles.jsonl")
out_sentences = os.path.join(output_dir, "simplewiki-sentences.txt")

NS = "http://www.mediawiki.org/xml/ns/export/0.10"

def clean_wikitext(text: str) -> str:
    """Remove wiki markup to get plain text."""
    # Remove templates {{...}}
    while "{{" in text:
        text = re.sub(r'\{\{[^{}]*\}\}', '', text)
    # Remove tables {|...|}
    text = re.sub(r'\{\|.*?\|\}', '', text, flags=re.DOTALL)
    # Remove file/image links
    text = re.sub(r'\[\[(?:File|Image|Category):[^\]]*\]\]', '', text, flags=re.IGNORECASE)
    # Unwrap [[link|display]] → display; [[link]] → link
    text = re.sub(r'\[\[(?:[^|\]]*\|)?([^\]]*)\]\]', r'\1', text)
    # Remove external links [url display] → display
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove headings markup
    text = re.sub(r'={2,6}([^=]+)={2,6}', r'\1', text)
    # Remove bold/italic
    text = re.sub(r"'{2,3}", '', text)
    # Normalize whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def is_valid_article(title: str, text: str) -> bool:
    """Filter out redirect pages, stubs with < 100 chars, and special pages."""
    if text.strip().lower().startswith("#redirect"):
        return False
    if len(text.strip()) < 100:
        return False
    skip_prefixes = ("Wikipedia:", "Talk:", "User:", "File:", "Template:", "Help:", "Category:")
    if any(title.startswith(p) for p in skip_prefixes):
        return False
    return True

count = 0
sentence_buffer = []

with bz2.open(dump_file, 'rt', encoding='utf-8') as f, \
     open(out_jsonl, 'w', encoding='utf-8') as jf, \
     open(out_sentences, 'w', encoding='utf-8') as sf:

    context = ET.iterparse(f, events=('end',))
    for event, elem in context:
        tag = elem.tag.replace(f'{{{NS}}}', '')
        if tag != 'page':
            continue

        title_el = elem.find(f'{{{NS}}}title')
        text_el = elem.find(f'.//{{{NS}}}text')
        id_el = elem.find(f'{{{NS}}}id')

        if title_el is None or text_el is None or text_el.text is None:
            elem.clear()
            continue

        title = title_el.text.strip()
        raw_text = text_el.text
        article_id = id_el.text if id_el is not None else str(count)

        clean = clean_wikitext(raw_text)
        if not is_valid_article(title, clean):
            elem.clear()
            continue

        record = {"id": article_id, "title": title, "text": clean}
        jf.write(json.dumps(record, ensure_ascii=False) + '\n')

        # Write sentences for tokenizer training
        for para in clean.split('\n\n'):
            para = para.strip()
            if len(para) > 20:
                sf.write(para + '\n')

        count += 1
        if count % 10000 == 0:
            print(f"    Processed {count} articles...", flush=True)

        if max_articles > 0 and count >= max_articles:
            break

        elem.clear()

print(f"    Done. {count} articles written to {out_jsonl}")
PYEOF

echo "==> Stats:"
wc -l "$OUTPUT_DIR/simplewiki-articles.jsonl" | awk '{print "    Articles:", $1}'
wc -c "$OUTPUT_DIR/simplewiki-sentences.txt" | awk '{printf "    Sentences file: %.1f MB\n", $1/1048576}'
echo "==> Done."
