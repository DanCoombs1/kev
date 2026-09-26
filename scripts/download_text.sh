#!/bin/bash
# Downloads the English text for the tokenizer and text pretraining into data/text/: WikiText-103 (Wikipedia
# articles, CC BY-SA 3.0), SQuAD's questions (CC BY-SA 4.0), and Simple and full English Wikipedia (CC BY-SA 3.0).
# The full Wikipedia (11.6 GB) is only kept long enough to pick out paragraphs on kev's topics. No login needed.
set -uo pipefail
cd "$(dirname "$0")/../data" || exit 1
mkdir -p text

if ! hf download Salesforce/wikitext --repo-type dataset --include "wikitext-103-raw-v1/*" --local-dir text/wikitext > /dev/null; then
  echo "wikitext: download failed, re-run later"
  exit 1
fi
if ! hf download rajpurkar/squad --repo-type dataset --include "plain_text/*" --local-dir text/squad > /dev/null; then
  echo "squad: download failed, re-run later"
  exit 1
fi
if ! hf download wikimedia/wikipedia --repo-type dataset --include "20231101.simple/*" --local-dir text/wikipedia > /dev/null; then
  echo "simple wikipedia: download failed, re-run later"
  exit 1
fi
if [ ! -f text/wikipedia_topics.parquet ]; then
  if ! hf download wikimedia/wikipedia --repo-type dataset --include "20231101.en/*" --local-dir text/wikipedia > /dev/null; then
    echo "wikipedia: download failed, re-run later"
    exit 1
  fi
  (cd .. && uv run python -m kev.data.text) || exit 1
  rm -r text/wikipedia/20231101.en
fi
du -sh text/wikitext text/squad text/wikipedia text/wikipedia_topics.parquet
