#!/bin/bash
# Downloads the English text for the tokenizer and text pretraining into data/text/:
# WikiText-103 (Wikipedia articles, CC BY-SA 3.0) and SQuAD's questions (CC BY-SA 4.0). No login needed.
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
du -sh text/wikitext text/squad
