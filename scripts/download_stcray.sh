#!/bin/bash
# Download STCray (CC BY 4.0, Khalifa University) from Hugging Face into data/stcray/.
#
# Needs a Hugging Face account that has accepted the dataset's terms at
# https://huggingface.co/datasets/Naoufel555/STCray-Dataset, logged in with `hf auth login`.
# Safe to re-run: finished parts are skipped.
set -uo pipefail
cd "$(dirname "$0")/../data" || exit 1
mkdir -p stcray

if ! hf auth whoami > /dev/null 2>&1; then
  echo "Not logged in to Hugging Face: run 'hf auth login' first"
  exit 1
fi

# Train and test first (2 GB, the core), then 21.6 GB of extra threat images made from CT scans.
for part in STCray_TrainSet STCray_TestSet STCray_Augmented; do
  if [ -f "stcray/.complete_$part" ]; then
    echo "$part: already done"
    continue
  fi
  echo "$part: downloading"
  if ! hf download Naoufel555/STCray-Dataset "$part.rar" --repo-type dataset --local-dir stcray > /dev/null; then
    echo "$part: download failed, re-run later"
    exit 1
  fi
  # macOS's built-in bsdtar reads .rar archives.
  if ! bsdtar -xf "stcray/$part.rar" -C stcray; then
    echo "$part: could not extract"
    exit 1
  fi
  rm "stcray/$part.rar"
  touch "stcray/.complete_$part"
  echo "$part: done ($(du -sh stcray | cut -f1) so far)"
done
