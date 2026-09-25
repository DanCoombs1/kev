#!/bin/bash
# Downloads STCray from Hugging Face into data/stcray/. Needs `hf auth login` with an account
# that has accepted the terms at https://huggingface.co/datasets/Naoufel555/STCray-Dataset.
set -uo pipefail
cd "$(dirname "$0")/../data" || exit 1
mkdir -p stcray

if ! hf auth whoami > /dev/null 2>&1; then
  echo "not logged in, run 'hf auth login' first"
  exit 1
fi

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
  if ! bsdtar -xf "stcray/$part.rar" -C stcray; then
    echo "$part: could not extract"
    exit 1
  fi
  rm "stcray/$part.rar"
  touch "stcray/.complete_$part"
  echo "$part: done ($(du -sh stcray | cut -f1))"
done
