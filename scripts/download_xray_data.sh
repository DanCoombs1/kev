#!/bin/bash
# Downloads DvXray, COMPASS-XP and PIDray into data/. Re-run to resume; Google Drive
# sometimes blocks popular files for up to a day ("quota exceeded").
set -uo pipefail
cd "$(dirname "$0")/../data" || exit 1

gdrive() { echo "https://drive.usercontent.google.com/download?id=$1&export=download&confirm=t"; }

get() {  # name url bytes dest
  local name=$1 url=$2 size=$3 dest=$4
  if [ -f "$dest/.complete_$name" ]; then
    echo "$name: already done"
    return 0
  fi
  # a blocked Drive file returns an HTML page instead
  if ! curl -sIL "$url" | tr -d '\r' | grep -qi '^content-type: application/octet-stream'; then
    echo "$name: not available right now, re-run later"
    return 1
  fi
  echo "$name: downloading $((size / 1000000000)) GB"
  if ! { curl -sSfL -o "$name.zip" "$url" && [ "$(wc -c < "$name.zip" | tr -d ' ')" = "$size" ] && unzip -tq "$name.zip" > /dev/null; }; then
    echo "$name: download incomplete or corrupt, re-run later"
    rm -f "$name.zip"
    return 1
  fi
  mkdir -p "$dest" && unzip -qo "$name.zip" -x '__MACOSX/*' '*.DS_Store' -d "$dest" && rm "$name.zip"
  touch "$dest/.complete_$name"
  echo "$name: done"
}

status=0
get dvxray_positive "$(gdrive 1NK1DWLMztROwRkJIlYAnexWLgyFv_gDF)" 2997873748 dvxray || status=1
get dvxray_negative "$(gdrive 18QJyRNVDG6jguNmV04GRuZM98IGdizUb)" 4110370849 dvxray || status=1
get compass_xp "https://zenodo.org/records/2654887/files/COMPASS-XP.zip?download=1" 5907597446 compass-xp || status=1
get pidray "$(gdrive 1UMq0CP20lKcraOTvsFMjiLjPfDam9jAp)" 11583282821 . || status=1
exit $status
