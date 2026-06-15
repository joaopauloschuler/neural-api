#!/usr/bin/env bash
# Download + prepare a SMALL slice of Google Speech Commands v2 for the
# SpeechCommands --full path. NOT run by the smoke (which is fully synthetic
# and needs no network). The full archive is ~2.3 GB; this script grabs it,
# keeps a handful of keyword folders, and (if ffmpeg is present) re-encodes
# every clip to 16 kHz mono 16-bit PCM -- the only format LoadWav16ToVolume
# accepts.
#
# Usage:   scripts/download_speech_commands.sh /path/to/out
# Then:    ./SpeechCommands --full /path/to/out/keywords
set -euo pipefail

OUT="${1:-./speech_commands}"
KEYWORDS=(yes no up down left right)
URL="http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"

mkdir -p "$OUT"
cd "$OUT"

if [ ! -f speech_commands_v0.02.tar.gz ]; then
  echo "Downloading Speech Commands v2 (~2.3 GB) ..."
  curl -L -o speech_commands_v0.02.tar.gz "$URL"
fi

echo "Extracting selected keyword folders ..."
mkdir -p raw
for kw in "${KEYWORDS[@]}"; do
  tar -xzf speech_commands_v0.02.tar.gz -C raw "$kw" 2>/dev/null || true
done

echo "Re-encoding to 16 kHz mono 16-bit PCM under keywords/ ..."
mkdir -p keywords
for kw in "${KEYWORDS[@]}"; do
  mkdir -p "keywords/$kw"
  for f in raw/"$kw"/*.wav; do
    [ -e "$f" ] || continue
    base="$(basename "$f")"
    if command -v ffmpeg >/dev/null 2>&1; then
      ffmpeg -y -loglevel error -i "$f" -ar 16000 -ac 1 -sample_fmt s16 \
        "keywords/$kw/$base"
    else
      # The originals are already 16 kHz mono 16-bit, so a copy works if you
      # trust the source format. ffmpeg is recommended to guarantee it.
      cp "$f" "keywords/$kw/$base"
    fi
  done
done

echo "Done. Train with:"
echo "    ./SpeechCommands --full \"$OUT/keywords\""
