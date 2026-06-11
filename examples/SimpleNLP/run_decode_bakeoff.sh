#!/usr/bin/env bash
# Driver for the decode-efficiency features bakeoff: runs every phase serially
# under a hard `timeout 270` cap (the program self-budgets, so the cap should
# normally NOT fire), tee-ing each phase's output to a log file, then prints a
# final summary table.
#
# Usage:  bash run_decode_bakeoff.sh
# Run it from examples/SimpleNLP (the program reads datasets/ relative paths).
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BIN="../../bin/$(uname -m)-linux/bin/DecodeFeaturesBakeoff"
if [ ! -x "$BIN" ]; then
  echo "Building DecodeFeaturesBakeoff with lazbuild..."
  lazbuild --build-mode=Release DecodeFeaturesBakeoff.lpi || exit 1
fi
if [ ! -x "$BIN" ]; then
  echo "ERROR: $BIN not found after build." >&2
  exit 1
fi

declare -A RESULT
for N in 1 2 3 4 5 6 7 8 9; do
  LOG="decode_bakeoff_phase${N}.log"
  echo "==================================================================="
  echo "=== Phase $N (cap: 270 s)  ->  $LOG"
  echo "==================================================================="
  # stdbuf -oL: line-buffer stdout so per-batch training logs survive the tee
  # even if the timeout cap fires mid-run.
  timeout 270 stdbuf -oL "$BIN" --phase "$N" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  if [ "$RC" -eq 124 ]; then
    RESULT[$N]="TIMEOUT (270 s cap fired)"
  elif [ "$RC" -eq 0 ]; then
    RESULT[$N]="OK"
  else
    RESULT[$N]="FAIL (exit $RC)"
  fi
done

echo
echo "==================== decode bakeoff summary ===================="
for N in 1 2 3 4 5 6 7 8 9; do
  printf "  phase %d : %s\n" "$N" "${RESULT[$N]}"
done
echo "Per-phase output: decode_bakeoff_phase{1..9}.log"
