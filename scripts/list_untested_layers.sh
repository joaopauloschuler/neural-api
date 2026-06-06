#!/usr/bin/env bash
# v0 coverage report: list TNNet* classes declared in neural/neuralnetwork.pas
# that have no reference anywhere under tests/.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/neural/neuralnetwork.pas"
TESTS_DIR="$REPO_ROOT/tests"

if [ ! -f "$SRC" ]; then
  echo "ERROR: source not found: $SRC" >&2
  exit 1
fi

# Extract unique TNNet* class names from "TNNetXxx = class(...)" or "TNNetXxx = class;" declarations.
classes=$(grep -Eo '\bTNNet[A-Za-z0-9_]+[[:space:]]*=[[:space:]]*class\b' "$SRC" \
  | sed -E 's/[[:space:]]*=.*$//' \
  | sort -u)

untested=()
while IFS= read -r cls; do
  [ -z "$cls" ] && continue
  if ! grep -RqsF --include='*.pas' --include='*.lpr' "$cls" "$TESTS_DIR"; then
    untested+=("$cls")
  fi
done <<< "$classes"

printf '%s\n' "${untested[@]}" | sort
echo "---"
echo "Untested TNNet* classes: ${#untested[@]}"
