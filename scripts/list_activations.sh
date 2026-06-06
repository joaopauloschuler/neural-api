#!/usr/bin/env bash
# list_activations.sh — enumerate every TNNetReLUBase descendant in
# neural/neuralnetwork.pas and mark each [tested] or [UNTESTED] based on
# whether any tests/*.pas test method body references the class name.
# Example output:
#   TNNetAbs                         [tested]
#   TNNetTanhExp                     [UNTESTED]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NN_PAS="$REPO_ROOT/neural/neuralnetwork.pas"
TESTS_DIR="$REPO_ROOT/tests"

if [[ ! -f "$NN_PAS" ]]; then
  echo "ERROR: $NN_PAS not found" >&2
  exit 1
fi

classes=$(grep -Eo '^[[:space:]]*TNNet[A-Za-z0-9_]+[[:space:]]*=[[:space:]]*class\(TNNetReLUBase\)' "$NN_PAS" \
  | sed -E 's/^[[:space:]]*//; s/[[:space:]]*=.*$//' \
  | sort -u)

tested=0
untested=0
while IFS= read -r cls; do
  [[ -z "$cls" ]] && continue
  if grep -RqsF --include='*.pas' "$cls" "$TESTS_DIR"; then
    printf '%-32s [tested]\n'   "$cls"
    tested=$((tested + 1))
  else
    printf '%-32s [UNTESTED]\n' "$cls"
    untested=$((untested + 1))
  fi
done <<< "$classes"

echo "---"
echo "TNNetReLUBase descendants: $((tested + untested))  tested: $tested  untested: $untested"
exit 0
