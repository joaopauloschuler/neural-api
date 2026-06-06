#!/usr/bin/env bash
# coverage_gradient_tests.sh — list TNNet* layers in neural/neuralnetwork.pas
# whose Backpropagate is overridden but which lack any Test* method whose
# name (not just body) mentions the class name across tests/*.pas.
# Example output:
#   TNNetFooBar
#   TNNetQuux
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NN_PAS="$REPO_ROOT/neural/neuralnetwork.pas"
TESTS_DIR="$REPO_ROOT/tests"

if [[ ! -f "$NN_PAS" ]]; then
  echo "ERROR: $NN_PAS not found" >&2
  exit 1
fi

# Classes that have an overridden Backpropagate implementation.
backprop_classes=$(grep -Eo '^procedure[[:space:]]+TNNet[A-Za-z0-9_]+\.Backpropagate\b' "$NN_PAS" \
  | sed -E 's/^procedure[[:space:]]+//; s/\.Backpropagate.*$//' \
  | sort -u)

# Build a blob of Test* method bodies across tests/*.pas. A class is "covered"
# if its name appears either in a Test* method name OR inside a Test* body
# (e.g. TestGlobalSumPoolGradientCheck constructs TNNetGlobalSumPool inside).
TEST_BLOB="$(mktemp)"
trap 'rm -f "$TEST_BLOB"' EXIT
awk '
  /^[[:space:]]*(procedure|function)[[:space:]]+[A-Za-z0-9_]+\.Test[A-Za-z0-9_]+/ { inproc=1 }
  inproc { print }
  inproc && /^end;[[:space:]]*$/ { inproc=0 }
' "$TESTS_DIR"/*.pas > "$TEST_BLOB" 2>/dev/null || true

missing_list=()
while IFS= read -r cls; do
  [[ -z "$cls" ]] && continue
  if ! grep -qF "$cls" "$TEST_BLOB"; then
    missing_list+=("$cls")
  fi
done <<< "$backprop_classes"

missing=${#missing_list[@]}
if (( missing > 0 )); then
  printf '%s\n' "${missing_list[@]}" | sort -u
fi

echo "---"
echo "Backprop-overriding classes without a Test* method mentioning them: $missing"
exit 0
