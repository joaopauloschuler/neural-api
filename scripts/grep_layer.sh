#!/usr/bin/env bash
# grep_layer.sh — print the declaration, Compute, Backpropagate and
# Test* references for a TNNet* class in neural/neuralnetwork.pas.
# Usage: scripts/grep_layer.sh TNNetReLU
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $(basename "$0") <TNNetClassName>" >&2
  exit 0
fi

CLASS="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
NN_PAS="$REPO_ROOT/neural/neuralnetwork.pas"
TESTS_DIR="$REPO_ROOT/tests"

section() { echo "=== $1 ==="; }

section "Declaration"
awk -v cls="$CLASS" '
  $0 ~ ("(^|[[:space:]])" cls "[[:space:]]*=[[:space:]]*class") {
    print FILENAME ":" NR ": " $0
    for (i=1; i<=6 && (getline l)>0; i++) print l
    f=1; exit
  } END { if(!f) print "(no override found)" }' "$NN_PAS"

for METHOD in Compute Backpropagate; do
  echo; section "$METHOD"
  awk -v pat="^(procedure|function)[[:space:]]+${CLASS}\\.${METHOD}([[:space:](]|\$)" '
    $0 ~ pat { b=1 }
    b { print }
    b && /^end;[[:space:]]*$/ { exit }
    END { if(!b) print "(no override found)" }' "$NN_PAS"
done

echo; section "Test methods"
# For each test file mentioning CLASS, list Test* methods whose body references it.
out=""
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  out+="$(awk -v cls="$CLASS" -v fname="$f" '
    /^[[:space:]]*(procedure|function)[[:space:]]+[A-Za-z0-9_]+\.Test[A-Za-z0-9_]*/ {
      name=$0; nameln=NR; inproc=1; mentions=0; next
    }
    inproc && index($0, cls) { mentions=1 }
    inproc && /^end;[[:space:]]*$/ {
      if (mentions) print fname ":" nameln ": " name
      inproc=0
    }' "$f")"$'\n'
done < <(grep -rlF --include='*.pas' "$CLASS" "$TESTS_DIR" 2>/dev/null || true)
out="$(printf '%s' "$out" | sed '/^$/d')"
if [[ -z "$out" ]]; then
  echo "(no Test* methods reference $CLASS)"
else
  printf '%s\n' "$out"
fi
exit 0
