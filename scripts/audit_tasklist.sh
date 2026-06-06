#!/usr/bin/env bash
#
# audit_tasklist.sh
#
# Scans tasklist.md for UNCHECKED tasks ("- [ ]") that mention TNNet*
# class names and flags any whose name is already present in
# neural/neuralnetwork.pas, either as a class declaration or in the
# CreateLayer dispatch table. Such entries are stale and likely should
# be checked off.
#
# Usage:
#   scripts/audit_tasklist.sh
#
# Always exits 0; this is a report, not a CI gate.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TASKLIST="$REPO_ROOT/tasklist.md"
NN_PAS="$REPO_ROOT/neural/neuralnetwork.pas"

if [[ ! -f "$TASKLIST" ]]; then
  echo "ERROR: tasklist.md not found at $TASKLIST" >&2
  exit 0
fi
if [[ ! -f "$NN_PAS" ]]; then
  echo "ERROR: neuralnetwork.pas not found at $NN_PAS" >&2
  exit 0
fi

# Build set of known TNNet* names: class declarations OR quoted strings
# anywhere in neuralnetwork.pas (the CreateLayer dispatch uses quoted
# 'TNNetFoo' literals, which this captures).
KNOWN="$(mktemp)"
trap 'rm -f "$KNOWN"' EXIT

grep -oE '\bTNNet[A-Za-z0-9_]+\b' "$NN_PAS" | sort -u > "$KNOWN"

stale_count=0
# Walk unchecked tasks, extract TNNet* tokens, check membership.
while IFS=: read -r lineno content; do
  # Skip checked items defensively (grep already filtered, but be safe).
  [[ "$content" =~ ^[[:space:]]*-[[:space:]]\[[xX]\] ]] && continue
  # Extract all TNNet* identifiers from the line.
  for name in $(grep -oE '\bTNNet[A-Za-z0-9_]+\b' <<< "$content" | sort -u); do
    if grep -qxF "$name" "$KNOWN"; then
      printf 'STALE: %s  (tasklist.md:%s)\n' "$name" "$lineno"
      stale_count=$((stale_count + 1))
    fi
  done
done < <(grep -nE '^[[:space:]]*-[[:space:]]\[[[:space:]]\]' "$TASKLIST" | grep -E '\bTNNet[A-Za-z0-9_]+\b' || true)

echo "---"
echo "Total stale entries: $stale_count"
exit 0
