#!/usr/bin/env bash
# Build and run the full test suite. Exits non-zero on any test failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LAZUTILS_PATH="${LAZUTILS_PATH:-/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux}"

if [ ! -d "$LAZUTILS_PATH" ]; then
  echo "ERROR: lazutils path not found: $LAZUTILS_PATH" >&2
  echo "Override with LAZUTILS_PATH=/path/to/lazutils/lib/<arch> $0" >&2
  exit 2
fi

echo "==> Building RunTests (LAZUTILS_PATH=$LAZUTILS_PATH)"
fpc -B -Fu../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 RunTests.pas

echo "==> Running tests"
exec ./RunTests -a -p
