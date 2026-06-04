#!/usr/bin/env bash
# Build and run the full test suite. Exits non-zero on any test failure.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate LazUtils (needed for utf8process). Honour an explicit LAZUTILS_PATH
# if set; otherwise auto-discover it the same way the CI workflow does, by
# finding the compiled utf8process.ppu under the Lazarus install tree.
if [ -z "${LAZUTILS_PATH:-}" ]; then
  LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1 || true)
fi

if [ -z "$LAZUTILS_PATH" ] || [ ! -d "$LAZUTILS_PATH" ]; then
  echo "ERROR: lazutils path not found (looked for utf8process.ppu under /usr/lib/lazarus and /usr/share/lazarus)" >&2
  echo "Override with LAZUTILS_PATH=/path/to/lazutils/lib/<arch> $0" >&2
  exit 2
fi

echo "==> Building RunTests (LAZUTILS_PATH=$LAZUTILS_PATH)"
fpc -B -Fu../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 RunTests.pas

echo "==> Running tests"
exec ./RunTests -a -p
