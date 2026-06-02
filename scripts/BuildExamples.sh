#!/usr/bin/env bash
# Batch-compile every example program under examples/ to catch build breakage.
#
# Compiles each examples/**/*.lpr with fpc against ../neural (same flags as
# tests/RunAll.sh). Console examples only: programs whose uses-clause pulls in
# the LCL widgetset (Interfaces/Forms/Graphics/...) need lazbuild + a GUI
# toolchain and are auto-skipped (listed at the end). Does NOT run the
# binaries — compile check only.
#
# Exit code is non-zero if any non-skipped example fails to compile.
#
# Usage:
#   bash scripts/BuildExamples.sh
#   LAZUTILS_PATH=/path/to/lazutils/lib/<arch> bash scripts/BuildExamples.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
NEURAL_DIR="$ROOT_DIR/neural"
EXAMPLES_DIR="$ROOT_DIR/examples"

# LazUtils is optional for the examples (none use UTF8Process), but pass it
# through when available so the flags mirror tests/RunAll.sh.
LAZUTILS_PATH="${LAZUTILS_PATH:-/usr/share/lazarus/4.4.0/components/lazutils/lib/x86_64-linux}"
LAZ_FLAG=()
if [ -d "$LAZUTILS_PATH" ]; then
  LAZ_FLAG=(-Fu"$LAZUTILS_PATH")
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "$WORK_DIR"' EXIT
mkdir -p "$WORK_DIR/units"   # fpc -FU does not create its output dir

# A program needs the LCL widgetset (cannot compile headless with plain fpc)
# when its uses-clause references a GUI unit.
needs_gui() {
  awk 'BEGIN{IGNORECASE=1} /uses/{u=1} u{print} /;/{if(u)exit}' "$1" \
    | grep -qiE '(^|[ ,(])(Interfaces|Forms|LCLType|LCLIntf|Graphics|Controls|ComCtrls|StdCtrls|ExtCtrls)([ ,;])'
}

# Pre-existing examples that need a special build setup beyond plain fpc + neural.
# Skipped so this script gates on the regular console examples. Each also fails
# to compile on master with the same flags — they are not regressions.
declare -A SPECIAL_SKIP=(
  [examples/SimpleImageClassifierGPU/SimpleImageClassifierGPU.lpr]="needs OpenCL build (-dOpenCL + easycl)"
  [examples/Cifar10Resize/Cifar10Resize.lpr]="depends on usuperresolutionexample from examples/SuperResolution"
  [examples/StringManipulation/StringManipulation.lpr]="calls GenerateStringFromChars, absent on master too"
)

declare -a OK_LIST=() FAIL_LIST=() SKIP_LIST=()

mapfile -t LPRS < <(find "$EXAMPLES_DIR" -name '*.lpr' | sort)
echo "==> Found ${#LPRS[@]} example programs under examples/"
[ -d "$LAZUTILS_PATH" ] && echo "==> Using LazUtils: $LAZUTILS_PATH" || echo "==> LazUtils not found (optional) — building without it"
echo

for lpr in "${LPRS[@]}"; do
  rel="${lpr#"$ROOT_DIR"/}"
  if needs_gui "$lpr"; then
    SKIP_LIST+=("$rel")
    printf 'SKIP (GUI) %s\n' "$rel"
    continue
  fi
  if [ -n "${SPECIAL_SKIP[$rel]:-}" ]; then
    SKIP_LIST+=("$rel")
    printf 'SKIP (env) %s  — %s\n' "$rel" "${SPECIAL_SKIP[$rel]}"
    continue
  fi
  out="$WORK_DIR/$(basename "${lpr%.lpr}")"
  log="$WORK_DIR/build.log"
  if fpc -B -Mobjfpc -Sh -O2 -Fu"$NEURAL_DIR" "${LAZ_FLAG[@]}" \
        -FU"$WORK_DIR/units" -o"$out" "$lpr" >"$log" 2>&1; then
    OK_LIST+=("$rel")
    printf 'OK         %s\n' "$rel"
  else
    FAIL_LIST+=("$rel")
    printf 'FAIL       %s\n' "$rel"
    sed 's/^/    | /' "$log" | grep -iE 'error|fatal' | head -5
  fi
done

echo
echo "==================== SUMMARY ===================="
echo "  Compiled OK : ${#OK_LIST[@]}"
echo "  Skipped     : ${#SKIP_LIST[@]}  (GUI widgetset / special build setup)"
echo "  Failed      : ${#FAIL_LIST[@]}"
if [ "${#FAIL_LIST[@]}" -gt 0 ]; then
  echo
  echo "  Failures:"
  printf '    - %s\n' "${FAIL_LIST[@]}"
  exit 1
fi
echo "  All non-GUI examples compiled cleanly."
