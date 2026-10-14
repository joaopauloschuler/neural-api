#!/usr/bin/env bash
# Build and run the test suite under a cgroup RAM cap, in the foreground.
#   ./run_suite_capped.sh <cpu|avx2|opencl> [--suite=TestName] [--no-build]
# Always ends with ONE line:
#   SUITE=<mode> EXIT=<code> COMPLETE=<yes|no> TESTS=<n> ERRORS=<n> FAILURES=<n> IGNORED=<n> LOG=<file>
# COMPLETE=no means the run ended without </TestResults> (crash, cap kill or timeout).
# Environment: MEMORY_MAX (default 2G, real RAM, no swap), SUITE_TIMEOUT (seconds,
# default 1800), SUITE_LOG_DIR (default $TMPDIR or /tmp), LAZUTILS_PATH.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODE="${1:-}"
shift || true
SUITE_ARG="-a"
DO_BUILD=1
for Arg in "$@"; do
  case "$Arg" in
    --suite=*) SUITE_ARG="$Arg" ;;
    --no-build) DO_BUILD=0 ;;
    *) echo "unknown argument: $Arg" >&2; exit 2 ;;
  esac
done

case "$MODE" in
  cpu)    DEFINES="";         BINARY="RunTests" ;;
  avx2)   DEFINES="-dAVX2";   BINARY="RunTestsAVX2" ;;
  opencl) DEFINES="-dOpenCL"; BINARY="RunTestsCL" ;;
  *) echo "usage: $0 <cpu|avx2|opencl> [--suite=TestName] [--no-build]" >&2; exit 2 ;;
esac

MEMORY_MAX="${MEMORY_MAX:-2G}"
SUITE_TIMEOUT="${SUITE_TIMEOUT:-1800}"
LOG_DIR="${SUITE_LOG_DIR:-${TMPDIR:-/tmp}}"
BUILD_LOG="$LOG_DIR/neural-suite-$MODE-build.log"
RUN_LOG="$LOG_DIR/neural-suite-$MODE.log"
mkdir -p "$LOG_DIR"

# One capped run at a time: PoCL dies on concurrent OpenCL runs, and two
# builds would overwrite each other's .ppu files.
exec 9>"$LOG_DIR/neural-suite.lock"
if ! flock -n 9; then
  echo "SUITE=$MODE EXIT=75 COMPLETE=no TESTS=? ERRORS=? FAILURES=? IGNORED=? LOG=another capped run holds $LOG_DIR/neural-suite.lock"
  exit 75
fi

# Caps real RAM (not virtual address space, which PoCL reserves in bulk);
# only the capped command is killed when it goes over.
capped() {
  systemd-run --user --scope -q -p MemoryMax="$MEMORY_MAX" -p MemorySwapMax=0 \
    timeout --signal=KILL "$SUITE_TIMEOUT" "$@"
}

summary() {
  local ExitCode="$1" Log="$2" Complete="no" Tests="?" Errors="?" Failures="?" Ignored="?"
  if grep -q '</TestResults>' "$Log" 2>/dev/null; then
    Complete="yes"
    Tests=$(grep -o '<NumberOfRunTests>[0-9]*' "$Log" | tail -1 | grep -o '[0-9]*$')
    Errors=$(grep -o '<NumberOfErrors>[0-9]*' "$Log" | tail -1 | grep -o '[0-9]*$')
    Failures=$(grep -o '<NumberOfFailures>[0-9]*' "$Log" | tail -1 | grep -o '[0-9]*$')
    Ignored=$(grep -o '<NumberOfIgnoredTests>[0-9]*' "$Log" | tail -1 | grep -o '[0-9]*$')
  fi
  echo "SUITE=$MODE EXIT=$ExitCode COMPLETE=$Complete TESTS=$Tests ERRORS=$Errors FAILURES=$Failures IGNORED=$Ignored LOG=$Log"
}

if [ "$DO_BUILD" = "1" ]; then
  if [ -z "${LAZUTILS_PATH:-}" ]; then
    LAZUTILS_PATH=$(find /usr/lib/lazarus /usr/share/lazarus -name "utf8process.ppu" -printf "%h\n" 2>/dev/null | head -1 || true)
  fi
  # shellcheck disable=SC2086
  capped fpc -B $DEFINES -Fu../neural -Fu"$LAZUTILS_PATH" -Mobjfpc -Sh -O2 \
    -o"$BINARY" RunTests.pas > "$BUILD_LOG" 2>&1
  BuildExit=$?
  if [ "$BuildExit" -ne 0 ]; then
    echo "SUITE=$MODE EXIT=$BuildExit COMPLETE=no TESTS=? ERRORS=? FAILURES=? IGNORED=? LOG=$BUILD_LOG (build failed)"
    exit "$BuildExit"
  fi
fi

capped "./$BINARY" "$SUITE_ARG" -p > "$RUN_LOG" 2>&1
RunExit=$?
summary "$RunExit" "$RUN_LOG"
exit "$RunExit"
