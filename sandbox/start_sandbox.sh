#!/usr/bin/env bash
# Start the adaptive sandbox API (no simpletir dependency).
# Same port/path as before: http://127.0.0.1:12345/faas/sandbox/
# Run from adaptive/sandbox/:  ./start_sandbox.sh
set -euo pipefail

SANDBOX_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SANDBOX_ROOT"

SANDBOX_PORT="${SANDBOX_PORT:-12345}"
SANDBOX_WORKERS="${SANDBOX_WORKERS:-4}"
SANDBOX_LOG="${SANDBOX_LOG:-sandbox.log}"

# Default: bwrap for isolation (restricts filesystem, proc, etc.; reduces impact of executed code).
# Set SANDBOX_BACKEND=plain to run without bwrap (e.g. if bwrap is not installed).
SANDBOX_BACKEND="${SANDBOX_BACKEND:-plain}"
if [[ "${SANDBOX_BACKEND}" == "bwrap" ]]; then
  echo "[info] Starting sandbox under bwrap (isolation enabled)"
  bwrap --ro-bind /usr /usr --tmpfs /tmp --proc /proc --dev /dev \
    --bind "$SANDBOX_ROOT" "$SANDBOX_ROOT" \
    --chdir "$SANDBOX_ROOT" \
    env -i HOME="$HOME" PATH="/usr/bin:/bin" \
    uvicorn sandbox_api:app --host 127.0.0.1 --port "$SANDBOX_PORT" --workers "$SANDBOX_WORKERS" \
    > "${SANDBOX_LOG}" 2>&1 &
else
  echo "[info] Starting sandbox without bwrap (SANDBOX_BACKEND=$SANDBOX_BACKEND)"
  uvicorn sandbox_api:app --host 127.0.0.1 --port "$SANDBOX_PORT" --workers "$SANDBOX_WORKERS" \
    > "${SANDBOX_LOG}" 2>&1 &
fi

echo "[info] Sandbox PID=$!"
echo "[info] Log: $SANDBOX_ROOT/${SANDBOX_LOG}"
echo "[info] URL: http://127.0.0.1:${SANDBOX_PORT}/faas/sandbox/"
echo "[info] Check: curl -s http://127.0.0.1:${SANDBOX_PORT}/health"
