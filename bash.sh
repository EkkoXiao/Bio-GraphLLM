#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-MolTC}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$#" -gt 0 ]]; then
  exec conda run --no-capture-output -n "${ENV_NAME}" \
    python "${SCRIPT_DIR}/gpu_saturate.py" "$@"
fi

exec conda run --no-capture-output -n "${ENV_NAME}" \
  python "${SCRIPT_DIR}/gpu_saturate.py" \
  --gpus "${GPUS:-all}" \
  --memory-fraction "${MEMORY_FRACTION:-0.90}" \
  --matrix-size "${MATRIX_SIZE:-8192}" \
  --dtype "${DTYPE:-fp16}" \
  --seconds "${RUN_SECONDS:-0}" \
  --report-every "${REPORT_EVERY:-30}"
