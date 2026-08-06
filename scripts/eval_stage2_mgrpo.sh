#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BASE_ROOT="${BASE_ROOT:-${STAGE2_RESULT_ROOT:-output_stage2_mgrpo/results}}"
RUN_TAG="${RUN_TAG:-stage2_mgrpo_qwen3_0_6b}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"

PYTHON=(python)
if [ -n "${CONDA_ENV_NAME}" ]; then
  command -v conda >/dev/null 2>&1 || {
    echo "conda not found, but CONDA_ENV_NAME is set." >&2
    exit 1
  }
  PYTHON=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON[@]}" -u scripts/eval_stage2_mgrpo.py \
  --base-root="${BASE_ROOT}" \
  --run-tag="${RUN_TAG}" "$@"
