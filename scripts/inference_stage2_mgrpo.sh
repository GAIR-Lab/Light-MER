#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="${CFG:-train_configs/stage2_mgrpo_qwen3_0_6b.yaml}"
STAGE2_CKPT_ROOT="${STAGE2_CKPT_ROOT:-checkpoints/stage2-mgrpo-qwen3-0.6b}"
DATASET="${DATASET:-inferenceData}"
BASE_ROOT="${BASE_ROOT:-${STAGE2_RESULT_ROOT:-output_stage2_mgrpo/results}}"
RUN_TAG="${RUN_TAG:-stage2_mgrpo_qwen3_0_6b}"
START_STEP="${START_STEP:-50}"
END_STEP="${END_STEP:-5000}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1200}"
TEMPERATURE="${TEMPERATURE:-0.9}"
TOP_P="${TOP_P:-0.9}"
MERGE_LORA="${MERGE_LORA:-1}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"
mkdir -p output logs checkpoints

PYTHON=(python)
if [ -n "${CONDA_ENV_NAME}" ]; then
  command -v conda >/dev/null 2>&1 || {
    echo "conda not found, but CONDA_ENV_NAME is set." >&2
    exit 1
  }
  PYTHON=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
fi

if [ ! -d "${STAGE2_CKPT_ROOT}" ]; then
  echo "Stage 2 checkpoint directory not found: ${STAGE2_CKPT_ROOT}" >&2
  echo "Set STAGE2_CKPT_ROOT=/path/to/stage2/checkpoints." >&2
  exit 1
fi

ARGS=(
  --zeroshot
  --dataset="${DATASET}"
  --cfg-path="${CFG}"
  --ckpt-dir="${STAGE2_CKPT_ROOT}"
  --max-new-tokens="${MAX_NEW_TOKENS}"
  --temperature="${TEMPERATURE}"
  --top-p="${TOP_P}"
  --start-step="${START_STEP}"
  --end-step="${END_STEP}"
  --base-root="${BASE_ROOT}"
  --run-tag="${RUN_TAG}"
  --quiet-prompt
  --compact-log
)

if [ "${MERGE_LORA}" = "1" ]; then
  ARGS+=(--merge-lora-for-inference)
else
  ARGS+=(--no-merge-lora-for-inference)
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON[@]}" -u inference_stage2_mgrpo.py "${ARGS[@]}" "$@"
