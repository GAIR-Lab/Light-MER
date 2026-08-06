#!/usr/bin/env bash

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="${CFG:-train_configs/stage2_mgrpo_qwen3_0_6b.yaml}"
STAGE1_CKPT="${STAGE1_CKPT:-checkpoints/stage1-swdh-qwen3-0.6b/checkpoint_000025_loss_1.549.pth}"
OUTPUT_DIR="${OUTPUT_DIR:-output/stage2_mgrpo_qwen3_0_6b}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_DIR}" logs checkpoints

PYTHON=(python)
if [ -n "${CONDA_ENV_NAME}" ]; then
  command -v conda >/dev/null 2>&1 || {
    echo "conda not found, but CONDA_ENV_NAME is set." >&2
    exit 1
  }
  PYTHON=(conda run --no-capture-output -n "${CONDA_ENV_NAME}" python)
fi

if [ ! -f "${STAGE1_CKPT}" ]; then
  echo "Stage 1 checkpoint not found: ${STAGE1_CKPT}" >&2
  echo "Set STAGE1_CKPT=/path/to/checkpoint_000025_loss_1.549.pth." >&2
  exit 1
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON[@]}" -u train_stage2_mgrpo.py \
  --cfg-path="${CFG}" \
  --options "model.grpo.ckpt=${STAGE1_CKPT}" "run.output_dir=${OUTPUT_DIR}" "$@"
