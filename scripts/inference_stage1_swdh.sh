#!/usr/bin/env bash
#SBATCH --job-name=swdh_inf
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=40G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="${CFG:-train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml}"
DATASET="${DATASET:-inferenceData}"
REPEAT="${REPEAT:-1}"
BASE_ROOT="${BASE_ROOT:-output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/repeat${REPEAT}/results}"
TEST_EPOCHS="${TEST_EPOCHS:-xxx-xxx}"
SKIP_EPOCH="${SKIP_EPOCH:-5}"
GPU="${GPU:-0}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"
mkdir -p output logs "${BASE_ROOT}"

if [ -n "${CONDA_ENV_NAME}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

OPTIONS=(
  "inference.base_root=${BASE_ROOT}"
  "inference.test_epochs=${TEST_EPOCHS}"
  "inference.skip_epoch=${SKIP_EPOCH}"
  "inference.gpu=${GPU}"
)

if [ -n "${CKPT_ROOT:-}" ]; then
  OPTIONS+=("inference.ckpt_root=${CKPT_ROOT}")
fi
if [ -n "${CKPT_NAME:-}" ]; then
  OPTIONS+=("inference.ckpt_name=${CKPT_NAME}")
fi
if [ -n "${TEST_EPOCH:-}" ]; then
  OPTIONS+=("inference.test_epoch=${TEST_EPOCH}")
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python -u inference_hybird.py \
  --zeroshot \
  --dataset="${DATASET}" \
  --cfg-path="${CFG}" \
  --options "${OPTIONS[@]}" "$@"
