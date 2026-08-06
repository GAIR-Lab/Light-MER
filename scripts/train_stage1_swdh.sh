#!/usr/bin/env bash
#SBATCH --job-name=swdh_q3_06b
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="${CFG:-train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml}"
TEACHER_CKPT="${TEACHER_CKPT:-checkpoints/qwen3_8b_teacher.pth}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"
mkdir -p output logs checkpoints

if [ -n "${CONDA_ENV_NAME}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

if [ ! -f "${TEACHER_CKPT}" ]; then
  echo "Teacher checkpoint not found: ${TEACHER_CKPT}" >&2
  echo "Set TEACHER_CKPT=/path/to/checkpoint.pth or place it at checkpoints/qwen3_8b_teacher.pth." >&2
  exit 1
fi

python -u train.py \
  --cfg-path="${CFG}" \
  --options "model.teacher.ckpt=${TEACHER_CKPT}" "$@"
