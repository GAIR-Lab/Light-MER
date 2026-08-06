#!/usr/bin/env bash
#SBATCH --job-name=swdh_tch_q3_8b
#SBATCH --time=120:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=80G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CFG="${CFG:-train_configs/stage1_teacher_qwen3_8b.yaml}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"
mkdir -p output logs checkpoints

if [ -n "${CONDA_ENV_NAME}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

python -u train.py --cfg-path="${CFG}" "$@"

echo
echo "Teacher checkpoints are under output/stage1_teacher_qwen3_8b/."
echo "Copy or symlink the selected checkpoint to checkpoints/qwen3_8b_teacher.pth before student distillation."
