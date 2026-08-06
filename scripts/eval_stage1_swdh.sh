#!/usr/bin/env bash
#SBATCH --job-name=swdh_eval
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --mem=60G

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-}"

cd "${PROJECT_ROOT}"

if [ -n "${CONDA_ENV_NAME}" ]; then
  eval "$(conda shell.bash hook)"
  conda activate "${CONDA_ENV_NAME}"
fi

python -u scripts/eval_stage1_swdh.py "$@"
