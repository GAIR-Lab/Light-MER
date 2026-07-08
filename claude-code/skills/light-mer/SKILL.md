---
name: light-mer
description: Deploy, train, infer, and evaluate Light-MER Stage 1 SWD-H workflows. Use when setting up Light-MER local paths, checking required checkpoints/models/datasets, preparing Hugging Face checkpoints, running Stage 1 training, running Stage 1 inference, or evaluating Stage 1 MER-UniBench outputs.
---

# Light-MER Stage 1

Use this skill to operate the public Light-MER Stage 1 SWD-H release. Keep the workflow practical: if the user only invokes the skill, show the Stage 1 menu; otherwise identify whether the user wants **train**, **inference**, or **evaluation**, run the automatic preflight check yourself, deploy missing resources through symlinks or environment variables, and only then run the project scripts.

Do not use this skill for Stage 2 M-GRPO implementation, GRPO/RLHF reward code, policy optimization, or private experiment cleanup.

## Repository Root

Work from the Light-MER GitHub repo root, identified by these files:

```text
train.py
inference_hybird.py
train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml
scripts/train_stage1_swdh.sh
scripts/inference_stage1_swdh.sh
```

If the current directory is not the repo root, locate it before continuing. Never move the user's original checkpoints or datasets. Prefer symlinks into `checkpoints/`, `models/`, or `dataset/`.

## Startup Menu

If the user invokes the Light-MER skill without choosing a workflow, show this menu and wait for a selection:

```text
Light-MER Stage 1 menu
1. Inference: set up or run the released SWD-H student checkpoint.
2. Train: train Stage 1 SWD-H from a teacher checkpoint.
3. Evaluation: evaluate existing Stage 1 inference outputs.

Reply with 1, 2, or 3. You can include any known local checkpoint, model, dataset, or output paths.
```

Do not show the menu when the user already asks for train, inference, or evaluation. Continue directly to the matching workflow and automatic preflight.

## Decision Flow

1. If the user asks to **train**, use the Train workflow.
2. If the user asks to **infer**, **run inference**, **test checkpoints**, or **evaluate a released Stage 1 checkpoint**, use the Inference workflow.
3. If the user asks to **evaluate**, first verify inference outputs exist; if not, run or request inference first.
4. If the user is unclear or only invokes the skill, show the Startup Menu.

## Automatic Preflight

Treat preflight as an internal first step, not as a manual command for the user. Before train, inference, or eval, run the bundled preflight script for the selected mode:

```bash
python claude-code/skills/light-mer/scripts/stage1_preflight.py <train|inference|eval>
```

Use `--json` when you need machine-readable output. The script exits with code `2` when required resources are missing; this is expected and should trigger the resource deployment flow.

When preflight reports missing resources, summarize the missing checkpoint/model/dataset/output paths and ask the user for those paths in one concise batch. If the user provides paths, deploy them with symlinks by default:

```bash
mkdir -p checkpoints models dataset
ln -s /path/to/source checkpoints/name-or-directory
```

Use copying only when the user explicitly asks for a physical copy. Do not overwrite existing files or links without asking.

## Train Workflow

Training Stage 1 requires:

```text
train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml
checkpoints/qwen3_8b_teacher.pth or TEACHER_CKPT=/path/to/teacher.pth
models/Qwen3-8B/
models/Qwen3-0.6B/
models/clip-vit-large-patch14/
models/clip-vit-base-patch16/
models/chinese-hubert-large/
models/chinese-hubert-base/
dataset/mer2025-dataset/
```

The MER2025 training dataset should include `video/`, `audio/`, `openface_face/`, `subtitle_chieng.csv`, `track2_train_mercaptionplus.csv`, and `track3_train_mercaptionplus.csv`.

If the teacher checkpoint is missing, ask whether to use a local checkpoint path or the released Hugging Face teacher checkpoint:

```text
https://huggingface.co/kevin233333/Light-MER/blob/main/light-mer-teacher-qwen3-8b.pth
```

After preflight passes, run:

```bash
CONDA_ENV_NAME=swdh-stage1 \
TEACHER_CKPT=checkpoints/qwen3_8b_teacher.pth \
bash scripts/train_stage1_swdh.sh
```

For Slurm clusters, use:

```bash
CONDA_ENV_NAME=swdh-stage1 \
TEACHER_CKPT=checkpoints/qwen3_8b_teacher.pth \
sbatch scripts/train_stage1_swdh.sh
```

Training outputs are expected under:

```text
output/stage1_swdh_qwen3_8b_to_qwen3_0_6b/
```

## Inference Workflow

Inference requires a Stage 1 student checkpoint directory, not a teacher checkpoint. Prefer the released Hugging Face Stage 1 directory:

```text
https://huggingface.co/kevin233333/Light-MER/tree/main/stage1-swdh-qwen3-0.6b
```

Recommended local layout:

```text
checkpoints/stage1-swdh-qwen3-0.6b/
├── checkpoint_000005_loss_2.097.pth
├── ...
└── checkpoint_000060_loss_1.291.pth
```

Inference also needs:

```text
models/Qwen3-0.6B/
models/clip-vit-base-patch16/
models/chinese-hubert-base/
dataset/<evaluation dataset folders>
```

After preflight passes, run one recommended checkpoint:

```bash
CKPT_ROOT=checkpoints/stage1-swdh-qwen3-0.6b \
TEST_EPOCH=60 \
CONDA_ENV_NAME=swdh-stage1 \
bash scripts/inference_stage1_swdh.sh
```

To run the full released checkpoint trajectory:

```bash
CKPT_ROOT=checkpoints/stage1-swdh-qwen3-0.6b \
TEST_EPOCHS=5-60 \
SKIP_EPOCH=5 \
CONDA_ENV_NAME=swdh-stage1 \
bash scripts/inference_stage1_swdh.sh
```

For a subset of datasets:

```bash
DATASET=MER2023,MELD \
CKPT_ROOT=checkpoints/stage1-swdh-qwen3-0.6b \
TEST_EPOCH=60 \
bash scripts/inference_stage1_swdh.sh
```

Inference outputs are expected under:

```text
output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/results-<dataset>/
```

## Evaluation Workflow

Evaluation requires inference outputs and the Qwen2.5 label extractor:

```text
models/Qwen2.5-7B-Instruct/
emotion_wheel/
output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/results-<dataset>/
```

Run:

```bash
CONDA_ENV_NAME=swdh-stage1 \
bash scripts/eval_stage1_swdh.sh \
  --base-root output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/results
```

Use `--debug` only for smoke checks that should avoid expensive label extraction where possible.

## Deployment Rules

- Use environment variables when the user wants to keep resources outside the repo: `SWDH_MODEL_ROOT`, `SWDH_DATASET_ROOT`, `SWDH_EMOTION_WHEEL_ROOT`, `SWDH_RESULT_ROOT`, `TEACHER_CKPT`, `CKPT_ROOT`, `CONDA_ENV_NAME`.
- Use symlinks when the user wants standard repo-relative paths.
- Do not commit large checkpoints, datasets, logs, or private absolute paths to GitHub.
- Do not edit `config.py` for local paths unless the user explicitly asks; prefer env vars.
- If a script fails, inspect the first missing path or import error, fix the environment/resource issue, and rerun the smallest command that validates the fix.
