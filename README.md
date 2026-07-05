# Official Implementation of AffectGPT SWD-H

> Stage 1 code release for Qwen3-8B to Qwen3-0.6B multimodal emotion distillation.

This repository is the public codebase for the AffectGPT SWD-H project. The current release contains the Stage 1 SWD-H distillation pipeline. Stage 2 will be released in a future update.

## News

- [2026-07-05] Stage 1 SWD-H training, inference, and evaluation code is released.
- [Coming Soon] Stage 2 code and instructions will be released here.

## Release Status

| Component | Status | Notes |
|---|---|---|
| Stage 1 teacher training | Released | Qwen3-8B AffectGPT teacher config and script |
| Stage 1 SWD-H distillation | Released | Qwen3-8B teacher to Qwen3-0.6B student |
| Stage 1 inference | Released | Multi-dataset zero-shot inference script |
| Stage 1 evaluation | Released | vLLM label extraction and Emotion Wheel metrics |
| Stage 2 | Coming Soon | Future update in this same repository |
| Model weights | Not included | Prepare locally or download from official model providers |
| Datasets | Not included | Prepare processed MER/MELD/IEMOCAP/CMU datasets locally |

## Overview

Stage 1 distills a Qwen3-8B AffectGPT teacher into a Qwen3-0.6B student with hidden-state Sliced Wasserstein Distance (SWD-H).

- Teacher: Qwen3-8B AffectGPT
- Student: Qwen3-0.6B AffectGPT
- Visual encoders: CLIP ViT-Large for teacher, CLIP ViT-Base for student
- Audio encoders: HuBERT-Large for teacher, HuBERT-Base for student
- Distillation: SWD-H on answer-token hidden states

## Directory Layout

```text
.
├── train.py
├── inference_hybird.py
├── eval_student_model.py
├── config.py
├── train_configs/
│   ├── stage1_teacher_qwen3_8b.yaml
│   └── stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml
├── scripts/
│   ├── train_teacher_qwen3_8b.sh
│   ├── train_stage1_swdh.sh
│   ├── inference_stage1_swdh.sh
│   ├── eval_stage1_swdh.py
│   └── eval_stage1_swdh.sh
├── my_affectgpt/
├── toolkit/
├── emotion_wheel/
├── requirements.txt
└── environment.yml
```

## Installation

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate swdh-stage1
```

or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The original experiments used one H100 80GB GPU. Smaller GPUs may require reducing batch size, sequence length, or enabling additional memory optimizations.

## Prepare Models

Place or symlink pretrained models under `models/`, or set `SWDH_MODEL_ROOT`.

Expected default layout:

```text
models/
├── Qwen3-8B/
├── Qwen3-0.6B/
├── Qwen2.5-7B-Instruct/
├── clip-vit-large-patch14/
├── clip-vit-base-patch16/
├── chinese-hubert-large/
└── chinese-hubert-base/
```

`Qwen2.5-7B-Instruct` is used by `eval_student_model.py` with vLLM to extract normalized emotion labels from generated text.

## Prepare Data

Place processed datasets under `dataset/`, or set `SWDH_DATASET_ROOT`.

Expected layout:

```text
dataset/
├── mer2025-dataset/
├── mer2023-dataset-process/
├── mer2024-dataset-process/
├── meld-process/
├── iemocap-process/
├── cmumosi-process/
├── cmumosei-process/
├── sims-process/
├── simsv2-process/
└── ovmerdplus-process/
```

Stage 1 training uses MERCaptionPlus files in `mer2025-dataset`, especially `track2_train_mercaptionplus.csv`, `track3_train_mercaptionplus.csv`, `subtitle_chieng.csv`, plus audio/video/OpenFace features.

## Getting Started

### 1. Train the Qwen3-8B Teacher

Skip this step if you already have a compatible teacher checkpoint.

```bash
CONDA_ENV_NAME=swdh-stage1 bash scripts/train_teacher_qwen3_8b.sh
```

After selecting the teacher checkpoint, copy or symlink it to:

```text
checkpoints/qwen3_8b_teacher.pth
```

### 2. Train the Qwen3-0.6B SWD-H Student

```bash
CONDA_ENV_NAME=swdh-stage1 \
TEACHER_CKPT=checkpoints/qwen3_8b_teacher.pth \
bash scripts/train_stage1_swdh.sh
```

Canonical config:

```text
train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml
```

Main SWD-H settings:

```yaml
model:
  teacher:
    use_swd: True
    swd_n_projections: 200
    swd_p: 1
    ot_weight: 0.5
    ot_ramp_steps: 10000
    kl_weight: 0.0
```

### 3. Run Inference

```bash
CKPT_ROOT=output/stage1_swdh_qwen3_8b_to_qwen3_0_6b/<run_dir> \
TEST_EPOCHS=5-60 \
SKIP_EPOCH=5 \
bash scripts/inference_stage1_swdh.sh
```

Use `TEST_EPOCH=60` to run one checkpoint.

### 4. Evaluate

```bash
bash scripts/eval_stage1_swdh.sh \
  --base-root output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/results
```

## Path Overrides

You can override default roots without editing source files:

```bash
export SWDH_MODEL_ROOT=/path/to/models
export SWDH_DATASET_ROOT=/path/to/dataset
export SWDH_EMOTION_WHEEL_ROOT=/path/to/emotion_wheel
export SWDH_RESULT_ROOT=/path/to/results
```

You can override YAML values directly:

```bash
python -u train.py \
  --cfg-path train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml \
  --options model.teacher.ckpt=/path/to/qwen3_8b_teacher.pth
```

## Citation

Citation information will be updated once the associated paper or technical report is available.

## Acknowledgement

This codebase builds on AffectGPT-style multimodal instruction tuning and open-source components from the PyTorch, Hugging Face Transformers, vLLM, CLIP, HuBERT, and BLIP/LAVIS ecosystems.
