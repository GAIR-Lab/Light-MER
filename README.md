# <img src="assets/lime-icon.png" alt="Light-MER lime icon" width="36"> Light-MER

Official implementation for the paper:

**Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters?**

<p align="center">🌟 If this project helps or inspires your research, please consider giving us a star! ⭐</p>

<p align="center">
  <a href="https://arxiv.org/abs/2607.12787"><img src="https://img.shields.io/badge/arXiv-Light--MER-b31b1b?style=flat&logo=arxiv&logoColor=white" alt="arXiv Light-MER"></a>
  <a href="https://pytorch.org/"><img src="https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c?style=flat&logo=pytorch&logoColor=white" alt="PyTorch 2.4+"></a>
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/Python-3.10-3776ab?style=flat&logo=python&logoColor=white" alt="Python 3.10"></a>
  <a href="https://huggingface.co/kevin233333/Light-MER"><img src="https://img.shields.io/badge/Hugging%20Face-Checkpoint-ffcc4d?style=flat" alt="Hugging Face checkpoint"></a>
  <a href="https://mp.weixin.qq.com/s/UPHYjmZgFyLggOysvpOOzA"><img src="https://img.shields.io/badge/Blog-%E6%96%B0%E6%99%BA%E5%85%83-07c160?style=flat" alt="Blog 新智元"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-2ea44f?style=flat" alt="Apache License 2.0"></a>
</p>

<p align="center">
  <img src="assets/light-mer-overview.png" alt="Light-MER Stage 1 overview: a large model becomes a small but strong student, and a large teacher teaches a small student through SWD-H." width="82%">
</p>

> Light-MER codebase for Stage 1 SWD-H distillation and Stage 2 M-GRPO refinement.

Light-MER revisits generative multimodal emotion recognition (MER) from an efficiency perspective. Instead of deploying a large 7B/8B multimodal emotion language model, Light-MER transfers the multimodal emotion reasoning ability of a strong teacher into a sub-1B deployment model.

This repository hosts the Light-MER open-source release, covering Stage 1 SWD-H distillation and Stage 2 M-GRPO refinement in one codebase.

## 📰 News

- **August 6, 2026**: Stage 2 M-GRPO checkpoints released.
- **August 6, 2026**: Stage 2 M-GRPO training, inference, evaluation, and agent-skill support released.
- **July 14, 2026**: Stage 1 SWD-H student checkpoints and Stage 1 agent skills released.
- **July 14, 2026**: README and public config aligned with the Light-MER paper; core model source is included in the release.
- **July 14, 2026**: Stage 1 SWD-H training, inference, and evaluation code released.
- **July 10, 2026**: 🎉 Our paper **Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters?** was accepted by the **ACM MM 2026 Main Track**.

**TODO list sorted by priority**

- [x] Release Stage 1 SWD-H training, inference, and evaluation code. (Completed on July 14, 2026)
- [x] Release Light-MER teacher and Stage 1 SWD-H student checkpoints. (Completed on July 14, 2026)
- [x] Release Codex and Claude Code Stage 1 deployment skills. (Completed on July 14, 2026)
- [x] Release Stage 2 M-GRPO refinement code and instructions. (Completed on August 6, 2026)
- [x] Release Stage 2 M-GRPO checkpoints. (Completed on August 6, 2026)

**If you encounter any questions or discover a bug within the paper or code, please do not hesitate to open an issue or submit a pull request.**

## 🚦 Release Status

| Component | Status | Notes |
|---|---|---|
| Stage 1 | Released | SWD-H distillation code for Qwen3-8B teacher to Qwen3-0.6B student |
| Stage 2 | Released | M-GRPO training, inference, and evaluation code/instructions |
| Evaluation | Released | Stage 1 and Stage 2 inference scripts, label extraction, and Emotion Wheel metrics |
| Model Checkpoint | Released | Teacher, Stage 1 SWD-H, and Stage 2 M-GRPO checkpoints released |
| AI Agent Skills | Released | Codex and Claude Code helpers with Stage 1/2 train, inference, and evaluation preflight checks |

## 🛠️ AI Agent Skills

Deployment helpers are included for both Codex and Claude Code. After installation, start the skill with one prompt; if no workflow is specified, it opens a Stage 1/2 menu and automatically checks checkpoints, pretrained models, datasets, configs, and inference outputs before running project scripts.

<details>
<summary><strong>Codex</strong></summary>

Install:

```bash
git clone https://github.com/GAIR-Lab/Light-MER.git
mkdir -p ~/.codex/skills
cp -r Light-MER/codex/skills/light-mer ~/.codex/skills/
```

After installation, open a new Codex session and invoke the installed Light-MER skill. Codex will open a six-option Stage 1/2 menu. The preflight checker runs inside the selected workflow; missing resources are requested together and deployed with symlinks or environment variables.

</details>

<details>
<summary><strong>Claude Code</strong></summary>

Install:

```bash
git clone https://github.com/GAIR-Lab/Light-MER.git
mkdir -p ~/.claude/skills
cp -r Light-MER/claude-code/skills/light-mer ~/.claude/skills/
```

After installation, invoke the installed Light-MER skill in Claude Code. Claude Code uses the same six-option Stage 1/2 menu and automatic resource checks.

</details>

Status: **Released for Stage 1 and Stage 2**.

## 🧠 Method Overview

Light-MER compresses a Qwen3-8B multimodal emotion teacher into a Qwen3-0.6B deployable student. Stage 1 uses SWD-H to align answer-token hidden-state geometry, while Stage 2 follows the M-GRPO refinement track for more concise and emotion-faithful generation.

<p align="center">
  <img src="assets/light-mer-model-overview.png" alt="Light-MER model overview showing SWD-H hidden-state distillation and M-GRPO refinement." width="96%">
</p>

## 🧩 Model Configuration

| Role | Language decoder | Visual encoder | Audio encoder |
|---|---|---|---|
| Teacher | Qwen3-8B | CLIP-ViT-Large-Patch14 | HuBERT-Large |
| Student | Qwen3-0.6B | CLIP-ViT-Base-Patch16 | HuBERT-Base |

The paper uses face-cropped visual inputs because facial regions carry salient affective cues. The current configs expose the same multimodal data path through `face_or_frame: "multiface_audio_face_text"`.

## ⚡ Efficiency Snapshot

Light-MER keeps the multimodal emotion reasoning pipeline compact: the Qwen3-0.6B student uses about **11x fewer FLOPs** and **2.54 GB peak memory**, while preserving the same MER generation interface.

| Model | Params | Peak Mem. | FLOPs | Direct | Descriptive |
|---|---:|---:|---:|---:|---:|
| Teacher | 9.00B | 20.04 GB | 10,902.6G | 0.901s | 6.138s |
| **SWD&#8209;H Student** | **854.93M** | **2.54 GB** | **988.8G (11.0x)** | **0.561s** | **4.621s** |
| **M&#8209;GRPO Student** | **854.93M** | **2.54 GB** | **988.8G (11.0x)** | **0.523s** | **3.105s** |

Direct and descriptive columns report latency per sample.

## ⚙️ Installation

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

Hardware requirements depend on the selected pretrained models, number of Stage 2 checkpoints, batch settings, and local software stack. Adjust training and inference parallelism according to your available GPUs and scheduler policy.

## 📦 Dataset

### 📝 MER-Caption+

Stage 1 distillation and Stage 2 M-GRPO refinement use MER-Caption+ from the MER2025 release:

- Download: [MERChallenge/MER2025](https://huggingface.co/datasets/MERChallenge/MER2025)

Expected layout:

```text
dataset/
└── mer2025-dataset/
    ├── video/
    ├── audio/
    ├── openface_face/
    ├── subtitle_chieng.csv
    ├── track2_train_mercaptionplus.csv
    └── track3_train_mercaptionplus.csv
```

### 🧪 MER-UniBench

Evaluation follows the MER-UniBench setting, covering basic emotion recognition, sentiment analysis, and open-vocabulary MER.

- MER2023/MER2024/SIMS/SIMS v2/CMU-MOSI/CMU-MOSEI/IEMOCAP/MELD: [Baidu Netdisk](https://pan.baidu.com/s/1kbfs5pG_hAri0QwvQl-Ecg?pwd=b9vn) / [TeraBox](https://1024terabox.com/s/1AE7uAU3Ib8aRBSyF1TMpow)
- OV-MERD+: [Baidu Netdisk](https://pan.baidu.com/s/1nBTw_ujSTQPAMyIs5Qv8Zw?pwd=k8tj) / [TeraBox](https://1024terabox.com/s/1O130fc81FVsGGsrjLuHyDA)

Expected layout:

```text
dataset/
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

## 🤖 Model Zoo

### 🧱 General Checkpoints

Place or symlink pretrained models under `models/`, or set `SWDH_MODEL_ROOT`.

| Model | Type | Used for | Link |
|---|---|---|---|
| Qwen3-8B | LLM | Teacher decoder | [Hugging Face](https://huggingface.co/Qwen/Qwen3-8B) |
| Qwen3-0.6B | LLM | Student decoder | [Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) |
| Qwen2.5-7B-Instruct | LLM | Evaluation label extraction | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| CLIP-ViT-Large-Patch14 | Visual Encoder | Teacher visual encoder | [Hugging Face](https://huggingface.co/openai/clip-vit-large-patch14) |
| CLIP-ViT-Base-Patch16 | Visual Encoder | Student visual encoder | [Hugging Face](https://huggingface.co/openai/clip-vit-base-patch16) |
| Chinese HuBERT-Large | Audio Encoder | Teacher audio encoder | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-hubert-large) |
| Chinese HuBERT-Base | Audio Encoder | Student audio encoder | [Hugging Face](https://huggingface.co/TencentGameMate/chinese-hubert-base) |

Expected layout:

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

### 🏁 Light-MER Checkpoints

| Model Name | Description | Link |
|---|---|---|
| Light-MER Teacher | Qwen3-8B teacher checkpoint | [Hugging Face](https://huggingface.co/kevin233333/Light-MER/blob/main/light-mer-teacher-qwen3-8b.pth) |
| Light-MER Stage 1 SWD-H | Qwen3-0.6B student after SWD-H distillation | [Hugging Face](https://huggingface.co/kevin233333/Light-MER/tree/main/stage1-swdh-qwen3-0.6b) |
| Light-MER Stage 2 M-GRPO | Qwen3-0.6B student checkpoints after M-GRPO refinement | [Hugging Face](https://huggingface.co/kevin233333/Light-MER/tree/main/stage2-mgrpo-qwen3-0.6b) |

## 🚀 Getting Started

### 🎓 Stage 1: SWD-H Distillation

#### 1. Train the Qwen3-8B Teacher

Skip this step if you already have a compatible teacher checkpoint.

```bash
CONDA_ENV_NAME=swdh-stage1 bash scripts/train_teacher_qwen3_8b.sh
```

After selecting the teacher checkpoint, copy or symlink it to:

```text
checkpoints/qwen3_8b_teacher.pth
```

#### 2. Train the Qwen3-0.6B SWD-H Student

```bash
CONDA_ENV_NAME=swdh-stage1 \
TEACHER_CKPT=checkpoints/qwen3_8b_teacher.pth \
bash scripts/train_stage1_swdh.sh
```

#### 3. Run Stage 1 Inference

```bash
CKPT_ROOT=output/stage1_swdh_qwen3_8b_to_qwen3_0_6b/<run_dir> \
REPEAT=1 \
BASE_ROOT=output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/repeat1/results \
TEST_EPOCH=60 \
bash scripts/inference_stage1_swdh.sh
```

This runs one task for one model/config, one repeat, and one epoch. To run a local serial epoch loop, set `TEST_EPOCHS=5-60` and `SKIP_EPOCH=5` explicitly.

#### 4. Evaluate Stage 1

```bash
bash scripts/eval_stage1_swdh.sh \
  --base-root output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/repeat1/results
```

### 🧭 Stage 2: M-GRPO Refinement

#### 1. Train the Stage 2 M-GRPO Student

Train Stage 2 from the Stage 1 SWD-H checkpoint:

```bash
STAGE1_CKPT=checkpoints/stage1-swdh-qwen3-0.6b/checkpoint_000025_loss_1.549.pth \
OUTPUT_DIR=output/stage2_mgrpo_qwen3_0_6b \
CONDA_ENV_NAME=swdh-stage1 \
bash scripts/train_stage2_mgrpo.sh
```

To skip training, download or symlink the [released checkpoints](https://huggingface.co/kevin233333/Light-MER/tree/main/stage2-mgrpo-qwen3-0.6b) into:

```text
checkpoints/stage2-mgrpo-qwen3-0.6b/
```

#### 2. Run Stage 2 Inference

```bash
STAGE2_CKPT_ROOT=checkpoints/stage2-mgrpo-qwen3-0.6b \
START_STEP=50 \
END_STEP=5000 \
STAGE2_RESULT_ROOT=output_stage2_mgrpo/results \
RUN_TAG=stage2_mgrpo_qwen3_0_6b \
TEMPERATURE=0.9 \
TOP_P=0.9 \
MERGE_LORA=1 \
bash scripts/inference_stage2_mgrpo.sh
```

You can also try `TEMPERATURE=0.3` for lower-temperature inference.

#### 3. Evaluate Stage 2

```bash
bash scripts/eval_stage2_mgrpo.sh \
  --base-root output_stage2_mgrpo/results \
  --run-tag stage2_mgrpo_qwen3_0_6b
```

The Stage 2 wrapper follows the original evaluation flow: it first creates any
missing Qwen label-extraction caches, then summarizes the Emotion Wheel metrics
for all nine MER-UniBench datasets.

> Security: evaluate only NPZ files generated locally by this repository or
> obtained from a trusted source. The original result format contains Python
> objects and therefore requires NumPy pickle loading.

### 🏁 Expected Results

<p align="center">
  <img src="assets/light-mer-main-results.png" alt="Light-MER results on MER-UniBench benchmarks" width="100%">
</p>

Inference and Qwen label extraction can be stochastic. Small differences
of about ±0.3% are normal when rerunning the pipeline.

## 🛠️ Path Overrides

You can override default roots without editing source files:

```bash
export SWDH_MODEL_ROOT=/path/to/models
export SWDH_DATASET_ROOT=/path/to/dataset
export SWDH_EMOTION_WHEEL_ROOT=/path/to/emotion_wheel
export SWDH_RESULT_ROOT=/path/to/results
export LIGHT_MER_ROOT=/path/to/Light-MER
export STAGE1_CKPT=/path/to/checkpoint_000025_loss_1.549.pth
export STAGE2_CKPT_ROOT=/path/to/stage2-mgrpo-qwen3-0.6b
export STAGE2_RESULT_ROOT=/path/to/stage2/results
export LIGHT_MER_EVAL_SEED=42
```

You can override YAML values directly:

```bash
python -u train.py \
  --cfg-path train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml \
  --options model.teacher.ckpt=/path/to/qwen3_8b_teacher.pth
```

## 📚 Citation

If you find Light-MER useful, please cite our arXiv preprint: [arXiv:2607.12787](https://arxiv.org/abs/2607.12787).

```bibtex
@misc{zheng2026lightmer,
  title = {Do We Really Need Multimodal Emotion Language Models Larger Than 1B Parameters?},
  author = {Zheng, Kaiwen and Fu, Junchen and Deng, Wenhao and Han, Hu and Jose, Joemon M. and Ge, Xuri},
  year = {2026},
  eprint = {2607.12787},
  archivePrefix = {arXiv},
  primaryClass = {cs.AI},
  url = {https://arxiv.org/abs/2607.12787},
  note = {Accepted by ACM MM 2026}
}
```

## 📄 License

This project is released under the [Apache License 2.0](LICENSE). Please also follow the licenses and usage terms of the external datasets, pretrained models, and checkpoints used with this codebase.

## 🙏 Acknowledgement

- Built on AffectGPT-style multimodal instruction tuning: [AffectGPT](https://github.com/zeroQiaoba/AffectGPT/tree/master/AffectGPT).
- Developed with [PyTorch](https://pytorch.org/) and [Hugging Face Transformers](https://github.com/huggingface/transformers).
- Uses [vLLM](https://github.com/vllm-project/vllm) for evaluation-time label extraction and efficient LLM inference.
- Uses [CLIP](https://github.com/openai/CLIP) visual encoders and HuBERT audio encoders for multimodal feature extraction.
- Reuses ideas and open-source components from [BLIP/LAVIS](https://github.com/salesforce/LAVIS) and [ImageBind](https://github.com/facebookresearch/ImageBind).
