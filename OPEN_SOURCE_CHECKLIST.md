# Light-MER Stage 2 Open-Source Readiness Checklist

本文件用于判断 Stage 2 是否已经可以公开到 GitHub。所有路径均相对于仓库根目录。

当前结论：**Stage 2 code/instructions 和 Stage 2 M-GRPO checkpoints 均已准备公开**。截至 2026-08-06，Stage 2 源码、公开命名、Skill、preflight、README 和基础安全检查已对齐；Stage 2 M-GRPO checkpoints 已通过 Hugging Face 单独发布，GitHub 仍不包含权重文件。

## 1. 当前判断

- [x] Stage 2 核心源码已经在候选目录中：`train_stage2_mgrpo.py`、`mgrpo_reward.py`、`inference_stage2_mgrpo.py`、Stage 2 scripts、preflight 和 reward unit test。
- [x] 候选仓库没有发现已跟踪的大模型权重、checkpoint、数据集、NPZ 结果、日志、`.env`、私钥或 MER_NOW 本地记录。
- [x] 现有 Stage 2 preflight 默认会隐藏本机绝对路径；需要排错时才使用 `--show-paths`。
- [x] 基础静态检查本次已通过：bash syntax、Python compileall、`tests.test_mgrpo_reward`、Codex skill validation 和 Claude skill validation。
- [x] Stage 2 公开命名已和 Stage 1 风格对齐：checkpoint 目录使用 `checkpoints/stage2-mgrpo-qwen3-0.6b`，结果 run tag 使用 `stage2_mgrpo_qwen3_0_6b`。
- [x] 公开文档和 skill 不再把 temperature、top-p 或 LoRA implementation option 编进 Stage 2 名字里。
- [x] README 的 News、TODO、Release Status 和 Model Zoo 已更新为 Stage 2 code/instructions + Stage 2 M-GRPO checkpoints released。
- [x] 完整复现已由项目维护者确认完成：Stage 2 M-GRPO checkpoints 训练、推理和 evaluation 已跑通。

## 2. Stage 2 标准公开口径

Stage 2 公开时只使用稳定 stage-level 命名。实验参数写在命令或配置中，不写进目录名、run tag、section 标题或 skill 菜单。

- [x] 从同一个 Stage 1 SWD-H step-25 checkpoint 初始化 Stage 2。
- [x] 先做 one-step smoke test，覆盖数据加载、generation、forward、backward、optimizer 和 checkpoint 写入。
- [x] 正式训练为单次 5000-step run，并在同一条训练轨迹上保存 step 50 与每 500 step checkpoint。
- [x] 长程训练每 500 step 保存一次。
- [x] checkpoint 目录恰好包含 11 个 periodic checkpoints：`50,500,1000,1500,2000,2500,3000,3500,4000,4500,5000`。
- [x] checkpoint 目录不包含 `checkpoint_final_*` 或任何额外 checkpoint。
- [x] inference 显式设置 `TEMPERATURE=0.9`、`TOP_P=0.9`、`MERGE_LORA=1`、`RUN_TAG=stage2_mgrpo_qwen3_0_6b`。
- [x] reproduction 的并行方式由使用者按本地硬件和调度系统决定；公开文档不写固定 GPU 数量或服务器限定。
- [x] evaluation 覆盖 9 datasets x 11 checkpoints，共 99 个 raw NPZ。
- [x] evaluation 生成 99 openset caches 和 44 sentiment caches，总计 242 NPZ。
- [x] 结果报告必须区分 official per-dataset best-checkpoint mean 和 best single-checkpoint mean，不能混写。

## 3. 必须同步审阅的公开文件

这些文件需要在同一个 Stage 2 release commit/PR 中一起审阅，不能只改一部分。

- [x] `README.md`
- [x] `OPEN_SOURCE_CHECKLIST.md`
- [x] `codex/skills/light-mer/SKILL.md`
- [x] `codex/skills/light-mer/agents/openai.yaml`
- [x] `codex/skills/light-mer/scripts/stage2_preflight.py`
- [x] `claude-code/skills/light-mer/SKILL.md`
- [x] `claude-code/skills/light-mer/scripts/stage2_preflight.py`
- [x] `scripts/train_stage2_mgrpo.sh`
- [x] `scripts/inference_stage2_mgrpo.sh`
- [x] `scripts/eval_stage2_mgrpo.sh`
- [x] `scripts/eval_stage2_mgrpo.py`
- [x] `inference_stage2_mgrpo.py`
- [x] `train_configs/stage2_mgrpo_qwen3_0_6b.yaml`
- [x] `.env.example`
- [x] `.gitignore`
- [x] `LICENSE`

发布说明：README News/TODO/Release Status/Model Zoo 可以写 Stage 2 code/instructions 和 Stage 2 M-GRPO checkpoints released；checkpoint 权重只放 Hugging Face，不进 GitHub。

## 4. Stage 2 应包含的源码

### 4.1 根目录入口与配置

- [x] `train_stage2_mgrpo.py` — Stage 2 M-GRPO 训练入口。
- [x] `mgrpo_reward.py` — M-GRPO reward 与格式奖励实现。
- [x] `inference_stage2_mgrpo.py` — Stage 2 M-GRPO checkpoint 推理入口。
- [x] `eval_student_model.py` — MER-UniBench/Emotion Wheel 评分依赖。
- [x] `config.py` — 数据集、模型及评测路径的公开配置接口。
- [x] `train_configs/stage2_mgrpo_qwen3_0_6b.yaml` — Stage 2 基础训练配置。
- [x] `requirements.txt`
- [x] `environment.yml`

Stage 2 配置默认使用单次 5000-step run，并用 `save_steps=[50]` 额外保存 step 50：

```text
single run:  train_configs/stage2_mgrpo_qwen3_0_6b.yaml defaults to max_steps=5000, save_interval=500, save_steps=[50]
inference:   TEMPERATURE=0.9, TOP_P=0.9, MERGE_LORA=1
```

### 4.2 Stage 2 wrapper、preflight 与 tests

- [x] `scripts/train_stage2_mgrpo.sh`
- [x] `scripts/inference_stage2_mgrpo.sh`
- [x] `scripts/eval_stage2_mgrpo.sh`
- [x] `scripts/eval_stage2_mgrpo.py`
- [x] `scripts/stage2_preflight.py`
- [x] `codex/skills/light-mer/scripts/stage2_preflight.py`
- [x] `claude-code/skills/light-mer/scripts/stage2_preflight.py`
- [x] `tests/test_mgrpo_reward.py`

### 4.3 共享运行时代码

以下目录是训练、generation、数据加载和 evaluation 的运行依赖，应随 Stage 2 一起保留已跟踪源码：

- [x] `my_affectgpt/common/`
- [x] `my_affectgpt/conversation/`
- [x] `my_affectgpt/datasets/`
- [x] `my_affectgpt/evaluation/`
- [x] `my_affectgpt/models/`
- [x] `my_affectgpt/processors/`
- [x] `my_affectgpt/runners/`
- [x] `my_affectgpt/tasks/`
- [x] `my_affectgpt/__init__.py`
- [x] `toolkit/`

### 4.4 Stage 1 依赖不能误删

完整仓库如果宣称支持 Stage 1 + Stage 2，下列 Stage 1 文件仍应保留：

- [x] `train.py`
- [x] `inference_hybird.py`
- [x] `train_configs/stage1_teacher_qwen3_8b.yaml`
- [x] `train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml`
- [x] `scripts/train_teacher_qwen3_8b.sh`
- [x] `scripts/train_stage1_swdh.sh`
- [x] `scripts/inference_stage1_swdh.sh`
- [x] `scripts/eval_stage1_swdh.sh`
- [x] `scripts/eval_stage1_swdh.py`

README 使用的公开图片也应保留：

- [x] `assets/light-mer-overview.png`
- [x] `assets/light-mer-model-overview.png`
- [x] `assets/light-mer-main-results.png`

## 5. 不应上传到 GitHub 的内容

### 模型、数据和结果

- [x] Stage 1 teacher checkpoint。
- [x] Stage 1 SWD-H checkpoint。
- [x] Stage 2 checkpoints，包括 step 50、500、1000、...、5000。
- [x] optimizer state、LoRA 权重、multimodal adapter/projection 权重。
- [x] Qwen、CLIP、HuBERT、Qwen2.5 evaluation LLM 权重。
- [x] MER-Caption+、MER-UniBench 视频、音频、face crop、subtitle、CSV 标注和中间处理文件。
- [x] inference raw NPZ、openset cache、sentiment cache。
- [x] prompts、生成文本、样本 ID、原始逐样本标签。
- [x] TensorBoard/W&B 文件、stdout/stderr、scheduler 日志、debug log、job ID 和节点信息。

公开 checkpoint 已放 Hugging Face 或独立模型存储；README/model card 应写清楚 license、上游模型许可、文件大小、SHA-256、对应 Git commit、step 和配置版本。

### 隐私和安全

- [x] `.env`、API key、token、password、cookie、credential、private key。
- [x] 绝对主目录、临时盘或挂载盘路径，以及用户名、邮箱、集群账号、partition、node 名称。
- [x] 私有调度脚本、内部调度命令和旧实验脚本。
- [x] `models/`、`dataset/`、`checkpoints/` 中的实际资源。
- [x] `output/`、`output_*/`、`logs/`、`wandb/`、`runs/`。
- [x] `*.pth`、`*.pt`、`*.bin`、`*.safetensors`。
- [x] `*.npz`、`*.npy`、`*.zip`、`*.tar`、`*.tar.gz`。
- [x] `*.out`、`*.err`、`*.log`。
- [x] `__pycache__/`、`*.pyc`、`.venv/`、`.ipynb_checkpoints/`。
- [x] `MER_NOW_*.md`、旧实验日志、debug log 和个人研究笔记。
- [x] 指向 clone 外资源的 symlink；即使目标文件未复制，symlink 也可能泄露本机路径。

## 6. 上传前检查命令

先查看准备上传的精确文件，不要直接执行未经审阅的 `git add .`：

```bash
git status --short
git diff --check
git diff --cached --name-only
git ls-files
```

检查大文件、敏感路径和常见 secret 形态：

```bash
find . -type f -size +25M -not -path './.git/*' -print
rg -n '(/home|/users|/scratch)/[^/]+|BEGIN .*[P]RIVATE KEY|g[h]p_|github_[p]at_|h[f]_' --glob '!.git/**'
```

检查脚本和 Python 源码：

```bash
bash -n scripts/*.sh
python -m compileall -q train.py train_stage2_mgrpo.py inference_hybird.py inference_stage2_mgrpo.py mgrpo_reward.py eval_student_model.py scripts my_affectgpt toolkit
python -m unittest tests.test_mgrpo_reward
```

检查 Skill：

```bash
python /path/to/skill-creator/scripts/quick_validate.py codex/skills/light-mer
```

## 7. 发布闸门

- [x] Stage 2 code/instructions 的阻塞项已完成，可以 commit 和 push。
- [x] 推送前必须人工确认 `git diff`、`git status --short` 和 staged file list。
- [x] Codex Skill 和 Claude Code Skill 的 Stage 2 标准流程必须一致。
- [x] 本次不再保持 Stage1-only；README 和 News 已按 Stage 2 code/instructions + Stage 2 M-GRPO checkpoints released 更新。
