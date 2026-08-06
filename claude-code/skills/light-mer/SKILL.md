---
name: light-mer
description: Set up, train, infer, and evaluate Light-MER Stage 1 SWD-H and Stage 2 M-GRPO workflows. Use when deploying Light-MER checkpoints, checking local models or MER datasets, running either training stage, running Stage 2 inference, or evaluating MER-UniBench outputs.
---

# Light-MER

Operate the public Light-MER release end to end. Detect the requested stage and
workflow, run its bundled preflight first, ask for all missing resource paths in
one concise message, deploy resources with environment variables or symlinks,
and then run the repository script.

Never move original checkpoints or datasets. Never copy checkpoints, datasets,
logs, inference outputs, credentials, or absolute private paths into Git.

## Locate the repository

Identify the repository by train.py, train_stage2_mgrpo.py, and train_configs/.
Search in this order:

1. An explicit path from the user.
2. LIGHT_MER_ROOT.
3. The current directory and its parents.

Set LIGHT_MER_ROOT when the skill is installed outside the repository.

## Show the startup menu

If the user invokes $light-mer without a stage or workflow, show:

    Light-MER menu
    1. Stage 1 inference
    2. Stage 1 SWD-H training
    3. Stage 1 evaluation
    4. Stage 2 inference
    5. Stage 2 M-GRPO training
    6. Stage 2 checkpoint-sweep evaluation

    Reply with 1-6 and include any known model, dataset, checkpoint, or output paths.

If the user names a workflow but not a stage, ask only which stage they mean.
Otherwise continue directly without showing the menu.

## Run automatic preflight

Resolve SKILL_DIR as the directory containing this SKILL.md. Run:

    python SKILL_DIR/scripts/stage1_preflight.py MODE --repo-root "$LIGHT_MER_ROOT"

or:

    python SKILL_DIR/scripts/stage2_preflight.py MODE \
      --repo-root "$LIGHT_MER_ROOT" \
      --dataset "${DATASET:-inferenceData}" \
      --steps STEPS \
      --run-tag "${RUN_TAG:-stage2_mgrpo_qwen3_0_6b}"

Replace MODE with train, inference, or eval. For the standard Stage 2 checkpoint
sweep, set STEPS to 50,500,1000,1500,2000,2500,3000,3500,4000,4500,5000. Exit code 2 means resources are missing and
should trigger deployment, not a generic failure response.

Keep Stage 2 names stable and stage-level: use checkpoints/stage2-mgrpo-qwen3-0.6b for checkpoints
and stage2_mgrpo_qwen3_0_6b for the result run tag. Do not encode generation parameters
or implementation options in directory names, run tags, README headings, or skill
menu labels.

Add --require-cuda when preflight runs inside the GPU allocation that will
execute the workflow. Without it, an unavailable GPU is reported as a warning
so deployment can still be prepared on a login node.

Preflight redacts local absolute paths by default. Use --show-paths only when
the user explicitly needs the full local path for troubleshooting.

When paths are missing:

1. Ask for all missing paths together.
2. Prefer environment variables when resources remain outside the repository.
3. Otherwise create symlinks under models/, dataset/, or checkpoints/.
4. Never overwrite an existing path without permission.
5. Rerun preflight and continue only after it passes.

## Stage 1

### Train SWD-H

Require the Qwen3-8B teacher checkpoint, Qwen3-8B and Qwen3-0.6B model
directories, CLIP Large/Base, HuBERT Large/Base, and MER-Caption+.

    TEACHER_CKPT=checkpoints/qwen3_8b_teacher.pth \
    CONDA_ENV_NAME=swdh-stage1 \
    bash scripts/train_stage1_swdh.sh

Use the released teacher checkpoint when requested:

    https://huggingface.co/kevin233333/Light-MER/blob/main/light-mer-teacher-qwen3-8b.pth

### Infer

Require a Stage 1 checkpoint directory, Qwen3-0.6B, CLIP Base, HuBERT Base,
and the selected evaluation datasets.

    CKPT_ROOT=checkpoints/stage1-swdh-qwen3-0.6b \
    REPEAT=1 \
    BASE_ROOT=output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/repeat1/results \
    TEST_EPOCH=60 \
    bash scripts/inference_stage1_swdh.sh

Run separate jobs per repeat and epoch unless the user explicitly requests a
serial sweep.

### Evaluate

Require inference outputs, Qwen2.5-7B-Instruct, and Emotion Wheel resources.

    bash scripts/eval_stage1_swdh.sh \
      --base-root output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/repeat1/results

## Stage 2

The Stage 2 checkpoint is not hosted in the public model repository yet. If it
is missing, ask whether the user has a local Stage 2 checkpoint directory or
wants to train Stage 2 first. Do not invent a download URL or silently fall
back to Stage 1 weights.

### Train M-GRPO

Initialize every Stage 2 run from the Stage 1 SWD-H step-25 checkpoint. Do not
resume the long run from a Stage 2 checkpoint and do not substitute a teacher
checkpoint. The entry point validates matching LoRA and multimodal projection
weights.

Run preflight, then a one-step smoke test in a separate output directory before
formal training. The smoke test must exercise loading, generation, forward,
backward, optimizer, and checkpoint writing. Never use smoke artifacts for
formal inference.

The standard checkpoint sweep is produced by one 5000-step run. The config saves
step 50 explicitly with save_steps=[50], then saves every 500 steps with
save_interval=500, so all public Stage 2 checkpoints come from the same training
trajectory.

    STAGE1_CKPT=checkpoints/stage1-swdh-qwen3-0.6b/checkpoint_000025_loss_1.549.pth \
    OUTPUT_DIR=output/stage2_mgrpo_qwen3_0_6b \
    CONDA_ENV_NAME=swdh-stage1 \
    bash scripts/train_stage2_mgrpo.sh

Keep the output directory outside normal Git tracking. Build a separate
checkpoint directory with symlinks, not copied weights. It must use the public
Stage 2 checkpoint name checkpoints/stage2-mgrpo-qwen3-0.6b and contain exactly these periodic
files:

    checkpoint_000050_grpo.pth
    checkpoint_000500_grpo.pth
    checkpoint_001000_grpo.pth
    checkpoint_001500_grpo.pth
    checkpoint_002000_grpo.pth
    checkpoint_002500_grpo.pth
    checkpoint_003000_grpo.pth
    checkpoint_003500_grpo.pth
    checkpoint_004000_grpo.pth
    checkpoint_004500_grpo.pth
    checkpoint_005000_grpo.pth

Do not link checkpoint_final_*. The inference entry point scans every periodic
checkpoint between START_STEP and END_STEP, so extra files in the checkpoint
directory change the experiment.

### Run Stage 2 inference

Run the Stage 2 checkpoint sweep with explicit generation parameters. These
parameters are part of the command, not part of the public output name.

    STAGE2_CKPT_ROOT=checkpoints/stage2-mgrpo-qwen3-0.6b \
    START_STEP=50 \
    END_STEP=5000 \
    DATASET=inferenceData \
    STAGE2_RESULT_ROOT=output_stage2_mgrpo/results \
    RUN_TAG=stage2_mgrpo_qwen3_0_6b \
    TEMPERATURE=0.9 \
    TOP_P=0.9 \
    MERGE_LORA=1 \
    bash scripts/inference_stage2_mgrpo.sh

Run inference preflight with the exact eleven-step list before allocation. Keep
compact and quiet logging enabled unless the user explicitly asks to inspect
sample content. DATASET=inferenceData runs all nine MER-UniBench datasets.

The Python entry point can load all eleven checkpoints sequentially in one
process. If the user wants to parallelize the sweep, split the checkpoint range
into independent jobs according to their local hardware and scheduler policy,
while keeping the same checkpoint root, result root, run tag, and generation
parameters for every job.

Use job-specific START_STEP and END_STEP values only to partition the same
standard checkpoint list. Do not start evaluation until every inference job
exits successfully.

### Evaluate Stage 2

Require all requested raw inference NPZ files, Qwen2.5-7B-Instruct, and Emotion
Wheel resources.

    bash scripts/eval_stage2_mgrpo.sh \
      --base-root output_stage2_mgrpo/results \
      --run-tag stage2_mgrpo_qwen3_0_6b

Run eval preflight with the exact eleven-step list first. The wrapper creates
missing Qwen label-extraction caches, then summarizes Emotion Wheel metrics
across all nine datasets. The official sweep summary selects the best
checkpoint separately for each dataset; it is not the score of one shared
checkpoint. If the user asks for the best single checkpoint, use the same
official scorer with its test_epochs filter for each step, average the nine
primary metrics per step, and report both results without conflating them.

Only evaluate NPZ files generated locally by this repository or otherwise
trusted by the user. A complete standard sweep contains 99 raw NPZ files,
99 openset caches, and 44 sentiment caches: 242 NPZ files total. Treat missing
files, duplicate steps, or an evaluation dependency that ran early as a failed
run.

## Deployment rules

- Use LIGHT_MER_ROOT, SWDH_MODEL_ROOT, SWDH_DATASET_ROOT,
  SWDH_EMOTION_WHEEL_ROOT, TEACHER_CKPT, STAGE1_CKPT, STAGE2_CKPT_ROOT, and
  STAGE2_RESULT_ROOT instead of editing private paths into source files.
- Keep model weights on Hugging Face or local storage, never in normal Git.
- Keep result NPZ files, logs, scheduler output, local notes, and datasets untracked.
- Run the smallest preflight or smoke test after fixing a path or dependency.
- Before publishing changes, run OPEN_SOURCE_CHECKLIST.md, the repository
  privacy scan, and inspect git status so local research notes are not staged.
- Never commit or push Stage 2 release changes without explicit user approval.
