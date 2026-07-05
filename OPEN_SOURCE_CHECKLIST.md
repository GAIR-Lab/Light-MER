# Open-Source Checklist

- [ ] Current code/config/scripts contain only Stage 1 SWD-H for Qwen3-8B teacher to Qwen3-0.6B student.
- [ ] README may say "Stage 2 Coming Soon", but no Stage 2 code/config/scripts are included yet.
- [ ] No model weights or checkpoints are included.
- [ ] No datasets or processed dataset artifacts are included.
- [ ] No `.npz`, `.npy`, `.pth`, `.pt`, `.bin`, `.safetensors`, logs, outputs, plots, or result summaries are included.
- [ ] No absolute private home, scratch, mount, or cluster-specific conda paths remain.
- [ ] No API keys, Hugging Face tokens, OpenAI keys, W&B keys, credentials, usernames, or cluster account details remain.
- [ ] README documents how to provide external model, dataset, teacher checkpoint, inference, and eval paths.
- [ ] License file is present and compatible with upstream AffectGPT code.
- [ ] `requirements.txt` or `environment.yml` installs in a clean environment.
- [ ] `python -m py_compile` passes for included Python files.
- [ ] A final grep for credentials and private absolute paths returns no actionable hits.
