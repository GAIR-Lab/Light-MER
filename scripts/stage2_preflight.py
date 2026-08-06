#!/usr/bin/env python3
"""Preflight checks for Light-MER Stage 2 train, inference, and evaluation."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path


MODEL_DIRS = ["Qwen3-0.6B", "clip-vit-base-patch16", "chinese-hubert-base"]
EVAL_MODEL_DIR = "Qwen2.5-7B-Instruct"
TRAIN_DATA_ITEMS = [
    "audio",
    "openface_face",
    "subtitle_chieng.csv",
    "track2_train_mercaptionplus.csv",
]
EVAL_DATASET_DIRS = {
    "MER2023": "mer2023-dataset-process",
    "MER2024": "mer2024-dataset-process",
    "MELD": "meld-process",
    "IEMOCAPFour": "iemocap-process",
    "CMUMOSI": "cmumosi-process",
    "CMUMOSEI": "cmumosei-process",
    "SIMS": "sims-process",
    "SIMSv2": "simsv2-process",
    "OVMERDPlus": "ovmerdplus-process",
}
RUNTIME_PACKAGES = [
    "torch",
    "torchvision",
    "transformers",
    "peft",
    "omegaconf",
    "decord",
    "numpy",
    "pandas",
]


def looks_like_repo(path: Path) -> bool:
    return (
        (path / "train.py").is_file()
        and (path / "train_stage2_mgrpo.py").is_file()
        and (path / "train_configs" / "stage2_mgrpo_qwen3_0_6b.yaml").is_file()
    )


def find_repo_root(explicit: str | None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    if os.environ.get("LIGHT_MER_ROOT"):
        candidates.append(Path(os.environ["LIGHT_MER_ROOT"]).expanduser())
    cwd = Path.cwd().resolve()
    candidates.extend([cwd, *cwd.parents])
    script = Path(__file__).resolve()
    candidates.extend([script.parent, *script.parents])
    for candidate in candidates:
        if looks_like_repo(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not locate the Light-MER repository. Set LIGHT_MER_ROOT or pass --repo-root."
    )


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def add_check(
    checks: list[dict],
    name: str,
    path: Path,
    *,
    required: bool = True,
    ok: bool | None = None,
    note: str = "",
) -> None:
    checks.append({
        "name": name,
        "path": str(path),
        "required": required,
        "ok": path.exists() if ok is None else ok,
        "note": note,
    })


def add_runtime_check(
    checks: list[dict],
    name: str,
    ok: bool,
    *,
    required: bool = True,
    note: str = "",
) -> None:
    checks.append({
        "name": name,
        "path": "runtime",
        "required": required,
        "ok": ok,
        "note": note,
    })


def writable_target(path: Path) -> bool:
    candidate = path.expanduser()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate.exists() and os.access(candidate, os.W_OK)


def runtime_checks(
    checks: list[dict],
    *,
    include_eval: bool,
    require_cuda: bool,
) -> None:
    packages = [*RUNTIME_PACKAGES, *(["vllm"] if include_eval else [])]
    for package in packages:
        add_runtime_check(
            checks,
            f"Python package:{package}",
            importlib.util.find_spec(package) is not None,
        )

    cuda_ok = False
    cuda_note = ""
    if importlib.util.find_spec("torch") is not None:
        try:
            import torch

            cuda_ok = torch.cuda.is_available()
            if cuda_ok:
                cuda_note = torch.cuda.get_device_name(0)
            else:
                cuda_note = "run again inside a GPU allocation"
        except Exception as exc:
            cuda_note = f"torch CUDA check failed: {exc}"
    else:
        cuda_note = "torch is not installed"
    add_runtime_check(
        checks,
        "CUDA available",
        cuda_ok,
        required=require_cuda,
        note=cuda_note,
    )


def csv_columns(path: Path, required: set[str]) -> tuple[bool, str]:
    if not path.is_file():
        return False, "file not found"
    try:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            columns = set(next(csv.reader(handle)))
    except Exception as exc:
        return False, f"cannot read header: {exc}"
    missing = sorted(required - columns)
    return not missing, "" if not missing else f"missing columns: {', '.join(missing)}"


def requested_datasets(raw: str) -> list[str]:
    if raw == "inferenceData":
        return list(EVAL_DATASET_DIRS)
    return [item.strip() for item in raw.split(",") if item.strip()]


def checkpoint_map(root: Path) -> dict[int, Path]:
    found = {}
    if not root.is_dir():
        return found
    for path in root.glob("checkpoint_[0-9]*_grpo.pth"):
        try:
            found[int(path.name.split("_")[1])] = path
        except (IndexError, ValueError):
            continue
    return found


def common_checks(repo: Path, checks: list[dict]) -> None:
    for relative in [
        "train_stage2_mgrpo.py",
        "mgrpo_reward.py",
        "inference_stage2_mgrpo.py",
        "scripts/eval_stage2_mgrpo.py",
        "scripts/train_stage2_mgrpo.sh",
        "scripts/inference_stage2_mgrpo.sh",
        "scripts/eval_stage2_mgrpo.sh",
        "train_configs/stage2_mgrpo_qwen3_0_6b.yaml",
    ]:
        add_check(checks, f"code:{relative}", repo / relative)


def model_checks(model_root: Path, checks: list[dict], include_eval: bool = False) -> None:
    for dirname in MODEL_DIRS:
        add_check(checks, f"model:{dirname}", model_root / dirname)
    if include_eval:
        add_check(checks, f"model:{EVAL_MODEL_DIR}", model_root / EVAL_MODEL_DIR)


def train_checks(args: argparse.Namespace, repo: Path) -> list[dict]:
    checks = []
    common_checks(repo, checks)
    model_root = env_path("SWDH_MODEL_ROOT", repo / "models")
    dataset_root = env_path("SWDH_DATASET_ROOT", repo / "dataset")
    model_checks(model_root, checks)
    stage1_ckpt = Path(
        args.stage1_ckpt
        or os.environ.get("STAGE1_CKPT", "")
        or repo / "checkpoints" / "stage1-swdh-qwen3-0.6b"
        / "checkpoint_000025_loss_1.549.pth"
    ).expanduser()
    add_check(checks, "Stage 1 initialization checkpoint", stage1_ckpt)
    train_root = dataset_root / "mer2025-dataset"
    for item in TRAIN_DATA_ITEMS:
        add_check(checks, f"MER-Caption+:{item}", train_root / item)
    for filename, columns in [
        ("track2_train_mercaptionplus.csv", {"name", "openset"}),
        ("subtitle_chieng.csv", {"name", "english"}),
    ]:
        path = train_root / filename
        ok, note = csv_columns(path, columns)
        add_check(checks, f"CSV schema:{filename}", path, ok=ok, note=note)
    output_dir = Path(
        args.output_dir
        or os.environ.get("OUTPUT_DIR", "")
        or repo / "output" / "stage2_mgrpo_qwen3_0_6b"
    ).expanduser()
    add_check(
        checks,
        "training output is writable",
        output_dir,
        ok=writable_target(output_dir),
    )
    return checks


def inference_checks(args: argparse.Namespace, repo: Path) -> list[dict]:
    checks = []
    common_checks(repo, checks)
    model_root = env_path("SWDH_MODEL_ROOT", repo / "models")
    dataset_root = env_path("SWDH_DATASET_ROOT", repo / "dataset")
    model_checks(model_root, checks)
    ckpt_root = Path(
        args.stage2_ckpt_root
        or os.environ.get("STAGE2_CKPT_ROOT", "")
        or repo / "checkpoints" / "stage2-mgrpo-qwen3-0.6b"
    ).expanduser()
    add_check(checks, "Stage 2 checkpoint directory", ckpt_root)
    checkpoints = checkpoint_map(ckpt_root)
    for step in args.steps:
        path = ckpt_root / f"checkpoint_{step:06d}_grpo.pth"
        add_check(checks, f"Stage 2 checkpoint step {step}", path, ok=step in checkpoints)
    for dataset in requested_datasets(args.dataset):
        dirname = EVAL_DATASET_DIRS.get(dataset)
        if dirname:
            add_check(checks, f"eval data:{dataset}", dataset_root / dirname)
        else:
            add_check(
                checks,
                f"eval data:{dataset}",
                Path("unknown-dataset"),
                ok=False,
                note="unknown dataset key",
            )
    result_root = Path(
        args.result_root
        or os.environ.get("STAGE2_RESULT_ROOT", "")
        or os.environ.get("BASE_ROOT", "")
        or repo / "output_stage2_mgrpo" / "results"
    ).expanduser()
    add_check(
        checks,
        "inference output is writable",
        result_root,
        ok=writable_target(result_root),
    )
    return checks


def eval_checks(args: argparse.Namespace, repo: Path) -> list[dict]:
    checks = []
    common_checks(repo, checks)
    model_root = env_path("SWDH_MODEL_ROOT", repo / "models")
    wheel_root = env_path("SWDH_EMOTION_WHEEL_ROOT", repo / "emotion_wheel")
    model_checks(model_root, checks, include_eval=True)
    add_check(checks, "Emotion Wheel resources", wheel_root)
    base_root = Path(
        args.result_root
        or os.environ.get("STAGE2_RESULT_ROOT", "")
        or os.environ.get("BASE_ROOT", "")
        or repo / "output_stage2_mgrpo" / "results"
    ).expanduser()
    for dataset in requested_datasets(args.dataset):
        result_dir = Path(f"{base_root}-{dataset.lower()}") / args.run_tag
        add_check(checks, f"results:{dataset}", result_dir)
        add_check(
            checks,
            f"results writable:{dataset}",
            result_dir,
            ok=result_dir.is_dir() and os.access(result_dir, os.W_OK),
        )
        for step in args.steps:
            raw = result_dir / f"checkpoint_{step:06d}_grpo.npz"
            add_check(checks, f"result:{dataset}:step {step}", raw)
    return checks


def print_summary(summary: dict) -> None:
    print(f"Light-MER Stage 2 preflight: {summary['mode']}")
    print("Status:", "OK" if summary["ok"] else "MISSING REQUIRED RESOURCES")
    for item in summary["checks"]:
        mark = "OK" if item["ok"] else ("MISSING" if item["required"] else "WARN")
        suffix = f" ({item['note']})" if item["note"] else ""
        print(f"[{mark}] {item['name']}: {item['path']}{suffix}")


def redact_summary_paths(summary: dict) -> dict:
    redacted = json.loads(json.dumps(summary))
    redacted["repo_root"] = "<redacted>"
    for key in ("missing", "checks"):
        for item in redacted[key]:
            value = item.get("path", "")
            if value not in {"runtime", "unknown-dataset"}:
                name = Path(value).name
                item["path"] = f"<redacted>/{name}" if name else "<redacted>"
    return redacted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["train", "inference", "eval"])
    parser.add_argument("--repo-root")
    parser.add_argument("--stage1-ckpt")
    parser.add_argument("--stage2-ckpt-root")
    parser.add_argument("--result-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--run-tag", default="stage2_mgrpo_qwen3_0_6b")
    parser.add_argument("--dataset", default=os.environ.get("DATASET", "inferenceData"))
    parser.add_argument("--steps", default="50,500,1000,1500,2000,2500,3000,3500,4000,4500,5000")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--show-paths", action="store_true")
    args = parser.parse_args()
    args.steps = [int(item.strip()) for item in args.steps.split(",") if item.strip()]

    repo = find_repo_root(args.repo_root)
    if args.mode == "train":
        checks = train_checks(args, repo)
    elif args.mode == "inference":
        checks = inference_checks(args, repo)
    else:
        checks = eval_checks(args, repo)
    runtime_checks(
        checks,
        include_eval=args.mode == "eval",
        require_cuda=args.require_cuda,
    )
    missing = [item for item in checks if item["required"] and not item["ok"]]
    summary = {
        "mode": args.mode,
        "repo_root": str(repo),
        "ok": not missing,
        "missing": missing,
        "checks": checks,
    }
    output_summary = summary if args.show_paths else redact_summary_paths(summary)
    if args.json:
        print(json.dumps(output_summary, indent=2))
    else:
        print_summary(output_summary)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
