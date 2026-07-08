#!/usr/bin/env python3
"""Preflight checks for Light-MER Stage 1 train/inference/eval workflows."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable


def find_repo_root(start: Path) -> Path:
    for path in [start, *start.parents]:
        if (path / "train.py").exists() and (path / "train_configs").exists():
            return path
    return start.parents[4]


REPO_ROOT = find_repo_root(Path(__file__).resolve())

MODEL_DIRS = {
    "qwen3_8b": "Qwen3-8B",
    "qwen3_0_6b": "Qwen3-0.6B",
    "qwen25": "Qwen2.5-7B-Instruct",
    "clip_large": "clip-vit-large-patch14",
    "clip_base": "clip-vit-base-patch16",
    "hubert_large": "chinese-hubert-large",
    "hubert_base": "chinese-hubert-base",
}

TRAIN_DATA_ITEMS = [
    "video",
    "audio",
    "openface_face",
    "subtitle_chieng.csv",
    "track2_train_mercaptionplus.csv",
    "track3_train_mercaptionplus.csv",
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


def env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else default


def exists(path: Path) -> bool:
    return path.exists()


def add_check(items: list[dict], name: str, path: Path, required: bool = True) -> None:
    items.append({
        "name": name,
        "path": str(path),
        "required": required,
        "ok": exists(path),
    })


def model_root(args: argparse.Namespace, repo: Path) -> Path:
    if args.model_root:
        return Path(args.model_root).expanduser()
    return env_path("SWDH_MODEL_ROOT", repo / "models")


def dataset_root(args: argparse.Namespace, repo: Path) -> Path:
    if args.dataset_root:
        return Path(args.dataset_root).expanduser()
    return env_path("SWDH_DATASET_ROOT", repo / "dataset")


def emotion_wheel_root(args: argparse.Namespace, repo: Path) -> Path:
    if args.emotion_wheel_root:
        return Path(args.emotion_wheel_root).expanduser()
    return env_path("SWDH_EMOTION_WHEEL_ROOT", repo / "emotion_wheel")


def result_root(args: argparse.Namespace, repo: Path) -> Path:
    if args.result_root:
        return Path(args.result_root).expanduser()
    return env_path("SWDH_RESULT_ROOT", repo / "output_stage1_swdh_qwen3_8b_to_qwen3_0_6b")


def teacher_ckpt(args: argparse.Namespace, repo: Path) -> Path:
    if args.teacher_ckpt:
        return Path(args.teacher_ckpt).expanduser()
    return env_path("TEACHER_CKPT", repo / "checkpoints" / "qwen3_8b_teacher.pth")


def checkpoint_root_candidates(args: argparse.Namespace, repo: Path) -> list[Path]:
    candidates: list[Path] = []
    for raw in [args.ckpt_root, os.environ.get("CKPT_ROOT")]:
        if raw:
            candidates.append(Path(raw).expanduser())
    candidates.extend([
        repo / "checkpoints" / "stage1-swdh-qwen3-0.6b",
        repo / "output" / "stage1_swdh_qwen3_8b_to_qwen3_0_6b",
    ])
    output_parent = repo / "output" / "stage1_swdh_qwen3_8b_to_qwen3_0_6b"
    if output_parent.exists():
        candidates.extend([p for p in output_parent.iterdir() if p.is_dir()])
    return candidates


def checkpoint_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    return sorted(root.glob("checkpoint_*.pth"))


def best_checkpoint_root(args: argparse.Namespace, repo: Path) -> tuple[Path | None, list[Path]]:
    seen: set[Path] = set()
    best_root: Path | None = None
    best_files: list[Path] = []
    for candidate in checkpoint_root_candidates(args, repo):
        candidate = candidate.resolve() if candidate.exists() else candidate
        if candidate in seen:
            continue
        seen.add(candidate)
        files = checkpoint_files(candidate)
        if len(files) > len(best_files):
            best_root = candidate
            best_files = files
    return best_root, best_files


def requested_eval_datasets(args: argparse.Namespace) -> Iterable[str]:
    raw = args.dataset or os.environ.get("DATASET") or "inferenceData"
    if raw == "inferenceData":
        return EVAL_DATASET_DIRS.keys()
    return [item.strip() for item in raw.split(",") if item.strip()]


def base_checks(repo: Path, cfg: Path) -> list[dict]:
    checks: list[dict] = []
    add_check(checks, "config", cfg)
    add_check(checks, "train.py", repo / "train.py")
    add_check(checks, "inference_hybird.py", repo / "inference_hybird.py")
    return checks


def train_checks(args: argparse.Namespace, repo: Path, cfg: Path) -> list[dict]:
    checks = base_checks(repo, cfg)
    root = model_root(args, repo)
    data_root = dataset_root(args, repo)
    add_check(checks, "train script", repo / "scripts" / "train_stage1_swdh.sh")
    add_check(checks, "teacher checkpoint", teacher_ckpt(args, repo))
    for key in ["qwen3_8b", "qwen3_0_6b", "clip_large", "clip_base", "hubert_large", "hubert_base"]:
        add_check(checks, f"model:{MODEL_DIRS[key]}", root / MODEL_DIRS[key])
    mer2025 = data_root / "mer2025-dataset"
    for item in TRAIN_DATA_ITEMS:
        add_check(checks, f"train data:{item}", mer2025 / item)
    return checks


def inference_checks(args: argparse.Namespace, repo: Path, cfg: Path) -> tuple[list[dict], dict]:
    checks = base_checks(repo, cfg)
    root = model_root(args, repo)
    data_root = dataset_root(args, repo)
    add_check(checks, "inference script", repo / "scripts" / "inference_stage1_swdh.sh")
    for key in ["qwen3_0_6b", "clip_base", "hubert_base"]:
        add_check(checks, f"model:{MODEL_DIRS[key]}", root / MODEL_DIRS[key])
    ckpt_root, ckpts = best_checkpoint_root(args, repo)
    checks.append({
        "name": "stage1 checkpoint root",
        "path": str(ckpt_root or (repo / "checkpoints" / "stage1-swdh-qwen3-0.6b")),
        "required": True,
        "ok": bool(ckpts),
        "checkpoint_count": len(ckpts),
        "recommended": str(ckpts[-1]) if ckpts else "",
    })
    for dataset in requested_eval_datasets(args):
        dirname = EVAL_DATASET_DIRS.get(dataset)
        if dirname:
            add_check(checks, f"eval data:{dataset}", data_root / dirname)
        else:
            checks.append({
                "name": f"eval data:{dataset}",
                "path": "unknown dataset key",
                "required": False,
                "ok": False,
            })
    details = {
        "ckpt_root": str(ckpt_root) if ckpt_root else "",
        "checkpoint_count": len(ckpts),
        "recommended_checkpoint": str(ckpts[-1]) if ckpts else "",
    }
    return checks, details


def eval_checks(args: argparse.Namespace, repo: Path, cfg: Path) -> list[dict]:
    checks = base_checks(repo, cfg)
    root = model_root(args, repo)
    results = result_root(args, repo)
    wheel = emotion_wheel_root(args, repo)
    add_check(checks, "eval script", repo / "scripts" / "eval_stage1_swdh.sh")
    add_check(checks, f"model:{MODEL_DIRS['qwen25']}", root / MODEL_DIRS["qwen25"])
    add_check(checks, "emotion wheel root", wheel)
    for dataset in requested_eval_datasets(args):
        add_check(checks, f"inference results:{dataset}", Path(f"{results}/results-{dataset.lower()}"))
    return checks


def summarize(mode: str, checks: list[dict], details: dict | None = None) -> dict:
    missing = [item for item in checks if item["required"] and not item["ok"]]
    warnings = [item for item in checks if not item["required"] and not item["ok"]]
    return {
        "mode": mode,
        "ok": not missing,
        "missing": missing,
        "warnings": warnings,
        "checks": checks,
        "details": details or {},
    }


def print_text(summary: dict) -> None:
    print(f"Light-MER Stage 1 preflight: {summary['mode']}")
    print("Status:", "OK" if summary["ok"] else "MISSING REQUIRED RESOURCES")
    print()
    for item in summary["checks"]:
        mark = "OK" if item["ok"] else ("MISSING" if item["required"] else "WARN")
        extra = ""
        if "checkpoint_count" in item:
            extra = f" ({item['checkpoint_count']} checkpoints)"
        print(f"[{mark}] {item['name']}: {item['path']}{extra}")
    if summary["missing"]:
        print("\nMissing paths to ask the user for:")
        for item in summary["missing"]:
            print(f"- {item['name']}: {item['path']}")
    if summary["details"]:
        print("\nDetails:")
        for key, value in summary["details"].items():
            print(f"- {key}: {value}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Light-MER Stage 1 resources.")
    parser.add_argument("mode", choices=["train", "inference", "eval"])
    parser.add_argument("--repo-root", default=str(REPO_ROOT))
    parser.add_argument("--cfg", default="train_configs/stage1_swdh_qwen3_8b_to_qwen3_0_6b.yaml")
    parser.add_argument("--teacher-ckpt")
    parser.add_argument("--ckpt-root")
    parser.add_argument("--model-root")
    parser.add_argument("--dataset-root")
    parser.add_argument("--emotion-wheel-root")
    parser.add_argument("--result-root")
    parser.add_argument("--dataset")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo = Path(args.repo_root).expanduser().resolve()
    cfg = Path(args.cfg).expanduser()
    if not cfg.is_absolute():
        cfg = repo / cfg

    if args.mode == "train":
        summary = summarize(args.mode, train_checks(args, repo, cfg))
    elif args.mode == "inference":
        checks, details = inference_checks(args, repo, cfg)
        summary = summarize(args.mode, checks, details)
    else:
        summary = summarize(args.mode, eval_checks(args, repo, cfg))

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print_text(summary)
    return 0 if summary["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
