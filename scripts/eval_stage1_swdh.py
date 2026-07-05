#!/usr/bin/env python
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval_student_model import main_zeroshot_scores


DATASETS = [
    "mer2023", "mer2024", "meld", "iemocapfour",
    "cmumosi", "cmumosei", "sims", "simsv2", "ovmerdplus",
]


def latest_dir(root):
    candidates = [p for p in glob.glob(os.path.join(root, "*")) if os.path.isdir(p)]
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def main():
    parser = argparse.ArgumentParser(description="Evaluate SWD-H stage1 inference outputs.")
    parser.add_argument(
        "--base-root",
        default="output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/results",
        help="Prefix used by inference_hybird.py before -<dataset> is appended.",
    )
    parser.add_argument("--datasets", nargs="*", default=DATASETS)
    parser.add_argument("--run-dir", default=None, help="Specific run directory name inside each results-<dataset> folder.")
    parser.add_argument("--debug", action="store_true", help="Skip vLLM label extraction calls where possible.")
    parser.add_argument("--summary", default="output_stage1_swdh_qwen3_8b_to_qwen3_0_6b/eval_summary.json")
    args = parser.parse_args()

    summary = []
    for dataset in args.datasets:
        dataset_root = f"{args.base_root}-{dataset.lower()}"
        if args.run_dir:
            input_dir = os.path.join(dataset_root, args.run_dir)
        else:
            input_dir = latest_dir(dataset_root)

        if input_dir is None or not os.path.isdir(input_dir):
            print(f"[skip] {dataset}: missing result directory under {dataset_root}")
            continue

        score1, score2, score3 = main_zeroshot_scores(
            input_dir,
            debug=args.debug,
            inter_print=True,
        )
        summary.append({
            "dataset": dataset,
            "input_dir": input_dir,
            "score1": score1,
            "score2": score2,
            "score3": score3,
        })

    if summary:
        avg = float(np.mean([item["score1"] for item in summary]))
    else:
        avg = None

    output = {"average_score1": avg, "results": summary}
    os.makedirs(os.path.dirname(args.summary), exist_ok=True)
    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print("\nSummary")
    for item in summary:
        print(f"{item['dataset']}: {item['score1'] * 100:.2f}")
    if avg is not None:
        print(f"Average: {avg * 100:.2f}")
    print(f"Saved: {args.summary}")


if __name__ == "__main__":
    main()
