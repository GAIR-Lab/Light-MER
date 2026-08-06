#!/usr/bin/env python3
"""Evaluate a Light-MER Stage 2 output tree."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from eval_student_model import main_zeroshot_scores


DATASETS = [
    "mer2023",
    "mer2024",
    "meld",
    "iemocapfour",
    "cmumosi",
    "cmumosei",
    "sims",
    "simsv2",
    "ovmerdplus",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Light-MER Stage 2 evaluator")
    parser.add_argument(
        "--base-root",
        required=True,
        help="Result prefix; the evaluator appends -<dataset>/<run-tag>",
    )
    parser.add_argument(
        "--run-tag",
        required=True,
        help="Directory below each dataset result root",
    )
    args = parser.parse_args()

    print("Step 1: build missing openset/sentiment caches")
    for dataset in DATASETS:
        process_root = f"{args.base_root}-{dataset}/{args.run_tag}"
        main_zeroshot_scores(
            process_root,
            debug=False,
            test_epochs="",
            inter_print=True,
        )

    print("\nStep 2: summarize cached scores")
    print_s1, print_s2, print_s3 = [], [], []
    avg_s1, avg_s2, avg_s3 = [], [], []
    raw_scores = {}
    for dataset in DATASETS:
        process_root = f"{args.base_root}-{dataset}/{args.run_tag}"
        score1, score2, score3 = main_zeroshot_scores(
            process_root,
            debug=True,
            test_epochs="",
            inter_print=False,
        )
        raw_scores[dataset] = (score1, score2, score3)
        print_s1.append("& %.2f" % (score1 * 100))
        print_s2.append("& %.2f" % (score2 * 100))
        print_s3.append("& %.2f" % (score3 * 100))
        avg_s1.append(score1)
        avg_s2.append(score2)
        avg_s3.append(score3)

    mean_s1 = float(np.mean(avg_s1))
    print_s1.append("& %.2f" % (mean_s1 * 100))
    print_s2.append("& %.2f" % (float(np.mean(avg_s2)) * 100))
    print_s3.append("& %.2f" % (float(np.mean(avg_s3)) * 100))

    print(args.run_tag + " [hitrate/fscore] " + " ".join(print_s1))
    print(args.run_tag + " [mscore/acc]     " + " ".join(print_s2))
    print(args.run_tag + " [ew_f1/recall]   " + " ".join(print_s3))
    print("mean_percent=%.4f" % (mean_s1 * 100))
    print(
        "per_dataset_percent="
        + ", ".join(
            f"{dataset}:{raw_scores[dataset][0] * 100:.4f}"
            for dataset in DATASETS
        )
    )


if __name__ == "__main__":
    main()
