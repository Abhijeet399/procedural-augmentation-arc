"""
leak_check.py — Verify there is no train/eval task-ID collision between the training
challenge set and the evaluation solutions set.

Background: this audit was run against assets/challenges.json (1307 training tasks) and
assets/solutions.json (400 ARC-1 public eval tasks). It found that task 070dd51e is
present in BOTH files — a genuine train/eval collision in the underlying ARC-1 asset
files, not a bug introduced by this project's data pipeline. Any checkpoint that solves
070dd51e is getting credit for a task it was trained on, which isn't a genuine
generalization result.

Usage:
    python experiments/05_leak_audit/leak_check.py \
        --train-challenges assets/challenges.json \
        --eval-solutions   assets/solutions.json
"""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-challenges", type=Path, required=True,
                   help="Training challenges.json (task IDs the embedding table/model sees).")
    p.add_argument("--eval-solutions", type=Path, required=True,
                   help="Evaluation solutions.json (task IDs scored at eval time).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_ids = set(json.load(open(args.train_challenges)).keys())
    eval_ids = set(json.load(open(args.eval_solutions)).keys())

    collisions = sorted(train_ids & eval_ids)

    print(f"Training tasks: {len(train_ids)}")
    print(f"Eval tasks:      {len(eval_ids)}")
    print(f"Collisions:      {len(collisions)}")
    for task_id in collisions:
        print(f"  LEAK: {task_id} appears in both the training set and the eval set")

    if collisions:
        print(
            f"\nTo report a leak-adjusted score, subtract credit for these "
            f"{len(collisions)} task(s) from the raw eval score before computing the "
            f"percentage."
        )


if __name__ == "__main__":
    main()
