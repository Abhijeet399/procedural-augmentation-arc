"""
prepare_dataset_rearc.py — Build a combined ARC + re-ARC training set.

Mirrors the NVARC prepare_dataset.py pattern, but for re-ARC:
  - re-ARC task IDs ARE the original ARC-1 training task IDs (filename = task_id),
    so pairs aggregate directly under existing embedding-table IDs. No parent
    extraction is needed.
  - re-ARC is in-distribution (verified regenerations of the ARC-1 training tasks)
    and has zero ID overlap with the ARC-1 evaluation set.

Safety guards:
  - Only augments task IDs that already exist in the original challenges.json
    (i.e. are in the model's fixed embedding table). re-ARC IDs absent from the
    1307 are silently skipped — they'd be dropped by build_model_and_data anyway.
  - Refuses to augment any task ID present in the eval solutions (leakage guard).
    With ARC-1 train/eval this is a no-op, but it is defense-in-depth.
  - Appends only to each task's 'train' key (never 'test').

Usage:
    python prepare_dataset_rearc.py \
        --orig-challenges      assets/challenges.json \
        --rearc-tasks-dir      assets/re_arc/tasks \
        --eval-solutions       assets/solutions.json \
        --max-rearc-pairs-per-task 12 \
        --seed 42 \
        --out-challenges       assets/combined_rearc_challenges.json
"""

import argparse
import json
import os
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build combined ARC + re-ARC challenges.json")
    p.add_argument("--orig-challenges", type=Path, required=True,
                   help="Original challenges.json (the 1307-task embedding-table set).")
    p.add_argument("--rearc-tasks-dir", type=Path, required=True,
                   help="Path to re_arc/tasks/ (one {task_id}.json per ARC-1 training task).")
    p.add_argument("--eval-solutions", type=Path, default=None,
                   help="solutions.json — its keys are eval tasks that must NOT be augmented.")
    p.add_argument("--max-rearc-pairs-per-task", type=int, default=12,
                   help="Cap on re-ARC pairs appended per task (controls orig:re-ARC ratio).")
    p.add_argument("--max-grid-dim", type=int, default=30,
                   help="Drop re-ARC pairs whose input/output grid exceeds this height or "
                        "width. re-ARC generators can emit grids larger than the canonical "
                        "ARC 30x30, which overflow the model's max_seq_len. Default 30 keeps "
                        "pairs within the original ARC size envelope.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-challenges", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading original challenges: {args.orig_challenges}")
    with open(args.orig_challenges) as f:
        challenges = json.load(f)
    orig_ids = set(challenges.keys())
    print(f"  {len(orig_ids)} tasks in embedding table")

    def _gp(inp, out):
        return json.dumps(inp) + ">" + json.dumps(out)

    eval_ids = set()
    eval_test_pairs = set()
    if args.eval_solutions and args.eval_solutions.exists():
        with open(args.eval_solutions) as f:
            solutions = json.load(f)
        eval_ids = set(solutions.keys())
        print(f"  {len(eval_ids)} eval task IDs loaded (will be excluded from augmentation)")
        # Content-level leak guard: hash every exact (input, output) eval TEST pair.
        # re-ARC can occasionally regenerate a training-task pair that coincidentally
        # equals an eval test pair; dropping those keeps the eval honest by construction.
        for tid, outs in solutions.items():
            tests = challenges.get(tid, {}).get("test", [])
            for i, out in enumerate(outs):
                if i < len(tests):
                    eval_test_pairs.add(_gp(tests[i]["input"], out))
        print(f"  {len(eval_test_pairs)} eval test pairs hashed (content leak guard)")

    rearc_files = {p.stem: p for p in args.rearc_tasks_dir.glob("*.json")}
    print(f"  {len(rearc_files)} re-ARC task files found")

    # Coverage diagnostics
    rearc_ids = set(rearc_files.keys())
    in_table = rearc_ids & orig_ids
    not_in_table = rearc_ids - orig_ids
    leakage = rearc_ids & eval_ids
    print(f"\nCoverage:")
    print(f"  re-ARC IDs in embedding table : {len(in_table)} / {len(rearc_ids)}")
    print(f"  re-ARC IDs NOT in table (skip): {len(not_in_table)}")
    print(f"  re-ARC ∩ eval (leakage guard) : {len(leakage)}  (must be 0)")
    if leakage:
        print(f"    LEAKAGE IDs (excluded): {sorted(leakage)[:10]}{' ...' if len(leakage) > 10 else ''}")

    cap = args.max_rearc_pairs_per_task
    mgd = args.max_grid_dim
    n_aug_tasks = 0
    n_pairs_added = 0
    n_skipped_eval = 0
    n_dropped_oversize = 0
    n_dropped_leak = 0
    short_tasks = []

    def _in_bounds(p):
        return (len(p["input"]) <= mgd and len(p["input"][0]) <= mgd
                and len(p["output"]) <= mgd and len(p["output"][0]) <= mgd)

    for task_id in sorted(in_table):
        if task_id in eval_ids:
            n_skipped_eval += 1
            continue
        with open(rearc_files[task_id]) as f:
            pairs = json.load(f)            # flat list of {"input","output"}
        if not pairs:
            continue
        valid = [p for p in pairs if _in_bounds(p)]   # drop grids > max_grid_dim
        n_dropped_oversize += len(pairs) - len(valid)
        if eval_test_pairs:                            # drop exact eval-test collisions
            before = len(valid)
            valid = [p for p in valid
                     if _gp(p["input"], p["output"]) not in eval_test_pairs]
            n_dropped_leak += before - len(valid)
        if not valid:
            continue
        k = min(cap, len(valid))
        if k < cap:
            short_tasks.append((task_id, k))
        sampled = rng.sample(valid, k)
        # Append to the task's 'train' key (create if missing for safety)
        challenges[task_id].setdefault("train", [])
        challenges[task_id]["train"].extend(
            {"input": p["input"], "output": p["output"]} for p in sampled
        )
        n_aug_tasks += 1
        n_pairs_added += k

    print(f"\nAugmentation summary:")
    print(f"  tasks augmented        : {n_aug_tasks}")
    print(f"  re-ARC pairs added     : {n_pairs_added}")
    print(f"  pairs dropped (>{mgd}px) : {n_dropped_oversize}")
    print(f"  pairs dropped (eval leak): {n_dropped_leak}")
    print(f"  eval tasks skipped     : {n_skipped_eval}")
    print(f"  avg re-ARC pairs/task  : {n_pairs_added / max(n_aug_tasks, 1):.1f}")
    if short_tasks:
        print(f"  tasks under cap ({cap}) : {len(short_tasks)}"
              f"  (e.g. {short_tasks[:5]})")

    args.out_challenges.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_challenges, "w") as f:
        json.dump(challenges, f)
    size_mb = os.path.getsize(args.out_challenges) / 1e6
    print(f"\nWrote {args.out_challenges}  ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
