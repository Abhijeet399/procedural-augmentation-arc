"""
prototype_ttt/convert_nvarc.py — Convert NVARC synthetic data to mdlARC format
================================================================================

NVARC JSON format (each file):
    [
      {"input": [[0,1,...], ...], "output": [[2,3,...], ...]},
      {"input": ..., "output": ...},
      ...   (30 pairs total per file)
    ]

mdlARC challenges.json format:
    {
      "task_id": {
        "train": [{"input": [...], "output": [...]}, ...],  # N demo pairs
        "test":  [{"input": [...]}]                         # 1 test input
      },
      ...
    }
mdlARC solutions.json format:
    {
      "task_id": [[[...]], ...]   # list of test outputs (one per test pair)
    }

Conversion strategy:
    Each NVARC file has 30 (input, output) pairs.
    We use the first --n-demos pairs as train demos, the last 1 as test.
    The test output goes into solutions.json (used for CE loss during training).

    Default: n_demos=4  → 4 train demos, 1 test pair per synthetic task.

Which split to use:
    nvarc_training/  716 task folders, ~66 files each → ~47k synthetic tasks
                     Subfolders named after real ARC task IDs — directly
                     relevant to our 400 eval tasks.
    nvarc_full/      120 task folders, ~466 files each → ~56k synthetic tasks
                     Broader coverage, may include harder/different patterns.

    Recommended: convert both, then train with a mixture.

Usage:
    # Convert nvarc_training (all files):
    python prototype_ttt/convert_nvarc.py \\
        --nvarc-dir   assets/nvarc_synthetic/nvarc_training \\
        --out-challenges assets/nvarc_training_challenges.json \\
        --out-solutions  assets/nvarc_training_solutions.json \\
        --n-demos 4

    # Convert nvarc_full:
    python prototype_ttt/convert_nvarc.py \\
        --nvarc-dir   assets/nvarc_synthetic/nvarc_full \\
        --out-challenges assets/nvarc_full_challenges.json \\
        --out-solutions  assets/nvarc_full_solutions.json \\
        --n-demos 4

    # Quick validation (first 100 files only):
        ... --max-files 100 --validate
"""

import json
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Convert NVARC synthetic data to mdlARC format")
    p.add_argument("--nvarc-dir",        required=True,
                   help="Path to nvarc_training/ or nvarc_full/")
    p.add_argument("--out-challenges",   required=True,
                   help="Output challenges.json path")
    p.add_argument("--out-solutions",    required=True,
                   help="Output solutions.json path")
    p.add_argument("--n-demos",          type=int, default=4,
                   help="Number of demo pairs per synthetic task (default: 4)")
    p.add_argument("--max-files",        type=int, default=None,
                   help="Cap on number of files to convert (for testing)")
    p.add_argument("--validate",         action="store_true",
                   help="Print detailed validation of first few tasks")
    return p.parse_args()


def convert_file(
    json_path:  Path,
    n_demos:    int,
) -> tuple[dict, dict]:
    """
    Convert one NVARC JSON file to a (challenge_entry, solution_entry) pair.

    Args:
        json_path: path to the NVARC JSON file
        n_demos:   number of demo pairs to use (rest become test pairs)

    Returns:
        (challenge_dict, solution_dict) suitable for challenges.json / solutions.json
        challenge_dict = {"train": [...], "test": [{"input": ...}]}
        solution_dict  = [output_grid]  (list of test outputs)
    """
    pairs = json.loads(json_path.read_text())

    if not isinstance(pairs, list) or len(pairs) < n_demos + 1:
        return None, None

    # Use stem of filename (without .json) as a unique task ID
    # e.g. "8dae5dfc_137f0df0" → task_id = "8dae5dfc_137f0df0"
    task_id = json_path.stem

    train_pairs = pairs[:n_demos]
    test_pairs  = pairs[n_demos:]   # everything after demos becomes test

    challenge = {
        "train": [
            {"input": p["input"], "output": p["output"]}
            for p in train_pairs
        ],
        "test": [
            {"input": p["input"]}
            for p in test_pairs
        ],
    }
    # Solutions: list of outputs for each test pair
    solutions = [p["output"] for p in test_pairs]

    return task_id, challenge, solutions


def validate_task(task_id: str, challenge: dict, solutions: list):
    """Print a human-readable summary of one converted task."""
    n_train = len(challenge.get("train", []))
    n_test  = len(challenge.get("test",  []))
    print(f"\n  task_id : {task_id}")
    print(f"  train   : {n_train} demo pairs")
    print(f"  test    : {n_test} test pairs")
    if challenge["train"]:
        ex = challenge["train"][0]
        inp, out = ex["input"], ex["output"]
        print(f"  demo[0] : input={len(inp)}×{len(inp[0]) if inp else 0}"
              f"  output={len(out)}×{len(out[0]) if out else 0}")
    if solutions:
        sol = solutions[0]
        print(f"  sol[0]  : {len(sol)}×{len(sol[0]) if sol else 0} grid")
    # Verify all grid values are ints
    bad_cells = []
    for key in ("train",):
        for ex in challenge[key]:
            for grid_key in ("input", "output"):
                for row in ex.get(grid_key, []):
                    for v in row:
                        if not isinstance(v, int):
                            bad_cells.append((key, grid_key, v))
    if bad_cells:
        print(f"  ✗ Non-int values found: {bad_cells[:5]}")
    else:
        print(f"  ✓ All grid values are ints")


def main():
    args = parse_args()
    nvarc_dir = Path(args.nvarc_dir)

    if not nvarc_dir.exists():
        print(f"ERROR: {nvarc_dir} does not exist")
        return

    # Collect all JSON files
    all_files = sorted(nvarc_dir.rglob("*.json"))
    if args.max_files:
        all_files = all_files[:args.max_files]

    print(f"Converting {len(all_files)} files from {nvarc_dir}")
    print(f"  n_demos={args.n_demos}  (→ each file yields {30 - args.n_demos} test pair(s))")

    challenges = {}
    solutions  = {}
    skipped    = 0
    n_validated = 0

    for i, json_path in enumerate(all_files):
        result = convert_file(json_path, args.n_demos)
        if result[0] is None:
            skipped += 1
            continue

        task_id, challenge, sol = result

        # Guard against duplicate task_ids (shouldn't happen with stem-based IDs)
        if task_id in challenges:
            task_id = f"{task_id}_{i}"

        challenges[task_id] = challenge
        solutions[task_id]  = sol

        if args.validate and n_validated < 3:
            validate_task(task_id, challenge, sol)
            n_validated += 1

        if (i + 1) % 5000 == 0:
            print(f"  [{i+1}/{len(all_files)}] converted {len(challenges)} tasks ...")

    # Write outputs
    out_c = Path(args.out_challenges)
    out_s = Path(args.out_solutions)
    out_c.parent.mkdir(parents=True, exist_ok=True)
    out_s.parent.mkdir(parents=True, exist_ok=True)

    out_c.write_text(json.dumps(challenges, separators=(",", ":")))
    out_s.write_text(json.dumps(solutions,  separators=(",", ":")))

    # Summary
    total_train_pairs = sum(len(v["train"]) for v in challenges.values())
    total_test_pairs  = sum(len(v["test"])  for v in challenges.values())
    print()
    print("=" * 60)
    print("CONVERSION COMPLETE")
    print("=" * 60)
    print(f"  Total tasks      : {len(challenges)}")
    print(f"  Skipped          : {skipped}")
    print(f"  Total train pairs: {total_train_pairs}")
    print(f"  Total test pairs : {total_test_pairs}")
    print(f"  challenges.json  → {out_c}  ({out_c.stat().st_size/1e6:.1f} MB)")
    print(f"  solutions.json   → {out_s}  ({out_s.stat().st_size/1e6:.1f} MB)")
    print()
    print("Next step — verify one task loads correctly:")
    print(f"  python3 -c \"")
    print(f"    import json")
    print(f"    c = json.load(open('{out_c}'))")
    print(f"    k = next(iter(c))")
    print(f"    print(k, 'train:', len(c[k]['train']), 'test:', len(c[k]['test']))\"")


if __name__ == "__main__":
    main()
