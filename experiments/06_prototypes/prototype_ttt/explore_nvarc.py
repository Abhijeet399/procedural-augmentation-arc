"""
explore_nvarc.py — Inspect NVARC synthetic dataset structure and format.

Answers:
  1. What does a single JSON file look like?
  2. Is the format compatible with mdlARC's challenges.json format?
  3. How many total synthetic tasks are there?
  4. Which of our 400 eval tasks have synthetic coverage?

Usage (run from the repo root):
    python prototype_ttt/explore_nvarc.py \
        --nvarc-dir  assets/nvarc_synthetic \
        --challenges assets/challenges.json \
        --solutions  assets/solutions.json
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nvarc-dir",   required=True,
                   help="Path to unzipped NVARC dir (contains nvarc_full/ and nvarc_training/)")
    p.add_argument("--challenges",  default="assets/challenges.json")
    p.add_argument("--solutions",   default="assets/solutions.json")
    return p.parse_args()


def load_one_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def check_format(data: dict, source_path: Path):
    """
    Check if a single NVARC JSON is compatible with mdlARC's expected format.

    mdlARC expects challenges.json to be:
        {task_id: {"train": [{"input": [[...]], "output": [[...]]}, ...],
                   "test":  [{"input": [[...]]}, ...]}}

    We check if the NVARC JSON matches this structure.
    """
    issues = []

    if not isinstance(data, dict):
        issues.append(f"Top-level is {type(data).__name__}, expected dict")
        return issues

    for key in ("train", "test"):
        if key not in data:
            issues.append(f"Missing key: '{key}'")
            continue
        if not isinstance(data[key], list):
            issues.append(f"'{key}' is {type(data[key]).__name__}, expected list")
            continue
        for i, ex in enumerate(data[key]):
            if "input" not in ex:
                issues.append(f"'{key}[{i}]' missing 'input'")
            if key == "train" and "output" not in ex:
                issues.append(f"'{key}[{i}]' missing 'output'")
            if "input" in ex:
                inp = ex["input"]
                if not isinstance(inp, list) or not all(isinstance(r, list) for r in inp):
                    issues.append(f"'{key}[{i}].input' not a 2D list")

    return issues


def summarize_grid(grid):
    if not grid:
        return "empty"
    return f"{len(grid)}×{len(grid[0])}"


def main():
    args = parse_args()
    nvarc_root = Path(args.nvarc_dir)

    # ── Discover subdirectories ───────────────────────────────────────────────
    subdirs = {
        name: nvarc_root / name
        for name in ("nvarc_full", "nvarc_training")
        if (nvarc_root / name).is_dir()
    }

    if not subdirs:
        # Maybe the user pointed directly at nvarc_full or nvarc_training
        subdirs = {"root": nvarc_root}

    print("=" * 70)
    print("NVARC SYNTHETIC DATASET EXPLORER")
    print("=" * 70)

    for split_name, split_dir in subdirs.items():
        task_folders = sorted(split_dir.iterdir()) if split_dir.is_dir() else []
        task_folders = [f for f in task_folders if f.is_dir()]
        all_json = list(split_dir.rglob("*.json"))

        print(f"\n── {split_name}/ ──")
        print(f"   Task subfolders : {len(task_folders)}")
        print(f"   Total JSON files: {len(all_json)}")

        if not all_json:
            print("   (empty)")
            continue

        # ── Show example folder structure ─────────────────────────────────
        ex_folder = task_folders[0] if task_folders else None
        if ex_folder:
            ex_files = sorted(ex_folder.glob("*.json"))[:3]
            print(f"\n   Example folder: {ex_folder.name}/")
            for ef in ex_files:
                print(f"     {ef.name}")
            if len(list(ex_folder.glob("*.json"))) > 3:
                n = len(list(ex_folder.glob("*.json")))
                print(f"     ... ({n} total files in this folder)")

        # ── Load and inspect one JSON file ────────────────────────────────
        sample_file = all_json[0]
        print(f"\n   Sample file: {sample_file.relative_to(nvarc_root)}")

        data = load_one_json(sample_file)
        print(f"   Top-level type : {type(data).__name__}")

        if isinstance(data, dict):
            print(f"   Top-level keys : {list(data.keys())}")

            # Case 1: file IS a task (has "train"/"test" at top level)
            if "train" in data or "test" in data:
                print(f"\n   ✓ File is a single task (train/test at top level)")
                issues = check_format(data, sample_file)
                if issues:
                    print(f"   ✗ Format issues:")
                    for iss in issues:
                        print(f"       - {iss}")
                else:
                    print(f"   ✓ Format is compatible with mdlARC challenges.json")

                n_train = len(data.get("train", []))
                n_test  = len(data.get("test", []))
                print(f"   Train demos: {n_train}")
                print(f"   Test  pairs: {n_test}")

                if n_train > 0:
                    inp = data["train"][0].get("input", [])
                    out = data["train"][0].get("output", [])
                    print(f"   Demo[0] input  shape: {summarize_grid(inp)}")
                    print(f"   Demo[0] output shape: {summarize_grid(out)}")
                if n_test > 0:
                    inp = data["test"][0].get("input", [])
                    print(f"   Test[0] input  shape: {summarize_grid(inp)}")

            # Case 2: file contains multiple tasks keyed by ID
            else:
                first_key = next(iter(data))
                first_val = data[first_key]
                print(f"\n   File contains {len(data)} tasks (keyed by task ID)")
                print(f"   Sample task key: '{first_key}'")
                if isinstance(first_val, dict):
                    print(f"   Sample task keys: {list(first_val.keys())}")
                    issues = check_format(first_val, sample_file)
                    if issues:
                        print(f"   ✗ Format issues:")
                        for iss in issues:
                            print(f"       - {iss}")
                    else:
                        print(f"   ✓ Format compatible with mdlARC challenges.json")

        # ── Files per folder distribution ─────────────────────────────────
        counts = [len(list(f.glob("*.json"))) for f in task_folders]
        if counts:
            print(f"\n   Files per subfolder: min={min(counts)} max={max(counts)} "
                  f"avg={sum(counts)/len(counts):.1f} total={sum(counts)}")

    # ── Cross-reference with our 400 eval tasks ───────────────────────────────
    print("\n" + "=" * 70)
    print("CROSS-REFERENCE WITH OUR 400 EVAL TASKS")
    print("=" * 70)

    try:
        with open(args.challenges) as f:
            challenges = json.load(f)
        eval_task_ids = set(challenges.keys())
        print(f"Loaded {len(eval_task_ids)} tasks from {args.challenges}")
    except Exception as e:
        print(f"Could not load challenges: {e}")
        eval_task_ids = set()

    for split_name, split_dir in subdirs.items():
        if not split_dir.is_dir():
            continue
        task_folders = [f for f in sorted(split_dir.iterdir()) if f.is_dir()]
        folder_names = {f.name for f in task_folders}

        # Check how many folder names match our eval task IDs
        matched = folder_names & eval_task_ids
        print(f"\n{split_name}/:")
        print(f"  Subfolders: {len(folder_names)}")
        if eval_task_ids:
            print(f"  Subfolders matching our eval tasks: {len(matched)} / {len(eval_task_ids)}")
            unmatched_eval = eval_task_ids - folder_names
            print(f"  Eval tasks WITHOUT synthetic coverage: {len(unmatched_eval)}")

        # Count total synthetic tasks available for matched eval tasks
        total_synth = sum(
            len(list((split_dir / t).glob("*.json")))
            for t in matched
        )
        if matched:
            print(f"  Total synthetic tasks for matched eval tasks: {total_synth}")
            avg = total_synth / len(matched)
            print(f"  Avg synthetic tasks per eval task: {avg:.1f}")

        # Show a few examples
        if matched:
            examples = sorted(matched)[:3]
            print(f"\n  Example coverage (first 3 matched eval tasks):")
            for t in examples:
                n = len(list((split_dir / t).glob("*.json")))
                print(f"    {t}: {n} synthetic variants")

    # ── Recommended next step ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("RECOMMENDED NEXT STEP")
    print("=" * 70)
    print("""
If format is compatible (train/test keys present, grids are 2D int lists):
  → Use nvarc_training/ for EXP Q.a (covers our 400 eval tasks directly)
  → Use nvarc_full/ for broader distribution coverage
  → Write a loader that yields (task_id, task_data) from the folder structure

If format needs adaptation:
  → Keys may need renaming, or grid values may be strings instead of ints
  → A thin converter script is needed before training
""")


if __name__ == "__main__":
    main()
