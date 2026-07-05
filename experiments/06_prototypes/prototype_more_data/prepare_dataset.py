"""
prototype_more_data/prepare_dataset.py
=======================================
Prepare a mixed training dataset for EXP Q fine-tuning.

Merges:
  - Original 1307 mdlARC tasks (assets/challenges.json)
  - Sampled NVARC synthetic tasks (assets/nvarc_training_challenges.json)

Mix ratio (EXP Q.a default): 80% NVARC synthetic, 20% original.

Why not use the full 1M+ NVARC tasks?
  The full nvarc_training_challenges.json is ~1GB. Loading it into mdlARC's
  dataset builder each training run is slow and memory-intensive.
  Instead we sample a fixed subset once, merge it with the originals, and
  save a single manageable combined challenges.json for training.

  Recommended sample size: 5000–15000 NVARC tasks (67–150× the original 1307).
  This gives enough diversity without impractical training times.

Usage:
    # Default: 10000 NVARC tasks + all 1307 originals → combined_challenges.json
    python prototype_more_data/prepare_dataset.py \\
        --original-challenges assets/challenges.json \\
        --original-solutions  assets/solutions.json \\
        --nvarc-challenges    assets/nvarc_training_challenges.json \\
        --nvarc-solutions     assets/nvarc_training_solutions.json \\
        --out-challenges      assets/combined_challenges.json \\
        --out-solutions       assets/combined_solutions.json \\
        --n-nvarc             10000 \\
        --seed                42

    # Smaller run for smoke test:
        ... --n-nvarc 500
"""

import json
import random
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Prepare mixed training dataset for EXP Q")
    p.add_argument("--original-challenges", required=True)
    p.add_argument("--original-solutions",  required=True)
    p.add_argument("--nvarc-challenges",    required=True)
    p.add_argument("--nvarc-solutions",     required=True)
    p.add_argument("--out-challenges",      required=True)
    p.add_argument("--out-solutions",       required=True)
    p.add_argument("--n-nvarc",  type=int, default=10000,
                   help="Number of NVARC tasks to sample (default: 10000)")
    p.add_argument("--max-nvarc-pairs-per-task", type=int, default=None,
                   help="Cap NVARC pairs added per parent task. "
                        "Recommended: 4-8 (same as original demo count) to prevent catastrophic forgetting. "
                        "Default: None (no cap, uses all pairs)")
    p.add_argument("--seed",     type=int, default=42)
    p.add_argument("--no-original", action="store_true",
                   help="Exclude original 1307 tasks (NVARC only)")
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)

    print("=" * 60)
    print("EXP Q — PREPARE COMBINED TRAINING DATASET")
    print("=" * 60)

    # ── Load originals ────────────────────────────────────────────────────────
    if not args.no_original:
        print(f"\nLoading original challenges: {args.original_challenges}")
        with open(args.original_challenges) as f:
            orig_c = json.load(f)
        with open(args.original_solutions) as f:
            orig_s = json.load(f)
        print(f"  → {len(orig_c)} original tasks")
    else:
        orig_c, orig_s = {}, {}
        print("\n  (original tasks excluded by --no-original)")

    # ── Sample NVARC ──────────────────────────────────────────────────────────
    print(f"\nLoading NVARC challenges: {args.nvarc_challenges}")
    print("  (this may take a moment for the 1GB file...)")
    with open(args.nvarc_challenges) as f:
        nvarc_c = json.load(f)
    with open(args.nvarc_solutions) as f:
        nvarc_s = json.load(f)
    print(f"  → {len(nvarc_c)} NVARC tasks available")

    n_sample = min(args.n_nvarc, len(nvarc_c))
    sampled_keys = random.sample(list(nvarc_c.keys()), n_sample)
    sampled_c = {k: nvarc_c[k] for k in sampled_keys}
    sampled_s = {k: nvarc_s[k] for k in sampled_keys}
    print(f"  → Sampled {n_sample} NVARC tasks (seed={args.seed})")

    # ── Merge by AGGREGATION ──────────────────────────────────────────────────
    # The model has a fixed embedding table for exactly 1307 task IDs.
    # New task IDs from NVARC (e.g. "00576224_025d127b") are silently dropped
    # by build_model_and_data because they're not in the embedding table.
    #
    # Fix: aggregate NVARC synthetic pairs UNDER the parent task ID.
    # e.g. "00576224_025d127b" → add its test pairs to task "00576224".
    # This lets the model train on new transformation patterns using the
    # existing task embedding, with zero architecture changes.
    #
    # Each NVARC file has format: task_id = "{parent}_{variant}"
    # We extract parent = task_id.split("_")[0]  (first 8-char hex)

    import copy as _copy
    combined_c = _copy.deepcopy(orig_c)
    combined_s = _copy.deepcopy(orig_s)

    # Quick sanity check on a few original tasks
    for _tid, _td in list(orig_c.items())[:3]:
        print(f"  Sample orig task '{_tid}' keys: {list(_td.keys())}")

    n_aggregated = 0
    n_dropped = 0
    for nvarc_task_id, nvarc_task in sampled_c.items():
        # Extract parent task ID (first component before underscore)
        parent_id = nvarc_task_id.split("_")[0]

        if parent_id not in combined_c:
            # Parent not in model's embedding table — skip
            n_dropped += 1
            continue

        # Append ALL NVARC pairs to the parent task's "train" split.
        # mdlARC training tasks only have "train" (no "test" key).
        # The model trains on "train" pairs (CE loss on output tokens).
        # We add both the demo pairs AND the test pairs from NVARC as
        # additional training examples — all are (input, output) pairs.
        nvarc_all_pairs = nvarc_task.get("train", []) + nvarc_task.get("test", [])
        # Apply per-task pair cap to prevent catastrophic forgetting.
        # Without a cap, 10k NVARC files × 30 pairs = 300k NVARC vs 5k original (55:1).
        # With cap=4: 10k files × 4 pairs = 40k NVARC vs 5k original (8:1).
        if args.max_nvarc_pairs_per_task is not None:
            nvarc_all_pairs = nvarc_all_pairs[:args.max_nvarc_pairs_per_task]
        combined_c[parent_id]["train"].extend(nvarc_all_pairs)

        # Solutions (answers) are only needed for eval "test" pairs.
        # Since we appended to "train", solutions are already embedded in
        # the training pairs themselves — no separate solutions entry needed.

        n_aggregated += 1

    print(f"  Aggregated into parent tasks: {n_aggregated}")
    if n_dropped:
        print(f"  Dropped (parent not in model): {n_dropped}")

    # Sanity: no new task IDs introduced
    new_ids = set(combined_c.keys()) - set(orig_c.keys())
    assert len(new_ids) == 0, f"Unexpected new task IDs: {new_ids}"

    print(f"\nCombined dataset:")
    print(f"  Original tasks : {len(orig_c)}")
    print(f"  NVARC tasks    : {n_sample}")
    print(f"  Total tasks    : {len(combined_c)}")

    # Count total examples (each test pair = 1 training example)
    total_train_ex = sum(len(v.get("test", [])) for v in combined_c.values())
    total_demo_ex  = sum(len(v.get("train", [])) for v in combined_c.values())
    print(f"  Total demo pairs: {total_demo_ex}")
    print(f"  Total test pairs (training examples): {total_train_ex:,}")

    # Mix ratio
    orig_ex  = sum(len(v.get("test", [])) for v in orig_c.values()) if orig_c else 0
    nvarc_ex = sum(len(v.get("test", [])) for v in sampled_c.values())
    if total_train_ex > 0:
        print(f"\n  Mix ratio by training examples:")
        print(f"    Original : {orig_ex:,}  ({orig_ex/total_train_ex*100:.1f}%)")
        print(f"    NVARC    : {nvarc_ex:,} ({nvarc_ex/total_train_ex*100:.1f}%)")

    # ── Write ─────────────────────────────────────────────────────────────────
    out_c = Path(args.out_challenges)
    out_s = Path(args.out_solutions)
    out_c.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting combined challenges → {out_c}")
    out_c.write_text(json.dumps(combined_c, separators=(",", ":")))
    print(f"  Size: {out_c.stat().st_size / 1e6:.1f} MB")

    print(f"Writing combined solutions  → {out_s}")
    out_s.write_text(json.dumps(combined_s, separators=(",", ":")))
    print(f"  Size: {out_s.stat().st_size / 1e6:.1f} MB")

    print("\n✓ Dataset ready. Next:")
    print(f"  python prototype_more_data/finetune_q.py \\")
    print(f"      --checkpoint  runs/tiny.pt \\")
    print(f"      --data-path   {out_c} \\")
    print(f"      --solutions   {out_s} \\")
    print(f"      --eval-data   assets/challenges.json \\")
    print(f"      --eval-solutions assets/solutions.json \\")
    print(f"      --out-checkpoint runs/q_a_finetuned.pt")


if __name__ == "__main__":
    main()
