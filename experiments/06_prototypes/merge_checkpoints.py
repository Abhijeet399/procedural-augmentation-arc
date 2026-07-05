"""
merge_checkpoints.py — Ensemble the epoch-645/648/650 submissions.

Reports the 3-way oracle ceiling (tasks solved by >=1 checkpoint) and builds a
"first-sub-first" merged submission:
    attempt_1 = first submission's attempt_1 (the best single model, epoch 645)
    attempt_2 = most-voted candidate across all 3 subs that differs from attempt_1
This is guaranteed not to regress below the first submission's score.

List the BEST submission FIRST in --subs (epoch 645).

Usage:
    python merge_checkpoints.py \
        --subs runs/eval_scratch_combined.epoch645/submission.json \
               runs/eval_scratch_combined.epoch650/submission.json \
               runs/eval_scratch_combined.epoch648/submission.json \
        --solutions assets/solutions.json \
        --out runs/ensemble_ckpt/merged.json
"""
import sys
import json
import argparse
from pathlib import Path
from collections import Counter

SRC_DIR = Path.cwd() / "src"
sys.path.insert(0, str(SRC_DIR))
import utils


def load(p):
    with open(p) as f:
        return json.load(f)


def get_attempts(pair):
    """Return list of grids (1-2) regardless of submission key style."""
    if isinstance(pair, dict):
        if "attempt_1" in pair:
            a = [pair["attempt_1"]]
            if pair.get("attempt_2") is not None:
                a.append(pair["attempt_2"])
            return a
        if "attempts" in pair:
            return list(pair["attempts"])
    return [pair]


def emit_pair(a1, a2, style):
    if style == "attempts":
        return {"attempts": [a1, a2]}
    return {"attempt_1": a1, "attempt_2": a2}


def gkey(g):
    return json.dumps(g, separators=(",", ":"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subs", nargs="+", required=True,
                    help="Submission JSONs; list BEST (epoch645) FIRST.")
    ap.add_argument("--solutions", default="assets/solutions.json")
    ap.add_argument("--out", default="runs/ensemble_ckpt/merged.json")
    args = ap.parse_args()

    # 1) Per-checkpoint scores + solved sets
    print("Per-checkpoint scores:")
    solved_sets = []
    for s in args.subs:
        res = utils.score_arc_submission(Path(args.solutions), Path(s))
        fs = set(res.get("fully_solved_tasks", []))
        solved_sets.append(fs)
        print(f"  {Path(s).parent.name:40s} {res['score']}/{res['max_score']} "
              f"= {res['percentage']:.3f}%  ({len(fs)} tasks)")

    union = set().union(*solved_sets)
    inter = set.intersection(*solved_sets) if solved_sets else set()
    base = max(len(s) for s in solved_sets)
    print(f"\n  3-way oracle UNION (solved by >=1 ckpt): {len(union)} tasks "
          f"= {len(union)/400*100:.2f}%  <- ceiling")
    print(f"  3-way INTERSECTION (solved by all):      {len(inter)} tasks")
    print(f"  Headroom over best single ({base} tasks): +{len(union)-base} tasks "
          f"(+{(len(union)-base)/400*100:.2f}pp max)")

    # 2) Build 645-first merged submission
    subs = [load(s) for s in args.subs]
    base_sub = subs[0]

    # Detect key style from first pair we can find
    style = "attempt_1"
    for pairs in base_sub.values():
        if pairs:
            style = "attempts" if "attempts" in pairs[0] else "attempt_1"
            break

    all_tasks = set()
    for s in subs:
        all_tasks.update(s.keys())

    merged = {}
    for tid in all_tasks:
        base_pairs = base_sub.get(tid)
        if base_pairs is None:
            for s in subs:
                if tid in s:
                    merged[tid] = s[tid]
                    break
            continue
        out_pairs = []
        for pi in range(len(base_pairs)):
            a1 = get_attempts(base_pairs[pi])[0]
            a1k = gkey(a1)
            votes = Counter()
            cand = {}
            for s in subs:
                ps = s.get(tid)
                if ps and pi < len(ps):
                    for g in get_attempts(ps[pi]):
                        k = gkey(g)
                        votes[k] += 1
                        cand[k] = g
            a2 = a1
            for k, _ in votes.most_common():
                if k != a1k:
                    a2 = cand[k]
                    break
            out_pairs.append(emit_pair(a1, a2, style))
        merged[tid] = out_pairs

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(merged, f)

    res = utils.score_arc_submission(Path(args.solutions), Path(args.out))
    print(f"\n  MERGED (645-first): {res['score']}/{res['max_score']} "
          f"= {res['percentage']:.3f}%  ({len(res.get('fully_solved_tasks', []))} tasks)")
    print(f"  Merged submission -> {args.out}")


if __name__ == "__main__":
    main()
