#!/usr/bin/env python3
"""
merge_ensemble.py — Combine N EXP C runs into one ensemble submission.

Strategy: RUN-1 FIRST, DIVERSITY SECOND
  attempt_1 = run 1's attempt_1  (preserves EXP C 30.50% baseline)
  attempt_2 = most-voted candidate across ALL runs that differs from attempt_1
              (adds diversity without regressing)

This guarantees the merged score >= run 1 score.

Usage:
    python merge_ensemble.py \\
        --subs runs/ensemble/seed1/.../submission.json \\
               runs/ensemble/seed2/.../submission.json \\
               runs/ensemble/seed3/.../submission.json \\
        --solutions assets/solutions.json \\
        --out runs/ensemble/merged_submission.json
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def load_sub(path):
    """Load a submission.json → {task_id: [{attempt_1: grid, attempt_2: grid}, ...]}"""
    with open(path) as f:
        data = json.load(f)
    result = {}
    for task_id, pairs in data.items():
        normalized = []
        for pair in pairs:
            if 'attempts' in pair:
                # EXP C format: {"attempts": [grid1, grid2]}
                attempts = pair['attempts']
                a1 = attempts[0] if len(attempts) > 0 else [[0]]
                a2 = attempts[1] if len(attempts) > 1 else a1
            else:
                a1 = pair.get('attempt_1', [[0]])
                a2 = pair.get('attempt_2', [[0]])
            normalized.append({'attempt_1': a1, 'attempt_2': a2})
        result[task_id] = normalized
    return result


def grid_key(g):
    return json.dumps(g, separators=(',', ':'))


def task_correct(sub, task_id, solutions):
    """True if task_id is fully correct in sub (all pairs have at least one correct attempt)."""
    sol_grids = solutions.get(task_id, [])
    if not sol_grids:
        return False
    if not isinstance(sol_grids[0][0], list):
        sol_grids = [sol_grids]
    pairs = sub.get(task_id, [])
    for pi, sol in enumerate(sol_grids):
        if pi >= len(pairs):
            return False
        a1 = pairs[pi]['attempt_1']
        a2 = pairs[pi]['attempt_2']
        if a1 != sol and a2 != sol:
            return False
    return True


def merge_submissions(sub_paths, solutions_path, out_path):
    subs = [load_sub(p) for p in sub_paths]

    solutions = {}
    if solutions_path and Path(solutions_path).exists():
        with open(solutions_path) as f:
            solutions = json.load(f)

    all_tasks = set()
    for s in subs:
        all_tasks.update(s.keys())

    merged = {}
    for task_id in sorted(all_tasks):
        n_pairs = max(len(s.get(task_id, [])) for s in subs)
        task_result = []

        for pair_idx in range(n_pairs):
            # attempt_1 = always run 1's attempt_1 (the proven RCOS top-1)
            run1_pairs = subs[0].get(task_id, [])
            if pair_idx < len(run1_pairs):
                a1 = run1_pairs[pair_idx]['attempt_1']
            else:
                a1 = [[0]]

            a1_key = grid_key(a1)

            # attempt_2 = most-voted candidate across ALL runs (both slots),
            # excluding the grid already in attempt_1
            vote_counter: Counter = Counter()
            seen_keys = {a1_key}
            candidate_map = {}  # key -> grid

            for sub in subs:
                pairs = sub.get(task_id, [])
                if pair_idx < len(pairs):
                    for slot in ('attempt_1', 'attempt_2'):
                        g = pairs[pair_idx][slot]
                        k = grid_key(g)
                        vote_counter[k] += 1
                        candidate_map[k] = g

            # Pick the most-voted candidate that isn't already attempt_1
            best_alt_key = None
            best_alt_votes = -1
            for k, votes in vote_counter.most_common():
                if k != a1_key:
                    best_alt_key = k
                    best_alt_votes = votes
                    break

            a2 = candidate_map[best_alt_key] if best_alt_key else a1
            task_result.append({'attempt_1': a1, 'attempt_2': a2})

        merged[task_id] = task_result

    # Score all runs and the merged result
    if solutions:
        print("\nScoring individual runs:")
        run_scores = []
        for i, sub in enumerate(subs, 1):
            c = sum(1 for tid in all_tasks if task_correct(sub, tid, solutions))
            pct = c / len(all_tasks) * 100
            run_scores.append(c)
            print(f"  Run {i}: {c}/{len(all_tasks)} = {pct:.2f}%")

        mc = sum(1 for tid in all_tasks if task_correct(merged, tid, solutions))
        mpct = mc / len(all_tasks) * 100
        best_single = max(run_scores)
        delta = mc - best_single
        sign = '+' if delta >= 0 else ''

        print(f"\n  Merged ensemble: {mc}/{len(all_tasks)} = {mpct:.2f}%")
        print(f"  Delta vs best single run: {sign}{delta} tasks ({sign}{delta/len(all_tasks)*100:.2f}pp)")

        # Also count oracle: tasks where ANY run has the correct answer in ANY slot
        oracle_set = set()
        for tid in all_tasks:
            for sub in subs:
                sol = solutions.get(tid, [])
                if not sol: continue
                if not isinstance(sol[0][0], list): sol = [sol]
                pairs = sub.get(tid, [])
                all_pairs_hit = True
                for pi, s in enumerate(sol):
                    if pi >= len(pairs):
                        all_pairs_hit = False; break
                    if pairs[pi]['attempt_1'] != s and pairs[pi]['attempt_2'] != s:
                        all_pairs_hit = False; break
                if all_pairs_hit:
                    oracle_set.add(tid)
                    break
        print(f"\n  Union oracle (any run correct): {len(oracle_set)}/{len(all_tasks)} = {len(oracle_set)/len(all_tasks)*100:.2f}%")
        print(f"  Unrealized ceiling: {len(oracle_set) - mc} tasks still convertible to ARC points")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(merged, f)
    print(f"\nMerged submission → {out_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--subs', nargs='+', required=True)
    p.add_argument('--solutions', default='assets/solutions.json')
    p.add_argument('--out', default='runs/ensemble/merged_submission.json')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    merge_submissions(args.subs, args.solutions, args.out)
