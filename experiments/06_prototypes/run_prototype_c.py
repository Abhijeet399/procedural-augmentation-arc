"""
run_prototype_c.py — Prototype C: Orientation Selection + RCOS candidate ranking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pipeline:
  1. Orientation selection  (8 passes, Prototype A)
  2. Candidate generation   (greedy + beam + temperature sampling)
  3. RCOS scoring           (one forward pass per candidate)
  4. Select argmax(RCS) → inverse-transform → submit

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX v3 — generation prompt (root cause of 0% oracle rate)
  The original code re-encoded manually-transformed GridPairs via
  _encode_full_pair(), then passed dihedral_id=best_d to the model.
  This applied the spatial transform TWICE:
    1. Physically: apply_dihedral_transform(grid, best_d) → canonical grid
    2. Via model's dihedral embedding: dihedral_ids=best_d → second transform

  Prototype A works because it uses tokens_by_dihedral[best_d] directly
  (pre-computed canonical tokens) and the dihedral embedding provides
  contextual conditioning — not a second physical transform.

  Fix: pass test_seq_ex (raw SequenceExample) to generate_all_candidates
  so the prompt is built from tokens_by_dihedral[best_d], matching exactly
  how Prototype A builds its generation prompt.

FIX v3 — RCOS demo pairs
  For CE scoring, demo GridPairs are now extracted from each demo
  example's tokens_by_dihedral[best_d] (same canonical token space
  as generation) rather than re-encoded from transformed grids.

FIX v2 — solutions / oracle
  Test examples have output=None in their token stream.  solutions.json
  is loaded at startup; the ground-truth grid is transformed into
  canonical space for oracle comparison.

FIX — arc_score
  Previously mixed pair-level correct count with task-level total.
  Now counts at task level: a task is correct iff ALL its pairs match.

Usage:
    python run_prototype_c.py \\
        --checkpoint  runs/tiny.pt \\
        --data-path   assets/challenges.json \\
        --solutions   assets/solutions.json \\
        --output-dir  runs/prototype_c

Quick smoke-test:
    python run_prototype_c.py \\
        --checkpoint  runs/tiny.pt \\
        --data-path   assets/challenges.json \\
        --solutions   assets/solutions.json \\
        --output-dir  runs/prototype_c_debug \\
        --no-beam --no-sample --max-tasks 20
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch

# ── path setup ─────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC  = _ROOT / "src"
for _p in [str(_ROOT), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build import build_model_and_data, load_checkpoint
from common import (
    apply_dihedral_transform,
    apply_inverse_dihedral_transform,
    split_grids_from_tokens,
)
from canonicalize import score_task_orientations
from rcos import (
    GridPair,
    compute_baseline_ce,
    compute_rcs_diagnostics,
    generate_all_candidates,
    grids_equal,
    rank_candidates_by_rcs,
)


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prototype C: Orientation Selection + RCOS"
    )
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data-path",    required=True)
    p.add_argument("--solutions",    default=None)
    p.add_argument("--output-dir",   default="runs/prototype_c")
    p.add_argument("--device",       default="cuda")

    p.add_argument("--no-greedy",    action="store_true")
    p.add_argument("--beam-width",   type=int,   default=10)
    p.add_argument("--no-beam",      action="store_true")
    p.add_argument("--n-sample",     type=int,   default=20)
    p.add_argument("--temps",        type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",        type=int,   default=None)
    p.add_argument("--no-sample",    action="store_true")

    p.add_argument("--max-tasks",    type=int,   default=None)
    p.add_argument("--task-id",      default=None)
    return p.parse_args()


def _args_for_build(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_path=Path(args.checkpoint),
        data_path=Path(args.data_path),
        seed=42,
        batch_size=1,
        device=args.device,
        enable_aug=False,
        enable_color_aug=False,
        enable_dihedral_aug=False,
        max_augments=0,
        color_apply_to_test=False,
        dihedral_apply_to_test=False,
    )


# =============================================================================
# Solutions loader
# =============================================================================

def load_solutions(solutions_path: Optional[str]) -> Dict[str, List[List[List[int]]]]:
    """
    Load solutions.json → {task_id: [grid_per_test_pair, …]} in original orientation.
    """
    if solutions_path is None:
        return {}
    path = Path(solutions_path)
    if not path.exists():
        print(f"  [WARNING] solutions.json not found at {path}.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    result: Dict[str, List[List[List[int]]]] = {}
    for task_id, grids in raw.items():
        # Wrap single-grid tasks so the format is always list-of-grids
        if grids and not isinstance(grids[0][0], list):
            grids = [grids]
        result[task_id] = grids
    return result


# =============================================================================
# Grid helpers
# =============================================================================

def _grids_from_seq_ex(ex, best_d: int) -> GridPair:
    """
    Extract a GridPair from a SequenceExample using tokens_by_dihedral[best_d].

    This is the canonical representation the model was trained to see for
    orientation best_d.  Using these tokens avoids re-encoding from scratch
    (which can produce subtly different sequences) and avoids the
    double-transform bug (physical transform + dihedral embedding).
    """
    if ex.tokens_by_dihedral is not None:
        toks = ex.tokens_by_dihedral[best_d].tolist()
    else:
        toks = ex.tokens.tolist()
    grids = split_grids_from_tokens(toks)
    inp = grids[0] if grids else []
    out = grids[1] if len(grids) > 1 else None
    return GridPair(input=inp, output=out)


def invert_d(grid: List[List[int]], d: int) -> List[List[int]]:
    return apply_inverse_dihedral_transform(grid, d)


# =============================================================================
# ARC-style scoring  (task-level: ALL pairs must match)
# =============================================================================

def arc_score(submission: Dict, solutions_path: str) -> Tuple[int, int]:
    """Return (n_correct_tasks, n_total_tasks).  A task is correct iff every
    test pair has at least one attempt that exactly matches ground truth."""
    with open(solutions_path) as f:
        solutions = json.load(f)

    total   = len(submission)
    correct = 0

    for task_id, attempt_list in submission.items():
        gt_list  = solutions.get(task_id, [])
        task_ok  = True
        for pair_idx, pair_attempts in enumerate(attempt_list):
            gt = gt_list[pair_idx] if pair_idx < len(gt_list) else None
            if gt is None:
                continue          # no GT → skip, don't penalise
            pair_ok = any(att == gt
                          for att in pair_attempts.get("attempts", []))
            if not pair_ok:
                task_ok = False
                break
        if task_ok:
            correct += 1

    return correct, total


# =============================================================================
# Per-task evaluation
# =============================================================================

def evaluate_task(
    model,
    dataset,
    task_id: str,
    device,
    args: argparse.Namespace,
    solutions: Dict[str, List[List[List[int]]]],
) -> Dict:
    """Run the full Prototype C pipeline on one task."""

    example_id   = dataset.task_id_to_example_id[task_id]
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs = [ex for ex in dataset.iter_examples(split="test")
                    if ex.task_id == task_id]

    if not demo_seq_exs or not test_seq_exs:
        return {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

    print(f"  demo={len(demo_seq_exs)} test={len(test_seq_exs)}")

    # ── Step 1: Orientation selection ─────────────────────────────────────────
    t0 = perf_counter()
    best_d, orient_losses = score_task_orientations(model, demo_seq_exs, device)
    t_orient = perf_counter() - t0
    _dnames = ['identity','rot90','rot180','rot270',
               'flip_h','flip_v','flip_main','flip_anti']
    print(f"  orientation: d{best_d} ({_dnames[best_d]})  "
          f"CE={orient_losses[best_d]:.4f}  ({t_orient:.1f}s)")

    # ── Step 2: Canonical demo GridPairs from tokens_by_dihedral[best_d] ──────
    #
    # KEY FIX: extract GridPairs from each demo's pre-computed canonical tokens
    # rather than re-encoding manually-transformed grids.  This guarantees the
    # token sequences fed to the model for RCOS CE scoring are in exactly the
    # same canonical representation that the generation prompt uses.
    #
    demo_pairs: List[GridPair] = [
        _grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs
    ]
    # Keep only demo pairs that have a valid output (sanity guard)
    demo_pairs = [gp for gp in demo_pairs if gp.output is not None]

    solution_grids_original = solutions.get(task_id, [])

    # ── Step 3–5: Per test pair ────────────────────────────────────────────────
    predicted_grids: List[Tuple[int, List[List[int]]]] = []
    all_diagnostics: List[Dict] = []

    for pair_idx, test_seq_ex in enumerate(test_seq_exs):
        t_pair = perf_counter()

        # Canonical test input extracted from tokens_by_dihedral[best_d]
        test_gp            = _grids_from_seq_ex(test_seq_ex, best_d)
        canonical_test_inp = test_gp.input

        # Ground truth in canonical space for oracle comparison
        gt_original = (solution_grids_original[pair_idx]
                       if pair_idx < len(solution_grids_original) else None)
        gt_canonical = (apply_dihedral_transform(gt_original, best_d)
                        if gt_original is not None else None)

        # ── Baseline CE ────────────────────────────────────────────────────────
        t_base = perf_counter()
        baseline_ce = compute_baseline_ce(
            model, demo_pairs, example_id, best_d, device
        )
        t_base = perf_counter() - t_base
        print(f"  [pair {pair_idx}] baseline_CE={baseline_ce:.4f}  ({t_base:.1f}s)")

        # ── Candidate generation ────────────────────────────────────────────────
        # FIX: pass test_seq_ex + demo_seq_exs so _build_test_prompt uses
        # tokens_by_dihedral[best_d] directly — identical to Prototype A's prompt.
        t_gen = perf_counter()
        candidates = generate_all_candidates(

            model, demo_pairs, canonical_test_inp,
            example_id, best_d, device,
            use_greedy   = not args.no_greedy,
            use_beam     = not args.no_beam and args.beam_width > 0,
            beam_width   = args.beam_width,
            use_sample   = not args.no_sample and args.n_sample > 0,
            n_per_temperature = args.n_sample,
            temperatures = tuple(args.temps),
            top_k        = args.top_k,
            # ← new kwargs wired through to _build_test_prompt
            test_seq_ex  = test_seq_ex,
            demo_seq_exs = demo_seq_exs,
        )
        t_gen = perf_counter() - t_gen
        print(f"  [pair {pair_idx}] {len(candidates)} unique candidates  ({t_gen:.1f}s gen)")

        # Capture greedy BEFORE RCOS re-ranks the list.
        # generate_all_candidates always puts greedy first (rcos.py line 697).
        greedy_cand = candidates[0] if candidates else []

        # ── RCOS scoring ────────────────────────────────────────────────────────
        t_score = perf_counter()
        ranked, _ = rank_candidates_by_rcs(
            model, demo_pairs, canonical_test_inp, candidates,
            example_id, best_d, device, baseline_ce=baseline_ce,
        )
        t_score = perf_counter() - t_score
        print(f"  [pair {pair_idx}] RCOS scored {len(ranked)} candidates  ({t_score:.1f}s)")

        if ranked:
            cand_grids = [r[0] for r in ranked]
            rcs_scores = [r[1] for r in ranked]
            aug_ces    = [r[2] for r in ranked]
        else:
            cand_grids, rcs_scores, aug_ces = [], [], []

        # Diagnostics — canonical candidates vs canonical gt
        diag = compute_rcs_diagnostics(
            cand_grids, rcs_scores, aug_ces, gt_canonical, baseline_ce
        )
        diag.update({
            "task_id":       task_id,
            "pair_idx":      pair_idx,
            "best_d":        best_d,
            "orient_losses": [round(x, 6) for x in orient_losses],
            "t_orient_s":    round(t_orient, 3),
            "t_gen_s":       round(t_gen, 3),
            "t_score_s":     round(t_score, 3),
        })
        all_diagnostics.append(diag)

        hit  = "✓ oracle" if diag["oracle_hit"] else "✗ oracle"
        sel  = "✓ rcs"   if diag["rcs_selects_correct"] else "✗ rcs"
        rank = diag["rcs_rank_of_correct"]
        rcs1 = diag["rcs_score_top1"]
        gap  = diag["score_gap"]
        rcs1_str = f"{rcs1:.4f}" if rcs1 is not None else "nan"
        print(f"  [{hit}] [{sel}]  rank={rank}  rcs_top1={rcs1_str}"
              + (f"  gap={gap:.4f}" if gap is not None else ""))

        # Option 2 (fixed): attempt_1 = greedy (Prototype A quality, ~27.8%),
        # attempt_2 = RCOS top-1 (adds bonus wins where greedy is wrong but
        # RCOS happens to rank the correct candidate first).
        # greedy_cand was captured before RCOS ranking above.
        attempt_1 = invert_d(greedy_cand, best_d) if greedy_cand else []
        attempt_2 = invert_d(cand_grids[0], best_d) if cand_grids else attempt_1
        predicted_grids.append((pair_idx, attempt_1, attempt_2))
        print(f"  pair done in {perf_counter() - t_pair:.1f}s\n")

    return {
        "task_id":         task_id,
        "predicted_grids": predicted_grids,
        "diagnostics":     all_diagnostics,
    }


# =============================================================================
# Summary printer
# =============================================================================

def print_summary(all_diag: List[Dict], solutions_path: Optional[str],
                  submission: Dict):
    n_pairs = len(all_diag)
    if n_pairs == 0:
        print("No diagnostics to summarise.")
        return

    n_tasks     = len({d["task_id"] for d in all_diag})
    oracle_n    = sum(1 for d in all_diag if d["oracle_hit"])
    rcs_n       = sum(1 for d in all_diag if d["rcs_selects_correct"])
    oracle_rate = oracle_n / n_pairs
    rcs_rate    = rcs_n   / n_pairs
    lift        = rcs_rate / oracle_rate if oracle_rate > 0 else 0.0

    cat_a = [d for d in all_diag if not d["oracle_hit"]]
    cat_b = [d for d in all_diag if d["oracle_hit"] and not d["rcs_selects_correct"]]
    cat_c = [d for d in all_diag if d["rcs_selects_correct"]]

    rank_counts: Dict[int, int] = defaultdict(int)
    for d in all_diag:
        if d["oracle_hit"] and d["rcs_rank_of_correct"] is not None:
            rank_counts[d["rcs_rank_of_correct"]] += 1
    oracle_total = sum(1 for d in all_diag if d["oracle_hit"])

    n_cands = [d["n_candidates"] for d in all_diag]

    print("\n" + "=" * 70)
    print("PROTOTYPE C — EXPERIMENT 1 SUMMARY")
    print("=" * 70)
    print(f"  Tasks evaluated      : {n_tasks}")
    print(f"  Test pairs           : {n_pairs}")
    print()
    print(f"  Oracle accuracy      : {oracle_n}/{n_pairs} = {oracle_rate*100:.1f}%")
    print(f"  RCOS accuracy        : {rcs_n}/{n_pairs} = {rcs_rate*100:.1f}%")
    print(f"  RCOS / oracle lift   : {lift*100:.1f}%")
    print()
    print("  Failure modes:")
    print(f"    A. Generation failure (correct ∉ candidates) : "
          f"{len(cat_a):4d}  ({len(cat_a)/n_pairs*100:.1f}%)")
    print(f"    B. Scoring failure    (in set, mis-ranked)   : "
          f"{len(cat_b):4d}  ({len(cat_b)/n_pairs*100:.1f}%)")
    print(f"    C. RCOS correct                              : "
          f"{len(cat_c):4d}  ({len(cat_c)/n_pairs*100:.1f}%)")
    print()
    print("  Rank distribution (oracle-hit tasks only):")
    for rank in sorted(rank_counts):
        cnt = rank_counts[rank]
        bar = "█" * min(cnt, 40)
        pct = cnt / max(oracle_total, 1) * 100
        print(f"    rank {rank:3d}: {cnt:4d}  {bar:40s} {pct:5.1f}%")
    print()

    gaps_right = [d["score_gap"] for d in cat_c if d["score_gap"] is not None]
    gaps_wrong  = [d["score_gap"] for d in cat_b if d["score_gap"] is not None]
    if gaps_right:
        print(f"  Avg score gap (correct selected) : "
              f"{sum(gaps_right)/len(gaps_right):+.4f}")
    if gaps_wrong:
        print(f"  Avg score gap (mis-ranked)       : "
              f"{sum(gaps_wrong)/len(gaps_wrong):+.4f}")
    print()
    print(f"  Avg candidates / pair: {sum(n_cands)/len(n_cands):.1f}"
          f"  (min={min(n_cands)} max={max(n_cands)})")
    print()

    score = None
    if solutions_path:
        correct, total = arc_score(submission, solutions_path)
        score = correct / max(total, 1)
        print(f"  Official ARC score   : {correct}/{total} = {score*100:.2f}%")
        print()

    print("  Comparison:")
    print(f"    Prototype A (orientation only)  : 27.8%")
    if score is not None:
        print(f"    Prototype C (+ RCOS, this run)  : {score*100:.1f}%")
        delta = score * 100 - 27.8
        sign  = "+" if delta >= 0 else ""
        print(f"    Δ                               : {sign}{delta:.1f}pp")
    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("PROTOTYPE C — Orientation Selection + RCOS")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Data        : {args.data_path}")
    print(f"  Greedy      : {not args.no_greedy}")
    print(f"  Beam width  : {args.beam_width}  (disabled={args.no_beam})")
    print(f"  Samples/temp: {args.n_sample}  temps={args.temps}"
          f"  (disabled={args.no_sample})")
    print("=" * 70 + "\n")

    solutions = load_solutions(args.solutions)
    if solutions:
        print(f"Loaded solutions for {len(solutions)} tasks from {args.solutions}")
    else:
        print("[WARNING] No solutions — oracle diagnostics disabled.")
    print()

    ckpt = load_checkpoint(Path(args.checkpoint))
    model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks, {len(dataset.examples)} examples\n")

    test_task_ids = sorted({ex.task_id for ex in dataset.iter_examples(split="test")})
    if args.task_id:
        test_task_ids = [args.task_id]
    elif args.max_tasks:
        test_task_ids = test_task_ids[:args.max_tasks]
    n_tasks = len(test_task_ids)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sub_dir = out_dir / "submission"
    sub_dir.mkdir(exist_ok=True)

    submission:      Dict        = {}
    all_diagnostics: List[Dict] = []
    t_total = perf_counter()

    for i, task_id in enumerate(test_task_ids, start=1):
        t_task = perf_counter()
        print(f"[{i}/{n_tasks}] {task_id}")
        try:
            result = evaluate_task(
                model, dataset, task_id, device, args, solutions=solutions
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            result = {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

        all_diagnostics.extend(result["diagnostics"])

        by_pair: Dict[int, Tuple] = {}
        for pair_idx, att1, att2 in result["predicted_grids"]:
            by_pair[pair_idx] = (att1, att2)
        task_sub = [{"attempts": list(by_pair[pi])} for pi in sorted(by_pair)]
        if task_sub:
            submission[task_id] = task_sub

        print(f"  task done in {perf_counter() - t_task:.1f}s\n")

    sub_path  = sub_dir / "submission.json"
    diag_path = out_dir / "diagnostics.json"
    with open(sub_path, "w")  as f: json.dump(submission,      f, indent=2)
    with open(diag_path, "w") as f: json.dump(all_diagnostics, f, indent=2)

    print(f"Submission  → {sub_path}")
    print(f"Diagnostics → {diag_path}")

    print_summary(all_diagnostics, args.solutions, submission)

    total = perf_counter() - t_total
    print(f"Total wall time : {total:.1f}s")
    print(f"Avg per task    : {total/max(n_tasks,1):.1f}s")


if __name__ == "__main__":
    main()
