"""
run_prototype_c.py — Prototype C v9: Orientation Selection + LoRA-TTT + RCOS

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VERSION HISTORY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

v8  27.0%   attempt_1=greedy (pre-RCOS), attempt_2=RCOS top-1
            Prototype A quality restored.  RCOS signal still weak because
            baseline_CE≈0 for all eval tasks (model memorised them).

v9.0  28.2%  TTT ran but loss=0.0000→0.0000 throughout.
              example_embedding was memorised → CE=0 → gradient=0 → LoRA
              learned nothing.

v9.1  target ≥ 38%
    FIX: zero example_embedding[example_id] before TTT starts.
    This breaks memorisation so loss > 0 and gradients actually flow.
    The embedding row is co-optimised with LoRA from a zero init.
    Adapted embedding + LoRA are kept for generation/RCOS scoring.
    Original embedding restored via restore_ttt_embedding() after task.

    Pipeline per task:
      1. Orientation selection           (unchanged from v8)
      2. apply_lora()                    inject trainable LoRA adapters
      3. run_ttt()                       ZEROS emb, trains LoRA+emb K steps
      4. compute_baseline_ce()           now > 0  →  RCOS signal restored
      5. generate_all_candidates()       adapted model → better candidates
      6. rank_candidates_by_rcs()        RCOS scoring with real signal
      7. remove_lora()                   restore attention layers
      8. restore_ttt_embedding()         restore original embedding

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY CLI FLAGS (new in v9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  --no-ttt              disable TTT (runs v8 behaviour)
  --ttt-steps   30      gradient steps per task (try 20–50)
  --ttt-lr      3e-4    AdamW learning rate  (try 1e-4 – 5e-4)
  --ttt-rank    8       LoRA rank r          (try 4, 8, 16)
  --ttt-alpha   16      LoRA alpha           (scale = alpha / rank)
  --ttt-verbose         print per-step TTT losses

Recommended full run:
    python run_prototype_c.py \\
        --checkpoint  runs/tiny.pt \\
        --data-path   assets/challenges.json \\
        --solutions   assets/solutions.json \\
        --no-sample \\
        --beam-width  10 \\
        --ttt-steps   30 \\
        --output-dir  runs/prototype_c_v9

Quick debug (20 tasks):
    python run_prototype_c.py \\
        --checkpoint  runs/tiny.pt \\
        --data-path   assets/challenges.json \\
        --solutions   assets/solutions.json \\
        --no-beam --no-sample \\
        --max-tasks   20 \\
        --ttt-steps   20 \\
        --ttt-verbose \\
        --output-dir  runs/prototype_c_v9_debug
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
    compute_baseline_ce_dihedral_avg,
    compute_rcs_diagnostics,
    generate_all_candidates,
    grids_equal,
    rank_candidates_by_rcs,
)
from lora_ttt import apply_lora, remove_lora, run_ttt, lora_summary


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prototype C v9: Orientation Selection + LoRA-TTT + RCOS"
    )
    # ── data / model ──────────────────────────────────────────────────────────
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--data-path",    required=True)
    p.add_argument("--solutions",    default=None)
    p.add_argument("--output-dir",   default="runs/prototype_c_v9")
    p.add_argument("--device",       default="cuda")

    # ── generation ────────────────────────────────────────────────────────────
    p.add_argument("--no-greedy",    action="store_true")
    p.add_argument("--beam-width",   type=int,   default=10)
    p.add_argument("--no-beam",      action="store_true")
    p.add_argument("--n-sample",     type=int,   default=20)
    p.add_argument("--temps",        type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",        type=int,   default=None)
    p.add_argument("--no-sample",    action="store_true")

    # ── LoRA TTT (NEW in v9) ──────────────────────────────────────────────────
    p.add_argument(
        "--no-ttt",
        action="store_true",
        help="Disable LoRA TTT — runs v8 behaviour (useful for ablation).",
    )
    p.add_argument(
        "--ttt-steps",
        type=int, default=30,
        help="Number of gradient steps per task (default: 30).",
    )
    p.add_argument(
        "--ttt-lr",
        type=float, default=3e-4,
        help="AdamW learning rate for TTT (default: 3e-4).",
    )
    p.add_argument(
        "--ttt-rank",
        type=int, default=8,
        help="LoRA rank r (default: 8).  Higher = more capacity, slower.",
    )
    p.add_argument(
        "--ttt-alpha",
        type=float, default=16.0,
        help="LoRA alpha.  Effective scale = alpha / rank (default: 16 → scale=2).",
    )
    p.add_argument(
        "--ttt-verbose",
        action="store_true",
        help="Print per-step TTT losses (useful for debugging).",
    )

    # ── misc ──────────────────────────────────────────────────────────────────
    p.add_argument("--max-tasks",    type=int,   default=None)
    p.add_argument("--task-id",      default=None)
    p.add_argument(
        "--dihedral-avg",
        action="store_true",
        help="Average RCOS CE over all 8 dihedral orientations (8× scoring "
             "cost, lower variance, recovers signal from overflow tasks).",
    )
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

def load_solutions(
    solutions_path: Optional[str],
) -> Dict[str, List[List[List[int]]]]:
    """Load solutions.json → {task_id: [grid_per_test_pair, …]}."""
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
    orientation best_d.
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
# ARC-style scoring
# =============================================================================

def arc_score(
    submission: Dict,
    solutions_path: str,
) -> Tuple[int, int]:
    """
    Count how many tasks are fully correct (ALL test pairs match).
    Each task has 2 attempts; a task is correct if attempt_1 OR attempt_2
    matches the ground truth for every test pair.
    """
    path = Path(solutions_path)
    if not path.exists():
        return 0, 0
    with open(path) as f:
        solutions = json.load(f)

    correct = 0
    total   = 0
    for task_id, sol_grids in solutions.items():
        total += 1
        if task_id not in submission:
            continue
        sub_pairs = submission[task_id]          # list of {"attempts": [a1, a2]}
        if isinstance(sol_grids[0][0], int):
            sol_grids = [sol_grids]              # single pair → wrap
        task_ok = True
        for pi, (sol_grid, sub_pair) in enumerate(zip(sol_grids, sub_pairs)):
            attempts = sub_pair.get("attempts", [])
            if not any(grids_equal(a, sol_grid) for a in attempts):
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
    task_id:   str,
    device,
    args:      argparse.Namespace,
    solutions: Dict[str, List[List[List[int]]]],
) -> Dict:
    """
    Run the full Prototype C v9 pipeline on one task.

    v9 adds LoRA TTT between orientation selection and generation.
    The TTT call wraps the entire candidate-generation + RCOS-scoring block.
    """

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
    _dnames = ['identity', 'rot90', 'rot180', 'rot270',
               'flip_h', 'flip_v', 'flip_main', 'flip_anti']
    print(f"  orientation: d{best_d} ({_dnames[best_d]})  "
          f"CE={orient_losses[best_d]:.4f}  ({t_orient:.1f}s)")

    # ── Step 2: Canonical demo GridPairs ──────────────────────────────────────
    demo_pairs: List[GridPair] = [
        _grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs
    ]
    demo_pairs = [gp for gp in demo_pairs if gp.output is not None]

    solution_grids_original = solutions.get(task_id, [])

    # ── Step 3: LoRA TTT  (NEW in v9) ─────────────────────────────────────────
    #
    # WHY HERE: After orientation selection (we need best_d for the token
    # sequences) but before generation and scoring (the adapted model must
    # be used for both).  A single TTT run per task adapts the model for
    # all test pairs in this task.
    #
    lora_state = None
    ttt_stats  = None

    if not args.no_ttt:
        t_ttt = perf_counter()
        lora_state = apply_lora(model, rank=args.ttt_rank, alpha=args.ttt_alpha,
                               example_id=example_id)   # v2: zero memorised embedding
        ttt_stats = run_ttt(
            model,
            demo_pairs    = demo_pairs,
            example_id    = example_id,
            dihedral_id   = best_d,
            device        = device,
            n_steps       = args.ttt_steps,
            lr            = args.ttt_lr,
            verbose       = args.ttt_verbose,
        )
        t_ttt = perf_counter() - t_ttt

        if ttt_stats["skipped"]:
            # No usable demo pairs — abort TTT and restore immediately
            remove_lora(model, lora_state)
            lora_state = None
            print(f"  TTT: skipped (no usable demo pairs)")
        else:
            l0 = ttt_stats['loss_start']
            l1 = ttt_stats['loss_end']
            n  = ttt_stats['n_examples']
            np = ttt_stats['n_lora_params']
            l0_s = f"{l0:.4f}" if l0 is not None else "—"
            l1_s = f"{l1:.4f}" if l1 is not None else "—"
            print(f"  TTT: {args.ttt_steps} steps  {n} demo pairs  "
                  f"loss {l0_s}→{l1_s}  "
                  f"+{np:,} LoRA params  ({t_ttt:.1f}s)")

    # ── Step 4–7: Per test pair ────────────────────────────────────────────────
    predicted_grids: List[Tuple[int, List[List[int]], List[List[int]]]] = []
    all_diagnostics: List[Dict] = []

    for pair_idx, test_seq_ex in enumerate(test_seq_exs):
        t_pair = perf_counter()

        # Canonical test input
        test_gp            = _grids_from_seq_ex(test_seq_ex, best_d)
        canonical_test_inp = test_gp.input

        # Ground truth in canonical space for oracle comparison
        gt_original  = (solution_grids_original[pair_idx]
                        if pair_idx < len(solution_grids_original) else None)
        gt_canonical = (apply_dihedral_transform(gt_original, best_d)
                        if gt_original is not None else None)

        # ── Baseline CE ────────────────────────────────────────────────────────
        # With TTT: the model is adapted so baseline_CE should be > 0.
        # Without TTT: baseline_CE ≈ 0 for memorised eval tasks.
        t_base = perf_counter()
        if args.dihedral_avg:
            baseline_ce = compute_baseline_ce_dihedral_avg(
                model, demo_pairs, example_id, device
            )
        else:
            baseline_ce = compute_baseline_ce(
                model, demo_pairs, example_id, best_d, device
            )
        t_base = perf_counter() - t_base
        print(f"  [pair {pair_idx}] baseline_CE={baseline_ce:.4f}  ({t_base:.1f}s)"
              + ("  [8-orientation avg]" if args.dihedral_avg else ""))

        # ── Candidate generation ────────────────────────────────────────────────
        t_gen = perf_counter()
        candidates = generate_all_candidates(
            model, demo_pairs, canonical_test_inp,
            example_id, best_d, device,
            use_greedy           = not args.no_greedy,
            use_beam             = not args.no_beam and args.beam_width > 0,
            beam_width           = args.beam_width,
            use_sample           = not args.no_sample and args.n_sample > 0,
            n_per_temperature    = args.n_sample,
            temperatures         = tuple(args.temps),
            top_k                = args.top_k,
            test_seq_ex          = test_seq_ex,
            demo_seq_exs         = demo_seq_exs,
        )
        t_gen = perf_counter() - t_gen
        print(f"  [pair {pair_idx}] {len(candidates)} unique candidates  ({t_gen:.1f}s gen)")

        # Capture greedy BEFORE RCOS re-ranks the list.
        greedy_cand = candidates[0] if candidates else []

        # ── RCOS scoring ────────────────────────────────────────────────────────
        t_score = perf_counter()
        ranked, _ = rank_candidates_by_rcs(
            model, demo_pairs, canonical_test_inp, candidates,
            example_id, best_d, device, baseline_ce=baseline_ce,
            dihedral_avg=args.dihedral_avg,
        )
        t_score = perf_counter() - t_score
        print(f"  [pair {pair_idx}] RCOS scored {len(ranked)} candidates  ({t_score:.1f}s)")

        if ranked:
            cand_grids = [r[0] for r in ranked]
            rcs_scores = [r[1] for r in ranked]
            aug_ces    = [r[2] for r in ranked]
        else:
            cand_grids, rcs_scores, aug_ces = [], [], []

        # ── Diagnostics ────────────────────────────────────────────────────────
        diag = compute_rcs_diagnostics(
            cand_grids, rcs_scores, aug_ces, gt_canonical, baseline_ce
        )
        diag.update({
            "task_id":        task_id,
            "pair_idx":       pair_idx,
            "best_d":         best_d,
            "orient_losses":  [round(x, 6) for x in orient_losses],
            "t_orient_s":     round(t_orient, 3),
            "t_gen_s":        round(t_gen, 3),
            "t_score_s":      round(t_score, 3),
            # TTT diagnostics (None when --no-ttt)
            "ttt_loss_start": ttt_stats["loss_start"]  if ttt_stats else None,
            "ttt_loss_end":   ttt_stats["loss_end"]    if ttt_stats else None,
            "ttt_n_examples": ttt_stats["n_examples"]  if ttt_stats else None,
        })
        all_diagnostics.append(diag)

        hit  = "✓ oracle" if diag["oracle_hit"]          else "✗ oracle"
        sel  = "✓ rcs"   if diag["rcs_selects_correct"]  else "✗ rcs"
        rank = diag["rcs_rank_of_correct"]
        rcs1 = diag["rcs_score_top1"]
        gap  = diag["score_gap"]
        rcs1_str = f"{rcs1:.4f}" if rcs1 is not None else "nan"
        print(f"  [{hit}] [{sel}]  rank={rank}  rcs_top1={rcs1_str}"
              + (f"  gap={gap:.4f}" if gap is not None else ""))

        # attempt_1 = greedy  (≈ Prototype A quality)
        # attempt_2 = RCOS top-1, but ONLY when the RCOS signal is trustworthy.
        #
        # Signal is considered dead / unreliable when:
        #   (a) baseline_CE < 0.1  — model still has memorised task context,
        #       so CE reductions are meaningless (near-zero floor).
        #   (b) >50% of scored candidates have a -inf RCS score — most augmented
        #       sequences overflowed the context window; ranking is noise.
        #       NOTE: compute_rcs() maps all NaN → -inf before returning, so
        #       the overflow signal lives in -inf, NOT in actual NaN values.
        #       The original `s != s` NaN test was a dead condition; we now
        #       use math.isinf(s) and s < 0 to catch -inf correctly.
        #
        # In either case fall back to greedy so we don't swap a correct greedy
        # answer for a randomly-ordered RCOS pick.
        _nan_frac = (sum(1 for s in rcs_scores if math.isinf(s) and s < 0) / len(rcs_scores)
                     if rcs_scores else 1.0)   # -inf = aug CE overflowed (NaN→-inf in compute_rcs)
        _rcos_reliable = (baseline_ce is not None
                          and baseline_ce >= 0.1
                          and _nan_frac <= 0.5)

        attempt_1 = invert_d(greedy_cand,  best_d) if greedy_cand  else []
        if _rcos_reliable and cand_grids:
            attempt_2 = invert_d(cand_grids[0], best_d)
        else:
            reason = (f"baseline_CE={baseline_ce:.4f}<0.1" if baseline_ce is not None and baseline_ce < 0.1
                      else f"inf_frac={_nan_frac:.0%}>50%" if _nan_frac > 0.5
                      else "no candidates")
            print(f"  [pair {pair_idx}] RCOS skipped ({reason}) → greedy fallback")
            attempt_2 = attempt_1   # fall back to greedy
        predicted_grids.append((pair_idx, attempt_1, attempt_2))
        print(f"  pair done in {perf_counter() - t_pair:.1f}s\n")

    # ── Step 8: Remove LoRA before returning ──────────────────────────────────
    if lora_state is not None:
        remove_lora(model, lora_state)

    return {
        "task_id":         task_id,
        "predicted_grids": predicted_grids,
        "diagnostics":     all_diagnostics,
    }


# =============================================================================
# Summary printer
# =============================================================================

def print_summary(
    all_diag:       List[Dict],
    solutions_path: Optional[str],
    submission:     Dict,
) -> None:
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

    # TTT stats
    ttt_losses_start = [d["ttt_loss_start"] for d in all_diag
                        if d.get("ttt_loss_start") is not None]
    ttt_losses_end   = [d["ttt_loss_end"] for d in all_diag
                        if d.get("ttt_loss_end") is not None]

    print("\n" + "=" * 70)
    print("PROTOTYPE C v9 — EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"  Tasks evaluated      : {n_tasks}")
    print(f"  Test pairs           : {n_pairs}")
    print()
    print(f"  Oracle accuracy      : {oracle_n}/{n_pairs} = {oracle_rate*100:.1f}%")
    print(f"  RCOS accuracy        : {rcs_n}/{n_pairs} = {rcs_rate*100:.1f}%")
    print(f"  RCOS / oracle lift   : {lift*100:.1f}%")
    print()
    if ttt_losses_start:
        avg_start = sum(ttt_losses_start) / len(ttt_losses_start)
        avg_end   = sum(ttt_losses_end)   / len(ttt_losses_end)
        print(f"  TTT loss (avg)       : {avg_start:.4f} → {avg_end:.4f}")
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
    gaps_wrong = [d["score_gap"] for d in cat_b if d["score_gap"] is not None]
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
    print(f"    Prototype A  (orientation only)   : 27.8%")
    print(f"    Prototype C v8 (greedy+RCOS)       : 27.0%")
    print(f"    Prototype C v9.0 (TTT, emb bug)   : 28.2%")
    if score is not None:
        print(f"    Prototype C v9.1 (+ emb-zero TTT)  : {score*100:.1f}%")
        delta = score * 100 - 27.8
        sign  = "+" if delta >= 0 else ""
        print(f"    Δ vs Prototype A                    : {sign}{delta:.1f}pp")
        delta_v8 = score * 100 - 27.0
        sign_v8  = "+" if delta_v8 >= 0 else ""
        print(f"    Δ vs v8                             : {sign_v8}{delta_v8:.1f}pp")
        if score * 100 >= 44.0:
            print(f"    ✓ BEATS Mithil baseline (44%)")
        else:
            gap_to_mithil = 44.0 - score * 100
            print(f"    Still {gap_to_mithil:.1f}pp below Mithil (44%)")
    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    ttt_str = (
        f"  LoRA-TTT: ENABLED  rank={args.ttt_rank}  alpha={args.ttt_alpha}"
        f"  steps={args.ttt_steps}  lr={args.ttt_lr}"
        if not args.no_ttt
        else "  LoRA-TTT: DISABLED (--no-ttt)"
    )

    print("=" * 70)
    print("PROTOTYPE C v9.1 — Orientation + LoRA-TTT (emb-zeroed) + RCOS")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Data        : {args.data_path}")
    print(f"  Greedy      : {not args.no_greedy}")
    print(f"  Beam width  : {args.beam_width}  (disabled={args.no_beam})")
    print(f"  Samples/temp: {args.n_sample}  temps={args.temps}"
          f"  (disabled={args.no_sample})")
    print(f"  Dihedral-avg RCOS: {'ENABLED (8× scoring)' if args.dihedral_avg else 'DISABLED'}")
    print(ttt_str)
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
    print(f"Dataset: {len(dataset.task_ids)} tasks, {len(dataset.examples)} examples")

    # Print LoRA layer summary before starting (informational)
    if not args.no_ttt:
        _probe_state = apply_lora(model, rank=args.ttt_rank, alpha=args.ttt_alpha)
        print(f"\nLoRA layer targets (rank={args.ttt_rank}):")
        print(lora_summary(model))
        remove_lora(model, _probe_state)
    print()

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

    submission:      Dict       = {}
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
            # Safety cleanup: remove any orphaned LoRA layers and restore embedding
            from lora_ttt import LoRALinear
            lora_mods = [(n, m) for n, m in model.named_modules()
                         if isinstance(m, LoRALinear)]
            if lora_mods:
                print(f"  [safety] removing {len(lora_mods)} orphaned LoRA layers")
                for _name, _lmod in lora_mods:
                    parts  = _name.split(".")
                    parent = model
                    for part in parts[:-1]:
                        parent = getattr(parent, part)
                    setattr(parent, parts[-1], _lmod.wrapped)
                for p in model.parameters():
                    p.requires_grad_(True)
                model.eval()
            # If lora_state exists, remove_lora will restore the embedding too
            if lora_state is not None:
                try:
                    remove_lora(model, lora_state)
                    lora_state = None
                except Exception:
                    pass
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
    with open(sub_path,  "w") as f: json.dump(submission,      f, indent=2)
    with open(diag_path, "w") as f: json.dump(all_diagnostics, f, indent=2)

    print(f"Submission  → {sub_path}")
    print(f"Diagnostics → {diag_path}")

    print_summary(all_diagnostics, args.solutions, submission)

    total = perf_counter() - t_total
    print(f"Total wall time : {total:.1f}s")
    print(f"Avg per task    : {total/max(n_tasks,1):.1f}s")


if __name__ == "__main__":
    main()
