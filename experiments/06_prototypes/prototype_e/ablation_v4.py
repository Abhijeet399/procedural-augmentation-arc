"""
ablation_v4.py — Three-way ablation on top of Prototype E v3 (29.5% baseline)

Runs three experiments back-to-back, sharing a single model load:

  EXP A  diversity bonus only          (beam=10, n_sample=20, lambda=0.3)
  EXP B  generation increase only      (beam=14, n_sample=30, lambda=0.0)
  EXP C  diversity bonus + bigger gen  (beam=14, n_sample=30, lambda=0.3)

Diversity bonus (λ > 0):
  For attempt_2 we want a candidate that is both plausible (high transition
  score) AND maximally different from greedy (maximises diversity of our two
  bets).  Combined score for non-greedy candidate c:

      score(c) = transition_score(c) + λ * dissimilarity(c, greedy)

  where dissimilarity = fraction of cells that differ from the greedy output.
  λ = 0 → pure transition ranking (current v3 behaviour).
  λ = 0.3 → moderate diversity push, keeping transition signal dominant.

Usage:
    python prototype_e/ablation_v4.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v4
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch

# ── path setup ───────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from candidate_filters import hard_filter
from transition_ranker  import rank_by_transition, transition_diagnostics, _grids_equal

# Import everything else from the main runner so we don't duplicate code
from run_prototype_e import (
    load_solutions,
    load_checkpoint,
    build_model_and_data,
    _args_for_build,
    score_task_orientations,
    generate_all_candidates,
    _grids_from_seq_ex,
    grids_equal,
    invert_d,
    apply_dihedral_transform,
    arc_score,
    GridPair,
)


# =============================================================================
# Diversity bonus: score for attempt_2 candidate selection
# =============================================================================

def pixel_dissimilarity(cand: List, greedy: List) -> float:
    """Fraction of cells that differ between cand and greedy (0..1)."""
    if not greedy or not cand:
        return 0.0
    total = sum(len(row) for row in greedy)
    if total == 0:
        return 0.0
    diff = 0
    for r in range(min(len(cand), len(greedy))):
        row_g = greedy[r]
        row_c = cand[r]
        for c in range(min(len(row_c), len(row_g))):
            if row_c[c] != row_g[c]:
                diff += 1
        # cells present in greedy but missing in cand count as different
        diff += max(0, len(row_g) - len(row_c))
    # rows present in greedy but missing in cand
    for r in range(len(cand), len(greedy)):
        diff += len(greedy[r])
    return diff / total


_PENALTY_THRESHOLD = -1e8   # legacy constant, kept for import compat


def _is_shape_mismatch(grid: List, test_input: List) -> bool:
    """True if candidate grid has different shape from test_input."""
    if not grid or not test_input:
        return False
    return len(grid) != len(test_input) or len(grid[0]) != len(test_input[0])


def pick_best_non_greedy(
    ranked_grids:    List,
    scored_pairs:    List[Tuple],   # (grid, transition_score)
    greedy_cand:     List,
    diversity_lambda: float,
    test_input:      Optional[List] = None,
) -> Optional[List]:
    """
    Return best non-greedy candidate using:
        combined_score = transition_score + λ * pixel_dissimilarity(c, greedy)

    ranked_grids: grids in transition-score order (for λ=0 fast path)
    scored_pairs: list of (grid, transition_score) in same order
    test_input:   used to detect shape-mismatch tasks

    Shape-mismatch tasks (output shape ≠ input shape):
        The new transition_ranker scores these via output-position frequency
        (log_prob_output_position) — a real, informative signal.
        Adding a diversity bonus on top of these scores introduces noise and
        hurts ranking.  We disable it when ALL non-greedy candidates are
        shape-mismatched relative to test_input.
    """
    if diversity_lambda == 0.0:
        return next((g for g in ranked_grids if not _grids_equal(g, greedy_cand)), None)

    # Detect shape-mismatch regime: disable diversity bonus
    if test_input is not None:
        non_greedy = [g for g, _ in scored_pairs if not _grids_equal(g, greedy_cand)]
        if non_greedy and all(_is_shape_mismatch(g, test_input) for g in non_greedy):
            # All candidates have a different shape — output-position scores are
            # already the best available signal; diversity bonus adds noise.
            return next((g for g in ranked_grids if not _grids_equal(g, greedy_cand)), None)

    # Normal path: diversity-boosted scoring
    best_score = None
    best_grid  = None
    for grid, t_score in scored_pairs:
        if _grids_equal(grid, greedy_cand):
            continue
        dis   = pixel_dissimilarity(grid, greedy_cand)
        score = t_score + diversity_lambda * dis
        if best_score is None or score > best_score:
            best_score = score
            best_grid  = grid
    return best_grid


# =============================================================================
# Per-task evaluation (extended to accept diversity_lambda + gen overrides)
# =============================================================================

def evaluate_task(
    model,
    dataset,
    task_id:          str,
    device,
    base_args:        argparse.Namespace,
    solutions:        Dict,
    diversity_lambda: float,
    beam_width:       int,
    n_sample:         int,
) -> Dict:
    example_id   = dataset.task_id_to_example_id[task_id]
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs = [ex for ex in dataset.iter_examples(split="test")
                    if ex.task_id == task_id]

    if not demo_seq_exs or not test_seq_exs:
        return {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

    # ── Orientation ──────────────────────────────────────────────────────────
    t0 = perf_counter()
    best_d, orient_losses = score_task_orientations(model, demo_seq_exs, device)
    t_orient = perf_counter() - t0

    demo_pairs = [_grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs]
    demo_pairs = [gp for gp in demo_pairs if gp.output is not None]
    raw_demo_pairs = [(gp.input, gp.output) for gp in demo_pairs]

    solution_grids_original = solutions.get(task_id, [])
    predicted_grids  = []
    all_diagnostics  = []

    for pair_idx, test_seq_ex in enumerate(test_seq_exs):
        t_pair = perf_counter()

        test_gp            = _grids_from_seq_ex(test_seq_ex, best_d)
        canonical_test_inp = test_gp.input

        gt_original  = (solution_grids_original[pair_idx]
                        if pair_idx < len(solution_grids_original) else None)
        gt_canonical = (apply_dihedral_transform(gt_original, best_d)
                        if gt_original is not None else None)

        # ── Generate ─────────────────────────────────────────────────────────
        t_gen = perf_counter()
        candidates = generate_all_candidates(
            model, demo_pairs, canonical_test_inp,
            example_id, best_d, device,
            use_greedy        = True,
            use_beam          = beam_width > 0,
            beam_width        = beam_width,
            use_sample        = n_sample > 0,
            n_per_temperature = n_sample,
            temperatures      = tuple(base_args.temps),
            top_k             = base_args.top_k,
            test_seq_ex       = test_seq_ex,
            demo_seq_exs      = demo_seq_exs,
        )
        t_gen = perf_counter() - t_gen
        greedy_cand = candidates[0] if candidates else []

        # ── Filter + transition rank ──────────────────────────────────────────
        t_rank = perf_counter()
        survivors, filter_stats = hard_filter(
            candidates, canonical_test_inp, raw_demo_pairs
        )
        if not survivors:
            survivors     = candidates[:]
            fallback_reason = "all_filtered"
        else:
            fallback_reason = None

        scored_list, _table = rank_by_transition(
            survivors, canonical_test_inp, raw_demo_pairs,
            greedy_cand   = greedy_cand,
            greedy_margin = 0.5,
        )
        ranked_grids = [g for g, _ in scored_list]
        t_rank = perf_counter() - t_rank

        # ── Oracle / correctness checks ───────────────────────────────────────
        oracle_hit = any(grids_equal(c, gt_canonical)
                         for c in candidates) if gt_canonical is not None else None
        e_selects  = (grids_equal(ranked_grids[0], gt_canonical)
                      if ranked_grids and gt_canonical is not None else None)
        greedy_ok  = (grids_equal(greedy_cand, gt_canonical)
                      if greedy_cand and gt_canonical is not None else None)

        # ── Attempt selection ─────────────────────────────────────────────────
        attempt_1      = invert_d(greedy_cand, best_d) if greedy_cand else []
        best_non_greedy = pick_best_non_greedy(
            ranked_grids, scored_list, greedy_cand, diversity_lambda,
            test_input=canonical_test_inp,
        )
        if best_non_greedy is not None:
            attempt_2 = invert_d(best_non_greedy, best_d)
        else:
            attempt_2 = attempt_1

        predicted_grids.append((pair_idx, attempt_1, attempt_2))

        a2_correct = (grids_equal(best_non_greedy, gt_canonical)
                      if best_non_greedy is not None and gt_canonical is not None else None)
        a2_label = "non-greedy" if best_non_greedy is not None else "same-as-greedy"

        n_shape_rej   = filter_stats.get("n_shape_rej",   0)
        n_palette_rej = filter_stats.get("n_palette_rej", 0)
        n_surv        = len(survivors)
        fb_str        = f"  fallback={fallback_reason}" if fallback_reason else ""
        oracle_sym    = "✓" if oracle_hit else ("✗" if oracle_hit is False else "?")
        e_sym         = "✓" if e_selects  else ("✗" if e_selects  is False else "?")
        print(f"  [pair {pair_idx}]"
              f"  oracle={oracle_sym}"
              f"  e_rank={e_sym}"
              f"  shape_rej={n_shape_rej}"
              f"  palette_rej={n_palette_rej}"
              f"  survivors={n_surv}"
              + fb_str
              + f"  ({t_rank:.2f}s rank)")
        print(f"  [pair {pair_idx}] src={a2_label}"
              f"  a1={'✓' if greedy_ok else '✗'}"
              f"  a2={'✓' if a2_correct else '✗'}"
              f"  ({perf_counter()-t_pair:.1f}s)")

        trans_agrees_g = (_grids_equal(ranked_grids[0], greedy_cand)
                         if ranked_grids else None)
        diag = {
            "task_id":            task_id,
            "pair_idx":           pair_idx,
            "best_d":             best_d,
            "n_candidates":       len(candidates),
            "oracle_hit":         oracle_hit,
            "e_selects_correct":  e_selects,
            "greedy_correct":     greedy_ok,
            "a2_correct":         a2_correct,
            "n_shape_rej":        n_shape_rej,
            "n_palette_rej":      n_palette_rej,
            "n_out":              n_surv,
            "fallback":           fallback_reason,
            "trans_agrees_greedy":trans_agrees_g,
            "t_orient_s":         round(t_orient, 3),
            "t_gen_s":            round(t_gen,    3),
            "t_rank_s":           round(t_rank,   4),
        }
        all_diagnostics.append(diag)

    return {
        "task_id":         task_id,
        "predicted_grids": predicted_grids,
        "diagnostics":     all_diagnostics,
    }


# =============================================================================
# Run one full experiment
# =============================================================================

def run_experiment(
    label:            str,
    model,
    dataset,
    device,
    base_args:        argparse.Namespace,
    solutions:        Dict,
    test_task_ids:    List[str],
    out_dir:          Path,
    diversity_lambda: float,
    beam_width:       int,
    n_sample:         int,
) -> Dict:
    print()
    print("=" * 70)
    print(f"EXPERIMENT: {label}")
    print(f"  beam={beam_width}  n_sample={n_sample}  diversity_lambda={diversity_lambda}")
    print("=" * 70)

    sub_dir = out_dir / "submission"
    sub_dir.mkdir(parents=True, exist_ok=True)

    submission:      Dict = {}
    all_diagnostics: List = []
    t_total = perf_counter()
    n_tasks = len(test_task_ids)

    for i, task_id in enumerate(test_task_ids, 1):
        t_task = perf_counter()
        print(f"[{i}/{n_tasks}] {task_id}")
        try:
            result = evaluate_task(
                model, dataset, task_id, device, base_args, solutions,
                diversity_lambda = diversity_lambda,
                beam_width       = beam_width,
                n_sample         = n_sample,
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            result = {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

        all_diagnostics.extend(result["diagnostics"])
        by_pair  = {pi: (a1, a2) for pi, a1, a2 in result["predicted_grids"]}
        task_sub = [{"attempts": list(by_pair[pi])} for pi in sorted(by_pair)]
        if task_sub:
            submission[task_id] = task_sub

        print(f"  task done in {perf_counter()-t_task:.1f}s\n")

    # Save
    sub_path  = sub_dir / "submission.json"
    diag_path = out_dir / "diagnostics.json"
    with open(sub_path,  "w") as f: json.dump(submission,      f, indent=2)
    with open(diag_path, "w") as f: json.dump(all_diagnostics, f, indent=2)
    print(f"Submission  → {sub_path}")
    print(f"Diagnostics → {diag_path}")

    # Collect summary stats
    n = len(all_diagnostics)
    n_oracle    = sum(1 for d in all_diagnostics if d.get("oracle_hit"))
    n_greedy    = sum(1 for d in all_diagnostics if d.get("greedy_correct"))
    n_e_correct = sum(1 for d in all_diagnostics if d.get("e_selects_correct"))
    n_a2_correct= sum(1 for d in all_diagnostics if d.get("a2_correct"))
    n_fallback  = sum(1 for d in all_diagnostics if d.get("fallback") == "all_filtered")

    arc_correct, arc_total = 0, n_tasks
    if base_args.solutions:
        arc_correct, arc_total = arc_score(submission, base_args.solutions)
    arc_pct = arc_correct / max(arc_total, 1) * 100

    elapsed = perf_counter() - t_total
    summary = {
        "label":            label,
        "beam_width":       beam_width,
        "n_sample":         n_sample,
        "diversity_lambda": diversity_lambda,
        "n_pairs":          n,
        "n_oracle":         n_oracle,
        "n_greedy":         n_greedy,
        "n_e_correct":      n_e_correct,
        "n_a2_correct":     n_a2_correct,
        "n_fallback":       n_fallback,
        "arc_correct":      arc_correct,
        "arc_total":        arc_total,
        "arc_pct":          arc_pct,
        "elapsed_s":        round(elapsed, 1),
    }

    print(f"\n  Oracle : {n_oracle}/{n} = {n_oracle/max(n,1)*100:.1f}%")
    print(f"  Greedy : {n_greedy}/{n} = {n_greedy/max(n,1)*100:.1f}%")
    print(f"  a2 hits: {n_a2_correct}/{n} = {n_a2_correct/max(n,1)*100:.1f}%")
    print(f"  ARC    : {arc_correct}/{arc_total} = {arc_pct:.2f}%")
    print(f"  Time   : {elapsed:.0f}s\n")

    return summary


# =============================================================================
# Comparison table
# =============================================================================

def print_comparison(summaries: List[Dict]) -> None:
    print()
    print("=" * 70)
    print("ABLATION COMPARISON")
    print("=" * 70)
    hdr = f"{'Experiment':<35}  {'beam':>4}  {'samp':>4}  {'λ':>4}  {'oracle':>6}  {'a2hits':>6}  {'ARC%':>6}"
    print(hdr)
    print("-" * 70)
    # Include v3 baseline for reference
    v3 = {"label": "v3 baseline (greedy+no-div)", "beam_width": 10, "n_sample": 20,
          "diversity_lambda": 0.0, "n_pairs": 419, "n_oracle": 153,
          "n_a2_correct": "~11", "arc_pct": 29.50}
    for s in [v3] + summaries:
        oracle_str  = f"{s['n_oracle']}/{s.get('n_pairs',419)}"
        a2_str      = str(s.get("n_a2_correct", "?"))
        arc_str     = f"{s['arc_pct']:.2f}%"
        lam_str     = f"{s['diversity_lambda']:.1f}" if isinstance(s['diversity_lambda'], float) else str(s['diversity_lambda'])
        print(f"{s['label']:<35}  {s['beam_width']:>4}  {s['n_sample']:>4}  "
              f"{lam_str:>4}  {oracle_str:>6}  {a2_str:>6}  {arc_str:>6}")
    print("=" * 70)

    best = max(summaries, key=lambda s: s["arc_pct"])
    print(f"\nBest experiment : {best['label']}  ({best['arc_pct']:.2f}%)")
    delta = best["arc_pct"] - 29.50
    sign  = "+" if delta >= 0 else ""
    print(f"Δ vs v3 baseline: {sign}{delta:.2f}pp")
    gap = 44.0 - best["arc_pct"]
    print(f"Gap to Mithil   : {gap:.1f}pp")


# =============================================================================
# Argument parsing + main
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-path",  required=True)
    p.add_argument("--solutions",  required=True)
    p.add_argument("--output-dir", default="runs/ablation_v4")
    p.add_argument("--max-tasks",  type=int, default=None)
    p.add_argument("--task-id",    default=None)
    # base generation defaults (used for EXP A; overridden per experiment)
    p.add_argument("--beam-width", type=int,   default=10)
    p.add_argument("--n-sample",   type=int,   default=20)
    p.add_argument("--temps",      type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",      type=int,   default=0)
    # unused by this script but needed by _args_for_build
    p.add_argument("--no-greedy",         action="store_true")
    p.add_argument("--no-beam",           action="store_true")
    p.add_argument("--no-sample",         action="store_true")
    p.add_argument("--no-shape-filter",   action="store_true")
    p.add_argument("--no-palette-filter", action="store_true")
    p.add_argument("--no-transition",     action="store_true")
    p.add_argument("--no-ttt",            action="store_true", default=True)
    p.add_argument("--device",            default=None)
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("ABLATION v4  — Diversity bonus × Generation increase")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Solutions  : {args.solutions}")
    print("=" * 70)

    solutions = load_solutions(args.solutions)
    print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt = load_checkpoint(Path(args.checkpoint))
    model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks")

    test_task_ids = sorted({ex.task_id for ex in dataset.iter_examples(split="test")})
    if args.task_id:
        test_task_ids = [args.task_id]
    elif args.max_tasks:
        test_task_ids = test_task_ids[:args.max_tasks]

    # ── Three experiments ────────────────────────────────────────────────────
    # EXP A: diversity bonus only (standard generation)
    # EXP B: bigger generation only (no diversity)
    # EXP C: both

    CONFIGS = [
        dict(label="A: diversity only (λ=0.3)",
             diversity_lambda=0.3,
             beam_width=10,
             n_sample=20),
        dict(label="B: bigger gen only (b14,s30)",
             diversity_lambda=0.0,
             beam_width=14,
             n_sample=30),
        dict(label="C: diversity + bigger gen",
             diversity_lambda=0.3,
             beam_width=14,
             n_sample=30),
    ]

    summaries = []
    for cfg in CONFIGS:
        exp_dir = Path(args.output_dir) / cfg["label"].replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", "").replace("=", "")
        s = run_experiment(
            label            = cfg["label"],
            model            = model,
            dataset          = dataset,
            device           = device,
            base_args        = args,
            solutions        = solutions,
            test_task_ids    = test_task_ids,
            out_dir          = exp_dir,
            diversity_lambda = cfg["diversity_lambda"],
            beam_width       = cfg["beam_width"],
            n_sample         = cfg["n_sample"],
        )
        summaries.append(s)

    # ── Final comparison ─────────────────────────────────────────────────────
    print_comparison(summaries)

    # Save comparison JSON
    comp_path = Path(args.output_dir) / "comparison.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()
