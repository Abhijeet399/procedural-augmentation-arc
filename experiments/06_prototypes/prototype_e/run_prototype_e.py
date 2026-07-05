"""
run_prototype_e.py — Prototype E: Shape+Palette Filter + Transition Ranking

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MOTIVATION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prototype C v9e analysis:
  Oracle (correct answer in pool): 36.5%  (153/419)
  Official score:                  29.0%  (121/419)
  Gap (correct in pool but not selected): ~7.5pp = ~31 pairs

RCOS fails here because the model memorised all eval tasks →
  baseline CE ≈ 0 → RCOS score differences are noise.

Prototype E replaces RCOS with THREE model-free filters/rankers:

  1. SHAPE FILTER (hard reject)
     Infer expected output shape from demo pairs (same/fixed/scaled/unknown).
     Hard-discard any candidate with the wrong shape.
     Covers: ALL tasks where output shape is predictable from demos.

  2. COLOR PALETTE FILTER (hard reject)
     Infer allowed output colors from demo outputs.
     Hard-discard candidates containing colors not seen in any demo output.
     Covers: recoloring, object tasks, many structural tasks.

  3. PIXEL-TRANSITION RANKER (soft ranking)
     Build (src→dst) color transition frequencies from demos.
     Score each surviving candidate by log-likelihood of its transitions.
     Covers: any task with a consistent color-mapping rule.

Pipeline per task:
  1. Orientation selection          (same as v9e)
  2. Generate candidates            (same as v9e: greedy + beam + sampling)
  3. E-rank: hard filter → transition score → pick top-1
  4. attempt_1 = greedy, attempt_2 = E-ranker top-1

Defaults match v9e exactly:
  --beam-width 10 --n-sample 20 --temps 0.7 1.0 --no-ttt

Run:
    python prototype_e/run_prototype_e.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/prototype_e_v1
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

import torch

# ── path setup ──────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent
_REPO_ROOT = _ROOT.parent.parent.parent   # repo root
_SRC_E = _ROOT / "src"
_SRC_CORE = _REPO_ROOT / "src"            # base pipeline's src/ folder

for _p in [str(_ROOT), str(_SRC_E), str(_SRC_CORE), str(_REPO_ROOT)]:
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
    generate_all_candidates,
    grids_equal,
)
# E-ranker modules (in prototype_e/src/)
from e_ranker import e_rank


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prototype E: Shape+Palette Filter + Transition Ranking"
    )
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-path",   required=True)
    p.add_argument("--solutions",   default=None)
    p.add_argument("--output-dir",  default="runs/prototype_e_v1")
    p.add_argument("--device",      default="cuda")

    # generation (match v9e defaults)
    p.add_argument("--no-greedy",   action="store_true")
    p.add_argument("--beam-width",  type=int,   default=10)
    p.add_argument("--no-beam",     action="store_true")
    p.add_argument("--n-sample",    type=int,   default=20)
    p.add_argument("--temps",       type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",       type=int,   default=None)
    p.add_argument("--no-sample",   action="store_true")

    # E-ranker options
    p.add_argument(
        "--no-shape-filter",
        action="store_true",
        help="Disable shape consistency filter (ablation).",
    )
    p.add_argument(
        "--no-palette-filter",
        action="store_true",
        help="Disable color palette filter (ablation).",
    )
    p.add_argument(
        "--no-transition",
        action="store_true",
        help="Disable transition ranking; use greedy order (ablation).",
    )

    p.add_argument("--max-tasks",   type=int,  default=None)
    p.add_argument("--task-id",     default=None)
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
# Helpers
# =============================================================================

def load_solutions(path: Optional[str]) -> Dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    result = {}
    for tid, grids in raw.items():
        if grids and not isinstance(grids[0][0], list):
            grids = [grids]
        result[tid] = grids
    return result


def _grids_from_seq_ex(ex, best_d: int) -> GridPair:
    if ex.tokens_by_dihedral is not None:
        toks = ex.tokens_by_dihedral[best_d].tolist()
    else:
        toks = ex.tokens.tolist()
    grids = split_grids_from_tokens(toks)
    inp = grids[0] if grids else []
    out = grids[1] if len(grids) > 1 else None
    return GridPair(input=inp, output=out)


def invert_d(grid, d: int):
    return apply_inverse_dihedral_transform(grid, d)


def arc_score(submission: Dict, solutions_path: str) -> Tuple[int, int]:
    p = Path(solutions_path)
    if not p.exists():
        return 0, 0
    with open(p) as f:
        solutions = json.load(f)
    correct, total = 0, 0
    for tid, sol_grids in solutions.items():
        total += 1
        if tid not in submission:
            continue
        if isinstance(sol_grids[0][0], int):
            sol_grids = [sol_grids]
        task_ok = True
        for pi, (sol, sub_pair) in enumerate(zip(sol_grids, submission[tid])):
            attempts = sub_pair.get("attempts", [])
            if not any(grids_equal(a, sol) for a in attempts):
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
    solutions: Dict,
) -> Dict:
    example_id   = dataset.task_id_to_example_id[task_id]
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs = [ex for ex in dataset.iter_examples(split="test")
                    if ex.task_id == task_id]

    if not demo_seq_exs or not test_seq_exs:
        return {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

    print(f"  demo={len(demo_seq_exs)} test={len(test_seq_exs)}")

    # ── Orientation selection ─────────────────────────────────────────────────
    t0 = perf_counter()
    best_d, orient_losses = score_task_orientations(model, demo_seq_exs, device)
    t_orient = perf_counter() - t0
    print(f"  orientation: d{best_d}  CE={orient_losses[best_d]:.4f}  ({t_orient:.1f}s)")

    # ── Canonical demo GridPairs ───────────────────────────────────────────────
    demo_pairs: List[GridPair] = [
        _grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs
    ]
    demo_pairs = [gp for gp in demo_pairs if gp.output is not None]

    # Raw demo pairs as plain lists (for E-ranker which doesn't use GridPair)
    raw_demo_pairs = [(gp.input, gp.output) for gp in demo_pairs]

    solution_grids_original = solutions.get(task_id, [])
    predicted_grids = []
    all_diagnostics = []

    for pair_idx, test_seq_ex in enumerate(test_seq_exs):
        t_pair = perf_counter()

        test_gp            = _grids_from_seq_ex(test_seq_ex, best_d)
        canonical_test_inp = test_gp.input

        gt_original  = (solution_grids_original[pair_idx]
                        if pair_idx < len(solution_grids_original) else None)
        gt_canonical = (apply_dihedral_transform(gt_original, best_d)
                        if gt_original is not None else None)

        # ── Generate candidates ───────────────────────────────────────────────
        t_gen = perf_counter()
        candidates = generate_all_candidates(
            model, demo_pairs, canonical_test_inp,
            example_id, best_d, device,
            use_greedy        = not args.no_greedy,
            use_beam          = not args.no_beam and args.beam_width > 0,
            beam_width        = args.beam_width,
            use_sample        = not args.no_sample and args.n_sample > 0,
            n_per_temperature = args.n_sample,
            temperatures      = tuple(args.temps),
            top_k             = args.top_k,
            test_seq_ex       = test_seq_ex,
            demo_seq_exs      = demo_seq_exs,
        )
        t_gen = perf_counter() - t_gen
        greedy_cand = candidates[0] if candidates else []
        print(f"  [pair {pair_idx}] {len(candidates)} candidates  ({t_gen:.1f}s gen)")

        # ── E-Ranking ─────────────────────────────────────────────────────────
        t_rank = perf_counter()

        if args.no_transition and args.no_shape_filter and args.no_palette_filter:
            # Full ablation: just use greedy order
            ranked_grids = candidates[:]
            e_diag = {"n_in": len(candidates), "n_out": len(candidates),
                      "fallback": "ablation_all_disabled"}
        else:
            # Apply filters / ranking (handle ablation flags)
            from candidate_filters import hard_filter
            from transition_ranker import rank_by_transition, transition_diagnostics

            # Stage 1: Hard filters
            if args.no_shape_filter and args.no_palette_filter:
                survivors = candidates[:]
                filter_stats = {"n_in": len(candidates), "n_out": len(candidates),
                                "n_shape_rej": 0, "n_palette_rej": 0}
            elif args.no_shape_filter:
                from candidate_filters import infer_allowed_colors, filter_by_palette
                allowed = infer_allowed_colors(raw_demo_pairs, canonical_test_inp)
                survivors, n_pal = filter_by_palette(candidates, allowed)
                filter_stats = {"n_in": len(candidates), "n_out": len(survivors),
                                "n_shape_rej": 0, "n_palette_rej": n_pal}
            elif args.no_palette_filter:
                from candidate_filters import filter_by_shape
                survivors, n_shp = filter_by_shape(candidates, canonical_test_inp, raw_demo_pairs)
                filter_stats = {"n_in": len(candidates), "n_out": len(survivors),
                                "n_shape_rej": n_shp, "n_palette_rej": 0}
            else:
                survivors, filter_stats = hard_filter(candidates, canonical_test_inp, raw_demo_pairs)

            fallback_reason = None
            if not survivors:
                fallback_reason = "all_filtered"
                survivors = [greedy_cand] if greedy_cand else candidates[:1]

            # Stage 2: Transition ranking
            if args.no_transition:
                scored_grids = survivors
                trans_diag = transition_diagnostics(
                    [(g, 0.0) for g in survivors], gt_canonical
                )
            else:
                scored, _ = rank_by_transition(
                    survivors, canonical_test_inp, raw_demo_pairs,
                    greedy_cand=greedy_cand,
                )
                scored_grids = [g for g, _ in scored]
                trans_diag = transition_diagnostics(scored, gt_canonical)

            e_diag = {}
            e_diag.update(filter_stats)
            e_diag.update(trans_diag)
            e_diag["fallback"] = fallback_reason
            from transition_ranker import _grids_equal
            e_diag["trans_agrees_greedy"] = (
                _grids_equal(scored_grids[0], greedy_cand) if scored_grids else None
            )
            ranked_grids = scored_grids

        t_rank = perf_counter() - t_rank

        # ── Oracle check ─────────────────────────────────────────────────────
        oracle_hit = any(grids_equal(c, gt_canonical)
                         for c in candidates) if gt_canonical is not None else None
        e_selects  = (grids_equal(ranked_grids[0], gt_canonical)
                      if ranked_grids and gt_canonical is not None else None)
        greedy_ok  = (grids_equal(greedy_cand, gt_canonical)
                      if greedy_cand and gt_canonical is not None else None)

        n_shape_rej   = e_diag.get("n_shape_rej",   0)
        n_palette_rej = e_diag.get("n_palette_rej", 0)
        n_out         = e_diag.get("n_out",          len(candidates))
        fb            = e_diag.get("fallback", None)

        oracle_sym = "✓" if oracle_hit else ("✗" if oracle_hit is False else "?")
        e_sym      = "✓" if e_selects  else ("✗" if e_selects  is False else "?")

        print(f"  [pair {pair_idx}]"
              f"  oracle={oracle_sym}"
              f"  e_rank={e_sym}"
              f"  shape_rej={n_shape_rej}"
              f"  palette_rej={n_palette_rej}"
              f"  survivors={n_out}"
              + (f"  fallback={fb}" if fb else "")
              + f"  ({t_rank:.2f}s rank)")

        diag = {
            "task_id":       task_id,
            "pair_idx":      pair_idx,
            "best_d":        best_d,
            "n_candidates":  len(candidates),
            "oracle_hit":    oracle_hit,
            "e_selects_correct":  e_selects,
            "greedy_correct":     greedy_ok,
            "t_orient_s":    round(t_orient, 3),
            "t_gen_s":       round(t_gen, 3),
            "t_rank_s":      round(t_rank, 4),
        }
        diag.update(e_diag)
        all_diagnostics.append(diag)

        # attempt_1 = greedy (always)
        # attempt_2 = BEST NON-GREEDY candidate by transition score.
        #   This maximises coverage: a1 covers greedy-correct cases,
        #   a2 independently covers cases where a non-greedy candidate is correct.
        #   Both pointing at the same grid wastes the second attempt slot.
        attempt_1 = invert_d(greedy_cand, best_d) if greedy_cand else []

        from transition_ranker import _grids_equal as _ge
        best_non_greedy = next(
            (g for g in ranked_grids if not _ge(g, greedy_cand)),
            None
        )
        if best_non_greedy is not None:
            attempt_2 = invert_d(best_non_greedy, best_d)
        else:
            attempt_2 = attempt_1  # all candidates identical to greedy
        predicted_grids.append((pair_idx, attempt_1, attempt_2))

        a2_is_non_greedy = best_non_greedy is not None
        a2_correct = (grids_equal(best_non_greedy, gt_canonical)
                      if best_non_greedy is not None and gt_canonical is not None else None)
        src_label = "non-greedy" if a2_is_non_greedy else "same-as-greedy"
        print(f"  [pair {pair_idx}] src={src_label}  a1={'✓' if greedy_ok else '✗'}  a2={'✓' if a2_correct else '✗'}  ({perf_counter()-t_pair:.1f}s)")

    return {
        "task_id":         task_id,
        "predicted_grids": predicted_grids,
        "diagnostics":     all_diagnostics,
    }


# =============================================================================
# Summary
# =============================================================================

def print_summary(
    all_diag: List[Dict],
    solutions_path: Optional[str],
    submission: Dict,
) -> None:
    n = len(all_diag)
    if n == 0:
        print("No diagnostics.")
        return

    n_tasks        = len({d["task_id"] for d in all_diag})
    n_oracle       = sum(1 for d in all_diag if d.get("oracle_hit"))
    n_e_correct    = sum(1 for d in all_diag if d.get("e_selects_correct"))
    n_greedy       = sum(1 for d in all_diag if d.get("greedy_correct"))

    # Filtering stats
    n_shape_rej    = sum(d.get("n_shape_rej",   0) for d in all_diag)
    n_palette_rej  = sum(d.get("n_palette_rej", 0) for d in all_diag)
    n_fallback     = sum(1 for d in all_diag if d.get("fallback") == "all_filtered")
    n_agrees_greedy= sum(1 for d in all_diag if d.get("trans_agrees_greedy") is True)

    # Recovery: oracle but not greedy, but e_rank correct
    n_recovered = sum(
        1 for d in all_diag
        if d.get("oracle_hit") and not d.get("greedy_correct") and d.get("e_selects_correct")
    )
    # Regressions: greedy correct but e_rank wrong
    n_regressed = sum(
        1 for d in all_diag
        if d.get("greedy_correct") and not d.get("e_selects_correct")
    )

    print("\n" + "=" * 70)
    print("PROTOTYPE E — SUMMARY")
    print("=" * 70)
    print(f"  Tasks evaluated      : {n_tasks}")
    print(f"  Test pairs           : {n}")
    print()
    print(f"  Oracle accuracy      : {n_oracle}/{n}  = {n_oracle/n*100:.1f}%")
    print(f"  Greedy accuracy      : {n_greedy}/{n} = {n_greedy/n*100:.1f}%")
    print(f"  E-ranker accuracy    : {n_e_correct}/{n} = {n_e_correct/n*100:.1f}%")
    print()
    print(f"  Recoveries  (greedy✗, e_rank✓) : {n_recovered}")
    print(f"  Regressions (greedy✓, e_rank✗) : {n_regressed}")
    print(f"  Net gain vs greedy             : {n_recovered - n_regressed:+d} pairs")
    print()
    print(f"  Filter stats:")
    print(f"    Shape-rejected candidates   : {n_shape_rej}")
    print(f"    Palette-rejected candidates : {n_palette_rej}")
    print(f"    Fallbacks (all filtered)    : {n_fallback}")
    print(f"    Trans agrees greedy         : {n_agrees_greedy}/{n}")
    print()

    if solutions_path:
        correct, total = arc_score(submission, solutions_path)
        score = correct / max(total, 1)
        print(f"  Official ARC score   : {correct}/{total} = {score*100:.2f}%")
        print()
        print(f"  Comparison:")
        print(f"    Prototype A (greedy)   : 27.8%")
        print(f"    Prototype C v9e        : 29.0%  ← previous best")
        print(f"    Prototype E (this)     : {score*100:.2f}%")
        delta = score * 100 - 29.0
        sign  = "+" if delta >= 0 else ""
        print(f"    Δ vs v9e               : {sign}{delta:.2f}pp")
        gap   = 44.0 - score * 100
        if gap <= 0:
            print(f"    ✓ BEATS Mithil 44% baseline!")
        else:
            print(f"    Still {gap:.1f}pp below Mithil (44%)")
    print("=" * 70 + "\n")


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    print("=" * 70)
    print("PROTOTYPE E — Shape+Palette Filter + Pixel-Transition Ranking")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Data        : {args.data_path}")
    print(f"  Beam width  : {args.beam_width}   (disabled={args.no_beam})")
    print(f"  Samples/temp: {args.n_sample}  temps={args.temps}")
    print(f"  Shape filter   : {'OFF' if args.no_shape_filter   else 'ON'}")
    print(f"  Palette filter : {'OFF' if args.no_palette_filter else 'ON'}")
    print(f"  Transition rank: {'OFF' if args.no_transition     else 'ON'}")
    print("=" * 70 + "\n")

    solutions = load_solutions(args.solutions)
    if solutions:
        print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt = load_checkpoint(Path(args.checkpoint))
    model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks\n")

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

    submission: Dict      = {}
    all_diagnostics: List = []
    t_total = perf_counter()

    for i, task_id in enumerate(test_task_ids, 1):
        t_task = perf_counter()
        print(f"[{i}/{n_tasks}] {task_id}")
        try:
            result = evaluate_task(model, dataset, task_id, device, args, solutions)
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            result = {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

        all_diagnostics.extend(result["diagnostics"])
        by_pair = {pi: (a1, a2) for pi, a1, a2 in result["predicted_grids"]}
        task_sub = [{"attempts": list(by_pair[pi])} for pi in sorted(by_pair)]
        if task_sub:
            submission[task_id] = task_sub

        print(f"  task done in {perf_counter()-t_task:.1f}s\n")

    sub_path  = sub_dir / "submission.json"
    diag_path = out_dir / "diagnostics.json"
    with open(sub_path,  "w") as f: json.dump(submission,      f, indent=2)
    with open(diag_path, "w") as f: json.dump(all_diagnostics, f, indent=2)

    print(f"Submission  → {sub_path}")
    print(f"Diagnostics → {diag_path}")

    print_summary(all_diagnostics, args.solutions, submission)

    total = perf_counter() - t_total
    hours = int(total // 3600)
    mins  = int((total % 3600) // 60)
    secs  = total % 60
    print(f"Total wall time : {total:.1f}s ({hours:02d}h{mins:02d}m{secs:04.1f}s)")
    print(f"Avg per task    : {total/max(n_tasks,1):.1f}s")


if __name__ == "__main__":
    main()
