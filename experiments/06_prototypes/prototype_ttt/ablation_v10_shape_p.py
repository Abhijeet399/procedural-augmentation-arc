"""
ablation_v10_shape_p.py — EXP P: Shape-Constrained Generation
=============================================================

Problem diagnosed from EXP C / EXP M:
  'fallback=all_filtered' — when the model generates ALL wrong-shape
  candidates, the current code dumps every wrong-shape candidate back
  to the transition ranker.  The ranker can't fix a shape it's never
  been designed to handle, so those tasks are effectively lost.

EXP P fix — Smart Fallback:
  When hard_filter rejects all candidates AND the shape rule is known
  (same / fixed / scaled), we reshape the greedy output to the expected
  (H, W) and use that as the single survivor instead.  This guarantees
  the ranker always receives at least one structurally valid candidate.

  When the shape rule is 'unknown' (no consistent demo-output shape),
  we fall back to the old behaviour (all unfiltered candidates) — we
  have no better information.

What this fixes:
  • Tasks where every beam/sample candidate has wrong shape
    → were guaranteed wrong before; now get the best available guess
  • Removes dependency on diversity_bonus fallback noise in these cases

What this does NOT fix:
  • Tasks where correct answer simply isn't in the model's distribution
    (oracle ceiling). That requires EXP Q (synthetic data fine-tuning).

Experiments:
  EXP K   No-TTT baseline  (reproduces EXP C)
  EXP P   No-TTT + smart fallback  (this experiment)

Usage:
    python prototype_ttt/ablation_v10_shape_p.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v10_shape_p

    # Quick smoke test:
        ... --max-tasks 20

    # Run only EXP P (skip baseline):
        ... --exp P
"""

import sys
import json
import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent.resolve()
# Insert in reverse priority order — last insert lands at sys.path[0] (highest priority)
# prototype_e/src goes in first (lowest priority among our additions)
# prototype_ttt/src goes in last  (highest priority — its candidate_filters overrides prototype_e's)
sys.path.insert(0, str(_HERE.parent / "prototype_e" / "src"))
sys.path.insert(0, str(_HERE.parent / "prototype_e"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE / "src"))  # ← wins: prototype_ttt/src searched first

from candidate_filters import (
    hard_filter,
    smart_fallback,
    infer_output_shape_rule,
    expected_output_shape,
    reshape_to_shape,
)
from ablation_v4 import (
    _grids_from_seq_ex,
    invert_d,
    grids_equal,
    apply_dihedral_transform,
    rank_by_transition,
    pick_best_non_greedy,
    pixel_dissimilarity,
    _grids_equal,
    GridPair,
)
from run_prototype_e import (
    load_solutions,
    load_checkpoint,
    build_model_and_data,
    _args_for_build,
    arc_score,
)
from rcos import generate_all_candidates
from canonicalize import score_task_orientations


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EXP P — shape-constrained generation")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-path",   required=True)
    p.add_argument("--solutions",   default=None)
    p.add_argument("--output-dir",  default="runs/ablation_v10_shape_p")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max-tasks",   type=int, default=None)
    p.add_argument("--task-id",     default=None)
    # Generation (EXP C best config)
    p.add_argument("--beam-width",  type=int,   default=14)
    p.add_argument("--n-sample",    type=int,   default=30)
    p.add_argument("--lambda",      type=float, default=0.3, dest="lam")
    p.add_argument("--temps",       type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",       type=int,   default=None)
    # Which experiments to run
    p.add_argument("--exp",         default=None,
                   help="Comma-separated subset to run: K,P  (default: both)")
    return p.parse_args()


# ─── Single-task evaluation ────────────────────────────────────────────────────

def evaluate_task_shape_p(
    model,
    dataset,
    task_id:         str,
    device,
    args,
    solutions:       Dict,
    use_smart_fallback: bool,
    diversity_lambda:   float = 0.3,
    beam_width:         int   = 14,
    n_sample:           int   = 30,
) -> Dict:
    """
    Like ablation_v4.evaluate_task but with the EXP P smart fallback.

    use_smart_fallback=False → EXP K (reproduces EXP C)
    use_smart_fallback=True  → EXP P (reshape greedy when all filtered)
    """
    example_id   = dataset.task_id_to_example_id[task_id]
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs = [ex for ex in dataset.iter_examples(split="test")
                    if ex.task_id == task_id]

    if not demo_seq_exs or not test_seq_exs:
        return {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

    best_d, _ = score_task_orientations(model, demo_seq_exs, device)

    demo_pairs     = [_grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs]
    demo_pairs     = [gp for gp in demo_pairs if gp.output is not None]
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
        candidates = generate_all_candidates(
            model, demo_pairs, canonical_test_inp,
            example_id, best_d, device,
            use_greedy        = True,
            use_beam          = beam_width > 0,
            beam_width        = beam_width,
            use_sample        = n_sample > 0,
            n_per_temperature = n_sample,
            temperatures      = tuple(args.temps),
            top_k             = args.top_k,
            test_seq_ex       = test_seq_ex,
            demo_seq_exs      = demo_seq_exs,
        )
        greedy_cand = candidates[0] if candidates else []

        # ── Filter ────────────────────────────────────────────────────────────
        survivors, filter_stats = hard_filter(
            candidates, canonical_test_inp, raw_demo_pairs
        )

        fallback_reason = None
        if not survivors:
            if use_smart_fallback:
                # ★ EXP P: reshape greedy to expected shape
                survivors, fallback_reason = smart_fallback(
                    candidates, greedy_cand, canonical_test_inp, raw_demo_pairs
                )
            else:
                # EXP K baseline: old behaviour
                survivors       = candidates[:]
                fallback_reason = "all_filtered"

        # ── Rank ──────────────────────────────────────────────────────────────
        scored_list, _ = rank_by_transition(
            survivors, canonical_test_inp, raw_demo_pairs,
            greedy_cand=greedy_cand, greedy_margin=0.5,
        )
        ranked_grids = [g for g, _ in scored_list]

        oracle_hit = (any(grids_equal(c, gt_canonical) for c in candidates)
                      if gt_canonical is not None else None)
        e_selects  = (grids_equal(ranked_grids[0], gt_canonical)
                      if ranked_grids and gt_canonical is not None else None)
        greedy_ok  = (grids_equal(greedy_cand, gt_canonical)
                      if greedy_cand and gt_canonical is not None else None)

        attempt_1 = invert_d(greedy_cand, best_d) if greedy_cand else []
        best_ng   = pick_best_non_greedy(
            ranked_grids, scored_list, greedy_cand, diversity_lambda,
            test_input=canonical_test_inp,
        )
        attempt_2 = invert_d(best_ng, best_d) if best_ng is not None else attempt_1
        predicted_grids.append((pair_idx, attempt_1, attempt_2))

        a2_correct = (grids_equal(best_ng, gt_canonical)
                      if best_ng is not None and gt_canonical is not None else None)
        src_label  = "non-greedy" if best_ng is not None else "same-as-greedy"

        n_shape_rej   = filter_stats.get("n_shape_rej",   0)
        n_palette_rej = filter_stats.get("n_palette_rej", 0)
        fb_str = f"  fallback={fallback_reason}" if fallback_reason else ""
        shape_rule = filter_stats.get("shape_rule", "?")

        print(f"  [pair {pair_idx}]"
              f"  oracle={'✓' if oracle_hit else '✗'}"
              f"  e_rank={'✓' if e_selects else '✗'}"
              f"  shape_rej={n_shape_rej}"
              f"  palette_rej={n_palette_rej}"
              f"  survivors={len(survivors)}"
              f"  rule={shape_rule}"
              + fb_str)
        print(f"  [pair {pair_idx}] src={src_label}"
              f"  a1={'✓' if greedy_ok else '✗'}"
              f"  a2={'✓' if a2_correct else '✗'}"
              f"  ({perf_counter()-t_pair:.1f}s)")

        all_diagnostics.append({
            "task_id":       task_id,
            "pair_idx":      pair_idx,
            "oracle_hit":    oracle_hit,
            "e_selects":     e_selects,
            "greedy_correct":greedy_ok,
            "a2_correct":    a2_correct,
            "n_shape_rej":   n_shape_rej,
            "n_palette_rej": n_palette_rej,
            "n_survivors":   len(survivors),
            "shape_rule":    shape_rule,
            "fallback":      fallback_reason,
        })

    return {"task_id": task_id, "predicted_grids": predicted_grids,
            "diagnostics": all_diagnostics}


# ─── Run one experiment ────────────────────────────────────────────────────────

def run_experiment(
    label:             str,
    model,
    dataset,
    device,
    args,
    solutions:         Dict,
    test_task_ids:     List[str],
    out_dir:           Path,
    use_smart_fallback: bool,
    diversity_lambda:  float = 0.3,
    beam_width:        int   = 14,
    n_sample:          int   = 30,
) -> Dict:
    print()
    print("=" * 70)
    print(f"EXPERIMENT: {label}")
    print(f"  beam={beam_width}  n_sample={n_sample}  λ={diversity_lambda}")
    print(f"  smart_fallback={'ENABLED' if use_smart_fallback else 'disabled'}")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    submission: Dict  = {}
    all_diag:   List  = []
    t_start = perf_counter()
    n_fallback_used = 0

    for i, task_id in enumerate(test_task_ids, 1):
        t_task = perf_counter()
        print(f"\n[{i}/{len(test_task_ids)}] {task_id}")
        try:
            result = evaluate_task_shape_p(
                model, dataset, task_id, device, args, solutions,
                use_smart_fallback=use_smart_fallback,
                diversity_lambda=diversity_lambda,
                beam_width=beam_width,
                n_sample=n_sample,
            )
        except Exception as exc:
            import traceback
            print(f"  ERROR: {exc}")
            traceback.print_exc()
            result = {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

        all_diag.extend(result["diagnostics"])
        by_pair = {pi: (a1, a2) for pi, a1, a2 in result["predicted_grids"]}
        task_sub = [{"attempts": list(by_pair[pi])} for pi in sorted(by_pair)]
        if task_sub:
            submission[task_id] = task_sub

        # Count smart fallback triggers
        for d in result["diagnostics"]:
            if d.get("fallback") == "reshaped_greedy":
                n_fallback_used += 1

        print(f"  task done in {perf_counter()-t_task:.1f}s")

    # Save
    sub_dir = out_dir / "submission"
    sub_dir.mkdir(exist_ok=True)
    with open(sub_dir / "submission.json", "w") as f:
        json.dump(submission, f, indent=2)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)

    # Score
    n_oracle  = sum(1 for d in all_diag if d.get("oracle_hit"))
    n_a2      = sum(1 for d in all_diag if d.get("a2_correct"))
    n_pairs   = len(all_diag)
    arc_correct = arc_pct = 0
    if args.solutions:
        arc_correct, arc_total = arc_score(submission, args.solutions)
        arc_pct = arc_correct / max(arc_total, 1) * 100

    elapsed = perf_counter() - t_start
    print(f"\n  Oracle         : {n_oracle}/{n_pairs} = {n_oracle/max(n_pairs,1)*100:.1f}%")
    print(f"  a2 hits        : {n_a2}/{n_pairs} = {n_a2/max(n_pairs,1)*100:.1f}%")
    print(f"  ARC            : {arc_correct}/{max(n_pairs,1)} = {arc_pct:.2f}%")
    if use_smart_fallback:
        print(f"  Reshaped fallbacks used: {n_fallback_used}")
    print(f"  Time           : {elapsed:.0f}s")
    print(f"\nSubmission  → {sub_dir/'submission.json'}")
    print(f"Diagnostics → {out_dir/'diagnostics.json'}")

    return {
        "label": label,
        "use_smart_fallback": use_smart_fallback,
        "n_pairs": n_pairs,
        "n_oracle": n_oracle,
        "n_a2_correct": n_a2,
        "arc_pct": arc_pct,
        "n_fallback_reshaped": n_fallback_used,
        "elapsed_s": round(elapsed),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("ABLATION v10 — EXP P: Shape-Constrained Generation")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print("=" * 70)

    solutions = load_solutions(args.solutions)
    print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt  = load_checkpoint(Path(args.checkpoint))
    model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks")

    test_task_ids = sorted({ex.task_id for ex in dataset.iter_examples(split="test")})
    if args.task_id:
        test_task_ids = [args.task_id]
    elif args.max_tasks:
        test_task_ids = test_task_ids[:args.max_tasks]

    EXPS = {
        "K": dict(label="K: No-TTT baseline (EXP C repro)", use_smart_fallback=False),
        "P": dict(label="P: Smart fallback (EXP P)",        use_smart_fallback=True),
    }

    if args.exp:
        keys = [k.upper() for k in args.exp.split(",")]
        EXPS = {k: v for k, v in EXPS.items() if k in keys}

    summaries = []
    for key, cfg in EXPS.items():
        exp_dir = Path(args.output_dir) / cfg["label"].replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        s = run_experiment(
            label              = cfg["label"],
            model              = model,
            dataset            = dataset,
            device             = device,
            args               = args,
            solutions          = solutions,
            test_task_ids      = test_task_ids,
            out_dir            = exp_dir,
            use_smart_fallback = cfg["use_smart_fallback"],
            diversity_lambda   = args.lam,
            beam_width         = args.beam_width,
            n_sample           = args.n_sample,
        )
        summaries.append(s)

    # ── Final comparison ──────────────────────────────────────────────────────
    EXP_C_REF = {
        "label": "C: EXP C reference (no TTT)",
        "n_pairs": 419, "n_oracle": 157, "n_a2_correct": 15, "arc_pct": 30.50,
        "n_fallback_reshaped": "—",
    }

    print()
    print("=" * 80)
    print("EXP P COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 80)
    hdr = f"{'Experiment':<40}  {'oracle':>8}  {'a2':>4}  {'ARC%':>6}  {'reshaped':>8}"
    print(hdr)
    print("-" * 80)
    for s in [EXP_C_REF] + summaries:
        n = s.get("n_pairs", 419)
        print(f"{s['label']:<40}  "
              f"{s['n_oracle']}/{n}  "
              f"{str(s.get('n_a2_correct','?')):>4}  "
              f"{s['arc_pct']:>5.2f}%  "
              f"{str(s.get('n_fallback_reshaped','—')):>8}")
    print("=" * 80)

    if summaries:
        best  = max(summaries, key=lambda s: s["arc_pct"])
        delta = best["arc_pct"] - 30.50
        sign  = "+" if delta >= 0 else ""
        print(f"\nBest : {best['label']}  ({best['arc_pct']:.2f}%)")
        print(f"Δ vs EXP C   : {sign}{delta:.2f}pp")
        print(f"Gap to Mithil: {44.0 - best['arc_pct']:.2f}pp")

    comp_path = Path(args.output_dir) / "comparison.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()
