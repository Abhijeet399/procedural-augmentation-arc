"""
ablation_v9_ttt.py — Test-Time Training ablation

Tests TTT at different step counts against the EXP C (no-TTT) baseline.
All experiments use beam=14, n_sample=30, lambda=0.3 (best config from EXP C).

TTT approach:
  For each eval task, make a copy of the model and fine-tune it for N gradient
  steps on the task's demo pairs (CE loss on output tokens only).  Then use
  the TTT model to generate candidates.  The base model is never modified.

This raises:
  1. Oracle ceiling — more correct answers appear in the candidate pool
  2. Greedy/beam quality — the TTT model generates the right answer more often

Experiments:
  EXP K  No TTT (reproduces EXP C at beam=14, n_sample=30, λ=0.3)
  EXP L  TTT 10 steps  (fast, lower quality)
  EXP M  TTT 30 steps  (recommended)
  EXP N  TTT 50 steps  (slower, may overfit)

Usage:
    python prototype_e/ablation_v9_ttt.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v9_ttt

    # Run only one experiment (e.g. just EXP M):
        ... --exp M

    # Quick smoke test on 5 tasks:
        ... --max-tasks 5
"""

import sys
import json
import copy
import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parent / "prototype_e"))
sys.path.insert(0, str(_HERE.parent / "prototype_e" / "src"))

from ablation_v4 import (
    parse_args as _base_parse_args,
    evaluate_task,
    _grids_from_seq_ex,
    invert_d,
    grids_equal,
    apply_dihedral_transform,
    rank_by_transition,
    pick_best_non_greedy,
    hard_filter,
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
from ttt import ttt_finetune, TttConfig


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="TTT ablation")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-path",   required=True)
    p.add_argument("--solutions",   default=None)
    p.add_argument("--output-dir",  default="runs/ablation_v9_ttt")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max-tasks",   type=int, default=None)
    p.add_argument("--task-id",     default=None)
    # Generation (fixed at EXP C best)
    p.add_argument("--beam-width",  type=int,   default=14)
    p.add_argument("--n-sample",    type=int,   default=30)
    p.add_argument("--temps",       type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",       type=int,   default=None)
    # TTT control
    p.add_argument("--exp",         default=None, help="Run only one exp: K/L/M/N")
    p.add_argument("--ttt-lr",      type=float, default=2e-4)
    p.add_argument("--ttt-layers",  type=int,   default=None,
                   help="Fine-tune only last N layers. None=all layers.")
    return p.parse_args()


# ─── Single-task TTT evaluation ───────────────────────────────────────────────

def evaluate_task_with_ttt(
    base_model,
    dataset,
    task_id:          str,
    device,
    args,
    solutions:        Dict,
    ttt_cfg:          Optional[TttConfig],   # None = no TTT
    diversity_lambda: float = 0.3,
    beam_width:       int   = 14,
    n_sample:         int   = 30,
) -> Dict:
    """
    Like ablation_v4.evaluate_task, but optionally runs TTT before generation.
    """
    example_id   = dataset.task_id_to_example_id[task_id]
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs = [ex for ex in dataset.iter_examples(split="test")
                    if ex.task_id == task_id]

    if not demo_seq_exs or not test_seq_exs:
        return {"task_id": task_id, "predicted_grids": [], "diagnostics": []}

    # ── Orientation (use base model — not TTT-adapted) ────────────────────────
    best_d, _ = score_task_orientations(base_model, demo_seq_exs, device)

    demo_pairs     = [_grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs]
    demo_pairs     = [gp for gp in demo_pairs if gp.output is not None]
    raw_demo_pairs = [(gp.input, gp.output) for gp in demo_pairs]

    # ── TTT fine-tuning ───────────────────────────────────────────────────────
    t_ttt = perf_counter()
    if ttt_cfg is not None:
        gen_model = ttt_finetune(
            base_model, demo_pairs, example_id, best_d, device, ttt_cfg,
            verbose=False,
        )
    else:
        gen_model = base_model
    t_ttt = perf_counter() - t_ttt

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

        # ── Generate with TTT model ───────────────────────────────────────────
        candidates = generate_all_candidates(
            gen_model, demo_pairs, canonical_test_inp,
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

        # ── Filter + rank (base model transition scores) ─────────────────────
        survivors, filter_stats = hard_filter(
            candidates, canonical_test_inp, raw_demo_pairs
        )
        if not survivors:
            survivors = candidates[:]
            fallback_reason = "all_filtered"
        else:
            fallback_reason = None

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
        fb = fallback_reason or ""

        print(f"  [pair {pair_idx}]"
              f"  oracle={'✓' if oracle_hit else '✗'}"
              f"  e_rank={'✓' if e_selects else '✗'}"
              f"  shape_rej={n_shape_rej}"
              f"  palette_rej={n_palette_rej}"
              f"  survivors={len(survivors)}"
              + (f"  fallback={fb}" if fb else "")
              + f"  (ttt={t_ttt:.1f}s)")

        print(f"  [pair {pair_idx}] src={src_label}"
              f"  a1={'✓' if greedy_ok else '✗'}"
              f"  a2={'✓' if a2_correct else '✗'}"
              f"  ({perf_counter()-t_pair:.1f}s)")

        all_diagnostics.append({
            "task_id": task_id, "pair_idx": pair_idx,
            "oracle_hit": oracle_hit, "e_selects_correct": e_selects,
            "greedy_correct": greedy_ok, "a2_correct": a2_correct,
            "n_shape_rej": n_shape_rej, "n_palette_rej": n_palette_rej,
            "ttt_s": round(t_ttt, 2),
        })

    # Free TTT model from GPU
    if ttt_cfg is not None:
        del gen_model
        if hasattr(device, 'type') and 'cuda' in str(device):
            import torch
            torch.cuda.empty_cache()

    return {"task_id": task_id, "predicted_grids": predicted_grids,
            "diagnostics": all_diagnostics}


# ─── Run one experiment ────────────────────────────────────────────────────────

def run_ttt_experiment(
    label:          str,
    base_model,
    dataset,
    device,
    args,
    solutions:      Dict,
    test_task_ids:  List[str],
    out_dir:        Path,
    ttt_cfg:        Optional[TttConfig],
    diversity_lambda: float = 0.3,
    beam_width:     int = 14,
    n_sample:       int = 30,
) -> Dict:
    print()
    print("=" * 70)
    print(f"EXPERIMENT: {label}")
    print(f"  beam={beam_width}  n_sample={n_sample}  λ={diversity_lambda}")
    if ttt_cfg:
        print(f"  TTT: {ttt_cfg.n_steps} steps  lr={ttt_cfg.lr}  layers={ttt_cfg.last_n_layers}")
    else:
        print(f"  TTT: disabled")
    print("=" * 70)

    out_dir.mkdir(parents=True, exist_ok=True)
    submission:     Dict = {}
    all_diag:       List = []
    t_start = perf_counter()

    for i, task_id in enumerate(test_task_ids, 1):
        t_task = perf_counter()
        print(f"\n[{i}/{len(test_task_ids)}] {task_id}")
        try:
            result = evaluate_task_with_ttt(
                base_model, dataset, task_id, device, args, solutions,
                ttt_cfg=ttt_cfg,
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
        print(f"  task done in {perf_counter()-t_task:.1f}s")

    # Save
    sub_dir = out_dir / "submission"
    sub_dir.mkdir(exist_ok=True)
    with open(sub_dir / "submission.json", "w") as f:
        json.dump(submission, f, indent=2)
    with open(out_dir / "diagnostics.json", "w") as f:
        json.dump(all_diag, f, indent=2)

    # Score
    n_oracle    = sum(1 for d in all_diag if d.get("oracle_hit"))
    n_a2        = sum(1 for d in all_diag if d.get("a2_correct"))
    n_pairs     = len(all_diag)
    arc_correct = arc_pct = 0
    if args.solutions:
        arc_correct, arc_total = arc_score(submission, args.solutions)
        arc_pct = arc_correct / max(arc_total, 1) * 100

    elapsed = perf_counter() - t_start
    print(f"\n  Oracle : {n_oracle}/{n_pairs} = {n_oracle/max(n_pairs,1)*100:.1f}%")
    print(f"  a2 hits: {n_a2}/{n_pairs} = {n_a2/max(n_pairs,1)*100:.1f}%")
    print(f"  ARC    : {arc_correct}/{400 if not args.max_tasks else args.max_tasks} = {arc_pct:.2f}%")
    print(f"  Time   : {elapsed:.0f}s")

    print(f"\nSubmission  → {sub_dir/'submission.json'}")
    print(f"Diagnostics → {out_dir/'diagnostics.json'}")

    return {
        "label": label,
        "ttt_steps": ttt_cfg.n_steps if ttt_cfg else 0,
        "ttt_lr": ttt_cfg.lr if ttt_cfg else 0,
        "beam_width": beam_width,
        "n_sample": n_sample,
        "diversity_lambda": diversity_lambda,
        "n_pairs": n_pairs,
        "n_oracle": n_oracle,
        "n_a2_correct": n_a2,
        "arc_pct": arc_pct,
        "elapsed_s": round(elapsed),
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("ABLATION v9 — Test-Time Training")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  TTT lr     : {args.ttt_lr}  layers={args.ttt_layers}")
    print("=" * 70)

    solutions = load_solutions(args.solutions)
    print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt  = load_checkpoint(Path(args.checkpoint))
    base_model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(args), checkpoint=ckpt, is_eval=True
    )
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad_(False)
    print(f"Model: {sum(p.numel() for p in base_model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks")

    test_task_ids = sorted({ex.task_id for ex in dataset.iter_examples(split="test")})
    if args.task_id:
        test_task_ids = [args.task_id]
    elif args.max_tasks:
        test_task_ids = test_task_ids[:args.max_tasks]

    # Experiment definitions
    EXPS = {
        "K": dict(label="K: No TTT (EXP C repro)",       ttt_cfg=None),
        "L": dict(label="L: TTT 10 steps",
                  ttt_cfg=TttConfig(n_steps=10,  lr=args.ttt_lr, last_n_layers=args.ttt_layers)),
        "M": dict(label="M: TTT 30 steps",
                  ttt_cfg=TttConfig(n_steps=30,  lr=args.ttt_lr, last_n_layers=args.ttt_layers)),
        "N": dict(label="N: TTT 50 steps",
                  ttt_cfg=TttConfig(n_steps=50,  lr=args.ttt_lr, last_n_layers=args.ttt_layers)),
    }

    if args.exp:
        exp_keys = [k.upper() for k in args.exp.split(",")]
        EXPS = {k: v for k, v in EXPS.items() if k in exp_keys}
        if not EXPS:
            print(f"ERROR: unknown exp '{args.exp}'. Choose from K,L,M,N")
            return

    summaries = []
    for key, cfg in EXPS.items():
        exp_dir = Path(args.output_dir) / cfg["label"].replace(" ", "_").replace(":", "").replace("(", "").replace(")", "")
        s = run_ttt_experiment(
            label           = cfg["label"],
            base_model      = base_model,
            dataset         = dataset,
            device          = device,
            args            = args,
            solutions       = solutions,
            test_task_ids   = test_task_ids,
            out_dir         = exp_dir,
            ttt_cfg         = cfg["ttt_cfg"],
            diversity_lambda= 0.3,
            beam_width      = args.beam_width,
            n_sample        = args.n_sample,
        )
        summaries.append(s)

    # ── Final comparison ─────────────────────────────────────────────────────
    exp_c = {
        "label": "C: EXP C reference (no TTT)",
        "ttt_steps": 0, "beam_width": 14, "n_sample": 30,
        "diversity_lambda": 0.3, "n_pairs": 419,
        "n_oracle": 157, "n_a2_correct": 15, "arc_pct": 30.50,
    }

    print()
    print("=" * 70)
    print("TTT ABLATION COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 70)
    hdr = f"{'Experiment':<42}  {'TTT':>4}  {'oracle':>7}  {'a2hits':>6}  {'ARC%':>6}  {'time':>6}"
    print(hdr)
    print("-" * 70)
    for s in [exp_c] + summaries:
        n = s.get('n_pairs', 419)
        print(f"{s['label']:<42}  {s['ttt_steps']:>4}  "
              f"{s['n_oracle']}/{n}  "
              f"{s.get('n_a2_correct','?'):>6}  "
              f"{s['arc_pct']:>5.2f}%"
              + (f"  {s.get('elapsed_s','?'):>5}s" if s.get('elapsed_s') else ""))
    print("=" * 70)

    if summaries:
        best  = max(summaries, key=lambda s: s["arc_pct"])
        delta = best["arc_pct"] - 30.50
        sign  = "+" if delta >= 0 else ""
        print(f"\nBest : {best['label']}  ({best['arc_pct']:.2f}%)")
        print(f"Δ vs EXP C : {sign}{delta:.2f}pp")
        print(f"Gap to Mithil : {44.0 - best['arc_pct']:.1f}pp")

    comp_path = Path(args.output_dir) / "comparison.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()
