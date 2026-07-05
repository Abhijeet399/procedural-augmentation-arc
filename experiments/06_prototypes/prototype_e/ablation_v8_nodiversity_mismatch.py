"""
ablation_v8_nodiversity_mismatch.py — Disable diversity bonus for shape-mismatch tasks

Builds on ablation_v7: the new transition_ranker already scores shape-mismatch
candidates via output-position frequency (real signal, not -1e9).

The remaining problem: the diversity bonus (λ * pixel_dissimilarity) adds noise
on top of these already-meaningful scores, pulling pick_best_non_greedy away
from the transition-ranked top candidate.

Fix (in ablation_v4.pick_best_non_greedy):
  When ALL non-greedy candidates have a different shape from test_input,
  set effective_lambda=0 and use pure output-position transition ranking.

Runs two experiments (new ranker, single model load):
  EXP I   new ranker + diversity ON  (reproduces EXP I from v7)
  EXP J   new ranker + diversity OFF for shape-mismatch tasks  (new)

Both at beam=14, n_sample=30, lambda=0.3.

Usage:
    python prototype_e/ablation_v8_nodiversity_mismatch.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v8_nodiversity_mismatch
"""

import sys
import json
import importlib
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from ablation_v4 import run_experiment, parse_args
from run_prototype_e import load_solutions, load_checkpoint, build_model_and_data, _args_for_build


def patch_ranker(module_path: str):
    spec = importlib.util.spec_from_file_location("transition_ranker", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ranker from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import ablation_v4 as av4
    av4.rank_by_transition = mod.rank_by_transition
    av4._grids_equal       = mod._grids_equal
    sys.modules["transition_ranker"] = mod


def main():
    args = parse_args()
    new_ranker = str(_HERE / "src" / "transition_ranker.py")

    print("=" * 70)
    print("ABLATION v8 — Disable diversity bonus for shape-mismatch tasks")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  New ranker : {new_ranker}")
    print(f"  Fix        : λ→0 when all non-greedy candidates have")
    print(f"               output shape ≠ test_input shape")
    print("=" * 70)

    solutions = load_solutions(args.solutions)
    print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt  = load_checkpoint(Path(args.checkpoint))
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

    patch_ranker(new_ranker)
    import ablation_v4 as av4

    # ── EXP I: new ranker, diversity ON (control) ─────────────────────────────
    # Temporarily monkey-patch pick_best_non_greedy to ignore test_input
    original_pick = av4.pick_best_non_greedy

    def pick_no_shape_detect(ranked_grids, scored_pairs, greedy_cand,
                             diversity_lambda, test_input=None):
        return original_pick(ranked_grids, scored_pairs, greedy_cand,
                             diversity_lambda, test_input=None)

    av4.pick_best_non_greedy = pick_no_shape_detect
    s_I = run_experiment(
        label            = "I: new ranker, diversity ON (control)",
        model            = model,
        dataset          = dataset,
        device           = device,
        base_args        = args,
        solutions        = solutions,
        test_task_ids    = test_task_ids,
        out_dir          = Path(args.output_dir) / "I_new_ranker_diversity_ON",
        diversity_lambda = 0.3,
        beam_width       = 14,
        n_sample         = 30,
    )

    # ── EXP J: new ranker, diversity OFF for shape-mismatch ───────────────────
    av4.pick_best_non_greedy = original_pick   # restore full version
    s_J = run_experiment(
        label            = "J: new ranker, diversity OFF (shape-mismatch)",
        model            = model,
        dataset          = dataset,
        device           = device,
        base_args        = args,
        solutions        = solutions,
        test_task_ids    = test_task_ids,
        out_dir          = Path(args.output_dir) / "J_new_ranker_diversity_OFF_mismatch",
        diversity_lambda = 0.3,
        beam_width       = 14,
        n_sample         = 30,
    )

    # ── Comparison ────────────────────────────────────────────────────────────
    exp_c = {
        "label": "C: EXP C reference (λ=0.3)",
        "beam_width": 14, "n_sample": 30, "diversity_lambda": 0.3,
        "n_pairs": 419, "n_oracle": 157, "n_a2_correct": 15, "arc_pct": 30.50,
    }
    summaries = [s_I, s_J]

    print()
    print("=" * 70)
    print("NO-DIVERSITY-MISMATCH COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 70)
    hdr = (f"{'Experiment':<45}  {'beam':>4}  {'samp':>4}  "
           f"{'oracle':>7}  {'a2hits':>6}  {'ARC%':>6}")
    print(hdr)
    print("-" * 70)
    for s in [exp_c] + summaries:
        oracle_str = f"{s['n_oracle']}/{s.get('n_pairs', 419)}"
        print(f"{s['label']:<45}  {s['beam_width']:>4}  {s['n_sample']:>4}  "
              f"{oracle_str:>7}  {s.get('n_a2_correct','?'):>6}  "
              f"{s['arc_pct']:>5.2f}%")
    print("=" * 70)

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
