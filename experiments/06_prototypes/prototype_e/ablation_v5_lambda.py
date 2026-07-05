"""
ablation_v5_lambda.py — λ tuning on top of the best config (EXP C: beam=14, n_sample=30, λ=0.3)

Runs two experiments back-to-back, sharing a single model load:

  EXP D  λ=0.5  (beam=14, n_sample=30) — stronger diversity push
  EXP E  λ=1.0  (beam=14, n_sample=30) — max diversity: transition + dissimilarity equally weighted

Reference (from ablation_v4 EXP C):
  Oracle : 157/419 = 37.5%
  Greedy : 117/419 = 27.9%
  a2 hits: 15/419  = 3.6%
  ARC    : 122/400 = 30.50%

Diversity scoring for attempt_2:

    combined(c) = transition_score(c) + λ × dissimilarity(c, greedy)

  λ=0.3 → dissimilarity adds up to 0.3 nats per fully-different grid
  λ=0.5 → dissimilarity adds up to 0.5 nats per fully-different grid
  λ=1.0 → dissimilarity and transition score weighted equally per nat

Usage:
    python prototype_e/ablation_v5_lambda.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v5_lambda
"""

# ── Re-use everything from ablation_v4 except the CONFIGS block ───────────
import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

# Import the shared machinery from ablation_v4 directly
from ablation_v4 import (
    pixel_dissimilarity,
    pick_best_non_greedy,
    evaluate_task,
    run_experiment,
    print_comparison,
    parse_args,
)

import json
from run_prototype_e import load_solutions, load_checkpoint, build_model_and_data, _args_for_build


def main():
    args = parse_args()

    print("=" * 70)
    print("ABLATION v5 — λ tuning (beam=14, n_sample=30)")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Solutions  : {args.solutions}")
    print(f"  Reference  : EXP C (λ=0.3) → 30.50%  ← beat this")
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

    # ── Two experiments: λ=0.5 then λ=1.0, both at beam=14, n_sample=30 ────
    CONFIGS = [
        dict(label="D: lambda=0.5 (b14,s30)",
             diversity_lambda=0.5,
             beam_width=14,
             n_sample=30),
        dict(label="E: lambda=1.0 (b14,s30)",
             diversity_lambda=1.0,
             beam_width=14,
             n_sample=30),
    ]

    summaries = []
    for cfg in CONFIGS:
        exp_dir = (
            Path(args.output_dir)
            / cfg["label"]
              .replace(" ", "_")
              .replace(":", "")
              .replace("(", "")
              .replace(")", "")
              .replace(",", "")
              .replace("=", "")
        )
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

    # ── Comparison (includes EXP C reference) ────────────────────────────────
    # Prepend EXP C as the reference row
    exp_c = {
        "label": "C: diversity + bigger gen (best)",
        "beam_width": 14,
        "n_sample": 30,
        "diversity_lambda": 0.3,
        "n_pairs": 419,
        "n_oracle": 157,
        "n_a2_correct": 15,
        "arc_pct": 30.50,
    }

    print()
    print("=" * 70)
    print("LAMBDA TUNING COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 70)
    hdr = (f"{'Experiment':<35}  {'beam':>4}  {'samp':>4}  {'λ':>4}  "
           f"{'oracle':>6}  {'a2hits':>6}  {'ARC%':>6}")
    print(hdr)
    print("-" * 70)
    for s in [exp_c] + summaries:
        oracle_str = f"{s['n_oracle']}/{s.get('n_pairs', 419)}"
        a2_str     = str(s.get("n_a2_correct", "?"))
        arc_str    = f"{s['arc_pct']:.2f}%"
        lam_str    = f"{s['diversity_lambda']:.1f}"
        print(f"{s['label']:<35}  {s['beam_width']:>4}  {s['n_sample']:>4}  "
              f"{lam_str:>4}  {oracle_str:>6}  {a2_str:>6}  {arc_str:>6}")
    print("=" * 70)

    best = max(summaries, key=lambda s: s["arc_pct"])
    print(f"\nBest new experiment : {best['label']}  ({best['arc_pct']:.2f}%)")
    delta = best["arc_pct"] - 30.50
    sign  = "+" if delta >= 0 else ""
    print(f"Δ vs EXP C (30.50%) : {sign}{delta:.2f}pp")
    gap = 44.0 - best["arc_pct"]
    print(f"Gap to Mithil       : {gap:.1f}pp")

    # Save
    comp_path = Path(args.output_dir) / "comparison.json"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)
    print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()
