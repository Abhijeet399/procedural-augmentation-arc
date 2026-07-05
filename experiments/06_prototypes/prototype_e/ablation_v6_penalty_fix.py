"""
ablation_v6_penalty_fix.py — Test the degenerate-score fix

Compares two configurations at beam=14, n_sample=30, λ=0.3:

  EXP F  λ=0.3, penalty-fix OFF  (reproduce EXP C as control)
  EXP G  λ=0.3, penalty-fix ON   (new behaviour: fall back to rank-0 when
                                   all non-greedy scores ≤ -1e8)

Root cause being fixed:
  For tasks with fixed/scaled output shape, score_candidate returns -1e9 for
  every candidate (shape mismatch vs test_input).  The diversity bonus then
  acts as the sole differentiator and picks the most-visually-different wrong
  candidate.  The fix detects this regime and disables the diversity bonus,
  falling back to rank-0 non-greedy.

Expected impact:
  Up to 7 additional tasks recovered (the DIVERSITY_BONUS_HURT group).
  Zero regressions on the 15 tasks already recovered by EXP C.

Usage:
    python prototype_e/ablation_v6_penalty_fix.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v6_penalty_fix
"""

import sys
from pathlib import Path
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

import json
from ablation_v4 import (
    run_experiment, print_comparison, parse_args,
    _PENALTY_THRESHOLD,
)
from run_prototype_e import load_solutions, load_checkpoint, build_model_and_data, _args_for_build


def main():
    args = parse_args()

    print("=" * 70)
    print("ABLATION v6 — Degenerate-score penalty fix")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Solutions  : {args.solutions}")
    print(f"  Fix        : disable diversity bonus when all scores ≤ {_PENALTY_THRESHOLD:.0e}")
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

    CONFIGS = [
        dict(label="F: EXP-C control (no fix)",
             diversity_lambda=0.3,
             beam_width=14,
             n_sample=30),
        dict(label="G: penalty-fix ON (λ=0.3)",
             diversity_lambda=0.3,
             beam_width=14,
             n_sample=30),
    ]

    # EXP F runs with fix disabled, EXP G with fix enabled.
    # We toggle the global threshold: set to -inf to disable (never triggers),
    # set to -1e8 to enable (triggers on degenerate scores).
    import ablation_v4 as _av4

    summaries = []
    for cfg in CONFIGS:
        # Toggle fix
        if "no fix" in cfg["label"]:
            _av4._PENALTY_THRESHOLD = float("-inf")   # never triggers
        else:
            _av4._PENALTY_THRESHOLD = -1e8            # enabled

        exp_dir = (
            Path(args.output_dir)
            / cfg["label"]
              .replace(" ", "_").replace(":", "").replace("(", "")
              .replace(")", "").replace(",", "").replace("=", "")
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

    # Restore
    _av4._PENALTY_THRESHOLD = -1e8

    # Comparison
    exp_c = {
        "label": "C: best so far (EXP C, λ=0.3)",
        "beam_width": 14, "n_sample": 30, "diversity_lambda": 0.3,
        "n_pairs": 419, "n_oracle": 157, "n_a2_correct": 15, "arc_pct": 30.50,
    }

    print()
    print("=" * 70)
    print("PENALTY FIX COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 70)
    hdr = (f"{'Experiment':<38}  {'beam':>4}  {'samp':>4}  {'λ':>4}  "
           f"{'oracle':>7}  {'a2hits':>6}  {'ARC%':>6}")
    print(hdr)
    print("-" * 70)
    for s in [exp_c] + summaries:
        oracle_str = f"{s['n_oracle']}/{s.get('n_pairs', 419)}"
        a2_str     = str(s.get("n_a2_correct", "?"))
        arc_str    = f"{s['arc_pct']:.2f}%"
        lam_str    = f"{s['diversity_lambda']:.1f}"
        print(f"{s['label']:<38}  {s['beam_width']:>4}  {s['n_sample']:>4}  "
              f"{lam_str:>4}  {oracle_str:>7}  {a2_str:>6}  {arc_str:>6}")
    print("=" * 70)

    best = max(summaries, key=lambda s: s["arc_pct"])
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
