"""
ablation_v7_shape_fix.py — Output-position scoring for shape-mismatch tasks

The fix (in transition_ranker.py):

  OLD: candidates with output shape != input shape get score -1e9
       -> all ties, diversity bonus picks most-different-from-greedy (wrong)

  NEW: shape-mismatch candidates scored by output-position color frequency
       P(color | output cell (r,c)) built from demo outputs
       -> real ranking signal for fixed/scaled output shape tasks

Runs two experiments back-to-back (single model load):
  EXP H  OLD ranker (transition_ranker.py.bak)
  EXP I  NEW ranker (transition_ranker.py)

Both at beam=14, n_sample=30, lambda=0.3.

Usage:
    python prototype_e/ablation_v7_shape_fix.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/ablation_v7_shape_fix
"""

import sys
import json
import shutil
import importlib
import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from ablation_v4 import run_experiment, parse_args
from run_prototype_e import load_solutions, load_checkpoint, build_model_and_data, _args_for_build


# ── Ranker hot-swap helper ────────────────────────────────────────────────────

def load_ranker_from_file(module_path: str):
    """Load transition_ranker from an explicit file path and return the module."""
    spec = importlib.util.spec_from_file_location("transition_ranker", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load ranker from: {module_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def patch_ranker(module_path: str):
    """
    Hot-swap the transition ranker that ablation_v4.run_experiment uses.
    ablation_v4 imports rank_by_transition and _grids_equal at module level,
    so we patch them directly in its namespace.
    """
    import ablation_v4 as av4
    mod = load_ranker_from_file(module_path)
    av4.rank_by_transition = mod.rank_by_transition
    av4._grids_equal       = mod._grids_equal
    sys.modules["transition_ranker"] = mod
    return mod


# =============================================================================

def main():
    args = parse_args()

    old_ranker = str(_HERE / "src" / "transition_ranker_old.py")
    new_ranker = str(_HERE / "src" / "transition_ranker.py")

    if not Path(old_ranker).exists():
        print(f"ERROR: backup ranker not found at {old_ranker}")
        print("Make sure transition_ranker.py.bak exists (the original version).")
        sys.exit(1)

    print("=" * 70)
    print("ABLATION v7 — Output-position scoring for shape-mismatch tasks")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Data       : {args.data_path}")
    print(f"  Solutions  : {args.solutions}")
    print(f"  Old ranker : {old_ranker}")
    print(f"  New ranker : {new_ranker}")
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

    CONFIGS = [
        dict(label="H: old ranker (shape-mismatch = -1e9)",   ranker=old_ranker),
        dict(label="I: new ranker (output-position fallback)", ranker=new_ranker),
    ]

    summaries = []
    for cfg in CONFIGS:
        print(f"\nPatching ranker: {Path(cfg['ranker']).name}")
        patch_ranker(cfg["ranker"])

        exp_dir = (
            Path(args.output_dir)
            / cfg["label"]
              .replace(" ", "_").replace(":", "").replace("(", "")
              .replace(")", "").replace("=", "").replace("-", "")
              .replace("/", "")
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
            diversity_lambda = 0.3,
            beam_width       = 14,
            n_sample         = 30,
        )
        summaries.append(s)

    # Restore new ranker
    patch_ranker(new_ranker)

    # ── Comparison ────────────────────────────────────────────────────────────
    exp_c = {
        "label": "C: EXP C reference (λ=0.3)",
        "beam_width": 14, "n_sample": 30, "diversity_lambda": 0.3,
        "n_pairs": 419, "n_oracle": 157, "n_a2_correct": 15, "arc_pct": 30.50,
    }

    print()
    print("=" * 70)
    print("SHAPE-FIX COMPARISON  (reference: EXP C = 30.50%)")
    print("=" * 70)
    hdr = (f"{'Experiment':<42}  {'beam':>4}  {'samp':>4}  "
           f"{'oracle':>7}  {'a2hits':>6}  {'ARC%':>6}")
    print(hdr)
    print("-" * 70)
    for s in [exp_c] + summaries:
        oracle_str = f"{s['n_oracle']}/{s.get('n_pairs', 419)}"
        print(f"{s['label']:<42}  {s['beam_width']:>4}  {s['n_sample']:>4}  "
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
