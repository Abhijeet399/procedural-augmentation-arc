"""
run_prototype_a.py — Prototype A evaluation: orientation scoring replaces AAIVR.

Usage
-----
After completing Mithil's training pipeline (run_script.py), run:

    python run_prototype_a.py \\
        --checkpoint runs/tiny.pt \\
        --data-path assets/challenges.json \\
        --solutions assets/solutions.json \\
        --use-color-canon          # optional: Prototype A+
        --output-dir runs/proto_a

This produces runs/proto_a/submission.json and prints the official ARC score
(if --solutions is provided).

What this does vs vanilla evaluate.py
--------------------------------------
  Vanilla:    800 augmented inference passes + AAIVR voting per task
  Prototype A: 8 orientation-scoring passes (on demo pairs, known outputs)
               + 1 inference pass (best orientation, no voting)
               ≈ 800× cheaper inference

  Color canonicalization (--use-color-canon / Prototype A+):
               Deterministically relabels colors by input-frequency rank.
               Removes color-permutation ambiguity without extra inference.
               Requires inverse mapping to restore original colors in output.
"""

import argparse
import sys
from pathlib import Path
from time import perf_counter

import torch


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prototype A: orientation-based canonical ARC inference."
    )
    p.add_argument(
        "--checkpoint", type=Path, required=True,
        help="Path to trained model checkpoint (.pt).",
    )
    p.add_argument(
        "--data-path", type=Path, default=Path("assets/challenges.json"),
        help="Path to challenges.json (default: assets/challenges.json).",
    )
    p.add_argument(
        "--solutions", type=Path, default=None,
        help="Path to solutions.json for scoring (optional).",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("runs/prototype_a"),
        help="Directory to write submission.json.",
    )
    p.add_argument(
        "--use-color-canon", action="store_true", default=False,
        help="Apply deterministic color canonicalization (Prototype A+).",
    )
    p.add_argument(
        "--max-new-tokens", type=int, default=931,
        help="Maximum tokens to generate per test pair.",
    )
    p.add_argument(
        "--device", type=str, default="cuda",
        help="Compute device (default: cuda).",
    )
    p.add_argument(
        "--task-ids", type=str, nargs="*", default=None,
        help="Restrict evaluation to these task IDs (default: all).",
    )
    p.add_argument(
        "--d-model", type=int, default=768,
        help="d_model of the loaded model (inferred from checkpoint if possible).",
    )
    p.add_argument(
        "--n-layers", type=int, default=8,
        help="n_layers of the loaded model (inferred from checkpoint if possible).",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Add src/ (repo root, three levels up from experiments/06_prototypes/) to path
    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root / "src"))

    import argparse as _argparse

    # Import Mithil's modules
    from build import build_model_and_data, load_checkpoint
    from canonicalize import run_prototype_a_evaluation, save_submission

    # ------------------------------------------------------------------
    # Build a minimal cfg that build_model_and_data expects
    # ------------------------------------------------------------------
    cfg = _argparse.Namespace(
        name="prototype_a",
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        # Architecture (will be overridden from checkpoint)
        d_model=args.d_model,
        n_heads=12,
        d_ff=args.d_model * 4,
        n_layers=args.n_layers,
        dropout=0.1,
        attention_dropout=None,
        seed=42,
        # No augmentation at eval time (we handle it ourselves)
        enable_aug=False,
        max_augments=0,
        enable_color_aug=False,
        color_apply_to_test=False,
        enable_dihedral_aug=False,
        dihedral_apply_to_test=False,
        batch_size=1,
        device=args.device,
        eval_only=True,
    )

    t_start = perf_counter()

    print("=" * 60)
    print("LCTE Prototype A — Orientation-based canonical ARC inference")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"Data       : {args.data_path}")
    print(f"Color canon: {args.use_color_canon}")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Load model + dataset
    # ------------------------------------------------------------------
    print("\nLoading model and dataset...")
    checkpoint = load_checkpoint(args.checkpoint)
    model, dataset, _, device, data_path = build_model_and_data(
        cfg, checkpoint=checkpoint, is_eval=True
    )
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Dataset: {len(dataset.task_ids)} tasks, {len(dataset.examples)} examples")

    # ------------------------------------------------------------------
    # Prototype A evaluation
    # ------------------------------------------------------------------
    print("\nRunning Prototype A evaluation...")
    t_eval = perf_counter()

    eval_result = run_prototype_a_evaluation(
        model=model,
        dataset=dataset,
        device=device,
        use_color_canon=args.use_color_canon,
        use_amp=True,
        max_new_tokens=args.max_new_tokens,
        task_ids=args.task_ids,
    )

    t_eval_done = perf_counter()
    print(f"\nEvaluation done in {t_eval_done - t_eval:.1f}s")
    print(f"Tasks evaluated : {eval_result['n_tasks']}")
    print(f"Grids predicted : {eval_result['n_predicted']}")

    # ------------------------------------------------------------------
    # Save submission
    # ------------------------------------------------------------------
    submission_path = save_submission(
        eval_result["submission"],
        output_dir=args.output_dir,
        run_name="submission",
    )

    # ------------------------------------------------------------------
    # Score (if solutions.json is available)
    # ------------------------------------------------------------------
    if args.solutions is not None and args.solutions.exists():
        from utils import score_arc_submission
        print("\nScoring submission...")
        score_result = score_arc_submission(args.solutions, submission_path)
        pct = score_result["percentage"]
        n = score_result["score"]
        total = score_result["max_score"]
        print(f"\n{'=' * 60}")
        print(f"PROTOTYPE A SCORE: {n}/{total} = {pct:.1f}%")
        print(f"{'=' * 60}")
    else:
        if args.solutions is not None:
            print(f"\n[Warning] solutions.json not found at {args.solutions}, skipping scoring.")
        print(f"\nSubmission written to: {submission_path}")

    t_total = perf_counter() - t_start
    print(f"\nTotal wall time: {t_total:.1f}s")

    # ------------------------------------------------------------------
    # Print orientation selection summary
    # ------------------------------------------------------------------
    print("\nOrientation selection summary (first 10 tasks):")
    stats = eval_result.get("orientation_stats", {})
    for i, (task_id, info) in enumerate(list(stats.items())[:10]):
        d = info["best_dihedral"]
        losses = info["losses"]
        loss_str = " | ".join(
            f"d{j}:{l:.3f}" + (" ←" if j == d else "") for j, l in enumerate(losses)
        )
        print(f"  {task_id}: {loss_str}")

    # Orientation distribution
    d_counts = [0] * 8
    for info in stats.values():
        d_counts[info["best_dihedral"]] += 1
    print("\nOrientation selection distribution:")
    dihedral_names = [
        "identity", "rot90", "rot180", "rot270",
        "flip_h", "flip_v", "flip_main", "flip_anti"
    ]
    for d, (name, count) in enumerate(zip(dihedral_names, d_counts)):
        bar = "█" * count
        print(f"  d{d} ({name:12s}): {count:4d} {bar}")


if __name__ == "__main__":
    main()
