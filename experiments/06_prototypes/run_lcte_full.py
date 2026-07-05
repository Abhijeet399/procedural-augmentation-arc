"""
run_lcte_full.py — Full LCTE pipeline: Prototype A + MetaEncoder warm-start.

Combines:
  - Prototype A: orientation selection via demo-pair CE scoring
  - Prototype B: meta-encoder warm-start for the per-task embedding,
                 optionally followed by K gradient steps on the embedding

Usage (Prototype A only — fast, zero new parameters)
-----------------------------------------------------
    python run_lcte_full.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json

Usage (Full LCTE with meta-encoder)
-------------------------------------
    python run_lcte_full.py \\
        --checkpoint     runs/tiny.pt \\
        --meta-checkpoint runs/meta_encoder/meta_enc.pt \\
        --data-path      assets/challenges.json \\
        --solutions      assets/solutions.json \\
        --finetune-steps 100

Ablation flags
--------------
  --no-orientation-selection   Use orientation 0 (identity) always
  --no-color-canon             Disable color canonicalization
  --no-meta-encoder            Disable meta-encoder warm-start (even if provided)
  --finetune-steps N           Number of embedding-only gradient steps (default 0)

Expected results (on 100GB VRAM — batch everything)
-----------------------------------------------------
  Vanilla Mithil (AAIVR 800×):  ~44% ARC-1  ~7% ARC-2
  Prototype A (1×, this script): ~30-38% ARC-1  (AAIVR contributes ~6-14pp)
  Prototype A + meta-encoder:   targeting to close the gap → ~38-44% ARC-1
"""

import argparse
import copy
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Full LCTE: orientation selection + optional meta-encoder warm-start."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--meta-checkpoint", type=Path, default=None,
                   help="MetaEncoder checkpoint (from train_meta_encoder.py). "
                        "If not provided, only Prototype A is run.")
    p.add_argument("--data-path", type=Path, default=Path("assets/challenges.json"))
    p.add_argument("--solutions", type=Path, default=None)
    p.add_argument("--output-dir", type=Path, default=Path("runs/lcte_full"))
    # Ablation flags
    p.add_argument("--no-orientation-selection", action="store_true", default=False)
    p.add_argument("--use-color-canon", action="store_true", default=False)
    p.add_argument("--no-meta-encoder", action="store_true", default=False)
    p.add_argument("--finetune-steps", type=int, default=0,
                   help="Embedding-only fine-tune steps after meta-encoder warm-start")
    p.add_argument("--finetune-lr", type=float, default=1e-2)
    p.add_argument("--max-new-tokens", type=int, default=931)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--task-ids", type=str, nargs="*", default=None)
    p.add_argument("--run-name", type=str, default=None,
                   help="Override the run directory name")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root / "src"))

    import argparse as _argparse
    from build import build_model_and_data, load_checkpoint
    from canonicalize import (
        build_color_canon_mapping,
        build_color_inverse_mapping,
        run_canonical_inference_for_task,
        save_submission,
        score_task_orientations,
    )
    from common import (
        IO_SEPARATOR_TOKEN_ID,
        SequenceExample,
        is_rectangular_grid,
        tokens_to_grid,
    )

    # ------------------------------------------------------------------
    # Determine run name
    # ------------------------------------------------------------------
    parts = ["lcte"]
    if args.meta_checkpoint and not args.no_meta_encoder:
        parts.append("metaenc")
    if args.use_color_canon:
        parts.append("colorcanon")
    if args.no_orientation_selection:
        parts.append("noorisel")
    if args.finetune_steps > 0:
        parts.append(f"ft{args.finetune_steps}")
    run_name = args.run_name or "_".join(parts)

    print("=" * 70)
    print("LCTE Full Pipeline")
    print(f"  Orientation selection : {not args.no_orientation_selection}")
    print(f"  Color canonicalization: {args.use_color_canon}")
    use_meta = bool(args.meta_checkpoint and not args.no_meta_encoder)
    print(f"  Meta-encoder warm-start: {use_meta}")
    print(f"  Embedding fine-tune steps: {args.finetune_steps}")
    print(f"  Run name: {run_name}")
    print("=" * 70)

    t_start = perf_counter()

    # ------------------------------------------------------------------
    # Load main model
    # ------------------------------------------------------------------
    cfg = _argparse.Namespace(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        d_model=768, n_heads=12, d_ff=3072, n_layers=8,
        dropout=0.1, attention_dropout=None, seed=42,
        enable_aug=False, max_augments=0,
        enable_color_aug=False, color_apply_to_test=False,
        enable_dihedral_aug=False, dihedral_apply_to_test=False,
        batch_size=1, device=args.device, eval_only=True,
    )

    print("\nLoading main model …")
    checkpoint = load_checkpoint(args.checkpoint)
    model, dataset, _, device, _ = build_model_and_data(
        cfg, checkpoint=checkpoint, is_eval=True
    )
    model.eval()
    model.to(dtype=torch.bfloat16)

    # Keep a clean copy of the original embeddings so we can restore them
    # per-task after each task's warm-start (important: don't contaminate
    # other tasks' embeddings when doing per-task operations)
    original_emb_weight = model.example_embedding.weight.data.clone()

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ------------------------------------------------------------------
    # Load meta-encoder (optional)
    # ------------------------------------------------------------------
    meta_encoder = None
    if use_meta:
        print(f"\nLoading meta-encoder from {args.meta_checkpoint} …")
        from meta_encoder import MetaEncoder, MetaEncoderConfig
        meta_data = torch.load(args.meta_checkpoint, map_location="cpu", weights_only=False)
        meta_cfg = MetaEncoderConfig(**meta_data["meta_config"])
        meta_encoder = MetaEncoder(meta_cfg)
        meta_encoder.load_state_dict(meta_data["meta_encoder_state"])
        meta_encoder = meta_encoder.to(device).eval()
        n_meta_params = sum(p.numel() for p in meta_encoder.parameters())
        print(f"MetaEncoder: {n_meta_params:,} params | val_mse={meta_data.get('val_mse', '?'):.4f}")

    # ------------------------------------------------------------------
    # Group examples by task and split
    # ------------------------------------------------------------------
    demo_by_task: Dict[str, List[SequenceExample]] = {}
    test_by_task: Dict[str, List[SequenceExample]] = {}

    for ex in dataset.examples:
        if args.task_ids is not None and ex.task_id not in args.task_ids:
            continue
        if ex.split == "train" and ex.has_output:
            demo_by_task.setdefault(ex.task_id, []).append(ex)
        elif ex.split == "test":
            test_by_task.setdefault(ex.task_id, []).append(ex)

    eval_tasks = sorted(test_by_task.keys())

    submission: Dict[str, list] = {}
    orientation_stats: Dict[str, object] = {}
    n_predicted = 0
    timing: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Per-task inference
    # ------------------------------------------------------------------
    for task_idx, task_id in enumerate(eval_tasks):
        t_task = perf_counter()
        demo_exs = demo_by_task.get(task_id, [])
        test_exs = sorted(test_by_task[task_id], key=lambda e: e.pair_index)

        print(
            f"\n[{task_idx + 1}/{len(eval_tasks)}] {task_id} | "
            f"demo={len(demo_exs)} test={len(test_exs)}"
        )

        # ---- Step 0: Restore the original embedding for this task ----
        example_id = int(demo_exs[0].example_id) if demo_exs else int(test_exs[0].example_id)
        with torch.no_grad():
            model.example_embedding.weight[example_id].copy_(
                original_emb_weight[example_id]
            )

        # ---- Step 1: Meta-encoder warm-start -------------------------
        if meta_encoder is not None and demo_exs:
            from meta_encoder import warm_start_embedding
            ws = warm_start_embedding(
                meta_encoder=meta_encoder,
                main_model=model,
                demo_examples=demo_exs,
                task_id=task_id,
                device=device,
                n_finetune_steps=args.finetune_steps,
                finetune_lr=args.finetune_lr,
                dihedral_id=0,
            )
            if args.finetune_steps > 0:
                print(
                    f"  meta-enc: pre_CE={ws.pre_finetune_loss:.4f} "
                    f"post_CE={ws.post_finetune_loss:.4f}"
                )

        # ---- Step 2: Color canonicalization (optional) ----------------
        color_mapping: Optional[List[int]] = None
        color_inv: Optional[List[int]] = None

        if args.use_color_canon and demo_exs:
            demo_input_grids = []
            for ex in demo_exs:
                tok = ex.tokens.tolist()
                input_toks = [t for t in tok[:tok.index(IO_SEPARATOR_TOKEN_ID)]]
                grid = tokens_to_grid(input_toks)
                if grid:
                    demo_input_grids.append(grid)
            if demo_input_grids:
                color_mapping = build_color_canon_mapping(demo_input_grids)
                color_inv = build_color_inverse_mapping(color_mapping)

        # ---- Step 3: Orientation scoring ------------------------------
        if args.no_orientation_selection or not demo_exs:
            best_d = 0
            losses = [float("inf")] * 8
        else:
            best_d, losses = score_task_orientations(
                model=model,
                demo_examples=demo_exs,
                device=device,
                use_amp=True,
                color_mapping=color_mapping,
            )

        orientation_stats[task_id] = {"best_dihedral": best_d, "losses": losses}
        print(f"  orientation: d{best_d} (CE={losses[best_d]:.4f})")

        # ---- Step 4: Inference ----------------------------------------
        predicted_grids = run_canonical_inference_for_task(
            model=model,
            demo_examples=demo_exs,
            test_examples=test_exs,
            device=device,
            best_dihedral=best_d,
            color_mapping=color_mapping,
            color_mapping_inv=color_inv,
            max_new_tokens=args.max_new_tokens,
        )

        # Build submission
        task_sub = []
        for grid in predicted_grids:
            if grid is None or not is_rectangular_grid(grid):
                attempt = [[0]]
            else:
                attempt = grid
                n_predicted += 1
            task_sub.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = task_sub

        t_task_done = perf_counter()
        timing[task_id] = t_task_done - t_task
        print(f"  task done in {timing[task_id]:.1f}s")

    # ------------------------------------------------------------------
    # Save submission
    # ------------------------------------------------------------------
    sub_path = save_submission(submission, output_dir=args.output_dir, run_name=run_name)

    # ------------------------------------------------------------------
    # Score
    # ------------------------------------------------------------------
    if args.solutions and Path(args.solutions).exists():
        from utils import score_arc_submission
        score_result = score_arc_submission(args.solutions, sub_path)
        print(f"\n{'=' * 70}")
        print(f"LCTE SCORE: {score_result['score']:.1f}/{score_result['max_score']}"
              f" = {score_result['percentage']:.1f}%")
        print(f"{'=' * 70}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    t_total = perf_counter() - t_start
    avg_task_time = sum(timing.values()) / max(1, len(timing))

    print(f"\nSummary")
    print(f"  Tasks evaluated    : {len(eval_tasks)}")
    print(f"  Grids predicted    : {n_predicted}")
    print(f"  Total wall time    : {t_total:.1f}s")
    print(f"  Avg time per task  : {avg_task_time:.1f}s")
    print(f"  Submission         : {sub_path}")

    # Orientation distribution
    d_counts = [0] * 8
    for info in orientation_stats.values():
        d_counts[info["best_dihedral"]] += 1
    print("\nOrientation selection distribution:")
    names = ["identity", "rot90", "rot180", "rot270", "flip_h", "flip_v", "flip_main", "flip_anti"]
    for d, (name, cnt) in enumerate(zip(names, d_counts)):
        print(f"  d{d} ({name:12s}): {cnt:4d} {'█' * cnt}")

    # Save detailed stats
    stats_path = Path(args.output_dir) / run_name / "orientation_stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with stats_path.open("w") as fh:
        json.dump({"tasks": orientation_stats, "timing": timing}, fh, default=str)
    print(f"Orientation stats saved to {stats_path}")


if __name__ == "__main__":
    main()
