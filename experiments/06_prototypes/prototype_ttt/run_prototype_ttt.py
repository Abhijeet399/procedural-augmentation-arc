"""
prototype_ttt/run_prototype_ttt.py
====================================
Production evaluation script integrating Test-Time Training (TTT).

Pipeline for each task:
  1. Score dihedral orientations with base model
  2. TTT-finetune a copy of the model on the task's demo pairs (N steps)
  3. Generate candidates with the fine-tuned model (beam + sampling)
  4. Hard filter (shape / palette)
  5. Transition ranker → pick attempt_1 (greedy) and attempt_2 (best non-greedy)
  6. Free the fine-tuned copy; base model is never modified

Why TTT closes the gap:
  The base model has memorised 1307 training tasks (CE ≈ 0), so RCOS-style
  reranking is blind.  But for the 400 eval tasks the model has never seen the
  specific transformation rule.  A few gradient steps on the 3-5 demo pairs
  steers next-token distributions toward that rule, raising:
    - Oracle ceiling: correct answer appears in the candidate pool
    - Greedy accuracy: model decodes the right answer more often

EXP C best config: beam=14, n_sample=30, λ=0.3
Recommended TTT config: n_steps=30, lr=2e-4, all layers

Usage:
    python prototype_ttt/run_prototype_ttt.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/prototype_ttt \\
        --ttt-steps  30 \\
        --ttt-lr     2e-4

    # Quick smoke test:
        ... --max-tasks 5 --ttt-steps 10

    # Ablation — test step counts K/L/M/N:
        python prototype_ttt/ablation_v9_ttt.py ...
"""

import sys
import json
import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

_HERE = Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent))
# prototype_e is the sibling that has run_prototype_e, ablation_v4, etc.
sys.path.insert(0, str(_HERE.parent / "prototype_e"))
sys.path.insert(0, str(_HERE.parent / "prototype_e" / "src"))

from ablation_v9_ttt import (
    evaluate_task_with_ttt,
    run_ttt_experiment,
)
from run_prototype_e import (
    load_solutions,
    load_checkpoint,
    build_model_and_data,
    _args_for_build,
    arc_score,
)
from ttt import TttConfig


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Prototype TTT — production runner")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-path",   required=True)
    p.add_argument("--solutions",   default=None)
    p.add_argument("--output-dir",  default="runs/prototype_ttt")
    p.add_argument("--device",      default="cuda")
    p.add_argument("--max-tasks",   type=int, default=None)
    p.add_argument("--task-id",     default=None)
    # Generation (EXP C best defaults)
    p.add_argument("--beam-width",  type=int,   default=14)
    p.add_argument("--n-sample",    type=int,   default=30)
    p.add_argument("--lambda",      type=float, default=0.3, dest="lam")
    p.add_argument("--temps",       type=float, nargs="+", default=[0.7, 1.0])
    p.add_argument("--top-k",       type=int,   default=None)
    # TTT
    p.add_argument("--ttt-steps",   type=int,   default=30)
    p.add_argument("--ttt-lr",      type=float, default=2e-4)
    p.add_argument("--ttt-layers",  type=int,   default=None,
                   help="Fine-tune only last N layers. Default: all layers.")
    p.add_argument("--no-ttt",      action="store_true",
                   help="Disable TTT (reproduces EXP C baseline)")
    return p.parse_args()


def _args_compat(args):
    """Adapter so _args_for_build gets the fields it expects."""
    args.data_path  = args.__dict__.get("data_path",  None)
    args.checkpoint = args.__dict__.get("checkpoint", None)
    args.top_k      = args.__dict__.get("top_k",      0) or 0
    return args


def main():
    args = parse_args()
    args = _args_compat(args)

    ttt_cfg = (None if args.no_ttt else
               TttConfig(
                   n_steps       = args.ttt_steps,
                   lr            = args.ttt_lr,
                   last_n_layers = args.ttt_layers,
               ))

    print("=" * 70)
    print("PROTOTYPE TTT — Test-Time Training Evaluation")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Data        : {args.data_path}")
    print(f"  Solutions   : {args.solutions}")
    print(f"  Generation  : beam={args.beam_width}  n_sample={args.n_sample}  λ={args.lam}")
    if ttt_cfg:
        print(f"  TTT         : {ttt_cfg.n_steps} steps  lr={ttt_cfg.lr:.0e}"
              f"  layers={'all' if ttt_cfg.last_n_layers is None else ttt_cfg.last_n_layers}")
    else:
        print(f"  TTT         : DISABLED (EXP C baseline)")
    print("=" * 70)

    solutions = load_solutions(args.solutions)
    print(f"Loaded solutions for {len(solutions)} tasks.")

    ckpt  = load_checkpoint(Path(args.checkpoint))
    import torch
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

    label = (f"TTT_{ttt_cfg.n_steps}steps_lr{ttt_cfg.lr:.0e}"
             if ttt_cfg else "no_ttt_EXP_C_baseline")

    summary = run_ttt_experiment(
        label           = label,
        base_model      = base_model,
        dataset         = dataset,
        device          = device,
        args            = args,
        solutions       = solutions,
        test_task_ids   = test_task_ids,
        out_dir         = Path(args.output_dir),
        ttt_cfg         = ttt_cfg,
        diversity_lambda= args.lam,
        beam_width      = args.beam_width,
        n_sample        = args.n_sample,
    )

    print()
    print("=" * 70)
    print("RESULT SUMMARY")
    print("=" * 70)
    print(f"  Oracle   : {summary['n_oracle']}/{summary['n_pairs']} = "
          f"{summary['n_oracle']/max(summary['n_pairs'],1)*100:.1f}%")
    print(f"  ARC%     : {summary['arc_pct']:.2f}%")
    delta = summary['arc_pct'] - 30.50
    sign  = "+" if delta >= 0 else ""
    print(f"  Δ vs EXP C (30.50%) : {sign}{delta:.2f}pp")
    print(f"  Gap to Mithil (44%) : {44.0 - summary['arc_pct']:.2f}pp")
    print("=" * 70)


if __name__ == "__main__":
    main()
