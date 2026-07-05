"""
prototype_more_data/finetune_q.py
==================================
EXP Q — Fine-tune mdlARC on combined original + NVARC synthetic data.

Why:
  The base model (runs/tiny.pt) has memorised 1307 tasks → CE ≈ 0 on training
  data → oracle ceiling locked at 37.5% on eval tasks.  The model simply does
  not know the transformation rules for 243 of the 400 eval tasks.

  Fine-tuning on NVARC's 103k synthetic puzzles (which cover the same eval task
  distribution) teaches the model new transformation patterns, raising the oracle
  ceiling.  Then TTT amplifies the already-improved prior.

Two variants:
  EXP Q.a  original checkpoint → fine-tune on 80% NVARC + 20% original
           (prepared by prepare_dataset.py)
  EXP Q.b  Q.a checkpoint → re-fine-tune on original 1307 only
           (re-specialize on the original distribution)

This script handles both:
  - Q.a: pass --data-path assets/combined_challenges.json
  - Q.b: pass --data-path assets/challenges.json  +  --checkpoint runs/q_a_finetuned.pt

Training loop:
  - AdamW, cosine LR with warmup
  - CE loss on output tokens only (same as mdlARC pre-training)
  - Eval oracle ceiling every --eval-interval steps on the 400 eval tasks
  - Saves best checkpoint (by oracle %) to --out-checkpoint

Usage:
    # EXP Q.a — fine-tune on combined dataset:
    python prototype_more_data/finetune_q.py \\
        --checkpoint     runs/tiny.pt \\
        --data-path      assets/combined_challenges.json \\
        --solutions      assets/combined_solutions.json \\
        --eval-data      assets/challenges.json \\
        --eval-solutions assets/solutions.json \\
        --out-checkpoint runs/q_a_finetuned.pt \\
        --lr             2e-5 \\
        --n-steps        20000 \\
        --eval-interval  2000

    # EXP Q.b — re-specialize on original 1307:
    python prototype_more_data/finetune_q.py \\
        --checkpoint     runs/q_a_finetuned.pt \\
        --data-path      assets/challenges.json \\
        --solutions      assets/solutions.json \\
        --eval-data      assets/challenges.json \\
        --eval-solutions assets/solutions.json \\
        --out-checkpoint runs/q_b_finetuned.pt \\
        --lr             5e-6 \\
        --n-steps        5000 \\
        --eval-interval  1000
"""

import sys
import json
import argparse
import math
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional

import torch

_HERE = Path(__file__).parent.resolve()
_REPO_ROOT = _HERE.parent.parent.parent   # repo root
# Insert in reverse priority — last insert lands at sys.path[0] and wins.
# repo root must be last so its build.py beats the PyPI 'build' package.
sys.path.insert(0, str(_HERE.parent / "prototype_e" / "src"))
sys.path.insert(0, str(_HERE.parent / "prototype_e"))
sys.path.insert(0, str(_REPO_ROOT / "src"))  # base pipeline src/ (common, rcos, etc.)
sys.path.insert(0, str(_REPO_ROOT))   # <- wins: repo root

# Load the base pipeline's build.py directly by path to avoid the PyPI 'build' package.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mdlarc_build", str(_REPO_ROOT / "src" / "build.py"))
_build_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_build_mod)
build_model_and_data = _build_mod.build_model_and_data
load_checkpoint      = _build_mod.load_checkpoint

# mdlARC's build.py has no save_checkpoint — roll our own.
# Strategy: load the base checkpoint once to get its full structure,
# then swap in the new model weights. This preserves config, vocab, etc.
_BASE_CKPT_CACHE = {}

def save_checkpoint(model, path, metadata=None):
    import torch, copy
    base_path = str(path)
    # Load base structure once (from the original tiny.pt we started from)
    if base_path not in _BASE_CKPT_CACHE:
        # First save: base checkpoint may not exist yet; start from empty dict
        if Path(path).exists():
            _BASE_CKPT_CACHE[base_path] = torch.load(path, map_location="cpu",
                                                       weights_only=False)
        else:
            _BASE_CKPT_CACHE[base_path] = {}
    ckpt = dict(_BASE_CKPT_CACHE[base_path])  # shallow copy of base structure
    # Overwrite weights with current model state
    ckpt["model"] = copy.deepcopy(model.state_dict())
    # Also try common alternative key names
    if "model_state_dict" in ckpt:
        ckpt["model_state_dict"] = ckpt["model"]
    if metadata:
        ckpt["finetune_metadata"] = metadata
    torch.save(ckpt, path)

from run_prototype_e import (
    _args_for_build,
    load_solutions,
)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="EXP Q — fine-tune on synthetic data")
    p.add_argument("--checkpoint",      required=True,
                   help="Base checkpoint to start from (runs/tiny.pt for Q.a)")
    p.add_argument("--data-path",       required=True,
                   help="Training challenges.json (combined or original)")
    p.add_argument("--solutions",       required=True,
                   help="Training solutions.json")
    p.add_argument("--eval-data",       required=True,
                   help="Eval challenges.json (assets/challenges.json)")
    p.add_argument("--eval-solutions",  required=True,
                   help="Eval solutions.json (assets/solutions.json)")
    p.add_argument("--out-checkpoint",  required=True,
                   help="Where to save the best fine-tuned checkpoint")
    p.add_argument("--device",          default="cuda")
    # Optimiser
    p.add_argument("--lr",              type=float, default=2e-5,
                   help="Peak learning rate (default: 2e-5)")
    p.add_argument("--weight-decay",    type=float, default=0.01)
    p.add_argument("--max-grad-norm",   type=float, default=1.0)
    p.add_argument("--warmup-steps",    type=int,   default=200)
    p.add_argument("--lr-min-ratio",    type=float, default=0.1,
                   help="Cosine decay floor = lr * lr_min_ratio")
    # Training schedule
    p.add_argument("--n-steps",         type=int,   default=20000,
                   help="Total gradient steps (default: 20000)")
    p.add_argument("--grad-accum",      type=int,   default=4,
                   help="Gradient accumulation steps (effective batch size)")
    p.add_argument("--eval-interval",   type=int,   default=2000,
                   help="Eval oracle ceiling every N steps")
    # Eval oracle (fast: just count oracle hits, no candidate generation)
    p.add_argument("--eval-max-tasks",  type=int,   default=400,
                   help="Max eval tasks for oracle check (default: 400 = full)")
    p.add_argument("--eval-beam",       type=int,   default=5,
                   help="Beam width during eval (smaller = faster)")
    p.add_argument("--eval-n-sample",   type=int,   default=10,
                   help="Samples per temp during eval")
    return p.parse_args()


def _cosine_lr(step, n_steps, warmup, lr, lr_min):
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(n_steps - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))


def _args_for_train(args):
    """Build args namespace for training dataloader."""
    import argparse
    a = argparse.Namespace(
        checkpoint_path   = Path(args.checkpoint),
        data_path         = Path(args.data_path),
        seed              = 42,
        batch_size        = 1,
        device            = args.device,
        enable_aug        = True,
        enable_color_aug  = True,
        enable_dihedral_aug = True,
        max_augments      = 2,
        color_apply_to_test = True,
        dihedral_apply_to_test = True,
    )
    return a


def _args_for_eval(args):
    """Build args namespace for eval (no augmentation)."""
    import argparse
    a = argparse.Namespace(
        checkpoint_path   = Path(args.checkpoint),
        data_path         = Path(args.eval_data),
        seed              = 42,
        batch_size        = 1,
        device            = args.device,
        enable_aug        = False,
        enable_color_aug  = False,
        enable_dihedral_aug = False,
        max_augments      = 0,
        color_apply_to_test  = False,
        dihedral_apply_to_test = False,
    )
    return a


def eval_oracle_ceiling(
    model,
    eval_dataset,
    eval_solutions: Dict,
    device,
    args,
    max_tasks: Optional[int] = None,
) -> float:
    """
    Fast oracle-ceiling check: for each eval task, generate a small candidate
    pool and count how many tasks have at least one correct candidate.
    Returns oracle % (0–100).
    """
    from rcos import generate_all_candidates, grids_equal
    from canonicalize import score_task_orientations
    from run_prototype_e import _grids_from_seq_ex
    from common import apply_dihedral_transform

    model.eval()
    with torch.no_grad():
        task_ids = sorted({ex.task_id for ex in eval_dataset.iter_examples(split="test")})
        if max_tasks:
            task_ids = task_ids[:max_tasks]

        n_oracle = 0
        for task_id in task_ids:
            demo_seq_exs = [ex for ex in eval_dataset.iter_examples(split="train")
                            if ex.task_id == task_id]
            test_seq_exs = [ex for ex in eval_dataset.iter_examples(split="test")
                            if ex.task_id == task_id]
            if not demo_seq_exs or not test_seq_exs:
                continue

            example_id = eval_dataset.task_id_to_example_id[task_id]
            try:
                best_d, _ = score_task_orientations(model, demo_seq_exs, device)
            except Exception:
                continue

            from rcos import GridPair
            demo_pairs = [_grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs]
            demo_pairs = [gp for gp in demo_pairs if gp.output is not None]

            sol_grids_orig = eval_solutions.get(task_id, [])

            task_ok = True
            for pair_idx, test_ex in enumerate(test_seq_exs):
                test_gp = _grids_from_seq_ex(test_ex, best_d)
                gt_orig = sol_grids_orig[pair_idx] if pair_idx < len(sol_grids_orig) else None
                if gt_orig is None:
                    continue
                gt_canon = apply_dihedral_transform(gt_orig, best_d)

                try:
                    candidates = generate_all_candidates(
                        model, demo_pairs, test_gp.input,
                        example_id, best_d, device,
                        use_greedy=True,
                        use_beam=args.eval_beam > 0,
                        beam_width=args.eval_beam,
                        use_sample=args.eval_n_sample > 0,
                        n_per_temperature=args.eval_n_sample,
                        temperatures=(0.7, 1.0),
                        top_k=None,
                        test_seq_ex=test_ex,
                        demo_seq_exs=demo_seq_exs,
                    )
                except Exception:
                    task_ok = False
                    break

                if not any(grids_equal(c, gt_canon) for c in candidates):
                    task_ok = False
                    break

            if task_ok:
                n_oracle += 1

    model.train()
    return n_oracle / max(len(task_ids), 1) * 100


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("EXP Q — FINE-TUNING ON SYNTHETIC DATA")
    print(f"  Base checkpoint : {args.checkpoint}")
    print(f"  Training data   : {args.data_path}")
    print(f"  Eval data       : {args.eval_data}")
    print(f"  LR              : {args.lr:.0e}  warmup={args.warmup_steps}")
    print(f"  Steps           : {args.n_steps}  (eval every {args.eval_interval})")
    print(f"  Grad accum      : {args.grad_accum}")
    print(f"  Out checkpoint  : {args.out_checkpoint}")
    print("=" * 70)

    # ── Load training data ────────────────────────────────────────────────────
    print("\nBuilding training dataset...")
    train_ckpt = load_checkpoint(Path(args.checkpoint))
    model, train_dataset, train_dl, device, _ = build_model_and_data(
        _args_for_train(args), checkpoint=train_ckpt, is_eval=False
    )
    print(f"Model          : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training tasks : {len(train_dataset.task_ids)}")

    # ── Load eval data (separate dataset, no augmentation) ────────────────────
    print("Building eval dataset...")
    eval_solutions = load_solutions(args.eval_solutions)
    _, eval_dataset, _, _, _ = build_model_and_data(
        _args_for_eval(args), checkpoint=train_ckpt, is_eval=True
    )
    eval_task_ids = sorted({ex.task_id for ex in eval_dataset.iter_examples(split="test")})
    n_eval = min(args.eval_max_tasks, len(eval_task_ids))
    print(f"Eval tasks     : {len(eval_task_ids)} (checking first {n_eval})")

    # ── Optimiser ─────────────────────────────────────────────────────────────
    model.train()
    for p in model.parameters():
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    lr_min = args.lr * args.lr_min_ratio

    # ── Initial oracle baseline ───────────────────────────────────────────────
    print("\nMeasuring initial oracle ceiling (base model)...")
    t0 = perf_counter()
    oracle_pct = eval_oracle_ceiling(
        model, eval_dataset, eval_solutions, device, args, max_tasks=n_eval
    )
    print(f"  Initial oracle : {oracle_pct:.1f}%  ({perf_counter()-t0:.0f}s)")
    print(f"  (EXP C baseline was 37.5%)")

    best_oracle = oracle_pct
    best_step   = 0
    out_ckpt    = Path(args.out_checkpoint)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)

    # Save initial checkpoint as baseline
    save_checkpoint(model, out_ckpt, metadata={"step": 0, "oracle_pct": oracle_pct})

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\nStarting training ({args.n_steps} steps)...")
    log_path = out_ckpt.parent / (out_ckpt.stem + "_training_log.jsonl")
    log_f    = open(log_path, "w")

    step       = 0
    accum_loss = 0.0
    accum_n    = 0
    t_start    = perf_counter()
    train_iter = iter(train_dl)

    opt.zero_grad()

    while step < args.n_steps:
        # ── Get next batch ────────────────────────────────────────────────────
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_dl)
            batch = next(train_iter)

        # Move batch to device
        batch = {k: v.to(device) if hasattr(v, "to") else v
                 for k, v in batch.items()}

        # ── Forward ───────────────────────────────────────────────────────────
        # Filter batch to only keys that forward() accepts.
        # Inspect accepted params once and cache.
        if not hasattr(model, "_fwd_params"):
            import inspect
            sig = inspect.signature(model.forward)
            model._fwd_params = set(sig.parameters.keys()) - {"kwargs", "self"}
        accepted = model._fwd_params
        fwd_batch = {k: v for k, v in batch.items() if k in accepted}
        outputs = model.forward(**fwd_batch, compute_input_loss=False)
        loss = outputs.get("loss")

        if loss is None or not torch.isfinite(loss):
            opt.zero_grad()
            continue

        # Scale for gradient accumulation
        (loss / args.grad_accum).backward()
        accum_loss += loss.item()
        accum_n    += 1

        # ── Optimiser step every grad_accum micro-batches ─────────────────────
        if accum_n % args.grad_accum == 0:
            step += 1
            lr_now = _cosine_lr(step, args.n_steps, args.warmup_steps, args.lr, lr_min)
            for pg in opt.param_groups:
                pg["lr"] = lr_now

            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            opt.zero_grad()

            avg_loss = accum_loss / args.grad_accum
            accum_loss = 0.0

            # ── Logging ───────────────────────────────────────────────────────
            if step % 100 == 0:
                elapsed = perf_counter() - t_start
                steps_per_sec = step / max(elapsed, 1)
                eta = (args.n_steps - step) / max(steps_per_sec, 1e-6)
                print(f"  step {step:5d}/{args.n_steps}"
                      f"  loss={avg_loss:.4f}"
                      f"  lr={lr_now:.2e}"
                      f"  eta={eta/60:.0f}m")
                log_f.write(json.dumps({
                    "step": step, "loss": round(avg_loss, 5),
                    "lr": round(lr_now, 8), "elapsed_s": round(elapsed, 1)
                }) + "\n")
                log_f.flush()

            # ── Periodic oracle eval ──────────────────────────────────────────
            if step % args.eval_interval == 0:
                print(f"\n  [Eval @ step {step}] measuring oracle ceiling...")
                t_eval = perf_counter()
                oracle_pct = eval_oracle_ceiling(
                    model, eval_dataset, eval_solutions, device, args, max_tasks=n_eval
                )
                t_eval = perf_counter() - t_eval
                print(f"  Oracle : {oracle_pct:.1f}%  (best so far: {best_oracle:.1f}% @ step {best_step})"
                      f"  ({t_eval:.0f}s)")
                log_f.write(json.dumps({
                    "step": step, "oracle_pct": round(oracle_pct, 2),
                    "eval_time_s": round(t_eval, 1)
                }) + "\n")
                log_f.flush()

                if oracle_pct > best_oracle:
                    best_oracle = oracle_pct
                    best_step   = step
                    save_checkpoint(model, out_ckpt,
                                    metadata={"step": step, "oracle_pct": oracle_pct})
                    print(f"  ✓ New best! Saved → {out_ckpt}")
                print()
                model.train()

    # ── Final eval ────────────────────────────────────────────────────────────
    print("\nFinal oracle evaluation...")
    oracle_pct = eval_oracle_ceiling(
        model, eval_dataset, eval_solutions, device, args, max_tasks=n_eval
    )
    print(f"Final oracle : {oracle_pct:.1f}%")
    if oracle_pct > best_oracle:
        best_oracle = oracle_pct
        best_step   = args.n_steps
        save_checkpoint(model, out_ckpt,
                        metadata={"step": args.n_steps, "oracle_pct": oracle_pct})

    log_f.close()
    elapsed = perf_counter() - t_start

    print()
    print("=" * 70)
    print("EXP Q FINE-TUNING COMPLETE")
    print("=" * 70)
    print(f"  Best oracle    : {best_oracle:.1f}%  (@ step {best_step})")
    print(f"  vs EXP C base  : 37.5%  Δ={best_oracle - 37.5:+.1f}pp")
    print(f"  Best checkpoint: {out_ckpt}")
    print(f"  Training log   : {log_path}")
    print(f"  Wall time      : {elapsed/60:.0f}m")
    print()
    print("Next step — run full eval with new checkpoint:")
    print(f"  python prototype_ttt/ablation_v9_ttt.py \\")
    print(f"      --checkpoint {out_ckpt} \\")
    print(f"      --data-path  assets/challenges.json \\")
    print(f"      --solutions  assets/solutions.json \\")
    print(f"      --output-dir runs/exp_q_eval \\")
    print(f"      --exp K,M --ttt-lr 1e-5 --ttt-layers 4")


if __name__ == "__main__":
    main()
