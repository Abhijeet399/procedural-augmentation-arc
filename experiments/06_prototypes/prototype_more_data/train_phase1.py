"""
prototype_more_data/train_phase1.py
=====================================
Phase 1: Train on NVARC combined dataset using the original NorMuon optimizer.

Strategy:
  Start from tiny.pt (for architecture config), train on combined NVARC data
  at FULL NorMuon LR (0.02). High LR quickly overwrites tiny.pt weights with
  NVARC-learned patterns — effectively training from scratch within a few epochs.

  After Phase 1, the model understands NVARC transformation rules but may have
  degraded on the original 1307 tasks. Phase 2 re-specializes on original data.

Architecture:
  Same as tiny.pt (76M params, 1307 task embeddings). No changes needed since
  NVARC pairs were aggregated under the 1307 parent task IDs in prepare_dataset.py.

Optimizer:
  SingleDeviceNorMuonWithAuxAdam — same as original mdlARC training.
  NorMuon for linear weights, AdamW for embeddings/biases.
  LR schedule: 2% linear warmup → flat → WSD decay (last 20%).

Usage (run from the repo root):
    python prototype_more_data/train_phase1.py \\
        --base-checkpoint  runs/tiny.pt \\
        --data-path        assets/combined_challenges.json \\
        --out-checkpoint   runs/phase1_nvarc.pt \\
        --epochs           5 \\
        --normuon-lr       0.02 \\
        --adamw-lr         3e-4

    # Quick smoke test (1 epoch):
        ... --epochs 1 --max-tasks 100
"""

import sys
import argparse
from pathlib import Path

_HERE = Path(__file__).parent.resolve()
_REPO_ROOT = _HERE.parent.parent.parent   # repo root
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("mdlarc_build", str(_REPO_ROOT / "src" / "build.py"))
_build = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_build)
build_model_and_data = _build.build_model_and_data
load_checkpoint      = _build.load_checkpoint

_spec2 = _ilu.spec_from_file_location("mdlarc_train", str(_REPO_ROOT / "src" / "train.py"))
_train = _ilu.module_from_spec(_spec2); _spec2.loader.exec_module(_train)
train_model    = _train.train_model
maybe_save_model = _train.maybe_save_model


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1 — NVARC training with NorMuon")
    p.add_argument("--base-checkpoint", required=True,
                   help="Base checkpoint for architecture config (runs/tiny.pt)")
    p.add_argument("--data-path",       required=True,
                   help="Combined NVARC challenges.json (assets/combined_challenges.json)")
    p.add_argument("--out-checkpoint",  required=True,
                   help="Where to save Phase 1 checkpoint (runs/phase1_nvarc.pt)")
    p.add_argument("--epochs",          type=int,   default=5)
    p.add_argument("--normuon-lr",      type=float, default=0.02,
                   help="NorMuon LR for linear weights (default: 0.02 = original training LR)")
    p.add_argument("--adamw-lr",        type=float, default=3e-4,
                   help="AdamW LR for embeddings/biases (default: 3e-4)")
    p.add_argument("--weight-decay",    type=float, default=0.1)
    p.add_argument("--grad-clip",       type=float, default=1.0)
    p.add_argument("--warmup-pct",      type=float, default=0.02)
    p.add_argument("--wsd-decay-start-pct", type=float, default=0.8)
    p.add_argument("--lr-floor",        type=float, default=0.01)
    p.add_argument("--grad-accum",      type=int,   default=1)
    p.add_argument("--device",          default="cuda")
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--max-tasks",       type=int, default=None,
                   help="Limit to first N tasks (for smoke testing)")
    p.add_argument("--checkpoint-every",type=int, default=None,
                   help="Save intermediate checkpoint every N epochs")
    return p.parse_args()


def _build_args(args):
    """Translate our CLI args into the namespace build_model_and_data expects."""
    import argparse
    a = argparse.Namespace(
        checkpoint_path   = Path(args.base_checkpoint),
        data_path         = Path(args.data_path),
        seed              = args.seed,
        batch_size        = 1,
        device            = args.device,
        # Augmentation enabled — same as original training
        enable_aug        = True,
        enable_color_aug  = True,
        enable_dihedral_aug = True,
        max_augments      = 2,
        color_apply_to_test  = True,
        dihedral_apply_to_test = True,
    )
    return a


def main():
    args = parse_args()

    print("=" * 70)
    print("PHASE 1 — NVARC Training with NorMuon")
    print(f"  Base checkpoint : {args.base_checkpoint}")
    print(f"  Data            : {args.data_path}")
    print(f"  Epochs          : {args.epochs}")
    print(f"  NorMuon LR      : {args.normuon_lr}")
    print(f"  AdamW LR        : {args.adamw_lr}")
    print(f"  Out checkpoint  : {args.out_checkpoint}")
    print("=" * 70)

    # ── Load architecture config from tiny.pt but DON'T restore weights ──────
    # We load the checkpoint for config/task_ids only, then reset model weights
    # to train from scratch on NVARC.
    import torch, copy
    base_ckpt = load_checkpoint(Path(args.base_checkpoint))

    # Build model + dataset. The checkpoint provides config + task_id whitelist.
    # Weights will be loaded from tiny.pt initially — they'll be overwritten
    # quickly by the high LR training on NVARC data.
    build_a = _build_args(args)
    model, dataset, dataloader, device, data_path = build_model_and_data(
        build_a, checkpoint=base_ckpt, is_eval=False  # weights loaded from base_ckpt
    )
    print(f"Model          : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Training tasks : {len(dataset.task_ids)}")
    print(f"Training seqs  : {len(dataset)}")
    print(f"Steps/epoch    : {len(dataloader)}")

    # ── Build train_model args ────────────────────────────────────────────────
    import argparse as _ap
    # train_model treats epochs as an ABSOLUTE target, not additional epochs.
    # Since tiny.pt is already at epoch=650, we must set target = 650 + our_epochs.
    # We also zero out the checkpoint's epoch/step counters so training starts fresh.
    import copy as _copy
    fresh_ckpt = _copy.deepcopy(base_ckpt)
    checkpoint_epoch = int(fresh_ckpt.get("epoch", 0))
    target_epochs = checkpoint_epoch + args.epochs
    # Reset step/epoch so train_model thinks we're starting from scratch
    # (but keep model weights and optimizer state)
    fresh_ckpt["epoch"] = 0
    fresh_ckpt["global_step"] = 0
    target_epochs = args.epochs  # now 0 + args.epochs = args.epochs
    print(f"  (checkpoint was at epoch {checkpoint_epoch}; "
          f"resetting counters, training {args.epochs} fresh epochs)")

    train_args = _ap.Namespace(
        epochs                      = args.epochs,
        optimizer                   = "normuon",
        normuon_lr                  = args.normuon_lr,
        adamw_lr                    = args.adamw_lr,
        weight_decay                = args.weight_decay,
        attention_weight_decay      = args.weight_decay,
        token_embedding_weight_decay= 0.0,
        task_embedding_weight_decay = 0.0,
        grad_clip                   = args.grad_clip,
        gradient_accumulation_steps = args.grad_accum,
        warmup_pct                  = args.warmup_pct,
        wsd_decay_start_pct         = args.wsd_decay_start_pct,
        lr_floor                    = args.lr_floor,
        save_path                   = Path(args.out_checkpoint),
        checkpoint_epochs           = args.checkpoint_every,
        do_validate                 = False,   # no val split in training data
        train_log_mode              = "10_steps",
        log_location                = "terminal",
        train_log_file              = None,
        batch_size                  = 1,
        val_batch_size              = 1,
    )

    print(f"\nStarting Phase 1 training ({args.epochs} epochs)...")
    print("NB: High LR will overwrite tiny.pt weights within 1-2 epochs.\n")

    train_model(
        args       = train_args,
        model      = model,
        dataloader = dataloader,
        dataset    = dataset,
        device     = device,
        data_path  = data_path,
        checkpoint = fresh_ckpt,  # counters reset to 0 so training actually runs
    )

    print("\n" + "=" * 70)
    print("PHASE 1 COMPLETE")
    print(f"  Checkpoint saved → {args.out_checkpoint}")
    print()
    print("Next — Phase 2 LR sweep:")
    print(f"  python prototype_more_data/train_phase2_lr_sweep.py \\")
    print(f"      --phase1-checkpoint {args.out_checkpoint} \\")
    print(f"      --data-path         assets/challenges.json \\")
    print(f"      --out-dir           runs/phase2_sweep \\")
    print(f"      --normuon-lrs       0.002 0.001 0.0005")
    print("=" * 70)


if __name__ == "__main__":
    main()
