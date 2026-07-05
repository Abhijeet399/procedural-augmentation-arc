"""
embedding_surgery.py — Prototype C v9 Embedding Surgery (Sub-option A2)

Problem: model.example_embedding[example_id] is memorised for all 1307 tasks
including the 400 ARC eval tasks.  RCOS has to surgically zero it at inference
time (inside _run_forward_ce) because the memorised vector shortcuts the CE
signal to ~0.

Fix: permanently zero the 400 eval task embedding rows, then run a short
fine-tuning pass on the remaining ~900 training tasks so the model
redistributes that context signal into the transformer weights.
After surgery, baseline_CE will be naturally > 0 for eval tasks without
any hacks, which means --dihedral-avg and the RCOS gate will both work
correctly without the surgical zeroing workaround.

Usage:
    python embedding_surgery.py \
        --checkpoint runs/tiny.pt \
        --eval-tasks  assets/challenges.json \
        --train-data  assets/arc_combined.json \
        --output      runs/tiny_surgery.pt \
        --steps       300 \
        --lr          2e-5 \
        --device      cuda
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Optional, Set

import torch
import torch.nn as nn
from torch.optim import AdamW

# ── project imports (same as run_prototype_c_v9.py) ──────────────────────────
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent.parent
for _p in [str(_HERE), str(_ROOT / "src")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from build import build_model_and_data, load_checkpoint
from common import ARCExampleDataset, compute_positions_3d

# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Zero eval-task embeddings + fine-tune to stabilise."
    )
    p.add_argument("--checkpoint",  required=True,
                   help="Path to tiny.pt (the checkpoint to operate on).")
    p.add_argument("--eval-tasks",  required=True,
                   help="challenges.json containing the 400 ARC eval task IDs.")
    p.add_argument("--train-data",  default=None,
                   help="JSON with training tasks. Defaults to the checkpoint's "
                        "recorded data_path (which usually contains all tasks).")
    p.add_argument("--output",      default="runs/tiny_surgery.pt",
                   help="Where to write the surgically-modified checkpoint.")
    p.add_argument("--steps",       type=int,   default=300,
                   help="Gradient steps for the stabilisation fine-tune.")
    p.add_argument("--lr",          type=float, default=2e-5,
                   help="AdamW learning rate for fine-tuning.")
    p.add_argument("--batch-size",  type=int,   default=4)
    p.add_argument("--grad-clip",   type=float, default=1.0)
    p.add_argument("--device",      default="cuda")
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--no-finetune", action="store_true",
                   help="Just zero the embeddings and save — skip fine-tuning.")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_eval_task_ids(eval_tasks_path: str) -> Set[str]:
    with open(eval_tasks_path) as f:
        return set(json.load(f).keys())


def find_example_embedding(model: nn.Module) -> Optional[nn.Embedding]:
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "example" in name.lower():
            return module
    return None


def zero_eval_embeddings(
    model: nn.Module,
    dataset: ARCExampleDataset,
    eval_task_ids: Set[str],
) -> List[int]:
    """
    Permanently zero the embedding rows that correspond to eval task IDs.
    Returns the list of zeroed example_ids (for re-zeroing during fine-tuning).
    """
    emb = find_example_embedding(model)
    if emb is None:
        raise RuntimeError("Could not find example_embedding in model.")

    zeroed_ids: List[int] = []
    missing = []
    for tid in eval_task_ids:
        if tid not in dataset.task_id_to_example_id:
            missing.append(tid)
            continue
        eid = dataset.task_id_to_example_id[tid]
        emb.weight.data[eid].zero_()
        zeroed_ids.append(eid)

    print(f"  Zeroed {len(zeroed_ids)} eval embedding rows.")
    if missing:
        print(f"  WARNING: {len(missing)} eval task IDs not found in dataset "
              f"(e.g. {missing[:3]}) — their embeddings are unchanged.")
    return zeroed_ids


def re_zero_embeddings(emb: nn.Embedding, zeroed_ids: List[int]) -> None:
    """Re-zero eval rows after each optimizer step (prevents AdamW from restoring them)."""
    with torch.no_grad():
        for eid in zeroed_ids:
            emb.weight.data[eid].zero_()


# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning loop
# ─────────────────────────────────────────────────────────────────────────────

def finetune(
    model: nn.Module,
    dataset: ARCExampleDataset,
    eval_task_ids: Set[str],
    zeroed_ids: List[int],
    steps: int,
    lr: float,
    batch_size: int,
    grad_clip: float,
    device: torch.device,
) -> None:
    """
    Short fine-tuning pass on training tasks only (eval tasks excluded).

    After each optimizer step, re-zero the eval embedding rows so AdamW
    cannot restore them.
    """
    from common import create_dataloader

    # Build a whitelist that excludes the 400 eval tasks
    train_task_ids = [tid for tid in dataset.task_ids if tid not in eval_task_ids]
    print(f"\n  Fine-tuning on {len(train_task_ids)} training tasks "
          f"(excluded {len(eval_task_ids)} eval tasks).")

    # Subset dataset to training tasks only
    train_dataset = ARCExampleDataset(
        json_path=dataset._json_path,
        splits=("train",),
        include_outputs=True,
        max_seq_len=dataset.max_seq_len,
        task_whitelist=train_task_ids,
    )
    print(f"  Training dataset: {len(train_dataset.task_ids)} tasks, "
          f"{len(train_dataset.examples)} examples.")

    loader = create_dataloader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )
    loader_iter = iter(loader)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    emb = find_example_embedding(model)
    model.train()

    losses: list[float] = []
    t0 = perf_counter()

    for step in range(1, steps + 1):
        # Replenish iterator when exhausted
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids    = batch["input_ids"].to(device)
        example_ids  = batch["example_ids"].to(device)
        dihedral_ids = batch["dihedral_ids"].to(device)
        positions_3d = batch["positions_3d"].to(device)
        # Support padded (attention_mask) and packed varlen (cu_seqlens) formats
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
        _cu = batch.get("cu_seqlens")
        cu_seqlens = _cu.to(device) if _cu is not None else None
        _ms = batch.get("max_seqlen")
        max_seqlen = int(_ms) if _ms is not None else None
        _si = batch.get("sep_indices")
        sep_indices = _si.to(device) if _si is not None else None

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            outputs = model(
                input_ids,
                example_ids,
                dihedral_ids,
                attention_mask=attention_mask,
                sep_indices=sep_indices,
                compute_input_loss=False,
                positions_3d=positions_3d,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        loss = outputs.get("output_loss") or outputs.get("loss")
        if loss is None:
            print(f"  [step {step}] WARNING: no loss in outputs — skipping.")
            continue

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        # ── KEY: re-zero eval rows after every optimizer step ──────────────
        if emb is not None:
            re_zero_embeddings(emb, zeroed_ids)

        losses.append(float(loss.detach()))

        if step % 50 == 0 or step == 1:
            avg = sum(losses[-50:]) / len(losses[-50:])
            elapsed = perf_counter() - t0
            print(f"  step {step:4d}/{steps}  loss={avg:.4f}  "
                  f"({elapsed:.0f}s elapsed)")

    model.eval()
    print(f"\n  Fine-tuning complete. "
          f"Final 50-step avg loss: {sum(losses[-50:])/max(len(losses[-50:]),1):.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("EMBEDDING SURGERY — Sub-option A2")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Eval tasks  : {args.eval_tasks}")
    print(f"  Output      : {args.output}")
    print(f"  Fine-tune   : {'DISABLED (--no-finetune)' if args.no_finetune else f'{args.steps} steps @ lr={args.lr}'}")
    print("=" * 70 + "\n")

    # ── Step 1: Load eval task IDs ────────────────────────────────────────
    eval_task_ids = load_eval_task_ids(args.eval_tasks)
    print(f"Loaded {len(eval_task_ids)} eval task IDs from {args.eval_tasks}")

    # ── Step 2: Load checkpoint + build model ────────────────────────────
    ckpt = load_checkpoint(Path(args.checkpoint))
    if ckpt is None:
        raise RuntimeError(f"Could not load checkpoint from {args.checkpoint}")

    # Determine training data path
    train_data_path = args.train_data or ckpt.get("data_path")
    if train_data_path is None:
        raise RuntimeError(
            "Cannot determine training data path. "
            "Pass --train-data explicitly."
        )
    print(f"Training data path: {train_data_path}")

    build_args = argparse.Namespace(
        checkpoint_path=Path(args.checkpoint),
        data_path=Path(train_data_path),
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        enable_aug=False,
        enable_color_aug=False,
        enable_dihedral_aug=False,
        max_augments=0,
        color_apply_to_test=False,
        dihedral_apply_to_test=False,
    )
    model, dataset, _dl, device, _dp = build_model_and_data(
        build_args, checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks, {len(dataset.examples)} examples")

    # Attach json path to dataset so fine-tuning can reload it
    dataset._json_path = Path(train_data_path)
    dataset.max_seq_len = model.config.max_seq_len

    # ── Step 3: Verify eval coverage ─────────────────────────────────────
    found_in_dataset = sum(
        1 for tid in eval_task_ids if tid in dataset.task_id_to_example_id
    )
    print(f"\nEval tasks found in dataset: {found_in_dataset}/{len(eval_task_ids)}")

    # ── Step 4: Zero eval embeddings ─────────────────────────────────────
    print("\nZeroing eval task embedding rows...")
    zeroed_ids = zero_eval_embeddings(model, dataset, eval_task_ids)

    # Sanity-check: baseline CE should now be > 0 for a sample eval task
    _sample_tid = next(iter(eval_task_ids))
    if _sample_tid in dataset.task_id_to_example_id:
        _eid = dataset.task_id_to_example_id[_sample_tid]
        emb = find_example_embedding(model)
        _norm = emb.weight.data[_eid].norm().item() if emb is not None else -1
        print(f"  Sanity check — embedding norm for '{_sample_tid}': {_norm:.6f}  "
              f"(should be 0.0)")

    # ── Step 5: Fine-tuning pass ──────────────────────────────────────────
    if not args.no_finetune:
        finetune(
            model=model,
            dataset=dataset,
            eval_task_ids=eval_task_ids,
            zeroed_ids=zeroed_ids,
            steps=args.steps,
            lr=args.lr,
            batch_size=args.batch_size,
            grad_clip=args.grad_clip,
            device=device,
        )

    # ── Step 6: Save surgically-modified checkpoint ───────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Carry over everything from the original checkpoint but replace model_state
    new_ckpt = {k: v for k, v in ckpt.items() if k != "model_state"}
    new_ckpt["model_state"] = model.state_dict()
    new_ckpt["surgery_notes"] = {
        "zeroed_eval_task_ids": sorted(eval_task_ids),
        "zeroed_example_ids": sorted(zeroed_ids),
        "finetune_steps": 0 if args.no_finetune else args.steps,
        "finetune_lr": args.lr,
    }
    torch.save(new_ckpt, out_path)
    print(f"\nSaved surgery checkpoint → {out_path}")
    print("=" * 70)

    # ── Step 7: Verification hint ─────────────────────────────────────────
    print("\nNext step — run the v9e pipeline with the new checkpoint:")
    print(f"  python run_prototype_c_v9.py \\")
    print(f"      --checkpoint {out_path} \\")
    print(f"      --data-path  assets/challenges.json \\")
    print(f"      --solutions  assets/solutions.json \\")
    print(f"      --beam-width 10 \\")
    print(f"      --no-ttt \\")
    print(f"      --output-dir runs/prototype_c_v9_surgery")
    print()
    print("NOTE: The surgical zeroing in rcos.py (_run_forward_ce) is now")
    print("redundant for these 400 tasks but harmless to leave in place.")


if __name__ == "__main__":
    main()
