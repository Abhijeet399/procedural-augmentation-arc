"""
prototype_e/ttt.py — Test-Time Training (TTT)

Fine-tunes a copy of the model on the current task's demo pairs for a small
number of gradient steps before generating candidates.

Why it works:
  The model has memorised 1307 training tasks → CE ≈ 0 at eval → RCOS fails.
  But for an *unseen* eval task the model's weights aren't tuned for that
  specific input→output pattern.  A few gradient steps on the 3-5 demo pairs
  pushes the model toward that task's transformation rule, raising both:
    1. Oracle ceiling  (correct answer now appears in beam/samples)
    2. Greedy accuracy (the modal next-token is the right one more often)

Design:
  - Deep-copies the model so the base weights are never touched.
  - Fine-tunes on ALL (demo_input → demo_output) pairs concatenated in a
    single sequence, masking loss to output tokens only.
  - Uses AdamW with a cosine LR schedule, bf16 parameters.
  - Optionally fine-tunes only the last N transformer layers.

Usage (in ablation scripts):
    from ttt import ttt_finetune, TttConfig

    cfg = TttConfig(n_steps=30, lr=2e-4, last_n_layers=None)
    ttt_model = ttt_finetune(model, demo_pairs, example_id, best_d, device, cfg)
    # ... generate with ttt_model ...
    del ttt_model  # free GPU memory
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# These imports resolve when running from the repo root
from rcos import GridPair, build_sequence_and_targets
from common import IGNORE_INDEX, compute_positions_3d


# ─── Config ───────────────────────────────────────────────────────────────────

@dataclass
class TttConfig:
    n_steps:       int   = 30        # gradient steps
    lr:            float = 2e-4      # peak learning rate
    weight_decay:  float = 0.01      # AdamW weight decay
    last_n_layers: Optional[int] = None  # None = fine-tune all params
    # cosine schedule: warm up for warmup_steps, then decay to lr_min_ratio * lr
    warmup_steps:  int   = 3
    lr_min_ratio:  float = 0.1
    # clip gradients
    max_grad_norm: float = 1.0
    # dihedral augmentation: also train with mirrored/rotated versions
    aug_dihedrals: bool  = False


# ─── Core ─────────────────────────────────────────────────────────────────────

def _get_trainable_params(model, last_n_layers: Optional[int]):
    """Return parameter iterator for the model, optionally restricted to last N layers."""
    if last_n_layers is None:
        return list(model.parameters())

    # Collect all named parameter groups, take the last N transformer blocks
    # Typical naming: model.transformer.h.{i}.* or model.layers.{i}.*
    all_named = list(model.named_parameters())

    # Find layer parameter groups by detecting repeating numeric indices
    import re
    layer_indices = set()
    for name, _ in all_named:
        m = re.search(r'\.(\d+)\.', name)
        if m:
            layer_indices.add(int(m.group(1)))

    if not layer_indices:
        # Can't find layer structure — fine-tune everything
        return [p for _, p in all_named]

    sorted_layers = sorted(layer_indices)
    cutoff = sorted_layers[-last_n_layers] if last_n_layers <= len(sorted_layers) else sorted_layers[0]

    trainable = []
    for name, p in all_named:
        m = re.search(r'\.(\d+)\.', name)
        if m and int(m.group(1)) >= cutoff:
            trainable.append(p)
        elif not m and any(k in name for k in ('norm', 'head', 'lm_head', 'embed')):
            # Also fine-tune embedding/head/norm layers
            trainable.append(p)

    return trainable if trainable else [p for _, p in all_named]


def _build_ttt_batch(
    demo_pairs: List[GridPair],
    example_id: int,
    dihedral_id: int,
    device: torch.device,
    model,
):
    """Build a single training batch from demo pairs."""
    token_list, targets, n_out = build_sequence_and_targets(demo_pairs)
    if n_out == 0 or len(token_list) > model.config.max_seq_len:
        return None

    seq_len    = len(token_list)
    input_ids  = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)
    targets_t  = torch.tensor(targets,    dtype=torch.long, device=device).unsqueeze(0)
    attn_mask  = torch.ones(1, seq_len, dtype=torch.bool, device=device)
    ex_ids     = torch.tensor([example_id],  dtype=torch.long, device=device)
    dih_ids    = torch.tensor([dihedral_id], dtype=torch.long, device=device)
    pos_3d     = compute_positions_3d(input_ids, attn_mask).to(device=device, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        example_ids=ex_ids,
        dihedral_ids=dih_ids,
        attention_mask=attn_mask,
        targets=targets_t,
        positions_3d=pos_3d,
        compute_input_loss=False,
    )


def _cosine_lr(step: int, n_steps: int, warmup: int, lr: float, lr_min: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    t = (step - warmup) / max(n_steps - warmup, 1)
    return lr_min + 0.5 * (lr - lr_min) * (1 + math.cos(math.pi * t))


def ttt_finetune(
    model,
    demo_pairs:  List[GridPair],
    example_id:  int,
    dihedral_id: int,
    device:      torch.device,
    cfg:         TttConfig = TttConfig(),
    verbose:     bool = False,
) -> object:
    """
    Return a fine-tuned copy of the model, trained on the task's demo pairs.
    The original model is never modified.

    Args:
        model:        base model (stays frozen / in eval mode)
        demo_pairs:   list of GridPair(input, output) for this task's demos
        example_id:   task example embedding index
        dihedral_id:  best dihedral orientation
        device:       CUDA/CPU device
        cfg:          TTT hyperparameters
        verbose:      print step losses

    Returns:
        fine-tuned model copy (in eval mode, same dtype as original)
    """
    if not demo_pairs:
        return copy.deepcopy(model)

    # ── Build training batch ──────────────────────────────────────────────────
    batch = _build_ttt_batch(demo_pairs, example_id, dihedral_id, device, model)
    if batch is None:
        if verbose:
            print("    [TTT] sequence too long or no output tokens — skipping")
        return copy.deepcopy(model)

    # ── Copy model ────────────────────────────────────────────────────────────
    ttt_model = copy.deepcopy(model)
    ttt_model.train()
    if next(ttt_model.parameters()).dtype != torch.bfloat16:
        ttt_model.to(dtype=torch.bfloat16)

    # ── Optimiser ─────────────────────────────────────────────────────────────
    trainable = _get_trainable_params(ttt_model, cfg.last_n_layers)
    for p in ttt_model.parameters():
        p.requires_grad_(False)
    for p in trainable:
        p.requires_grad_(True)

    opt = torch.optim.AdamW(
        [p for p in trainable if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    lr_min = cfg.lr * cfg.lr_min_ratio

    # ── Training loop ─────────────────────────────────────────────────────────
    for step in range(cfg.n_steps):
        lr_now = _cosine_lr(step, cfg.n_steps, cfg.warmup_steps, cfg.lr, lr_min)
        for pg in opt.param_groups:
            pg["lr"] = lr_now

        opt.zero_grad()
        outputs = ttt_model.forward(**batch)
        loss = outputs.get("loss")
        if loss is None or not torch.isfinite(loss):
            break

        loss.backward()
        if cfg.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for p in trainable if p.grad is not None],
                cfg.max_grad_norm,
            )
        opt.step()

        if verbose:
            print(f"    [TTT] step {step+1:3d}/{cfg.n_steps}  lr={lr_now:.2e}  loss={loss.item():.4f}")

    ttt_model.eval()
    # Disable gradients on all params now that training is done
    for p in ttt_model.parameters():
        p.requires_grad_(False)

    return ttt_model
