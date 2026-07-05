"""
lora_ttt.py  —  LoRA Test-Time Training (TTT) for Prototype C  [v2]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY v2 FIXES THE ZERO-LOSS PROBLEM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

In v1, every task showed:
    TTT: 30 steps  loss 0.0000→0.0000
    baseline_CE = 0.0000

The LoRA adapters received zero gradient. Root cause:

  model.example_embedding[example_id] is a lookup table memorised during
  training on ALL 1307 eval tasks.  Calling model.forward(example_ids=X)
  retrieves the memorised context → loss ≈ 0 immediately → no gradient →
  LoRA learns nothing → baseline_CE stays 0 → RCOS signal stays dead.

v2 FIX:
  In apply_lora(), if example_id is supplied:
    1. Save the memorised embedding row
    2. Zero it out (model can no longer cheat)
    3. Make it trainable alongside LoRA

  run_ttt() then trains LoRA + the zeroed embedding jointly.
  After 30 steps both have specialised to this task from scratch.
  baseline_CE is now > 0 → RCOS signal is alive.

  remove_lora() restores the original embedding so the next task is clean.
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


# =============================================================================
# 1.  LoRALinear
# =============================================================================

class LoRALinear(nn.Module):
    """Wraps a frozen nn.Linear with a trainable low-rank bypass."""

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float) -> None:
        super().__init__()
        self.wrapped = wrapped
        d_out, d_in = wrapped.weight.shape
        self.rank  = rank
        self.scale = alpha / rank
        dev = wrapped.weight.device

        self.lora_A = nn.Parameter(
            torch.empty(rank, d_in, device=dev, dtype=torch.float32)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(d_out, rank, device=dev, dtype=torch.float32)
        )
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base     = self.wrapped(x)
        lora_out = (x.float() @ self.lora_A.T @ self.lora_B.T) * self.scale
        return base + lora_out.to(base.dtype)

    def extra_repr(self) -> str:
        d_out, d_in = self.wrapped.weight.shape
        return f"in={d_in}, out={d_out}, rank={self.rank}, scale={self.scale:.3f}"


# =============================================================================
# 2.  Layer discovery
# =============================================================================

_ATTENTION_HINTS: Tuple[str, ...] = (
    "q_proj", "k_proj", "v_proj", "out_proj", "o_proj",
    "query",  "key",    "value",
    "c_attn", "c_proj", "c_fc",
    "qkv",    "attn",
)

_MLP_HINTS: Tuple[str, ...] = (
    "fc1", "fc2", "mlp", "ffn", "feed_forward",
    "up_proj", "down_proj", "gate_proj",
)


def _find_target_linear_names(model: nn.Module) -> List[str]:
    all_linears: List[str] = []
    attn_names: List[str]  = []
    attn_mlp:   List[str]  = []

    for name, mod in model.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        if min(mod.weight.shape) < 16:
            continue
        all_linears.append(name)
        lname = name.lower()
        is_attn = any(kw in lname for kw in _ATTENTION_HINTS)
        is_mlp  = any(kw in lname for kw in _MLP_HINTS)
        if is_attn:
            attn_names.append(name)
            attn_mlp.append(name)
        elif is_mlp:
            attn_mlp.append(name)

    if attn_names:
        return attn_names
    if attn_mlp:
        return attn_mlp
    if all_linears:
        def _sz(n: str) -> int:
            return max(dict(model.named_modules())[n].weight.shape)  # type: ignore
        srt = sorted(all_linears, key=_sz, reverse=True)
        return srt[1:] or all_linears
    return []


def _get_parent_and_attr(model: nn.Module, dotpath: str) -> Tuple[nn.Module, str]:
    parts  = dotpath.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent, parts[-1]


def _find_example_embedding(model: nn.Module) -> Optional[nn.Embedding]:
    """Find the task-level context embedding (example_embedding / task_embedding)."""
    for attr in ("example_embedding", "task_embedding", "task_embeddings",
                 "puzzle_embedding", "task_emb"):
        obj = getattr(model, attr, None)
        if isinstance(obj, nn.Embedding):
            return obj
    for name, mod in model.named_modules():
        if isinstance(mod, nn.Embedding):
            lname = name.lower()
            if any(kw in lname for kw in ("example", "task", "puzzle")):
                return mod
    return None


# =============================================================================
# 3.  apply_lora / remove_lora
# =============================================================================

def apply_lora(
    model:        nn.Module,
    rank:         int   = 8,
    alpha:        float = 16.0,
    target_names: Optional[List[str]] = None,
    example_id:   Optional[int] = None,
) -> Dict[str, Any]:
    """
    Inject LoRA adapters in-place AND (v2) break the memorised embedding.

    example_id:  if provided, the corresponding row in model.example_embedding
                 is saved and zeroed so TTT sees real loss → real gradients.
    """
    for p in model.parameters():
        p.requires_grad_(False)

    names = target_names or _find_target_linear_names(model)
    replaced: Dict[str, nn.Linear] = {}

    for dotpath in names:
        parent, attr = _get_parent_and_attr(model, dotpath)
        orig = getattr(parent, attr)
        if not isinstance(orig, nn.Linear):
            continue
        setattr(parent, attr, LoRALinear(orig, rank=rank, alpha=alpha))
        replaced[dotpath] = orig

    log.info("apply_lora: LoRA rank=%d α=%.1f into %d layers",
             rank, alpha, len(replaced))

    # ── v2: zero the example_embedding row ───────────────────────────────────
    emb_saved:      Optional[torch.Tensor] = None
    emb_example_id: Optional[int]          = None

    if example_id is not None:
        emb = _find_example_embedding(model)
        if emb is not None and example_id < emb.num_embeddings:
            emb_saved      = emb.weight.data[example_id].clone()
            emb_example_id = example_id
            emb.weight.data[example_id].zero_()
            emb.weight.requires_grad_(True)
            log.info("apply_lora: zeroed example_embedding[%d]", example_id)
        else:
            log.warning("apply_lora: could not find/zero example_embedding "
                        "(example_id=%s)", example_id)

    return {
        "replaced":        replaced,
        "target_names":    names,
        "rank":            rank,
        "alpha":           alpha,
        "emb_saved":       emb_saved,
        "emb_example_id":  emb_example_id,
    }


def remove_lora(model: nn.Module, state: Dict[str, Any]) -> None:
    """Remove LoRA adapters and restore the embedding row."""
    for dotpath, orig in state["replaced"].items():
        parent, attr = _get_parent_and_attr(model, dotpath)
        cur = getattr(parent, attr)
        if isinstance(cur, LoRALinear):
            setattr(parent, attr, orig)

    for p in model.parameters():
        p.requires_grad_(True)

    # ── v2: restore embedding ─────────────────────────────────────────────────
    emb_saved      = state.get("emb_saved")
    emb_example_id = state.get("emb_example_id")

    if emb_saved is not None and emb_example_id is not None:
        emb = _find_example_embedding(model)
        if emb is not None:
            emb.weight.data[emb_example_id] = emb_saved
            emb.weight.requires_grad_(False)
            log.info("remove_lora: restored example_embedding[%d]", emb_example_id)

    log.info("remove_lora: restored %d Linear layers", len(state["replaced"]))


def get_lora_parameters(model: nn.Module) -> List[nn.Parameter]:
    params = []
    for mod in model.modules():
        if isinstance(mod, LoRALinear):
            params.append(mod.lora_A)
            params.append(mod.lora_B)
    return params


def count_lora_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in get_lora_parameters(model))


# =============================================================================
# 4.  run_ttt
# =============================================================================

def run_ttt(
    model:        nn.Module,
    demo_pairs:   List,
    example_id:   int,
    dihedral_id:  int,
    device:       torch.device,
    n_steps:      int   = 30,
    lr:           float = 3e-4,
    weight_decay: float = 1e-4,
    grad_clip:    float = 1.0,
    verbose:      bool  = False,
) -> Dict[str, Any]:
    """
    Fine-tune LoRA adapters + zeroed example_embedding row on demo pairs.

    v2 change: the example_embedding row was zeroed by apply_lora, so
    the model starts with no task context → loss > 0 → real gradients.
    We include the embedding in the optimizer and mask gradients to only
    update the example_id row (other rows get zero gradient).

    The model is put back in eval mode after training.
    """
    from rcos import build_sequence_and_targets, IGNORE_INDEX
    from common import compute_positions_3d

    lora_params = get_lora_parameters(model)
    if not lora_params:
        log.warning("run_ttt: no LoRA parameters found.")
        return {"n_examples": 0, "loss_start": None, "loss_end": None,
                "loss_history": [], "skipped": True, "n_lora_params": 0}

    n_lora = count_lora_parameters(model)

    # Build training examples
    max_seq  = model.config.max_seq_len
    examples: List[Tuple[List[int], List[int]]] = []

    for gp in demo_pairs:
        if gp.output is None:
            continue
        try:
            tok_list, targets, n_out = build_sequence_and_targets([gp])
        except Exception as e:
            log.debug("TTT build error: %s", e)
            continue
        if len(tok_list) > max_seq or n_out == 0:
            continue
        examples.append((tok_list, targets))

    if not examples:
        log.warning("run_ttt: no usable demo pairs (example_id=%d)", example_id)
        return {"n_examples": 0, "loss_start": None, "loss_end": None,
                "loss_history": [], "skipped": True, "n_lora_params": n_lora}

    # Build optimizer: LoRA params + example_embedding (if trainable)
    emb        = _find_example_embedding(model)
    has_emb_grad = (emb is not None and emb.weight.requires_grad)

    opt_params: List[nn.Parameter] = list(lora_params)
    if has_emb_grad:
        opt_params.append(emb.weight)

    optimizer = torch.optim.AdamW(
        opt_params, lr=lr, weight_decay=weight_decay, betas=(0.9, 0.999)
    )

    example_ids_t  = torch.tensor([example_id],  dtype=torch.long, device=device)
    dihedral_ids_t = torch.tensor([dihedral_id], dtype=torch.long, device=device)

    model.train()
    loss_history: List[float] = []
    n = len(examples)

    for step in range(n_steps):
        tok_list, targets = examples[step % n]
        seq_len   = len(tok_list)
        input_ids = torch.tensor(tok_list,  dtype=torch.long, device=device).unsqueeze(0)
        targets_t = torch.tensor(targets,   dtype=torch.long, device=device).unsqueeze(0)
        attn_mask = torch.ones(1, seq_len,  dtype=torch.bool,  device=device)

        positions_3d = compute_positions_3d(input_ids, attn_mask).to(
            device=device, dtype=torch.long
        )

        outputs = model.forward(
            input_ids=input_ids,
            example_ids=example_ids_t,
            dihedral_ids=dihedral_ids_t,
            attention_mask=attn_mask,
            targets=targets_t,
            compute_input_loss=False,
            positions_3d=positions_3d,
        )

        loss = outputs.get("loss") or outputs.get("output_loss")
        if loss is None or not torch.isfinite(loss):
            continue

        optimizer.zero_grad()
        loss.backward()

        # Mask embedding gradients: only update the example_id row
        if has_emb_grad and emb.weight.grad is not None:
            mask = torch.zeros_like(emb.weight.grad)
            mask[example_id] = 1.0
            emb.weight.grad.mul_(mask)

        nn.utils.clip_grad_norm_(lora_params, grad_clip)
        optimizer.step()

        val = float(loss.item())
        loss_history.append(val)

        if verbose and step % 10 == 0:
            print(f"    [TTT] step {step:3d}/{n_steps}  loss={val:.4f}")

    model.eval()

    emb_params = (emb.weight[example_id].numel()
                  if has_emb_grad and emb else 0)

    return {
        "n_examples":    n,
        "loss_start":    loss_history[0]  if loss_history else None,
        "loss_end":      loss_history[-1] if loss_history else None,
        "loss_history":  loss_history,
        "skipped":       False,
        "n_lora_params": n_lora + emb_params,
    }


# =============================================================================
# 5.  Diagnostic helpers
# =============================================================================

def lora_summary(model: nn.Module) -> str:
    """Human-readable summary of injected LoRA layers."""
    lines = []
    total = 0
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinear):
            d_out, d_in = mod.wrapped.weight.shape
            n = mod.rank * d_in + d_out * mod.rank  # actual param count
            total += n
            lines.append(f"  {name:<55s}  rank={mod.rank}  "
                         f"({d_out}×{d_in})  +{n:,} params")
    lines.append(f"  Total LoRA parameters: {total:,}")
    return "\n".join(lines)
