"""
poe_scorer.py — Product of Experts (PoE) candidate scoring.

poe_score(y) = (1/T) * sum_i log P_model(aug_i(y) | aug_i(prompt))

where T = number of output tokens (per-token normalization).

KEY FIX: model.example_embedding[task_example_id] is memorized for all
1307 ARC tasks including the 400 eval tasks.  Without zeroing it, CE ≈ 0
regardless of candidate quality, making PoE scores near-random.
We zero the embedding before each forward pass and restore it after.
"""
from __future__ import annotations
import inspect
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from augmentations import AugParams, sample_poe_augs, hash_grid
from seq_builder import build_augmented_sequence, to_forward_kwargs, build_task_sequence

MAX_SEQ_LEN = 1863


def _accepted(model):
    return set(inspect.signature(model.forward).parameters.keys())


@torch.no_grad()
def _score_single(
    model,
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    candidate:       np.ndarray,
    task_example_id: int,
    aug:             AugParams,
    device:          str,
) -> float:
    """Per-token log P_model(aug(candidate) | aug(prompt)), with embedding zeroed."""
    # Truncate train pairs to fit context
    pairs = train_pairs
    while len(pairs) > 0:
        check = build_task_sequence(
            train_pairs=[(aug.apply_to_grid(i), aug.apply_to_grid(o)) for i, o in pairs],
            test_input=aug.apply_to_grid(test_input),
            test_output=aug.apply_to_grid(candidate),
            task_example_id=task_example_id, dihedral_id=aug.d8_idx, device='cpu',
        )
        if int(check['max_seqlen']) <= MAX_SEQ_LEN:
            break
        pairs = pairs[1:]

    if not pairs:
        return 0.0

    from augmentations import AugParams as _AP
    aug_no_order = _AP(d8_idx=aug.d8_idx, color_perm=aug.color_perm, example_order=None)

    seq = build_augmented_sequence(
        train_pairs=pairs, test_input=test_input,
        candidate_output=candidate, task_example_id=task_example_id,
        aug=aug_no_order, device=device,
    )
    output_mask = seq['output_mask']
    out_positions = output_mask.nonzero(as_tuple=True)[0]
    if len(out_positions) == 0:
        return 0.0

    fwd = {k: v for k, v in to_forward_kwargs(seq).items() if k in _accepted(model)}

    # ---- KEY FIX: zero the memorized task embedding before scoring ----
    # The model memorizes eval task embeddings -> CE ≈ 0 without this fix.
    # Zero the embedding so CE reflects actual candidate quality vs. the demos.
    emb = model.example_embedding.weight.data
    saved = emb[task_example_id].clone()
    emb[task_example_id].zero_()

    try:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**fwd)
    finally:
        # Always restore, even if an exception occurs
        emb[task_example_id].copy_(saved)

    if isinstance(out, tuple): out = out[0]
    if isinstance(out, dict):
        for key in ('logits', 'lm_logits', 'output_logits'):
            if key in out:
                out = out[key]; break
        else:
            out = next(v for v in out.values() if isinstance(v, torch.Tensor) and v.dim() >= 2)
    if out.dim() == 3: out = out.squeeze(0)
    if isinstance(out, tuple): out = out[0]

    input_ids      = seq['input_ids']
    pred_positions = out_positions - 1
    target_tokens  = input_ids[out_positions]
    log_probs = F.log_softmax(out[pred_positions].float(), dim=-1)
    total_log_prob = float(log_probs[
        torch.arange(len(target_tokens), device=device), target_tokens
    ].sum().item())

    # ---- Per-token normalization ----
    # Normalize by output token count so scores are comparable across different
    # candidate sizes and are not biased toward smaller grids.
    n_tokens = len(out_positions)
    return total_log_prob / n_tokens if n_tokens > 0 else 0.0


@torch.no_grad()
def score_candidates_poe(
    model,
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    candidates:      list[np.ndarray],
    task_example_id: int,
    n_aug:           int = 16,
    seed:            int = 42,
    device:          str = 'cuda',
    verbose:         bool = False,
) -> list[tuple[np.ndarray, float]]:
    """Score candidates via PoE. Returns sorted (grid, score) list (higher = better)."""
    if not candidates:
        return []

    model.eval()
    rng  = np.random.default_rng(seed)
    augs = sample_poe_augs(n=n_aug, rng=rng, n_train=len(train_pairs), permute_examples=True)

    # Deduplicate candidates
    seen:   dict[bytes, int]  = {}
    unique: list[np.ndarray]  = []
    for c in candidates:
        h = hash_grid(c)
        if h not in seen:
            seen[h] = len(unique); unique.append(c)

    results = []
    for cand in unique:
        total = 0.0
        for aug in augs:
            try:
                total += _score_single(model, train_pairs, test_input,
                                       cand, task_example_id, aug, device)
            except Exception as e:
                if verbose: print(f'    [PoE] aug d8={aug.d8_idx} error: {e}')
        results.append((cand, total))
        if verbose:
            print(f'    [PoE] shape={cand.shape} poe={total:.3f}')

    results.sort(key=lambda x: -x[1])
    return results


def select_top2(scored):
    a1 = scored[0][0] if len(scored) > 0 else None
    a2 = scored[1][0] if len(scored) > 1 else None
    return a1, a2
