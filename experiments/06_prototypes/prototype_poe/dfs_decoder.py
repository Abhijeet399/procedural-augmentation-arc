"""
dfs_decoder.py — Threshold-filtered DFS candidate generation.

ARChitects/NVARC: DFS over output tokens, keeping branches with
conditional probability >= threshold T (default ~9%).
"""
from __future__ import annotations
import inspect
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from augmentations import AugParams, sample_poe_augs
from seq_builder import build_task_sequence, to_forward_kwargs, SEP_TOKEN, PLANE_OUTPUT

MAX_SEQ_LEN = 1863  # model's hard context limit


def _infer_output_shape(train_pairs):
    shapes = Counter(out.shape for _, out in train_pairs)
    return shapes.most_common(1)[0][0] if shapes else None


def _accepted(model):
    return set(inspect.signature(model.forward).parameters.keys())


@torch.no_grad()
def dfs_generate_single_aug(
    model,
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    output_shape:    tuple[int, int],
    task_example_id: int,
    aug:             AugParams,
    threshold:       float = 0.09,
    max_live:        int = 64,
    device:          str = 'cuda',
) -> list[tuple[np.ndarray, float]]:
    """DFS generation for ONE augmentation. Returns (canonical_grid, log_prob) pairs."""
    H, W = output_shape
    aug_pairs   = [(aug.apply_to_grid(i), aug.apply_to_grid(o)) for i, o in train_pairs]
    aug_test_in = aug.apply_to_grid(test_input)
    # Augmented output shape (D8 may swap H/W)
    aug_H, aug_W = aug.apply_to_grid(np.zeros((H, W), dtype=np.int64)).shape
    n_out = aug_H * aug_W

    acc = _accepted(model)

    # Truncate demo pairs to fit within model context limit.
    # Drop from the front (oldest pairs first) until prompt fits.
    pairs_to_use = aug_pairs
    while len(pairs_to_use) > 0:
        test_seq = build_task_sequence(
            train_pairs=pairs_to_use, test_input=aug_test_in, test_output=None,
            task_example_id=task_example_id, dihedral_id=aug.d8_idx, device='cpu',
        )
        if int(test_seq['max_seqlen']) <= MAX_SEQ_LEN:
            break
        pairs_to_use = pairs_to_use[1:]  # drop oldest pair

    # Build the base prompt (all demo pairs + test input + SEP, no test output)
    base = build_task_sequence(
        train_pairs=pairs_to_use, test_input=aug_test_in,
        test_output=None, task_example_id=task_example_id,
        dihedral_id=aug.d8_idx, device=device,
    )
    prompt_ids   = base['input_ids'].cpu().numpy()
    prompt_pos   = base['positions_3d'].cpu().numpy()
    prompt_len   = int(base['prompt_len'])
    last_sep_abs = int(base['sep_indices'][0])

    def output_pos(step):
        return (step // aug_W, step % aug_W, PLANE_OUTPUT)

    def _fwd(ids_np, pos_np):
        n = len(ids_np)
        fwd = {
            'input_ids':    torch.tensor(ids_np,  dtype=torch.long,  device=device),
            'positions_3d': torch.tensor(pos_np,  dtype=torch.long,  device=device),
            'example_ids':  base['example_ids'],
            'dihedral_ids': base['dihedral_ids'],
            'cu_seqlens':   torch.tensor([0, n],  dtype=torch.int32, device=device),
            'max_seqlen':   n,
            'sep_indices':  base['sep_indices'],
        }
        fwd = {k: v for k, v in fwd.items() if k in acc}
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**fwd)
        # Model may return a dict (when sep_indices triggers output_loss)
        # or a plain tensor (logits). Extract logits robustly.
        if isinstance(out, tuple):
            out = out[0]
        if isinstance(out, dict):
            for key in ('logits', 'lm_logits', 'output_logits'):
                if key in out:
                    out = out[key]
                    break
            else:
                # Find first 2D+ tensor value
                for v in out.values():
                    if isinstance(v, torch.Tensor) and v.dim() >= 2:
                        out = v
                        break
        if out.dim() == 3:
            out = out.squeeze(0)
        return out  # [N, vocab_size]

    # DFS state: (partial_output_tokens, cumulative_log_prob)
    active:    list[tuple[list[int], float]] = [([], 0.0)]
    completed: list[tuple[list[int], float]] = []

    for step in range(n_out):
        if not active:
            break
        next_active = []
        for partial, acc_lp in active:
            ids = list(prompt_ids) + partial
            pos = list(prompt_pos.tolist()) + [list(output_pos(i)) for i in range(len(partial))]
            try:
                out = _fwd(np.array(ids, dtype=np.int64), np.array(pos, dtype=np.int64))
            except Exception as e:
                print(f"  [DFS] fwd error step={step}: {e}")
                continue

            probs = F.softmax(out[-1, :10].float(), dim=0)
            if step == 0 and partial == []:
                top3 = probs.topk(3)
                print(f"  [DFS diag] step=0 aug d8={aug.d8_idx} "
                      f"top3_probs={[f'{p:.3f}' for p in top3.values.tolist()]} "
                      f"top3_toks={top3.indices.tolist()} "
                      f"threshold={threshold}")
            for tok in range(10):
                p = float(probs[tok])
                if p >= threshold:
                    new_p = partial + [tok]
                    new_lp = acc_lp + float(np.log(max(p, 1e-12)))
                    if len(new_p) == n_out:
                        completed.append((new_p, new_lp))
                    else:
                        next_active.append((new_p, new_lp))

        if len(next_active) > max_live:
            next_active.sort(key=lambda x: -x[1])
            next_active = next_active[:max_live]
        active = next_active

    for partial, lp in active:
        if len(partial) == n_out:
            completed.append((partial, lp))

    results = []
    seen: set[bytes] = set()
    for tok_list, lp in completed:
        aug_grid  = np.array(tok_list, dtype=np.int64).reshape(aug_H, aug_W)
        canonical = aug.apply_inverse_to_grid(aug_grid)
        key = canonical.tobytes()
        if key not in seen:
            seen.add(key)
            results.append((canonical, lp))
    return results


@torch.no_grad()
def generate_candidates_dfs(
    model,
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    task_example_id: int,
    output_shape:    Optional[tuple[int, int]] = None,
    n_aug_generate:  int = 8,
    threshold:       float = 0.09,
    max_live:        int = 64,
    seed:            int = 42,
    device:          str = 'cuda',
    verbose:         bool = False,
) -> list[np.ndarray]:
    """Generate diverse candidates via DFS across multiple augmentations."""
    if output_shape is None:
        output_shape = _infer_output_shape(train_pairs)
    if output_shape is None:
        raise ValueError("Could not infer output shape")

    rng  = np.random.default_rng(seed)
    augs = sample_poe_augs(n=n_aug_generate, rng=rng,
                           n_train=len(train_pairs), permute_examples=False)

    all_candidates: list[np.ndarray] = []
    seen: set[bytes] = set()

    for i, aug in enumerate(augs):
        if verbose:
            print(f"  [DFS] aug {i+1}/{len(augs)} d8={aug.d8_idx}")
        try:
            results = dfs_generate_single_aug(
                model=model, train_pairs=train_pairs, test_input=test_input,
                output_shape=output_shape, task_example_id=task_example_id,
                aug=aug, threshold=threshold, max_live=max_live, device=device,
            )
        except Exception as e:
            if verbose: print(f"  [DFS] aug {i+1} failed: {e}")
            continue

        for cand, _ in results:
            key = cand.tobytes()
            if key not in seen:
                seen.add(key); all_candidates.append(cand)

        if verbose:
            print(f"         -> {len(results)} completions, {len(all_candidates)} unique total")

    return all_candidates


@torch.no_grad()
def generate_candidates_sample(
    model,
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    task_example_id: int,
    output_shape:    tuple[int, int],
    n_samples:       int = 30,
    temperatures:    tuple[float, ...] = (0.7, 0.8, 1.0, 1.2),
    seed:            int = 42,
    device:          str = 'cuda',
) -> list[np.ndarray]:
    """Greedy + temperature sampling fallback."""
    H, W = output_shape
    rng = np.random.default_rng(seed)
    acc = _accepted(model)

    # Truncate to model context limit
    pairs_for_sample = train_pairs
    while len(pairs_for_sample) > 0:
        test_seq = build_task_sequence(
            train_pairs=pairs_for_sample, test_input=test_input, test_output=None,
            task_example_id=task_example_id, dihedral_id=0, device='cpu',
        )
        if int(test_seq['max_seqlen']) <= MAX_SEQ_LEN:
            break
        pairs_for_sample = pairs_for_sample[1:]

    base = build_task_sequence(
        train_pairs=pairs_for_sample, test_input=test_input,
        test_output=None, task_example_id=task_example_id,
        dihedral_id=0, device=device,
    )
    prompt_ids  = base['input_ids'].cpu().numpy()
    prompt_pos  = base['positions_3d'].cpu().numpy().tolist()
    prompt_len  = int(base['prompt_len'])

    def output_pos(step): return (step // W, step % W, PLANE_OUTPUT)

    def _fwd(ids_np, pos_np):
        n = len(ids_np)
        fwd = {
            'input_ids':    torch.tensor(ids_np, dtype=torch.long,  device=device),
            'positions_3d': torch.tensor(pos_np, dtype=torch.long,  device=device),
            'example_ids':  base['example_ids'],
            'dihedral_ids': base['dihedral_ids'],
            'cu_seqlens':   torch.tensor([0, n], dtype=torch.int32, device=device),
            'max_seqlen':   n,
            'sep_indices':  base['sep_indices'],
        }
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            out = model(**{k: v for k, v in fwd.items() if k in acc})
        if isinstance(out, tuple): out = out[0]
        if isinstance(out, dict):
            for key in ('logits', 'lm_logits', 'output_logits'):
                if key in out:
                    out = out[key]; break
            else:
                out = next(v for v in out.values() if isinstance(v, torch.Tensor) and v.dim() >= 2)
        if out.dim() == 3: out = out.squeeze(0)
        return out

    candidates: list[np.ndarray] = []
    seen: set[bytes] = set()

    def run_one(temp):
        ids = list(prompt_ids)
        pos = list(prompt_pos)
        for step in range(H * W):
            out = _fwd(np.array(ids, dtype=np.int64), np.array(pos, dtype=np.int64))
            logits = out[-1, :10].float()
            if temp is None:
                next_tok = int(logits.argmax())
            else:
                next_tok = int(torch.multinomial(F.softmax(logits / temp, dim=0), 1))
            ids.append(next_tok)
            pos.append(list(output_pos(step)))
        return np.array(ids[prompt_len:], dtype=np.int64).reshape(H, W)

    try:
        g = run_one(None)
        key = g.tobytes()
        if key not in seen: seen.add(key); candidates.append(g)
    except Exception: pass

    for _ in range(n_samples):
        try:
            g = run_one(float(rng.choice(temperatures)))
            key = g.tobytes()
            if key not in seen: seen.add(key); candidates.append(g)
        except Exception: pass

    return candidates
