"""
seq_builder.py — Converts raw ARC grids into tensor dicts for mdlARC.

Format confirmed from real dataloader batch:

  The model treats the ENTIRE task context (all demo pairs + test pair)
  as ONE flat sequence. There is ONE example_id (task-level, not pair-level),
  ONE dihedral_id, and ONE sep_index pointing to the last SEP token.

  input_ids    [total_len]       — all tokens concatenated
  positions_3d [total_len, 3]    — (row, col, plane) per token
  example_ids  [1]               — task-level embedding index
  dihedral_ids [1]               — D8 transform index for whole context
  cu_seqlens   [2]               — always [0, total_len]
  max_seqlen   int               — = total_len
  sep_indices  [1]               — absolute position of LAST sep token

  Planes: PLANE_INPUT=1, PLANE_SEP=2, PLANE_OUTPUT=3
  SEP token id: 10
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

SEP_TOKEN    = 10
PLANE_INPUT  = 1
PLANE_SEP    = 2
PLANE_OUTPUT = 3

_EXTRA_KEYS = {'output_mask', 'prompt_len'}


def _encode_pair(
    input_grid: np.ndarray,
    output_grid: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Encode (input, output) pair → (tokens, positions, sep_idx).
    sep_idx is the index of the SEP token WITHIN this pair's tokens.
    If output_grid is None, only input+SEP is encoded.
    """
    h_in, w_in = input_grid.shape
    n_in = h_in * w_in

    in_toks = input_grid.astype(np.int64).flatten()
    rows_in = np.repeat(np.arange(h_in, dtype=np.int64), w_in)
    cols_in = np.tile(np.arange(w_in, dtype=np.int64), h_in)
    in_pos  = np.stack([rows_in, cols_in,
                        np.full(n_in, PLANE_INPUT, dtype=np.int64)], axis=1)

    sep_tok = np.array([SEP_TOKEN], dtype=np.int64)
    sep_pos = np.array([[0, 0, PLANE_SEP]], dtype=np.int64)
    sep_idx = n_in   # SEP is at index n_in within this pair

    parts_t = [in_toks, sep_tok]
    parts_p = [in_pos,  sep_pos]

    if output_grid is not None:
        h_out, w_out = output_grid.shape
        n_out = h_out * w_out
        out_toks = output_grid.astype(np.int64).flatten()
        rows_out = np.repeat(np.arange(h_out, dtype=np.int64), w_out)
        cols_out = np.tile(np.arange(w_out, dtype=np.int64), h_out)
        out_pos  = np.stack([rows_out, cols_out,
                             np.full(n_out, PLANE_OUTPUT, dtype=np.int64)], axis=1)
        parts_t.append(out_toks)
        parts_p.append(out_pos)

    return np.concatenate(parts_t), np.concatenate(parts_p), sep_idx


def build_task_sequence(
    train_pairs:       list[tuple[np.ndarray, np.ndarray]],
    test_input:        np.ndarray,
    test_output:       Optional[np.ndarray],
    task_example_id:   int,
    dihedral_id:       int = 0,
    device:            str = 'cuda',
) -> dict:
    """
    Build the model input dict for one task.

    All demo pairs + the test pair are concatenated into ONE flat sequence.
    The model receives a single example_id (task-level), single dihedral_id,
    and a single sep_index pointing to the last SEP in the sequence.

    Returns dict with:
        input_ids    [total_len]   int64
        positions_3d [total_len,3] int64
        example_ids  [1]           int64
        dihedral_ids [1]           int64
        cu_seqlens   [2]           int32  — always [0, total_len]
        max_seqlen   int
        sep_indices  [1]           int64  — absolute position of last SEP
        output_mask  [total_len]   bool   — True at test output positions
        prompt_len   int           — tokens before test output
    """
    all_toks: list[np.ndarray] = []
    all_pos:  list[np.ndarray] = []
    running = 0

    # Demo pairs
    for inp, out in train_pairs:
        toks, pos, _ = _encode_pair(inp, out)
        all_toks.append(toks)
        all_pos.append(pos)
        running += len(toks)

    # Test pair
    test_toks, test_pos, test_sep_local = _encode_pair(test_input, test_output)
    all_toks.append(test_toks)
    all_pos.append(test_pos)
    last_sep_abs = running + test_sep_local  # absolute position of last SEP

    input_ids    = np.concatenate(all_toks)
    positions_3d = np.concatenate(all_pos)
    total_len    = len(input_ids)

    # Output mask: test output tokens = everything after last_sep_abs + 1
    output_mask = np.zeros(total_len, dtype=bool)
    if test_output is not None:
        n_out = test_output.size
        output_mask[-n_out:] = True

    prompt_len = total_len - (test_output.size if test_output is not None else 0)

    def t(arr, dtype=torch.long):
        return torch.tensor(arr, dtype=dtype, device=device)

    return {
        'input_ids':    t(input_ids),
        'positions_3d': t(positions_3d),
        'example_ids':  t(np.array([task_example_id], dtype=np.int64)),
        'dihedral_ids': t(np.array([dihedral_id],      dtype=np.int64)),
        'cu_seqlens':   t(np.array([0, total_len],     dtype=np.int32), torch.int32),
        'max_seqlen':   total_len,
        'sep_indices':  t(np.array([last_sep_abs],     dtype=np.int64)),
        'output_mask':  t(output_mask, torch.bool),
        'prompt_len':   prompt_len,
    }


def build_augmented_sequence(
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    candidate_output: np.ndarray,
    task_example_id: int,
    aug,
    device:          str = 'cuda',
) -> dict:
    """Build model input for (aug(task), aug(candidate))."""
    pairs = train_pairs
    if aug.example_order is not None:
        pairs = [train_pairs[i] for i in aug.example_order]
    aug_pairs = [(aug.apply_to_grid(inp), aug.apply_to_grid(out)) for inp, out in pairs]
    aug_test_in   = aug.apply_to_grid(test_input)
    aug_candidate = aug.apply_to_grid(candidate_output)
    return build_task_sequence(
        train_pairs=aug_pairs, test_input=aug_test_in,
        test_output=aug_candidate, task_example_id=task_example_id,
        dihedral_id=aug.d8_idx, device=device,
    )


def build_prompt_sequence(
    train_pairs:     list[tuple[np.ndarray, np.ndarray]],
    test_input:      np.ndarray,
    task_example_id: int,
    aug,
    device:          str = 'cuda',
) -> dict:
    """Build prompt-only sequence (no candidate output appended)."""
    pairs = train_pairs
    if aug.example_order is not None:
        pairs = [train_pairs[i] for i in aug.example_order]
    aug_pairs   = [(aug.apply_to_grid(inp), aug.apply_to_grid(out)) for inp, out in pairs]
    aug_test_in = aug.apply_to_grid(test_input)
    return build_task_sequence(
        train_pairs=aug_pairs, test_input=aug_test_in,
        test_output=None, task_example_id=task_example_id,
        dihedral_id=aug.d8_idx, device=device,
    )


def to_forward_kwargs(seq_dict: dict) -> dict:
    """Strip metadata keys before calling model.forward()."""
    return {k: v for k, v in seq_dict.items() if k not in _EXTRA_KEYS}


_REPO_ROOT = str(Path(__file__).resolve().parent.parent.parent.parent)


def validate_format(
    checkpoint_path: str = f'{_REPO_ROOT}/runs/tiny.pt',
    data_path:       str = f'{_REPO_ROOT}/assets/challenges.json',
    mdlarc_root:     str = _REPO_ROOT,
) -> None:
    """Compare a real dataloader batch to what build_task_sequence produces."""
    import sys, importlib.util as ilu, types, torch
    for p in [mdlarc_root, f"{mdlarc_root}/src"]:
        if p not in sys.path:
            sys.path.insert(0, p)

    def load_mod(name, rel):
        spec = ilu.spec_from_file_location(name, f"{mdlarc_root}/{rel}")
        mod  = ilu.module_from_spec(spec); spec.loader.exec_module(mod)
        return mod

    build = load_mod('mdlarc_build', 'src/build.py')
    ckpt  = build.load_checkpoint(Path(checkpoint_path))

    args = types.SimpleNamespace(data_path=data_path, batch_size=1, num_workers=0, seed=42)
    _ret    = build.build_model_and_data(args, ckpt, is_eval=True)
    model   = _ret[0]; dataset = _ret[1]; loader = _ret[2]

    batch = next(iter(loader))
    print("=== Real dataloader batch ===")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: shape={str(v.shape):20s} dtype={v.dtype} "
                  f"min={v.min().item()} max={v.max().item()}")
        else:
            print(f"  {k:20s}: {type(v).__name__} = {v}")

    with open(data_path) as f:
        challenges = json.load(f)

    task_ids = ckpt['task_ids']
    first_eval_id = next(
        tid for tid in task_ids if tid in challenges and 'test' in challenges[tid]
    )
    task_data   = challenges[first_eval_id]
    train_pairs = [(np.array(p['input'], dtype=np.int64),
                    np.array(p['output'], dtype=np.int64))
                   for p in task_data['train']]
    test_input  = np.array(task_data['test'][0]['input'], dtype=np.int64)

    # Get the task-level example_id from dataset
    task_example_id = int(dataset.task_id_to_example_id[first_eval_id])

    from augmentations import identity_aug
    seq = build_task_sequence(
        train_pairs=train_pairs, test_input=test_input,
        test_output=None, task_example_id=task_example_id,
        dihedral_id=0, device='cpu',
    )

    print(f"\n=== Our build_task_sequence (task={first_eval_id}) ===")
    for k, v in seq.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:20s}: shape={str(v.shape):20s} dtype={v.dtype} "
                  f"min={v.min().item()} max={v.max().item()}")
        else:
            print(f"  {k:20s}: {v}")

    print("\n=== Checks ===")
    print(f"  example_ids shape [1]?  {seq['example_ids'].shape}")
    print(f"  dihedral_ids shape [1]? {seq['dihedral_ids'].shape}")
    print(f"  cu_seqlens shape [2]?   {seq['cu_seqlens'].shape}")
    print(f"  sep_indices shape [1]?  {seq['sep_indices'].shape}")
    sep_pos = int(seq['sep_indices'][0])
    sep_tok = int(seq['input_ids'][sep_pos])
    print(f"  Token at sep_indices[0]={sep_pos}: {sep_tok}  (should be {SEP_TOKEN})")
    print(f"  task_example_id: {task_example_id}")


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'validate':
        validate_format()
