"""
src/canonicalize.py — Orientation scoring + canonical inference (Prototype A core).

Provides the full public API used by run_prototype_a.py, run_lcte_full.py,
and run_prototype_c.py:

    score_task_orientations(model, demo_examples, device)
        → (best_d, losses_per_orientation)

    run_canonical_inference_for_task(model, demo_examples, test_examples,
                                     best_d, device, color_mapping,
                                     color_mapping_inv, max_new_tokens, ...)
        → List[Optional[predicted_grid]]

    run_prototype_a_evaluation(model, dataset, device, ...)
        → dict with 'submission', 'orientation_stats', 'n_tasks', 'n_predicted'

    save_submission(submission, output_dir, run_name)
        → Path

    build_color_canon_mapping(demo_input_grids)  → List[int]
    build_color_inverse_mapping(fwd_mapping)     → List[int]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KEY FIX (v3): _build_prompt_from_canonical_demos_and_test now reads
tokens_by_dihedral[best_d] directly from each SequenceExample instead
of re-encoding grids via encode_example().  Re-encoding was causing:
  1. Subtly different token sequences from the pre-computed canonical tokens.
  2. A double-transform: physical rotation already applied when GridPairs
     were extracted, plus the model's dihedral embedding applying it again.
Prototype A always used tokens_by_dihedral — this function now matches
that behaviour exactly.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common import (
    IGNORE_INDEX,
    IO_SEPARATOR_TOKEN_ID,
    VOCAB_SIZE,
    SequenceExample,
    apply_color_permutation_to_grid,
    apply_inverse_dihedral_transform,
    compute_positions_3d,
    extract_output_tokens,
    is_rectangular_grid,
    tokens_to_grid,
)
from evaluate import (
    DEFAULT_MAX_NEW_TOKENS,
    batched_greedy_generate,
    _build_prompt_from_tokens,
)


_DIHEDRAL_NAMES = [
    "identity", "rot90", "rot180", "rot270",
    "flip_h", "flip_v", "flip_main", "flip_anti",
]


# =============================================================================
# 1. Color canonicalization
# =============================================================================

def build_color_canon_mapping(demo_input_grids: Sequence[Sequence[Sequence[int]]]) -> List[int]:
    """
    Build a deterministic color permutation from demo input grids.

    Colors are ranked by descending frequency across *all* demo input grids.
    Ties are broken by original color value (lower → lower canonical index).
    Color 0 (background) is always preserved.  Only colors 1-9 participate.

    Returns a list of length VOCAB_SIZE where mapping[original_color] = canonical_color.
    """
    freq: Counter = Counter()
    used: set = set()
    for grid in demo_input_grids:
        for row in grid:
            for val in row:
                v = int(val)
                if 1 <= v <= 9:
                    freq[v] += 1
                    used.add(v)

    sorted_used = sorted(used, key=lambda c: (-freq[c], c))
    mapping = list(range(VOCAB_SIZE))   # identity for everything by default
    for canonical_idx, original_color in enumerate(sorted_used, start=1):
        mapping[original_color] = canonical_idx
    return mapping


def build_color_inverse_mapping(fwd_mapping: Sequence[int]) -> List[int]:
    """Invert a color mapping: inverse[fwd_mapping[c]] = c."""
    inv = list(range(len(fwd_mapping)))
    for original, canonical in enumerate(fwd_mapping):
        if 0 <= canonical < len(inv):
            inv[canonical] = original
    return inv


# =============================================================================
# 2. Orientation CE scoring
# =============================================================================

@torch.no_grad()
def score_task_orientations(
    model,
    demo_examples: List[SequenceExample],
    device: torch.device,
    use_amp: bool = True,
    color_mapping: Optional[List[int]] = None,
) -> Tuple[int, List[float]]:
    """
    Score all 8 dihedral orientations on the demo pairs by output CE.

    For each orientation d, compute the mean output cross-entropy over all
    demo pairs using the pre-computed tokens_by_dihedral[d].  Select the
    orientation with minimum CE.

    Args:
        model            : trained TinyTransformer (eval mode)
        demo_examples    : SequenceExample objects with tokens_by_dihedral
        device           : target device
        use_amp          : use bfloat16 autocast
        color_mapping    : optional color-remapping applied to tokens

    Returns:
        (best_d, losses)  where best_d = argmin(losses), losses has length 8
    """
    model.eval()
    if not demo_examples:
        return 0, [math.inf] * 8

    example_id = int(getattr(demo_examples[0], "example_id", 0))
    color_map_t: Optional[torch.Tensor] = None
    if color_mapping is not None:
        color_map_t = torch.tensor(color_mapping, dtype=torch.long, device=device)

    losses: List[float] = []

    for d in range(8):
        total_loss   = 0.0
        total_tokens = 0

        for ex in demo_examples:
            if not getattr(ex, "has_output", True):
                continue

            # Use pre-computed canonical tokens
            if ex.tokens_by_dihedral is not None:
                tok = ex.tokens_by_dihedral[d].to(device)
                pos = (ex.cached_positions_by_dihedral[d].to(device)
                       if ex.cached_positions_by_dihedral is not None
                       else None)
                sep = (ex.sep_index_by_dihedral[d]
                       if getattr(ex, "sep_index_by_dihedral", None) is not None
                       else ex.sep_index)
            else:
                tok = ex.tokens.to(device)
                pos = (getattr(ex, "cached_positions", None))
                if pos is not None:
                    pos = pos.to(device)
                sep = ex.sep_index

            if color_map_t is not None:
                tok = color_map_t[tok]

            seq_len = tok.size(0)
            if seq_len > model.config.max_seq_len:
                continue

            input_ids = tok.unsqueeze(0)
            attn_mask = torch.ones(1, seq_len, dtype=torch.bool, device=device)
            ex_ids_t  = torch.tensor([example_id], dtype=torch.long, device=device)
            dih_ids_t = torch.tensor([d], dtype=torch.long, device=device)
            sep_t     = torch.tensor([sep], dtype=torch.long, device=device)

            if pos is None:
                pos = compute_positions_3d(input_ids, attn_mask).squeeze(0)
            positions_3d = pos.unsqueeze(0)

            if next(model.parameters()).dtype != torch.bfloat16:
                model.to(dtype=torch.bfloat16)

            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                 enabled=use_amp):
                out = model.forward(
                    input_ids=input_ids,
                    example_ids=ex_ids_t,
                    dihedral_ids=dih_ids_t,
                    attention_mask=attn_mask,
                    targets=input_ids.clone(),
                    compute_input_loss=False,
                    positions_3d=positions_3d,
                    sep_indices=sep_t,
                )

            output_loss = out.get("output_loss")
            n_out_toks  = out.get("num_output_tokens")
            if output_loss is None or n_out_toks is None:
                continue
            n = int(n_out_toks.item())
            if n == 0:
                continue
            total_loss   += float(output_loss.item()) * n
            total_tokens += n

        losses.append(total_loss / total_tokens if total_tokens > 0 else math.inf)

    best_d = int(min(range(8), key=lambda i: losses[i]))
    return best_d, losses


# =============================================================================
# 3. Single-task canonical inference
# =============================================================================

def _build_prompt_from_canonical_demos_and_test(
    demo_examples: List[SequenceExample],
    test_ex: SequenceExample,
    best_d: int,
) -> List[int]:
    """
    Build the generation prompt for one test pair.

    CRITICAL: The prompt is ONLY the test input tokens (up to <sep>).
    Demo examples are NOT concatenated into the token sequence.

    The model conditions on the demo pairs through its per-task embedding
    (example_embedding[example_id]), not through in-context token sequences.
    Including all demo tokens in the prompt would typically exceed max_seq_len.
    This matches exactly how Prototype A builds its generation prompt.

    Reads tokens_by_dihedral[best_d] directly to use the pre-computed
    canonical token sequences.
    """
    if test_ex.tokens_by_dihedral is not None:
        test_toks = test_ex.tokens_by_dihedral[best_d].tolist()
    else:
        test_toks = test_ex.tokens.tolist()

    return _build_prompt_from_tokens(test_toks)


def run_canonical_inference_for_task(
    model,
    demo_examples:    List[SequenceExample],
    test_examples:    List[SequenceExample],
    device:           torch.device,
    best_dihedral:    int,
    color_mapping:    Optional[List[int]] = None,
    color_mapping_inv: Optional[List[int]] = None,
    max_new_tokens:   int = DEFAULT_MAX_NEW_TOKENS,
    temperature:      Optional[float] = None,
    top_k:            Optional[int] = None,
) -> List[Optional[List[List[int]]]]:
    """
    Generate predictions for all test pairs using orientation best_dihedral.

    The prompt is built from tokens_by_dihedral[best_dihedral] directly,
    matching how the model was trained.

    Returns a list of predicted output grids (in original orientation),
    one per test pair.  Returns None for pairs where the model output is
    non-rectangular or empty.
    """
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    example_id = int(getattr(demo_examples[0], "example_id",
                              getattr(test_examples[0], "example_id", 0)))

    results: List[Optional[List[List[int]]]] = []

    for test_ex in test_examples:
        prompt = _build_prompt_from_canonical_demos_and_test(
            demo_examples, test_ex, best_dihedral
        )

        generated = batched_greedy_generate(
            model=model,
            prompts=[prompt],
            example_ids=[example_id],
            dihedral_ids=[best_dihedral],
            device=device,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )

        out_toks       = extract_output_tokens(generated[0])
        predicted_grid = tokens_to_grid(out_toks)

        # Inverse color mapping (optional)
        if color_mapping_inv is not None and predicted_grid:
            predicted_grid = apply_color_permutation_to_grid(
                predicted_grid, color_mapping_inv
            )

        # Inverse dihedral transform → original orientation
        if best_dihedral != 0 and predicted_grid and is_rectangular_grid(predicted_grid):
            predicted_grid = apply_inverse_dihedral_transform(
                predicted_grid, best_dihedral
            )

        results.append(predicted_grid if is_rectangular_grid(predicted_grid) else None)

    return results


# =============================================================================
# 4. Full Prototype A evaluation pipeline
# =============================================================================

def run_prototype_a_evaluation(
    model,
    dataset,
    device: torch.device,
    use_color_canon: bool = False,
    use_amp: bool = True,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: Optional[float] = None,
    top_k_sample: Optional[int] = None,
    task_ids: Optional[Sequence[str]] = None,
) -> Dict:
    """
    Run Prototype A evaluation on all (or a subset of) tasks.

    For each task:
      1. Gather demo examples (split='train', has_output=True).
      2. (Optional) Build canonical color mapping from demo inputs.
      3. Score 8 dihedral orientations by demo-pair output CE.
      4. Run inference ONCE for each test pair using best orientation.
      5. Collect results for submission.json.

    Returns dict with keys:
        'submission', 'orientation_stats', 'n_tasks', 'n_predicted'
    """
    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    demo_by_task: Dict[str, List[SequenceExample]] = {}
    test_by_task: Dict[str, List[SequenceExample]] = {}

    for ex in dataset.examples:
        if task_ids is not None and ex.task_id not in task_ids:
            continue
        if ex.split == "train" and ex.has_output:
            demo_by_task.setdefault(ex.task_id, []).append(ex)
        elif ex.split == "test":
            test_by_task.setdefault(ex.task_id, []).append(ex)

    eval_task_ids = sorted(test_by_task.keys())
    if task_ids is not None:
        eval_task_ids = [t for t in eval_task_ids if t in task_ids]

    submission:        Dict[str, List[Dict]] = {}
    orientation_stats: Dict[str, object]     = {}
    n_predicted = 0

    for task_idx, task_id in enumerate(eval_task_ids):
        demo_exs = demo_by_task.get(task_id, [])
        test_exs = sorted(test_by_task[task_id], key=lambda e: e.pair_index)

        print(f"[{task_idx + 1}/{len(eval_task_ids)}] {task_id} | "
              f"demo={len(demo_exs)} test={len(test_exs)}")

        # (Optional) Color canonicalization
        color_mapping: Optional[List[int]] = None
        color_inv:     Optional[List[int]] = None

        if use_color_canon and demo_exs:
            demo_input_grids = []
            for ex in demo_exs:
                tok = ex.tokens.tolist()
                try:
                    sep_pos = tok.index(IO_SEPARATOR_TOKEN_ID)
                    grid = tokens_to_grid(tok[:sep_pos])
                except ValueError:
                    grid = []
                if grid:
                    demo_input_grids.append(grid)
            if demo_input_grids:
                color_mapping = build_color_canon_mapping(demo_input_grids)
                color_inv     = build_color_inverse_mapping(color_mapping)

        # Orientation scoring
        if demo_exs:
            best_d, losses = score_task_orientations(
                model, demo_exs, device, use_amp=use_amp,
                color_mapping=color_mapping,
            )
        else:
            best_d, losses = 0, [math.inf] * 8

        orientation_stats[task_id] = {"best_dihedral": best_d, "losses": losses}
        print(f"  → orientation d{best_d}  CE={losses[best_d]:.4f}")

        # Inference
        predicted_grids = run_canonical_inference_for_task(
            model=model,
            demo_examples=demo_exs,
            test_examples=test_exs,
            device=device,
            best_dihedral=best_d,
            color_mapping=color_mapping,
            color_mapping_inv=color_inv,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k_sample,
        )

        task_sub: List[Dict] = []
        for grid in predicted_grids:
            if grid is None or not is_rectangular_grid(grid):
                attempt = [[0]]
            else:
                attempt = grid
                n_predicted += 1
            task_sub.append({"attempt_1": attempt, "attempt_2": attempt})
        submission[task_id] = task_sub

    return {
        "submission":        submission,
        "orientation_stats": orientation_stats,
        "n_tasks":           len(eval_task_ids),
        "n_predicted":       n_predicted,
    }


# =============================================================================
# 5. Submission helper
# =============================================================================

def save_submission(submission: Dict, output_dir, run_name: str = "prototype_a") -> Path:
    """Save submission dict as submission.json inside output_dir/run_name/."""
    out_dir = Path(output_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "submission.json"
    with out_path.open("w") as fh:
        json.dump(submission, fh)
    print(f"Submission saved to {out_path}")
    return out_path
