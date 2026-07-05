"""
Rule Consistency Scoring (RCOS) for ARC-AGI candidate selection.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Core insight:
  Every ARC task has a hidden rule R. If candidate ŷ for test input x_test
  is correct (matches R), then adding (x_test, ŷ) as an additional "demo"
  should reduce the model's uncertainty on the existing demos — because the
  model now has one more example of the SAME rule.

  If ŷ is WRONG, adding it introduces a contradictory example, which should
  INCREASE the model's uncertainty on the original demos.

Formally:
    RCS(ŷ) = CE_baseline − CE_augmented

where:
    CE_baseline  = CE of demo outputs given demo inputs only
    CE_augmented = CE of demo outputs when (x_test, ŷ) is prepended as a
                   new demo (so demos can attend back to it via causal attn)

Positive RCS → candidate is rule-consistent.
Negative RCS → candidate contradicts the observed demos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FIX v3 — prompt construction (_build_test_prompt)

  Original code always built the generation prompt by manually encoding
  GridPairs via _encode_full_pair().  This caused a double-transform when
  the caller had already physically rotated the grids to canonical space
  AND the model also applied a dihedral transformation via its embedding.

  Fix: _build_test_prompt now accepts optional `demo_seq_exs` and
  `test_seq_ex` (raw SequenceExample objects).  When provided, it reads
  tokens_by_dihedral[dihedral_id] directly — exactly the same pre-computed
  canonical token sequences that Prototype A uses — and falls back to the
  GridPair re-encoding only when SequenceExample objects are unavailable.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Module structure:
  1. Token sequence utilities
  2. CE computation (baseline + augmented)
  3. Candidate generation  (greedy / temperature sampling / beam search)
  4. Full RCOS ranking
  5. Diagnostic utilities for Experiment 1
"""

from __future__ import annotations

import sys
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ─── Local imports (assumes we run from mdlARC root) ─────────────────────────
_SRC = Path(__file__).parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from common import (
    IGNORE_INDEX,
    IO_SEPARATOR_TOKEN_ID,
    END_TOKEN_ID,
    START_TOKEN_ID,
    NEXT_LINE_TOKEN_ID,
    VOCAB_SIZE,
    apply_dihedral_transform,
    compute_positions_3d,
    encode_example,
    extract_output_tokens,
    is_rectangular_grid,
    split_grids_from_tokens,
    tokens_to_grid,
)
from evaluate import (
    DEFAULT_MAX_NEW_TOKENS,
    BatchGridState,
    _build_prompt_from_tokens,
    _derive_initial_state_from_prompt,
    _left_pad_sequences,
    _select_next_token,
    batched_greedy_generate,
)


# =============================================================================
# Lightweight container for a grid pair (used throughout RCOS)
# =============================================================================

@dataclass
class GridPair:
    """Holds the input and output grids for one ARC example."""
    input:  List[List[int]]
    output: Optional[List[List[int]]] = None  # None for test examples


def grids_from_sequence_example(ex) -> GridPair:
    """
    Extract raw grids from a SequenceExample (which stores pre-tokenised data).

    Uses `split_grids_from_tokens` to recover the 2D arrays from the token
    stream.  The first grid is the input; the second (if present) is the
    output.
    """
    token_list = (ex.tokens.tolist()
                  if isinstance(ex.tokens, torch.Tensor)
                  else list(ex.tokens))
    grids = split_grids_from_tokens(token_list)
    inp  = grids[0] if grids else []
    out  = grids[1] if len(grids) > 1 else None
    return GridPair(input=inp, output=out)


# =============================================================================
# 1.  Token sequence utilities
# =============================================================================

def _encode_full_pair(input_grid: List[List[int]],
                      output_grid: List[List[int]]) -> List[int]:
    """Tokenise one complete (input, output) pair: <start> … <sep> … <end>."""
    return encode_example(input_grid, output_grid,
                          include_output=True, append_end=True)


def _encode_input_only(input_grid: List[List[int]]) -> List[int]:
    """Tokenise just the input side of a pair (no output, no <end>)."""
    return encode_example(input_grid, output_grid=None,
                          include_output=False, append_end=False)


def _find_output_positions_in_pair(pair_tokens: List[int]) -> List[int]:
    """
    Return the 0-indexed positions of output (cell-value) tokens inside a
    single pair's token list.

    A pair looks like:
        <start> [input cells + <nl>…] <sep> [output cells + <nl>…] <end>

    Output positions = everything strictly between <sep> and <end>.
    """
    in_output = False
    positions: List[int] = []
    for i, tok in enumerate(pair_tokens):
        if tok == IO_SEPARATOR_TOKEN_ID:
            in_output = True
            continue
        if tok == END_TOKEN_ID:
            break
        if in_output:
            positions.append(i)
    return positions


def build_sequence_and_targets(
    demo_pairs: List[GridPair],
    synthetic_prepend: Optional[Tuple[List[List[int]], List[List[int]]]] = None,
) -> Tuple[List[int], List[int], int]:
    """
    Build a flat token sequence and a custom target list for RCOS CE scoring.

    The targets list has IGNORE_INDEX everywhere EXCEPT at positions belonging
    to the ORIGINAL demo output tokens.  Passing this as `targets` to
    `model.forward()` makes the model report CE only on those positions.

    Args:
        demo_pairs:       list of GridPair objects (must have both .input and .output)
        synthetic_prepend: optional (input_grid, output_grid) prepended as an
                           extra "demo" BEFORE the real demos.  Its output tokens
                           are masked out (IGNORE_INDEX) so they don't contribute
                           to the CE.

    Returns:
        token_list:         flat list of token IDs for the whole sequence
        targets:            same length as token_list; IGNORE_INDEX except at
                            demo-output positions
        n_demo_output_toks: how many demo output tokens are in the targets
    """
    token_list: List[int] = []

    # ── optional synthetic pair (prepended before demos) ─────────────────────
    if synthetic_prepend is not None:
        in_g, out_g = synthetic_prepend
        synth_tokens = _encode_full_pair(in_g, out_g)
        token_list.extend(synth_tokens)

    # ── real demo pairs ───────────────────────────────────────────────────────
    demo_output_positions: List[int] = []
    for gp in demo_pairs:
        if gp.output is None:
            raise ValueError(
                "GridPair passed to build_sequence_and_targets must have .output"
            )
        pair_tokens    = _encode_full_pair(gp.input, gp.output)
        local_out_pos  = _find_output_positions_in_pair(pair_tokens)
        offset         = len(token_list)
        for p in local_out_pos:
            demo_output_positions.append(offset + p)
        token_list.extend(pair_tokens)

    # ── build targets ─────────────────────────────────────────────────────────
    targets = [IGNORE_INDEX] * len(token_list)
    for pos in demo_output_positions:
        if pos < len(token_list):
            targets[pos] = token_list[pos]

    return token_list, targets, len(demo_output_positions)


# =============================================================================
# 2.  CE computation
# =============================================================================

def _find_example_embedding(model) -> Optional[torch.nn.Embedding]:
    """
    Find the task-level example embedding in the model.

    The model was trained on all 1307 tasks (including the 400 eval tasks),
    so example_embedding[example_id] has near-zero loss for every eval task.
    This makes baseline_CE ≈ 0, killing the RCOS signal entirely.

    We locate the embedding here so _run_forward_ce can temporarily zero the
    relevant row during CE scoring — restoring a meaningful baseline_CE —
    without touching the embedding during generation (where the model still
    needs task context to produce good candidates).
    """
    for attr in ("example_embedding", "task_embedding", "puzzle_embedding"):
        obj = getattr(model, attr, None)
        if isinstance(obj, torch.nn.Embedding):
            return obj
    return None


@torch.no_grad()
def _run_forward_ce(
    model,
    token_list:   List[int],
    targets:      List[int],
    example_id:   int,
    dihedral_id:  int,
    device:       torch.device,
) -> float:
    """
    Run one padded forward pass and return the mean CE over non-ignored targets.

    Returns float('inf') if the sequence is too long or has no valid targets.

    SURGICAL EMBEDDING ZEROING
    --------------------------
    The model memorised every eval task via example_embedding[example_id],
    so without intervention baseline_CE ≈ 0 and RCOS has no signal.

    Fix: temporarily zero example_embedding[example_id] for THIS forward pass
    only, then restore the original weights immediately after.  Generation
    calls never go through this function, so candidates are unaffected.
    """
    if not token_list:
        return math.inf

    if len(token_list) > model.config.max_seq_len:
        return math.inf

    has_valid = any(t != IGNORE_INDEX for t in targets)
    if not has_valid:
        return math.inf

    seq_len    = len(token_list)
    input_ids  = torch.tensor(token_list, dtype=torch.long, device=device).unsqueeze(0)
    targets_t  = torch.tensor(targets,    dtype=torch.long, device=device).unsqueeze(0)
    attn_mask  = torch.ones(1, seq_len, dtype=torch.bool, device=device)

    example_ids_t  = torch.tensor([example_id],  dtype=torch.long, device=device)
    dihedral_ids_t = torch.tensor([dihedral_id], dtype=torch.long, device=device)

    positions_3d = compute_positions_3d(input_ids, attn_mask).to(
        device=device, dtype=torch.long
    )

    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    # Temporarily zero the memorised embedding row so the model must rely on
    # the token sequence alone — giving a meaningful (non-zero) CE signal.
    emb = _find_example_embedding(model)
    saved_emb = None
    if emb is not None and example_id < emb.num_embeddings:
        saved_emb = emb.weight.data[example_id].clone()
        emb.weight.data[example_id].zero_()

    try:
        outputs = model.forward(
            input_ids=input_ids,
            example_ids=example_ids_t,
            dihedral_ids=dihedral_ids_t,
            attention_mask=attn_mask,
            targets=targets_t,
            compute_input_loss=False,
            positions_3d=positions_3d,
        )
    finally:
        # Always restore — even if forward() raises
        if saved_emb is not None:
            emb.weight.data[example_id] = saved_emb

    # The model may return the relevant loss under "loss" or "output_loss"
    loss = outputs.get("loss")
    if loss is None:
        loss = outputs.get("output_loss")
    if loss is None:
        return math.inf
    val = float(loss.item())
    return val if math.isfinite(val) else math.inf


def compute_baseline_ce(
    model,
    demo_pairs:  List[GridPair],
    example_id:  int,
    dihedral_id: int,
    device:      torch.device,
) -> float:
    """
    CE of demo output tokens — demos concatenated, with length-aware truncation.

    The original approach concatenated ALL demo pairs, which exceeded max_seq_len
    (~1863 tokens) for tasks with many demos.  Fix: iteratively drop the OLDEST
    demo pairs until the concatenated sequence fits in the context window.
    This preserves the causal cross-demo attention that gives RCOS its signal
    (baseline_CE ≈ 0 from per-pair scoring destroys the signal).
    """
    if not demo_pairs:
        return math.inf

    # Try with as many demos as possible, dropping oldest first if too long
    for start in range(len(demo_pairs)):
        subset = [gp for gp in demo_pairs[start:] if gp.output is not None]
        if not subset:
            continue
        token_list, targets, n_out = build_sequence_and_targets(subset)
        if n_out == 0:
            continue
        if len(token_list) > model.config.max_seq_len:
            continue   # still too long, try fewer demos
        return _run_forward_ce(
            model, token_list, targets, example_id, dihedral_id, device
        )

    return math.inf


def compute_augmented_ce(
    model,
    demo_pairs:       List[GridPair],
    test_input_grid:  List[List[int]],
    candidate_grid:   List[List[int]],
    example_id:       int,
    dihedral_id:      int,
    device:           torch.device,
) -> float:
    """
    CE of demo output tokens when (test_input, candidate) is PREPENDED.

    Same length-aware truncation as compute_baseline_ce: drop oldest demos
    until the full sequence (synthetic pair + remaining demos) fits in the
    context window.  The synthetic pair is always kept (it's the signal).
    """
    if not demo_pairs:
        return math.inf

    for start in range(len(demo_pairs)):
        subset = [gp for gp in demo_pairs[start:] if gp.output is not None]
        if not subset:
            continue
        token_list, targets, n_out = build_sequence_and_targets(
            subset,
            synthetic_prepend=(test_input_grid, candidate_grid),
        )
        if n_out == 0:
            continue
        if len(token_list) > model.config.max_seq_len:
            continue
        return _run_forward_ce(
            model, token_list, targets, example_id, dihedral_id, device
        )

    return math.inf


def compute_baseline_ce_dihedral_avg(
    model,
    demo_pairs:  List[GridPair],
    example_id:  int,
    device:      torch.device,
) -> float:
    """
    Average baseline CE across all 8 dihedral orientations.

    For each d in 0..7:
      - Rotate all demo pair grids (input+output) by dihedral d
      - Run compute_baseline_ce with dihedral_id=d
    Average the finite results.  Returns inf if all orientations overflow.

    This 8× averaging reduces variance in the CE signal and mitigates the
    context-overflow problem: orientations that produce shorter token sequences
    (e.g. transpositions of tall grids) are likely to succeed even when the
    canonical orientation overflows the context window.
    """
    values: List[float] = []
    for d in range(8):
        rotated: List[GridPair] = []
        for gp in demo_pairs:
            if gp.output is None:
                continue
            rotated.append(GridPair(
                input=apply_dihedral_transform(gp.input, d),
                output=apply_dihedral_transform(gp.output, d),
            ))
        if not rotated:
            continue
        ce = compute_baseline_ce(model, rotated, example_id, d, device)
        if math.isfinite(ce):
            values.append(ce)
    return sum(values) / len(values) if values else math.inf


def compute_augmented_ce_dihedral_avg(
    model,
    demo_pairs:       List[GridPair],
    test_input_grid:  List[List[int]],
    candidate_grid:   List[List[int]],
    example_id:       int,
    device:           torch.device,
) -> float:
    """
    Average augmented CE across all 8 dihedral orientations.

    For each d in 0..7:
      - Rotate demo pairs, test_input, and candidate by dihedral d
      - Run compute_augmented_ce with dihedral_id=d
    Average the finite results.  Returns inf if all orientations overflow.
    """
    values: List[float] = []
    for d in range(8):
        rotated: List[GridPair] = []
        for gp in demo_pairs:
            if gp.output is None:
                continue
            rotated.append(GridPair(
                input=apply_dihedral_transform(gp.input, d),
                output=apply_dihedral_transform(gp.output, d),
            ))
        if not rotated:
            continue
        rt = apply_dihedral_transform(test_input_grid, d)
        rc = apply_dihedral_transform(candidate_grid, d)
        ce = compute_augmented_ce(model, rotated, rt, rc, example_id, d, device)
        if math.isfinite(ce):
            values.append(ce)
    return sum(values) / len(values) if values else math.inf


def compute_rcs(baseline_ce: float, augmented_ce: float) -> float:
    """
    RCS(ŷ) = CE_baseline − CE_augmented.

    Special cases:
      baseline=inf, aug=finite → inf  (candidate reduced uncertainty from zero;
                                        very strong positive signal — rank first)
      baseline=finite, aug=inf → -inf (candidate broke the model — rank last)
      baseline=inf, aug=inf   → nan  → mapped to -inf

    We map nan → -inf so nan-scored candidates rank last rather than being
    placed randomly by Python's sort (nan comparisons are undefined, which
    destroyed RCOS lift by ~30pp in testing).
    """
    result = float(baseline_ce) - float(augmented_ce)
    if math.isnan(result):
        return -math.inf
    return result


# =============================================================================
# 3.  Candidate generation
# =============================================================================

def _build_test_prompt(
    demo_pairs:    List[GridPair],
    test_input:    List[List[int]],
    dihedral_id:   int,
    demo_seq_exs:  Optional[Any] = None,   # unused, kept for API compat
    test_seq_ex:   Optional[Any] = None,   # SequenceExample when available
) -> List[int]:
    """
    Build the token prompt for decoding the test output.

    CRITICAL: The generation prompt is ONLY the test input tokens (up to and
    including <sep>).  Demo pairs are NOT concatenated here.

    This matches exactly how Prototype A builds its generation prompt:
    the model conditions on the demo pairs through its per-task
    example_embedding (indexed by example_id), NOT by including the demo
    token sequences in the context window.  Concatenating demo tokens would
    inflate the prompt to thousands of tokens, exceeding max_seq_len (~1863).

    Preferred path: use pre-computed tokens_by_dihedral[dihedral_id] from
    test_seq_ex — the same canonical token sequences the model was trained on.

    Fallback path: encode test_input directly via _encode_input_only().
    """
    if test_seq_ex is not None:
        if test_seq_ex.tokens_by_dihedral is not None:
            test_toks = test_seq_ex.tokens_by_dihedral[dihedral_id].tolist()
        else:
            test_toks = test_seq_ex.tokens.tolist()
        return _build_prompt_from_tokens(test_toks)

    # Fallback: encode from canonical test_input grid
    test_toks = _encode_input_only(test_input)
    if not test_toks or test_toks[-1] != IO_SEPARATOR_TOKEN_ID:
        test_toks.append(IO_SEPARATOR_TOKEN_ID)
    return _build_prompt_from_tokens(test_toks)


def generate_greedy(
    model,
    demo_pairs:   List[GridPair],
    test_input:   List[List[int]],
    example_id:   int,
    dihedral_id:  int,
    device:       torch.device,
    demo_seq_exs: Optional[Any] = None,
    test_seq_ex:  Optional[Any] = None,
) -> List[List[List[int]]]:
    """Generate exactly one greedy-decode candidate."""
    return _generate_n(
        model, demo_pairs, test_input, example_id, dihedral_id, device,
        n=1, temperature=None, top_k=None,
        demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
    )


def generate_sampled(
    model,
    demo_pairs:        List[GridPair],
    test_input:        List[List[int]],
    example_id:        int,
    dihedral_id:       int,
    device:            torch.device,
    n_per_temperature: int = 20,
    temperatures:      Tuple[float, ...] = (0.7, 1.0),
    top_k:             Optional[int] = None,
    demo_seq_exs:      Optional[Any] = None,
    test_seq_ex:       Optional[Any] = None,
) -> List[List[List[int]]]:
    """Generate temperature-sampled candidates (diverse exploration)."""
    grids: List[List[List[int]]] = []
    for temp in temperatures:
        grids.extend(_generate_n(
            model, demo_pairs, test_input, example_id, dihedral_id, device,
            n=n_per_temperature, temperature=temp, top_k=top_k,
            demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
        ))
    return grids


@torch.no_grad()
def generate_beam(
    model,
    demo_pairs:   List[GridPair],
    test_input:   List[List[int]],
    example_id:   int,
    dihedral_id:  int,
    device:       torch.device,
    beam_width:   int = 10,
    demo_seq_exs: Optional[Any] = None,
    test_seq_ex:  Optional[Any] = None,
) -> List[List[List[int]]]:
    """
    Beam search with KV-caching.

    Algorithm:
      1. Run the prompt through the model once → get first-token logits + KV cache.
      2. From the first-token logits, initialise B beams with the top-B tokens.
      3. For each subsequent step, run B decode steps in parallel (batched),
         expand to B×V candidates, keep top-B globally.
      4. Collect finished beams and return their decoded grids.
    """
    if beam_width <= 0:
        return []

    model.eval()
    if next(model.parameters()).dtype != torch.bfloat16:
        model.to(dtype=torch.bfloat16)

    prompt  = _build_test_prompt(
        demo_pairs, test_input, dihedral_id,
        demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
    )
    max_new = DEFAULT_MAX_NEW_TOKENS

    # ── 1.  Full prompt pass ───────────────────────────────────────────────
    example_ids_t  = torch.tensor([example_id],  dtype=torch.long, device=device)
    dihedral_ids_t = torch.tensor([dihedral_id], dtype=torch.long, device=device)

    input_ids, attn_mask = _left_pad_sequences([prompt], END_TOKEN_ID, device)
    prompt_len   = input_ids.size(1)
    padded_len   = min(
        (prompt_len + max_new + 127) // 128 * 128, model.config.max_seq_len
    )

    positions_3d = compute_positions_3d(input_ids, attn_mask).to(
        device=device, dtype=torch.long
    )
    example_emb = model.example_embedding(example_ids_t).to(dtype=torch.bfloat16)

    outputs      = model.forward_generate(
        input_ids=input_ids,
        example_ids=example_ids_t,
        dihedral_ids=dihedral_ids_t,
        past_key_values=None,
        positions_3d=positions_3d,
        attention_mask=attn_mask,
        example_embeds=example_emb,
    )
    first_logits = outputs["logits"]            # [1, prompt_len, V]
    prompt_kvs   = outputs["past_key_values"]   # list of (k, v) per layer

    # Build padded KV buffer
    kv_1: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for k, v in prompt_kvs:
        B, H, L, D = k.shape
        k_buf = torch.zeros(B, H, padded_len, D, dtype=torch.bfloat16, device=device)
        v_buf = torch.zeros(B, H, padded_len, D, dtype=torch.bfloat16, device=device)
        k_buf[:, :, :L, :] = k
        v_buf[:, :, :L, :] = v
        kv_1.append((k_buf, v_buf))

    # ── 2.  Initialise B beams from first logits ───────────────────────────
    last_logits   = first_logits[0, -1, :]
    log_probs     = F.log_softmax(last_logits.float(), dim=-1)
    top_lp, top_toks = torch.topk(log_probs, min(beam_width, VOCAB_SIZE))

    init_state, _ = _derive_initial_state_from_prompt(input_ids, positions_3d)

    beams: List[Dict] = []
    for i in range(top_toks.size(0)):
        tok = int(top_toks[i].item())
        clp = float(top_lp[i].item())

        gs      = BatchGridState(init_state.clone())
        tok_t   = torch.tensor([tok], dtype=torch.long, device=device)
        last_pos = gs.update(tok_t).unsqueeze(1)   # [1, 1, 3]

        beam_kv = [(kb.clone(), vb.clone()) for kb, vb in kv_1]

        bam = torch.zeros(1, padded_len, dtype=torch.bool, device=device)
        bam[:, :prompt_len] = attn_mask
        bam[:, prompt_len]  = True

        beams.append({
            "cum_log_prob": clp,
            "tokens":       list(prompt) + [tok],
            "cache_pos":    prompt_len + 1,
            "kv":           beam_kv,
            "attn":         bam,
            "grid_state":   gs,
            "last_pos":     last_pos,
            "finished":     (tok == END_TOKEN_ID),
        })

    # Ensure compiled decoder
    if not hasattr(model, "_compiled_decode"):
        print("Compiling model for decoding step...")
        model._compiled_decode = torch.compile(
            model.forward_generate, mode="default", fullgraph=True
        )

    finished_beams: List[Dict] = [b for b in beams if b["finished"]]
    active_beams:   List[Dict] = [b for b in beams if not b["finished"]]

    # ── 3.  Beam search steps ──────────────────────────────────────────────
    for _step in range(max_new - 1):
        if not active_beams:
            break
        if len(finished_beams) >= beam_width:
            break

        step_candidates: List[Tuple[float, int, int]] = []

        for b_idx, beam in enumerate(active_beams):
            cp         = torch.tensor([beam["cache_pos"]], dtype=torch.long, device=device)
            last_tok_t = torch.tensor([[beam["tokens"][-1]]],
                                      dtype=torch.long, device=device)

            out = model._compiled_decode(
                input_ids=last_tok_t,
                example_ids=example_ids_t,
                dihedral_ids=dihedral_ids_t,
                past_key_values=tuple(beam["kv"]),
                positions_3d=beam["last_pos"],
                attention_mask=beam["attn"],
                cache_position=cp,
                example_embeds=example_emb,
            )
            step_logits = out["logits"]          # [1, 1, V]

            lp    = F.log_softmax(step_logits[0, 0, :].float(), dim=-1)
            top_n = min(beam_width, VOCAB_SIZE)
            top_lp_s, top_tok_s = torch.topk(lp, top_n)

            for j in range(top_n):
                step_candidates.append((
                    beam["cum_log_prob"] + float(top_lp_s[j].item()),
                    int(top_tok_s[j].item()),
                    b_idx,
                ))

        step_candidates.sort(key=lambda x: -x[0])
        selected = step_candidates[:beam_width]

        new_active: List[Dict] = []
        for new_clp, next_tok, b_idx in selected:
            parent = active_beams[b_idx]

            new_gs   = BatchGridState(parent["grid_state"].state.clone())
            tok_t    = torch.tensor([next_tok], dtype=torch.long, device=device)
            new_lpos = new_gs.update(tok_t).unsqueeze(1)

            new_kv  = [(kb.clone(), vb.clone()) for kb, vb in parent["kv"]]
            new_bam = parent["attn"].clone()
            new_cp  = parent["cache_pos"] + 1
            if new_cp < padded_len:
                new_bam[:, new_cp] = True

            new_beam = {
                "cum_log_prob": new_clp,
                "tokens":       parent["tokens"] + [next_tok],
                "cache_pos":    new_cp,
                "kv":           new_kv,
                "attn":         new_bam,
                "grid_state":   new_gs,
                "last_pos":     new_lpos,
                "finished":     (next_tok == END_TOKEN_ID),
            }
            if new_beam["finished"]:
                finished_beams.append(new_beam)
            else:
                new_active.append(new_beam)

        active_beams = new_active

    finished_beams.extend(active_beams)
    finished_beams.sort(key=lambda b: -b["cum_log_prob"])

    grids: List[List[List[int]]] = []
    for beam in finished_beams[:beam_width]:
        out_toks = extract_output_tokens(beam["tokens"])
        grid     = tokens_to_grid(out_toks)
        if is_rectangular_grid(grid):
            grids.append(grid)

    return grids


def deduplicate_grids(grids: List[List[List[int]]]) -> List[List[List[int]]]:
    """Remove duplicate grids while preserving insertion order."""
    seen: set = set()
    result: List[List[List[int]]] = []
    for g in grids:
        key = tuple(tuple(int(c) for c in row) for row in g)
        if key not in seen:
            seen.add(key)
            result.append(g)
    return result


def generate_all_candidates(
    model,
    demo_pairs:        List[GridPair],
    test_input:        List[List[int]],
    example_id:        int,
    dihedral_id:       int,
    device:            torch.device,
    *,
    use_greedy:        bool = True,
    use_beam:          bool = True,
    beam_width:        int = 10,
    use_sample:        bool = True,
    n_per_temperature: int = 20,
    temperatures:      Tuple[float, ...] = (0.7, 1.0),
    top_k:             Optional[int] = None,
    # FIX v3: optional SequenceExample objects for correct prompt construction
    demo_seq_exs:      Optional[Any] = None,
    test_seq_ex:       Optional[Any] = None,
) -> List[List[List[int]]]:
    """
    Generate candidates from all three strategies and deduplicate.

    When `test_seq_ex` and `demo_seq_exs` are provided, all generation
    methods build the prompt from tokens_by_dihedral[dihedral_id] (the
    pre-computed canonical token sequences), which is the same approach
    used by Prototype A.  This avoids the double-transform bug.

    Strategy order (for diversity):
        greedy → beam search → temperature sampling
    """
    all_grids: List[List[List[int]]] = []

    if use_greedy:
        all_grids.extend(generate_greedy(
            model, demo_pairs, test_input, example_id, dihedral_id, device,
            demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
        ))

    if use_beam and beam_width > 0:
        all_grids.extend(generate_beam(
            model, demo_pairs, test_input, example_id, dihedral_id, device,
            beam_width=beam_width,
            demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
        ))

    if use_sample and n_per_temperature > 0 and temperatures:
        all_grids.extend(generate_sampled(
            model, demo_pairs, test_input, example_id, dihedral_id, device,
            n_per_temperature=n_per_temperature,
            temperatures=temperatures,
            top_k=top_k,
            demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
        ))

    return deduplicate_grids(all_grids)


# =============================================================================
# 4.  Full RCOS ranking
# =============================================================================

def rank_candidates_by_rcs(
    model,
    demo_pairs:      List[GridPair],
    test_input:      List[List[int]],
    candidates:      List[List[List[int]]],
    example_id:      int,
    dihedral_id:     int,
    device:          torch.device,
    baseline_ce:     Optional[float] = None,
    dihedral_avg:    bool = False,
) -> Tuple[List[Tuple[List[List[int]], float, float]], float]:
    """
    Score every candidate by RCS, return them sorted best-first.

    Args:
        dihedral_avg: if True, average CE over all 8 orientations for both
                      baseline and augmented passes.  This reduces variance and
                      recovers signal from tasks whose canonical orientation
                      overflows the context window.

    Returns:
        ranked:      list of (grid, rcs_score, aug_ce) sorted by desc RCS
        baseline_ce: the CE of demos without any synthetic pair
    """
    if not candidates:
        return [], baseline_ce or math.inf

    if baseline_ce is None:
        if dihedral_avg:
            baseline_ce = compute_baseline_ce_dihedral_avg(
                model, demo_pairs, example_id, device
            )
        else:
            baseline_ce = compute_baseline_ce(
                model, demo_pairs, example_id, dihedral_id, device
            )

    scored: List[Tuple[List[List[int]], float, float]] = []
    for cand in candidates:
        if dihedral_avg:
            aug_ce = compute_augmented_ce_dihedral_avg(
                model, demo_pairs, test_input, cand, example_id, device
            )
        else:
            aug_ce = compute_augmented_ce(
                model, demo_pairs, test_input, cand,
                example_id, dihedral_id, device,
            )
        rcs = compute_rcs(baseline_ce, aug_ce)
        scored.append((cand, rcs, aug_ce))

    scored.sort(key=lambda x: -x[1])   # descending RCS
    return scored, baseline_ce


# =============================================================================
# 5.  Diagnostic utilities — Experiment 1
# =============================================================================

def grids_equal(g1: Optional[List[List[int]]],
                g2: Optional[List[List[int]]]) -> bool:
    """Return True iff two grids are identical (element-wise)."""
    if g1 is None or g2 is None:
        return False
    if len(g1) != len(g2):
        return False
    return all(
        len(r1) == len(r2) and all(a == b for a, b in zip(r1, r2))
        for r1, r2 in zip(g1, g2)
    )


def compute_rcs_diagnostics(
    candidates:   List[List[List[int]]],
    rcs_scores:   List[float],
    aug_ces:      List[float],
    ground_truth: Optional[List[List[int]]],
    baseline_ce:  float,
) -> Dict:
    """
    Compute per-task diagnostic metrics for Experiment 1.

    Metrics:
      oracle_hit          — is the correct answer anywhere in the candidate set?
      rcs_selects_correct — does the top-RCS candidate equal the ground truth?
      rcs_rank_of_correct — 1-indexed rank of the correct answer; None if absent
      rcs_score_correct   — RCS score of the correct candidate
      rcs_score_top1      — RCS score of the best candidate
      score_gap           — rcs_correct − rcs_top1  (<0 means mis-ranked)
                            None when both scores are -inf (indeterminate)
      aug_ce_correct      — augmented CE for the correct candidate
      aug_ce_top1         — augmented CE for the top-ranked candidate
      baseline_ce         — CE without any synthetic pair
      n_candidates        — total number of (unique, rectangular) candidates
    """
    n         = len(candidates)
    top1_rcs  = rcs_scores[0] if rcs_scores else -math.inf
    top1_aug  = aug_ces[0]    if aug_ces    else  math.inf

    oracle_hit     = False
    correct_rank   = None
    correct_rcs    = None
    correct_aug_ce = None

    for rank, (cand, rcs, ac) in enumerate(
        zip(candidates, rcs_scores, aug_ces), start=1
    ):
        if grids_equal(cand, ground_truth):
            oracle_hit     = True
            correct_rank   = rank
            correct_rcs    = rcs
            correct_aug_ce = ac
            break

    rcs_selects_correct = (correct_rank == 1)

    score_gap = None
    if oracle_hit and correct_rcs is not None:
        gap = correct_rcs - top1_rcs   # 0 if rank==1, negative otherwise
        # -inf - (-inf) = nan: both candidates had no RCS signal (inf CE baseline
        # with inf aug CE for both). Report None for indeterminate gap.
        score_gap = None if math.isnan(gap) else gap

    return {
        "oracle_hit":           oracle_hit,
        "rcs_selects_correct":  rcs_selects_correct,
        "rcs_rank_of_correct":  correct_rank,
        "rcs_score_correct":    correct_rcs if (correct_rcs is not None and not math.isinf(correct_rcs)) else None,
        "rcs_score_top1":       top1_rcs    if not math.isinf(top1_rcs)    else None,
        "score_gap":            score_gap,
        "aug_ce_correct":       correct_aug_ce,
        "aug_ce_top1":          top1_aug,
        "baseline_ce":          baseline_ce,
        "n_candidates":         n,
    }


# =============================================================================
# Private helpers
# =============================================================================

def _generate_n(
    model,
    demo_pairs:   List[GridPair],
    test_input:   List[List[int]],
    example_id:   int,
    dihedral_id:  int,
    device:       torch.device,
    n:            int = 1,
    temperature:  Optional[float] = None,
    top_k:        Optional[int] = None,
    demo_seq_exs: Optional[Any] = None,
    test_seq_ex:  Optional[Any] = None,
) -> List[List[List[int]]]:
    """Internal: generate n candidates via batched_greedy_generate."""
    prompt = _build_test_prompt(
        demo_pairs, test_input, dihedral_id,
        demo_seq_exs=demo_seq_exs, test_seq_ex=test_seq_ex,
    )
    seqs = batched_greedy_generate(
        model=model,
        prompts=[prompt] * n,
        example_ids=[example_id] * n,
        dihedral_ids=[dihedral_id] * n,
        device=device,
        max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
        temperature=temperature,
        top_k=top_k,
    )
    grids: List[List[List[int]]] = []
    for seq in seqs:
        out_toks = extract_output_tokens(seq)
        grid     = tokens_to_grid(out_toks)
        if is_rectangular_grid(grid):
            grids.append(grid)
    return grids
