"""
prototype_e/src/e_ranker.py

E-Ranker: Combined Filter + Transition Scoring Pipeline
=========================================================
Applies hard filters then transition ranking to a pool of candidates.

Decision logic:
  1. Hard-filter by shape (wrong-shape candidates discarded)
  2. Hard-filter by color palette (forbidden-color candidates discarded)
  3. Rank survivors by pixel-transition log-likelihood
  4. If >=1 survivor remains → top-1 is attempt_2
  5. If ALL candidates were filtered out → fall back to greedy (attempt_1)
     (This is safe: would rather keep the greedy answer than nothing)

The ranker is deliberately conservative:
  - Never discards the greedy candidate (it's always attempt_1 regardless)
  - Only replaces attempt_2 when we have real signal
  - Falls back gracefully at every stage

Usage:
    from e_ranker import e_rank
    ranked, diag = e_rank(candidates, greedy_cand, test_input, demo_pairs, gt=None)
    attempt_2 = ranked[0] if ranked else greedy_cand
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from candidate_filters import hard_filter
from transition_ranker import rank_by_transition, transition_diagnostics

Grid = List[List[int]]


def e_rank(
    candidates: List[Grid],
    greedy_cand: Grid,
    test_input:  Grid,
    demo_pairs:  List[Tuple[Grid, Grid]],
    gt_grid:     Optional[Grid] = None,
) -> Tuple[List[Grid], dict]:
    """
    Full E-ranking pipeline.

    Args:
        candidates:  all generated candidates (deduplicated pool)
        greedy_cand: the greedy candidate (always preserved as attempt_1)
        test_input:  the test input grid
        demo_pairs:  list of (input, output) demo grids
        gt_grid:     optional ground truth for diagnostics (canonical space)

    Returns:
        (ranked_grids, diagnostics_dict)
        ranked_grids is sorted best-first; may be empty if all filtered.
    """
    diag: dict = {}

    # ── Stage 1: Hard filters ─────────────────────────────────────────────────
    survivors, filter_stats = hard_filter(candidates, test_input, demo_pairs)
    diag.update(filter_stats)

    fallback_reason = None
    if not survivors:
        # All candidates rejected — fall back to greedy to avoid empty result
        fallback_reason = "all_filtered"
        survivors = [greedy_cand] if greedy_cand else candidates[:1]
        diag["fallback"] = fallback_reason

    # ── Stage 2: Transition ranking ───────────────────────────────────────────
    scored, table = rank_by_transition(survivors, test_input, demo_pairs)
    ranked_grids = [g for g, _ in scored]

    # ── Stage 3: Diagnostics ─────────────────────────────────────────────────
    trans_diag = transition_diagnostics(scored, gt_grid)
    diag.update(trans_diag)
    diag["fallback"] = fallback_reason

    # Extra: did transition ranking agree with greedy?
    if ranked_grids:
        from transition_ranker import _grids_equal
        diag["trans_agrees_greedy"] = _grids_equal(ranked_grids[0], greedy_cand)
    else:
        diag["trans_agrees_greedy"] = None

    return ranked_grids, diag
