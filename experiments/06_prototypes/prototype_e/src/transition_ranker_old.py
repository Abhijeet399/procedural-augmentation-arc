"""
prototype_e/src/transition_ranker.py

Pixel-Transition Re-Ranker
===========================
From demo (input, output) pairs, build a frequency table of per-cell
color transitions: (src_color → dst_color).

For each candidate output, score it by the log-likelihood of each cell's
transition under the learned table.  Higher score = better match to the
transformation pattern.

Why this works:
  Many ARC tasks are expressible as simple color-mapping rules, e.g.:
    - "everywhere color 3, output color 7"
    - "cells adjacent to color 1 become color 2, rest stay"
  The transition table captures exactly this signal without any model.

Why this beats RCOS here:
  RCOS requires non-zero baseline CE for signal.  For memorized eval tasks,
  CE ≈ 0 → RCOS is noise.  Transition scoring is model-free and always works.

Scoring formula (per candidate c given test_input t):
  score(c) = Σ_{r,c} log P(c[r][c] | t[r][c])
  where P(dst | src) = (count(src→dst) + α) / (Σ_d count(src→d) + 10α)
  with Laplace smoothing α (default 0.1).

Tie-breaking / robustness extras:
  1. Position-aware bonus: weight transitions at positions that changed in
     >50% of demo pairs more highly (these are the "interesting" cells).
  2. Exact-demo bonus: candidates identical to a demo output get a small
     bonus (the rule might be identity for some tasks).
  3. Diversity penalty: if top-K candidates are all identical, keep only one
     (deduplication already handled upstream in generate_all_candidates, but
     the ranker should still be diversity-aware).
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]

# ARC colors
N_COLORS = 10
LAPLACE_ALPHA = 0.1


# ─── TRANSITION TABLE ─────────────────────────────────────────────────────────

class TransitionTable:
    """
    (src_color → dst_color) frequency table built from demo pairs.
    Also tracks which grid positions tend to change.
    """

    def __init__(self, demo_pairs: List[Tuple[Grid, Grid]]):
        # counts[src][dst] = number of times src mapped to dst
        self.counts: Dict[int, Counter] = defaultdict(Counter)
        # pos_change_frac[(r, c)] = fraction of demos where cell (r,c) changed
        self.pos_change_frac: Dict[Tuple[int,int], float] = {}
        # demo output set for exact-match bonus
        self.demo_outputs = []
        # total demo pairs
        self.n_demos = len(demo_pairs)

        self._build(demo_pairs)

    def _build(self, demo_pairs: List[Tuple[Grid, Grid]]) -> None:
        if not demo_pairs:
            return

        # Per-position change tracking
        # Only track positions common to all pairs
        pos_changed: Dict[Tuple[int,int], int] = defaultdict(int)
        pos_total:   Dict[Tuple[int,int], int] = defaultdict(int)

        for inp, out in demo_pairs:
            self.demo_outputs.append(out)
            R_in, C_in   = len(inp),  len(inp[0])  if inp else 0
            R_out, C_out = len(out), len(out[0]) if out else 0
            R = min(R_in, R_out)
            C = min(C_in, C_out)
            for r in range(R):
                for c in range(C):
                    src = inp[r][c]
                    dst = out[r][c]
                    self.counts[src][dst] += 1
                    pos_total[(r, c)] += 1
                    if src != dst:
                        pos_changed[(r, c)] += 1

        for pos, total in pos_total.items():
            self.pos_change_frac[pos] = pos_changed.get(pos, 0) / total

    def log_prob_transition(self, src: int, dst: int) -> float:
        """Log P(dst | src) with Laplace smoothing."""
        row = self.counts[src]
        total = sum(row.values()) + N_COLORS * LAPLACE_ALPHA
        count = row.get(dst, 0) + LAPLACE_ALPHA
        return math.log(count / total)

    def log_prob_stay(self, src: int) -> float:
        """Log P(src → src) — convenience wrapper."""
        return self.log_prob_transition(src, src)

    def is_change_position(self, r: int, c: int, threshold: float = 0.5) -> bool:
        """True if this cell changes in >threshold fraction of demos."""
        return self.pos_change_frac.get((r, c), 0.0) > threshold


# ─── CANDIDATE SCORING ────────────────────────────────────────────────────────

def score_candidate(
    candidate: Grid,
    test_input: Grid,
    table: TransitionTable,
    pos_weight: float = 2.0,
) -> float:
    """
    Score a candidate grid using the transition table.

    Args:
        candidate:  proposed output grid
        test_input: test input grid
        table:      TransitionTable built from demos
        pos_weight: extra weight for cells that frequently change in demos

    Returns:
        log-likelihood score (higher = better)
    """
    R_in,  C_in  = len(test_input), len(test_input[0]) if test_input else 0
    R_out, C_out = len(candidate),  len(candidate[0])  if candidate  else 0
    R = min(R_in, R_out)
    C = min(C_in, C_out)

    if R == 0 or C == 0:
        return -1e9

    # Shape mismatch penalty (should be pre-filtered, but just in case)
    if R_in != R_out or C_in != C_out:
        return -1e9

    score = 0.0
    for r in range(R):
        for c in range(C):
            src = test_input[r][c]
            dst = candidate[r][c]
            lp  = table.log_prob_transition(src, dst)
            w   = pos_weight if table.is_change_position(r, c) else 1.0
            score += w * lp

    # Exact-demo bonus: if candidate == a demo output, boost slightly
    for demo_out in table.demo_outputs:
        if _grids_equal(candidate, demo_out):
            score += 1.0   # small bonus
            break

    return score


def _grids_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        if ra != rb:
            return False
    return True


# ─── PUBLIC RANKING API ───────────────────────────────────────────────────────

def rank_by_transition(
    candidates:   List[Grid],
    test_input:   Grid,
    demo_pairs:   List[Tuple[Grid, Grid]],
    greedy_cand:  Optional[Grid] = None,
    greedy_margin: float = 0.5,
) -> Tuple[List[Tuple[Grid, float]], TransitionTable]:
    """
    Score and sort candidates by pixel-transition log-likelihood.

    greedy_cand: if provided and in the pool, only displace it when another
        candidate's per-cell score advantage > greedy_margin log-units.
        This prevents swapping a correct greedy answer for a marginally
        better-scoring wrong one.

    Returns:
        (sorted_list_of_(grid, score), table)
        sorted descending (highest score first)
    """
    if not candidates:
        return [], TransitionTable([])

    table = TransitionTable(demo_pairs)
    scored = [
        (c, score_candidate(c, test_input, table))
        for c in candidates
    ]

    # Greedy-preference: only displace greedy if the gap is substantial
    if greedy_cand is not None:
        R = len(test_input)
        C = len(test_input[0]) if test_input else 1
        n_cells = max(R * C, 1)
        greedy_score = next((s for c, s in scored if _grids_equal(c, greedy_cand)), None)
        if greedy_score is not None:
            best_score = max(s for _, s in scored)
            per_cell_gap = (best_score - greedy_score) / n_cells
            if per_cell_gap < greedy_margin:
                # Gap too small — keep greedy first, sort rest by score
                scored.sort(key=lambda x: (not _grids_equal(x[0], greedy_cand), -x[1]))
                return scored, table

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, table


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

def transition_diagnostics(
    scored: List[Tuple[Grid, float]],
    gt_grid: Optional[Grid],
) -> dict:
    """Compute diagnostic info given ranked candidates and optional ground truth."""
    if not scored:
        return {
            "n_candidates": 0,
            "oracle_hit":   False,
            "trans_selects_correct": False,
            "trans_rank_of_correct": None,
            "trans_score_top1":      None,
            "trans_score_gap":       None,
        }

    n = len(scored)
    oracle_hit = False
    correct_rank = None
    correct_score = None

    if gt_grid is not None:
        for rank, (grid, score) in enumerate(scored):
            if _grids_equal(grid, gt_grid):
                oracle_hit = True
                correct_rank = rank
                correct_score = score
                break

    top_score = scored[0][1]
    gap = (correct_score - top_score) if correct_score is not None else None

    return {
        "n_candidates":          n,
        "oracle_hit":            oracle_hit,
        "trans_selects_correct": correct_rank == 0,
        "trans_rank_of_correct": correct_rank,
        "trans_score_top1":      top_score,
        "trans_score_gap":       gap,
    }
