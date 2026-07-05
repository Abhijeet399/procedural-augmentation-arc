"""
prototype_e/src/transition_ranker.py

Pixel-Transition Re-Ranker
===========================
From demo (input, output) pairs, build a frequency table of per-cell
color transitions: (src_color -> dst_color).

For each candidate output, score it by the log-likelihood of each cell's
transition under the learned table.  Higher score = better match to the
transformation pattern.

Scoring -- two modes (auto-selected per candidate):

  1. SAME-SHAPE (output shape == input shape):
       score(c) = Σ w(r,c) * log P(c[r][c] | test_input[r][c])
       where P(dst | src) from (src->dst) counts over demo pairs.

  2. SHAPE-MISMATCH (fixed/scaled output tasks):
       score(c) = Σ log P(c[r][c] at output position (r,c))
       where the position-color distribution is built from demo *outputs* only.
       Previously returned hard -1e9, making ranking useless for these tasks.

Tie-breaking extras:
  - Position-aware bonus: weight frequently-changing cells more (same-shape only).
  - Exact-demo bonus: +1.0 if candidate equals a demo output.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

Grid = List[List[int]]

N_COLORS = 10
LAPLACE_ALPHA = 0.1


# ─── TRANSITION TABLE ─────────────────────────────────────────────────────────

class TransitionTable:
    """
    Dual scoring table:
      - counts[(src,dst)]:        input-to-output color transitions (same-shape tasks)
      - output_pos_counts[(r,c)]: per-position output color frequencies (mismatch fallback)
    """

    def __init__(self, demo_pairs: List[Tuple[Grid, Grid]]):
        self.counts: Dict[int, Counter] = defaultdict(Counter)
        self.pos_change_frac: Dict[Tuple[int,int], float] = {}
        self.output_pos_counts: Dict[Tuple[int,int], Counter] = defaultdict(Counter)
        self.demo_outputs = []
        self.n_demos = len(demo_pairs)
        self._build(demo_pairs)

    def _build(self, demo_pairs):
        if not demo_pairs:
            return
        pos_changed: Dict[Tuple[int,int], int] = defaultdict(int)
        pos_total:   Dict[Tuple[int,int], int] = defaultdict(int)

        for inp, out in demo_pairs:
            self.demo_outputs.append(out)
            R_in  = len(inp);   C_in  = len(inp[0])  if inp else 0
            R_out = len(out);   C_out = len(out[0]) if out else 0

            # Output position frequency -- always built regardless of shape
            for r in range(R_out):
                for c in range(C_out):
                    self.output_pos_counts[(r, c)][out[r][c]] += 1

            # Input->output transitions -- only where shapes overlap
            R = min(R_in, R_out);  C = min(C_in, C_out)
            for r in range(R):
                for c in range(C):
                    src = inp[r][c];  dst = out[r][c]
                    self.counts[src][dst] += 1
                    pos_total[(r, c)] += 1
                    if src != dst:
                        pos_changed[(r, c)] += 1

        for pos, total in pos_total.items():
            self.pos_change_frac[pos] = pos_changed.get(pos, 0) / total

    def log_prob_transition(self, src: int, dst: int) -> float:
        row = self.counts[src]
        total = sum(row.values()) + N_COLORS * LAPLACE_ALPHA
        return math.log((row.get(dst, 0) + LAPLACE_ALPHA) / total)

    def log_prob_output_position(self, r: int, c: int, color: int) -> float:
        """Fallback scorer: P(color at output cell (r,c)), built from demo outputs only."""
        counts = self.output_pos_counts[(r, c)]
        total  = sum(counts.values()) + N_COLORS * LAPLACE_ALPHA
        return math.log((counts.get(color, 0) + LAPLACE_ALPHA) / total)

    def log_prob_stay(self, src: int) -> float:
        return self.log_prob_transition(src, src)

    def is_change_position(self, r: int, c: int, threshold: float = 0.5) -> bool:
        return self.pos_change_frac.get((r, c), 0.0) > threshold


# ─── CANDIDATE SCORING ────────────────────────────────────────────────────────

def score_candidate(
    candidate: Grid,
    test_input: Grid,
    table: TransitionTable,
    pos_weight: float = 2.0,
) -> float:
    """
    Score a candidate output grid.

    Automatically switches scoring mode based on shape:
      - Same shape as test_input  -> transition scoring (original method)
      - Different shape           -> output-position frequency (new fallback)

    The fallback replaces the hard -1e9 penalty that previously made all
    candidates in fixed/scaled output tasks score identically, leaving
    diversity bonus as the only differentiator (which hurt rather than helped).
    """
    R_in  = len(test_input); C_in  = len(test_input[0]) if test_input else 0
    R_out = len(candidate);  C_out = len(candidate[0])  if candidate  else 0

    if R_out == 0 or C_out == 0:
        return -1e9

    # ── Shape-mismatch: output-position frequency scoring ─────────────────────
    if R_in != R_out or C_in != C_out:
        score = 0.0
        for r in range(R_out):
            for c in range(C_out):
                score += table.log_prob_output_position(r, c, candidate[r][c])
        for demo_out in table.demo_outputs:
            if _grids_equal(candidate, demo_out):
                score += 1.0
                break
        return score

    # ── Same-shape: input->output transition scoring ──────────────────────────
    score = 0.0
    for r in range(R_out):
        for c in range(C_out):
            lp = table.log_prob_transition(test_input[r][c], candidate[r][c])
            w  = pos_weight if table.is_change_position(r, c) else 1.0
            score += w * lp
    for demo_out in table.demo_outputs:
        if _grids_equal(candidate, demo_out):
            score += 1.0
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
    candidates:    List[Grid],
    test_input:    Grid,
    demo_pairs:    List[Tuple[Grid, Grid]],
    greedy_cand:   Optional[Grid] = None,
    greedy_margin: float = 0.5,
) -> Tuple[List[Tuple[Grid, float]], TransitionTable]:
    """
    Score and sort candidates by transition log-likelihood.

    Greedy preference: only displace greedy if best competitor leads by
    > greedy_margin nats per cell.  Uses candidate shape for normalisation
    (correct for both same-shape and shape-mismatch tasks).
    """
    if not candidates:
        return [], TransitionTable([])

    table  = TransitionTable(demo_pairs)
    scored = [(c, score_candidate(c, test_input, table)) for c in candidates]

    if greedy_cand is not None:
        greedy_score = next((s for c, s in scored if _grids_equal(c, greedy_cand)), None)
        if greedy_score is not None:
            best_score = max(s for _, s in scored)
            R_c = len(greedy_cand);  C_c = len(greedy_cand[0]) if greedy_cand else 1
            n_cells = max(R_c * C_c, 1)
            if (best_score - greedy_score) / n_cells < greedy_margin:
                scored.sort(key=lambda x: (not _grids_equal(x[0], greedy_cand), -x[1]))
                return scored, table

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored, table


# ─── DIAGNOSTICS ──────────────────────────────────────────────────────────────

def transition_diagnostics(
    scored: List[Tuple[Grid, float]],
    gt_grid: Optional[Grid],
) -> dict:
    if not scored:
        return {
            "n_candidates": 0, "oracle_hit": False,
            "trans_selects_correct": False, "trans_rank_of_correct": None,
            "trans_score_top1": None, "trans_score_gap": None,
        }
    oracle_hit = False;  correct_rank = None;  correct_score = None
    if gt_grid is not None:
        for rank, (grid, score) in enumerate(scored):
            if _grids_equal(grid, gt_grid):
                oracle_hit = True;  correct_rank = rank;  correct_score = score;  break
    top_score = scored[0][1]
    return {
        "n_candidates":          len(scored),
        "oracle_hit":            oracle_hit,
        "trans_selects_correct": correct_rank == 0,
        "trans_rank_of_correct": correct_rank,
        "trans_score_top1":      top_score,
        "trans_score_gap":       (correct_score - top_score) if correct_score is not None else None,
    }
