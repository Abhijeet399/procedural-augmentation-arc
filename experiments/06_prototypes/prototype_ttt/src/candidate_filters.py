"""
prototype_ttt/src/candidate_filters.py

Hard-Reject Candidate Filters  (EXP P additions marked with ★)
================================================================
Two deterministic filters that eliminate structurally impossible candidates
before scoring, plus a smart fallback for when everything gets filtered.

Filter 1 — Shape Consistency
    Infer the expected output shape rule from demo pairs:
      'same'   → output shape == input shape
      'fixed'  → output shape is a fixed (H, W)
      'scaled' → output is exactly k× input (scale2x/3x tasks)
      'unknown'→ no consistent rule; keep all candidates
    Candidates whose shape violates the inferred rule are discarded.

Filter 2 — Color Palette
    Collect the set of colors that appear in demo outputs.
    Candidates containing colors outside this allowed set are discarded.

★ Smart Fallback (EXP P)
    When hard_filter rejects ALL candidates ('fallback=all_filtered'),
    instead of dumping all wrong-shape candidates back to the ranker we:
      1. Compute the expected output shape from demo pairs.
      2. Reshape the greedy candidate to that shape (crop excess rows/cols,
         pad missing rows/cols with 0).
      3. Use the reshaped greedy as the single survivor.
    When the shape rule is 'unknown' we fall back to the original behaviour
    (use all unfiltered candidates) because we have no shape target.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

Grid = List[List[int]]


# ─── SHAPE UTILITIES ──────────────────────────────────────────────────────────

def _shape(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)


def infer_output_shape_rule(
    demo_pairs: List[Tuple[Grid, Grid]]
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Infer the output shape rule from demo pairs.

    Returns:
        ('same',    None)       all demos have output.shape == input.shape
        ('fixed',   (H, W))     all demos have the same fixed output shape
        ('scaled',  (sy, sx))   output = sy×input_H  ×  sx×input_W
        ('unknown', None)       no consistent rule
    """
    if not demo_pairs:
        return 'unknown', None

    if all(_shape(o) == _shape(i) for i, o in demo_pairs):
        return 'same', None

    out_shapes = {_shape(o) for _, o in demo_pairs}
    if len(out_shapes) == 1:
        return 'fixed', out_shapes.pop()

    for sy, sx in [(2, 2), (3, 3), (2, 1), (1, 2), (3, 1), (1, 3)]:
        if all(
            _shape(o)[0] == sy * _shape(i)[0] and
            _shape(o)[1] == sx * _shape(i)[1]
            for i, o in demo_pairs
        ):
            return 'scaled', (sy, sx)

    return 'unknown', None


def expected_output_shape(
    rule:  str,
    param: Optional[Tuple[int, int]],
    test_input: Grid,
) -> Optional[Tuple[int, int]]:
    """Compute expected output shape given the rule and test input shape."""
    ih, iw = _shape(test_input)
    if rule == 'same':
        return (ih, iw)
    elif rule == 'fixed':
        return param
    elif rule == 'scaled':
        sy, sx = param
        return (sy * ih, sx * iw)
    return None


def filter_by_shape(
    candidates: List[Grid],
    test_input:  Grid,
    demo_pairs:  List[Tuple[Grid, Grid]],
) -> Tuple[List[Grid], int]:
    """
    Remove candidates whose shape contradicts the demo-inferred shape rule.
    Returns (surviving_candidates, n_rejected).
    """
    rule, param = infer_output_shape_rule(demo_pairs)
    exp = expected_output_shape(rule, param, test_input)
    if exp is None:
        return candidates, 0
    kept     = [c for c in candidates if _shape(c) == exp]
    rejected = len(candidates) - len(kept)
    return kept, rejected


# ─── COLOR PALETTE ────────────────────────────────────────────────────────────

def infer_allowed_colors(
    demo_pairs: List[Tuple[Grid, Grid]],
    test_input: Grid,
) -> frozenset:
    """
    Build the set of colors legal in any output grid.
    Allowed = all colors seen anywhere in demos + test_input + {0}.
    """
    allowed: set = {0}
    for inp, out in demo_pairs:
        for row in inp:
            allowed.update(row)
        for row in out:
            allowed.update(row)
    for row in test_input:
        allowed.update(row)
    return frozenset(allowed)


def filter_by_palette(
    candidates:     List[Grid],
    allowed_colors: frozenset,
) -> Tuple[List[Grid], int]:
    """Remove candidates containing colours outside the allowed palette."""
    def _valid(g: Grid) -> bool:
        for row in g:
            for v in row:
                if v not in allowed_colors:
                    return False
        return True

    kept     = [c for c in candidates if _valid(c)]
    rejected = len(candidates) - len(kept)
    return kept, rejected


# ─── ★ SMART FALLBACK (EXP P) ─────────────────────────────────────────────────

def reshape_to_shape(grid: Grid, target_h: int, target_w: int) -> Grid:
    """
    Crop or zero-pad grid to exactly (target_h, target_w).

    Rows beyond target_h are discarded.
    Missing rows are filled with zeros.
    Each row is cropped or padded with zeros to target_w.
    """
    result: Grid = []
    for r in range(target_h):
        if r < len(grid):
            row = list(grid[r][:target_w])          # crop cols
            row.extend([0] * (target_w - len(row))) # pad cols
        else:
            row = [0] * target_w                     # pad row
        result.append(row)
    return result


def smart_fallback(
    candidates:  List[Grid],
    greedy_cand: Grid,
    test_input:  Grid,
    demo_pairs:  List[Tuple[Grid, Grid]],
) -> Tuple[List[Grid], str]:
    """
    ★ EXP P smart fallback for when hard_filter rejects all candidates.

    Strategy (in order):
      1. Infer expected output shape from demo pairs.
      2. If shape is known:
           a. Use greedy reshaped to expected (H, W) as single survivor.
              → fallback reason = 'reshaped_greedy'
      3. If shape is unknown:
           a. Fall back to original behaviour: use all unfiltered candidates.
              → fallback reason = 'all_filtered_unknown_shape'

    Returns (survivors, fallback_reason_string)
    """
    rule, param = infer_output_shape_rule(demo_pairs)
    exp = expected_output_shape(rule, param, test_input)

    if exp is not None and greedy_cand:
        target_h, target_w = exp
        reshaped = reshape_to_shape(greedy_cand, target_h, target_w)
        return [reshaped], "reshaped_greedy"
    else:
        return candidates[:], "all_filtered_unknown_shape"


# ─── COMBINED HARD FILTER ─────────────────────────────────────────────────────

def hard_filter(
    candidates: List[Grid],
    test_input:  Grid,
    demo_pairs:  List[Tuple[Grid, Grid]],
) -> Tuple[List[Grid], dict]:
    """
    Apply shape + palette filters in sequence.

    Returns (survivors, filter_stats).
    NOTE: When all candidates are filtered, returns an empty list — the caller
    decides what fallback to use.  Use smart_fallback() for EXP P behaviour.
    """
    n_in = len(candidates)

    after_shape, n_shape   = filter_by_shape(candidates, test_input, demo_pairs)
    allowed                = infer_allowed_colors(demo_pairs, test_input)
    after_palette, n_palette = filter_by_palette(after_shape, allowed)

    stats = {
        "n_in":              n_in,
        "n_shape_rej":       n_shape,
        "n_palette_rej":     n_palette,
        "n_out":             len(after_palette),
        "shape_rule":        infer_output_shape_rule(demo_pairs)[0],
        "n_allowed_colors":  len(allowed),
    }
    return after_palette, stats
