"""
prototype_e/src/candidate_filters.py

Hard-Reject Candidate Filters
==============================
Two deterministic filters that eliminate structurally impossible candidates
before scoring.  Neither requires model inference.

Filter 1 — Shape Consistency
    From demo (input, output) pairs, infer the expected output shape rule:
      'same'   → output shape == input shape (most common)
      'fixed'  → output shape is a fixed (H, W) regardless of input
      'scaled' → output is exactly 2× or 3× input (scale2x/3x tasks)
      'unknown'→ no consistent rule detected; keep all candidates
    Candidates whose shape violates the inferred rule are discarded.

Filter 2 — Color Palette
    Collect the set of colors that appear in demo outputs.
    Also include colors that appear in demo inputs AND test input (these are
    "task-relevant" colors that the output might echo).
    Candidates containing colors outside this allowed set are discarded.
    Color 0 (background) is always allowed.
"""

from __future__ import annotations

from collections import Counter
from typing import List, Optional, Tuple

Grid = List[List[int]]


# ─── SHAPE UTILITIES ──────────────────────────────────────────────────────────

def _shape(g: Grid) -> Tuple[int, int]:
    return len(g), len(g[0]) if g else 0


def infer_output_shape_rule(
    demo_pairs: List[Tuple[Grid, Grid]]
) -> Tuple[str, Optional[Tuple[int, int]]]:
    """
    Infer the output shape rule from demo pairs.

    Returns:
        ('same',    None)       all demos have output.shape == input.shape
        ('fixed',   (H, W))     all demos have the same fixed output shape
        ('scaled',  (sy, sx))   output = sy×input_H  ×  sx×input_W  (2× or 3×)
        ('unknown', None)       no consistent rule
    """
    if not demo_pairs:
        return 'unknown', None

    # Check 'same'
    if all(_shape(o) == _shape(i) for i, o in demo_pairs):
        return 'same', None

    # Check 'fixed'
    out_shapes = {_shape(o) for _, o in demo_pairs}
    if len(out_shapes) == 1:
        return 'fixed', out_shapes.pop()

    # Check 'scaled' — output = k × input for k in {2, 3}
    for sy, sx in [(2, 2), (3, 3), (2, 1), (1, 2), (3, 1), (1, 3)]:
        if all(
            _shape(o)[0] == sy * _shape(i)[0] and
            _shape(o)[1] == sx * _shape(i)[1]
            for i, o in demo_pairs
        ):
            return 'scaled', (sy, sx)

    return 'unknown', None


def expected_output_shape(
    rule: str,
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
    return None  # unknown


def filter_by_shape(
    candidates: List[Grid],
    test_input: Grid,
    demo_pairs: List[Tuple[Grid, Grid]],
) -> Tuple[List[Grid], int]:
    """
    Remove candidates whose shape contradicts the demo-inferred shape rule.

    Returns:
        (surviving_candidates, n_rejected)
    """
    rule, param = infer_output_shape_rule(demo_pairs)
    exp = expected_output_shape(rule, param, test_input)

    if exp is None:
        return candidates, 0  # unknown rule — keep all

    kept     = [c for c in candidates if _shape(c) == exp]
    rejected = len(candidates) - len(kept)
    return kept, rejected


# ─── COLOR PALETTE ────────────────────────────────────────────────────────────

def infer_allowed_colors(
    demo_pairs: List[Tuple[Grid, Grid]],
    test_input: Grid,
) -> frozenset:
    """
    Build the set of colors that are legal in any output grid.

    Allowed = (colors in any demo INPUT or OUTPUT) ∪ (colors in test input) ∪ {0}

    We include demo INPUTS and test_input because many ARC tasks map input
    colors to output — a color from the input may appear in the output even
    if it never appeared in a demo output.  We only block colors that have
    NEVER appeared anywhere in the task (truly hallucinated colors).
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
    candidates: List[Grid],
    allowed_colors: frozenset,
) -> Tuple[List[Grid], int]:
    """
    Remove candidates containing colors outside the allowed palette.

    Returns:
        (surviving_candidates, n_rejected)
    """
    def _valid(g: Grid) -> bool:
        for row in g:
            for v in row:
                if v not in allowed_colors:
                    return False
        return True

    kept     = [c for c in candidates if _valid(c)]
    rejected = len(candidates) - len(kept)
    return kept, rejected


# ─── COMBINED HARD FILTER ─────────────────────────────────────────────────────

def hard_filter(
    candidates: List[Grid],
    test_input:  Grid,
    demo_pairs:  List[Tuple[Grid, Grid]],
) -> Tuple[List[Grid], dict]:
    """
    Apply shape + palette filters in sequence.

    Returns:
        (survivors, filter_stats)
    """
    n_in = len(candidates)

    # Shape filter
    after_shape, n_shape = filter_by_shape(candidates, test_input, demo_pairs)

    # Palette filter
    allowed = infer_allowed_colors(demo_pairs, test_input)
    after_palette, n_palette = filter_by_palette(after_shape, allowed)

    stats = {
        "n_in":          n_in,
        "n_shape_rej":   n_shape,
        "n_palette_rej": n_palette,
        "n_out":         len(after_palette),
        "shape_rule":    infer_output_shape_rule(demo_pairs)[0],
        "n_allowed_colors": len(allowed),
    }
    return after_palette, stats
