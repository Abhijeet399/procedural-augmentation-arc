"""
prototype_d/src/rule_search.py  (v2)

Rule Search Engine
==================
Exhaustive search for a primitive program consistent with all demo pairs.

Search budget (fastest → most expensive):
  Depth 1a: geometric + structural + object (~50 candidates,     ~0.01s)
  Depth 1b: + task color + task conditional (~100–1000 new,     ~0.1s)
  Depth 2a: geometric × geometric pairs    (~64 candidates,      ~0.01s)
  Depth 2b: geometric × color              (~geo × color,        ~0.1s)
  Depth 2c: object × color                 (~obj × color,        ~0.2s)
  Depth 2d: conditional depth-1 only       (each cond prim alone,~0.5s)
  Depth 3:  geometric triples (optional)   (~512,                ~2s)

Conditional primitives are tested at depth-1 before depth-2 compositions
because many ARC tasks are solved by a single conditional rule.
"""

from __future__ import annotations

import itertools
from typing import List, Optional, Tuple

from primitives import (
    Grid, Primitive,
    ALL_SINGLE_PRIMITIVES,
    GEOMETRIC_PRIMITIVES,
    STRUCTURAL_PRIMITIVES,
    OBJECT_PRIMITIVES,
    COLOR_PRIMITIVES,
    build_color_primitives_for_task,
    build_conditional_primitives_for_task,
    build_all_task_primitives,
    grid_equal, compose,
)

# ─── TYPES ────────────────────────────────────────────────────────────────────

DemoPair   = Tuple[Grid, Grid]
RuleResult = Tuple[Optional[Primitive], str, int]


# ─── CONSISTENCY CHECK ────────────────────────────────────────────────────────

def rule_is_consistent(rule: Primitive, demo_pairs: List[DemoPair]) -> bool:
    for inp, out in demo_pairs:
        try:
            predicted = rule(inp)
        except Exception:
            return False
        if not grid_equal(predicted, out):
            return False
    return True


# ─── SEARCH ───────────────────────────────────────────────────────────────────

def find_rule(
    demo_pairs: List[DemoPair],
    max_depth: int = 2,
    verbose: bool = False,
) -> RuleResult:
    """
    Search for a primitive program consistent with all demo pairs.

    Returns (rule, name, depth) or (None, "NOT_FOUND", -1).

    Search order:
      depth 1:  all single primitives (geometric, structural, object, color)
                + task-scoped color primitives
                + task-scoped conditional primitives  ← NEW
      depth 2:  geo×geo, geo×color, obj×color, cond(depth-1 only)
      depth 3:  geometric triples (optional)
    """
    if not demo_pairs:
        return None, "NO_DEMOS", -1

    demo_inputs  = [p[0] for p in demo_pairs]
    demo_outputs = [p[1] for p in demo_pairs]

    # Task-scoped parameterised primitives
    color_prims = build_color_primitives_for_task(demo_inputs, demo_outputs)
    cond_prims  = build_conditional_primitives_for_task(demo_inputs, demo_outputs)

    # ── Depth 1 ───────────────────────────────────────────────────────────────
    d1_static = ALL_SINGLE_PRIMITIVES          # geometric + structural + object + color
    d1_all    = d1_static + color_prims + cond_prims

    if verbose:
        print(f"    [search] depth 1 — {len(d1_all)} candidates "
              f"(static={len(d1_static)} color={len(color_prims)} cond={len(cond_prims)}) ...")

    for rule in d1_all:
        if rule_is_consistent(rule, demo_pairs):
            name = getattr(rule, "__name__", str(rule))
            if verbose:
                print(f"    [search] ✓ FOUND depth 1: {name}")
            return rule, name, 1

    if max_depth < 2:
        return None, "NOT_FOUND", -1

    # ── Depth 2 ───────────────────────────────────────────────────────────────
    # 2a: geometric pairs (cheap, 64 candidates)
    d2_geo = [compose(f, g) for f, g in itertools.product(GEOMETRIC_PRIMITIVES, repeat=2)]

    # 2b: geometric + structural combos (covers e.g. rotate → crop)
    geo_struct = GEOMETRIC_PRIMITIVES + STRUCTURAL_PRIMITIVES
    d2_geo_struct = [
        compose(f, g)
        for f in GEOMETRIC_PRIMITIVES
        for g in STRUCTURAL_PRIMITIVES
    ] + [
        compose(f, g)
        for f in STRUCTURAL_PRIMITIVES
        for g in GEOMETRIC_PRIMITIVES
    ]

    # 2c: (geometric|object) × color  and  color × (geometric|object)
    geo_obj   = GEOMETRIC_PRIMITIVES + OBJECT_PRIMITIVES
    d2_obj_color = [
        compose(f, g)
        for f in geo_obj
        for g in color_prims
    ] + [
        compose(f, g)
        for f in color_prims
        for g in geo_obj
    ]

    # 2d: geometric × conditional and conditional × geometric
    #     (only if conditional set is not huge — cap at 200 for speed)
    cond_subset = cond_prims[:200]
    d2_geo_cond = [
        compose(f, g)
        for f in GEOMETRIC_PRIMITIVES
        for g in cond_subset
    ] + [
        compose(f, g)
        for f in cond_subset
        for g in GEOMETRIC_PRIMITIVES
    ]

    # 2e: object × object
    d2_obj_obj = [
        compose(f, g)
        for f in OBJECT_PRIMITIVES
        for g in OBJECT_PRIMITIVES
        if f is not g
    ]

    d2_all = d2_geo + d2_geo_struct + d2_obj_color + d2_geo_cond + d2_obj_obj

    if verbose:
        print(f"    [search] depth 2 — {len(d2_all)} candidates ...")

    for rule in d2_all:
        if rule_is_consistent(rule, demo_pairs):
            name = getattr(rule, "__name__", str(rule))
            if verbose:
                print(f"    [search] ✓ FOUND depth 2: {name}")
            return rule, name, 2

    if max_depth < 3:
        return None, "NOT_FOUND", -1

    # ── Depth 3 (optional, geometric only) ───────────────────────────────────
    d3 = [
        compose(f, g, h)
        for f, g, h in itertools.product(GEOMETRIC_PRIMITIVES, repeat=3)
    ]
    if verbose:
        print(f"    [search] depth 3 — {len(d3)} candidates ...")
    for rule in d3:
        if rule_is_consistent(rule, demo_pairs):
            name = getattr(rule, "__name__", str(rule))
            if verbose:
                print(f"    [search] ✓ FOUND depth 3: {name}")
            return rule, name, 3

    return None, "NOT_FOUND", -1


# ─── DIAGNOSTICS: return ALL valid depth-1 rules ─────────────────────────────

def find_all_rules(
    demo_pairs: List[DemoPair],
    max_depth: int = 1,
) -> List[Tuple[Primitive, str, int]]:
    demo_inputs  = [p[0] for p in demo_pairs]
    demo_outputs = [p[1] for p in demo_pairs]
    color_prims  = build_color_primitives_for_task(demo_inputs, demo_outputs)
    cond_prims   = build_conditional_primitives_for_task(demo_inputs, demo_outputs)
    all_d1 = ALL_SINGLE_PRIMITIVES + color_prims + cond_prims
    results = []
    for rule in all_d1:
        if rule_is_consistent(rule, demo_pairs):
            results.append((rule, getattr(rule, "__name__", "?"), 1))
    return results
