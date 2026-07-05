"""
prototype_d/src/primitives.py

ARC Transformation Primitives Library
======================================
All functions operate on grids: List[List[int]]
  - Values 0–9 are ARC colors
  - Grids are row-major: grid[row][col]

Every primitive is a function of signature:
    f(grid: Grid) -> Grid

Additions in v2:
  - Object-level primitives: connected-component extraction, bounding-box
    crop, object fill, keep-largest / keep-smallest object, duplicate
    objects, sort objects by size/color, object count → output grid
  - Conditional coloring: neighbour-triggered recolor, flood-fill from
    seed color, color cells that touch a given color
"""

from __future__ import annotations

from collections import Counter, deque
from typing import Callable, Dict, List, Optional, Set, Tuple

Grid = List[List[int]]
Primitive = Callable[[Grid], Grid]


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def grid_equal(a: Grid, b: Grid) -> bool:
    if len(a) != len(b):
        return False
    return all(ra == rb for ra, rb in zip(a, b))


def grid_size(g: Grid) -> Tuple[int, int]:
    return len(g), (len(g[0]) if g else 0)


def copy_grid(g: Grid) -> Grid:
    return [row[:] for row in g]


def colors_used(g: Grid) -> List[int]:
    seen: set = set()
    for row in g:
        seen.update(row)
    return sorted(seen)


def most_common_color(g: Grid) -> int:
    cnt: Counter = Counter()
    for row in g:
        cnt.update(row)
    return cnt.most_common(1)[0][0]


def least_common_color(g: Grid) -> int:
    cnt: Counter = Counter()
    for row in g:
        cnt.update(row)
    return cnt.most_common()[-1][0]


def background_color(g: Grid) -> int:
    """Most-common colour is treated as background."""
    return most_common_color(g)


# ─── CONNECTED COMPONENTS ────────────────────────────────────────────────────

def _flood_fill_component(
    g: Grid, r0: int, c0: int, visited: List[List[bool]]
) -> List[Tuple[int, int]]:
    """BFS flood-fill returning all cells of the connected component."""
    R, C = grid_size(g)
    color = g[r0][c0]
    cells: List[Tuple[int, int]] = []
    queue = deque([(r0, c0)])
    visited[r0][c0] = True
    while queue:
        r, c = queue.popleft()
        cells.append((r, c))
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 0 <= nr < R and 0 <= nc < C and not visited[nr][nc] and g[nr][nc] == color:
                visited[nr][nc] = True
                queue.append((nr, nc))
    return cells


def get_objects(g: Grid, background: int = -1) -> List[Tuple[int, List[Tuple[int,int]]]]:
    """
    Return list of (color, cells) for each connected component of non-background colour.
    If background==-1, auto-detect as most-common colour.
    """
    R, C = grid_size(g)
    bg = background if background >= 0 else most_common_color(g)
    visited = [[False] * C for _ in range(R)]
    objects = []
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and g[r][c] != bg:
                cells = _flood_fill_component(g, r, c, visited)
                objects.append((g[r][c], cells))
    return objects


def _bounding_box(cells: List[Tuple[int,int]]) -> Tuple[int,int,int,int]:
    """Returns (min_r, min_c, max_r, max_c)."""
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


# ─── OBJECT-LEVEL PRIMITIVES ─────────────────────────────────────────────────

def keep_largest_object(g: Grid) -> Grid:
    """Keep only the largest (non-background) connected component; zero the rest."""
    R, C = grid_size(g)
    bg = background_color(g)
    objects = get_objects(g, bg)
    if not objects:
        return copy_grid(g)
    largest = max(objects, key=lambda x: len(x[1]))
    keep: Set[Tuple[int,int]] = set(largest[1])
    out = [[bg] * C for _ in range(R)]
    for r, c in keep:
        out[r][c] = g[r][c]
    return out


def keep_smallest_object(g: Grid) -> Grid:
    """Keep only the smallest connected component; zero the rest."""
    R, C = grid_size(g)
    bg = background_color(g)
    objects = get_objects(g, bg)
    if not objects:
        return copy_grid(g)
    smallest = min(objects, key=lambda x: len(x[1]))
    keep: Set[Tuple[int,int]] = set(smallest[1])
    out = [[bg] * C for _ in range(R)]
    for r, c in keep:
        out[r][c] = g[r][c]
    return out


def fill_objects_with_color(g: Grid) -> Grid:
    """Fill every object's bounding box with its own colour (solid rectangle)."""
    R, C = grid_size(g)
    bg = background_color(g)
    out = [[bg] * C for _ in range(R)]
    for color, cells in get_objects(g, bg):
        r0, c0, r1, c1 = _bounding_box(cells)
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                out[r][c] = color
    return out


def crop_to_content(g: Grid) -> Grid:
    """Crop grid to the tightest bounding box of all non-background cells."""
    bg = background_color(g)
    R, C = grid_size(g)
    rs = [r for r in range(R) for c in range(C) if g[r][c] != bg]
    cs = [c for r in range(R) for c in range(C) if g[r][c] != bg]
    if not rs:
        return copy_grid(g)
    r0, r1, c0, c1 = min(rs), max(rs), min(cs), max(cs)
    return [g[r][c0:c1+1] for r in range(r0, r1+1)]


def sort_objects_by_size(g: Grid) -> Grid:
    """
    Re-draw objects sorted by ascending size, left-to-right, keeping same rows.
    Each object is placed at the same row as original but packed left-to-right
    with 1-cell gaps.  Useful for tasks that reorder objects by size.
    """
    R, C = grid_size(g)
    bg = background_color(g)
    objects = get_objects(g, bg)
    if not objects:
        return copy_grid(g)
    objects_sorted = sorted(objects, key=lambda x: len(x[1]))
    out = [[bg] * C for _ in range(R)]
    col_cursor = 0
    for color, cells in objects_sorted:
        r0, c0, r1, c1 = _bounding_box(cells)
        w = c1 - c0 + 1
        h = r1 - r0 + 1
        if col_cursor + w > C:
            break
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if g[r][c] == color:
                    out[r][col_cursor + (c - c0)] = color
        col_cursor += w + 1
    return out


def count_objects_to_row(g: Grid) -> Grid:
    """
    Output a single-row grid whose length equals the number of non-background
    objects, filled with the most common non-background colour.
    Covers tasks like 'count the shapes → output a 1×N bar'.
    """
    bg = background_color(g)
    objects = get_objects(g, bg)
    n = len(objects)
    if n == 0:
        return [[bg]]
    fill = objects[0][0]  # colour of first object
    # try to use most common non-bg colour
    cnt: Counter = Counter()
    for color, cells in objects:
        cnt[color] += len(cells)
    fill = cnt.most_common(1)[0][0]
    return [[fill] * n]


def _make_object_factory(
    select_fn,  # (objects) -> selected_object_cells
    name: str,
) -> Primitive:
    """Generic factory for single-object extraction into a tight crop."""
    def _fn(g: Grid) -> Grid:
        bg = background_color(g)
        objects = get_objects(g, bg)
        if not objects:
            return copy_grid(g)
        color, cells = select_fn(objects)
        r0, c0, r1, c1 = _bounding_box(cells)
        R, C = r1 - r0 + 1, c1 - c0 + 1
        out = [[bg] * C for _ in range(R)]
        for r, c in cells:
            out[r - r0][c - c0] = color
        return out
    _fn.__name__ = name
    return _fn


crop_largest_object = _make_object_factory(
    lambda objs: max(objs, key=lambda x: len(x[1])),
    "crop_largest_object",
)
crop_smallest_object = _make_object_factory(
    lambda objs: min(objs, key=lambda x: len(x[1])),
    "crop_smallest_object",
)


def duplicate_objects_h(g: Grid) -> Grid:
    """Mirror each object horizontally within the same grid."""
    R, C = grid_size(g)
    bg = background_color(g)
    out = copy_grid(g)
    for color, cells in get_objects(g, bg):
        r0, c0, r1, c1 = _bounding_box(cells)
        w = c1 - c0
        new_c0 = C - 1 - c1
        for r, c in cells:
            nc = new_c0 + (c - c0)
            if 0 <= nc < C:
                out[r][nc] = color
    return out


def duplicate_objects_v(g: Grid) -> Grid:
    """Mirror each object vertically within the same grid."""
    R, C = grid_size(g)
    bg = background_color(g)
    out = copy_grid(g)
    for color, cells in get_objects(g, bg):
        r0, c0, r1, c1 = _bounding_box(cells)
        new_r0 = R - 1 - r1
        for r, c in cells:
            nr = new_r0 + (r - r0)
            if 0 <= nr < R:
                out[nr][c] = color
    return out


def _recolor_objects_by_size_factory() -> List[Primitive]:
    """
    Recolor: largest object → color X, smallest → color Y.
    Returns a list of primitives, one per (largest_color, smallest_color) combo.
    Called at task time so we can scope to the task's colors.
    """
    # This is a task-specific factory — see build_object_primitives_for_task
    return []


def object_count_to_grid(g: Grid) -> Grid:
    """
    Output a grid of size (n_objects × n_objects) filled with the non-bg colour.
    Covers 'count → fill square' tasks.
    """
    bg = background_color(g)
    objects = get_objects(g, bg)
    n = len(objects)
    if n == 0:
        return [[bg]]
    cnt: Counter = Counter(color for color, _ in objects)
    fill = cnt.most_common(1)[0][0]
    return [[fill] * n for _ in range(n)]


# ─── CONDITIONAL COLORING ────────────────────────────────────────────────────

def _neighbor_recolor_factory(
    trigger_color: int, target_color: int, new_color: int
) -> Primitive:
    """
    Recolor cells of `target_color` that are 4-adjacent to at least one
    cell of `trigger_color` → set them to `new_color`.
    """
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        out = copy_grid(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] == target_color:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == trigger_color:
                            out[r][c] = new_color
                            break
        return out
    _fn.__name__ = f"if_adj_{trigger_color}_recolor_{target_color}_to_{new_color}"
    return _fn


def _flood_recolor_factory(seed_color: int, new_color: int) -> Primitive:
    """
    Flood-fill: starting from ALL cells of `seed_color`, paint outward
    through background cells using `new_color`.  Like 'spread the color'.
    """
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        bg = background_color(g)
        if seed_color == bg:
            return copy_grid(g)
        out = copy_grid(g)
        visited = [[False] * C for _ in range(R)]
        queue: deque = deque()
        for r in range(R):
            for c in range(C):
                if g[r][c] == seed_color:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == bg and not visited[nr][nc]:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
        while queue:
            r, c = queue.popleft()
            out[r][c] = new_color
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == bg and not visited[nr][nc]:
                    visited[nr][nc] = True
                    queue.append((nr, nc))
        return out
    _fn.__name__ = f"flood_from_{seed_color}_to_{new_color}"
    return _fn


def _color_touching_factory(border_color: int, fill_color: int) -> Primitive:
    """
    Color ALL background cells that are 4-adjacent to the border_color
    with fill_color (single-step expansion, not full flood).
    """
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        bg = background_color(g)
        out = copy_grid(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] == bg:
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r+dr, c+dc
                        if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == border_color:
                            out[r][c] = fill_color
                            break
        return out
    _fn.__name__ = f"color_touching_{border_color}_with_{fill_color}"
    return _fn


def _enclosed_fill_factory(wall_color: int, fill_color: int) -> Primitive:
    """
    Flood from all grid borders through NON-wall cells.
    Any non-wall cell not reachable from the border is 'enclosed' -> fill_color.
    Works correctly even when wall_color == most_common_color (background).
    """
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        reachable = [[False] * C for _ in range(R)]
        queue: deque = deque()
        for r in range(R):
            for c in range(C):
                if (r == 0 or r == R-1 or c == 0 or c == C-1) and g[r][c] != wall_color:
                    if not reachable[r][c]:
                        reachable[r][c] = True
                        queue.append((r, c))
        while queue:
            r, c = queue.popleft()
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < R and 0 <= nc < C and not reachable[nr][nc] and g[nr][nc] != wall_color:
                    reachable[nr][nc] = True
                    queue.append((nr, nc))
        out = copy_grid(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] != wall_color and not reachable[r][c]:
                    out[r][c] = fill_color
        return out
    _fn.__name__ = f"fill_enclosed_by_{wall_color}_with_{fill_color}"
    return _fn


def _diagonal_recolor_factory(trigger_color: int, target_color: int, new_color: int) -> Primitive:
    """
    Like neighbor_recolor but uses 8-connectivity (including diagonals).
    """
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        out = copy_grid(g)
        for r in range(R):
            for c in range(C):
                if g[r][c] == target_color:
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == dc == 0:
                                continue
                            nr, nc = r+dr, c+dc
                            if 0 <= nr < R and 0 <= nc < C and g[nr][nc] == trigger_color:
                                out[r][c] = new_color
                                break
        return out
    _fn.__name__ = f"if_diag_adj_{trigger_color}_recolor_{target_color}_to_{new_color}"
    return _fn


# ─── GEOMETRIC TRANSFORMS ─────────────────────────────────────────────────────

def identity(g: Grid) -> Grid:
    return copy_grid(g)


def rotate90(g: Grid) -> Grid:
    R, C = grid_size(g)
    return [[g[R - 1 - c][r] for c in range(R)] for r in range(C)]


def rotate180(g: Grid) -> Grid:
    return rotate90(rotate90(g))


def rotate270(g: Grid) -> Grid:
    return rotate90(rotate90(rotate90(g)))


def flip_h(g: Grid) -> Grid:
    return [row[::-1] for row in g]


def flip_v(g: Grid) -> Grid:
    return g[::-1]


def transpose(g: Grid) -> Grid:
    R, C = grid_size(g)
    return [[g[r][c] for r in range(R)] for c in range(C)]


def antitranspose(g: Grid) -> Grid:
    R, C = grid_size(g)
    return [[g[C - 1 - c][R - 1 - r] for r in range(R)] for c in range(C)]


DIHEDRAL: List[Primitive] = [
    identity, rotate90, rotate180, rotate270,
    flip_h, flip_v, transpose, antitranspose,
]
DIHEDRAL_NAMES = [
    "identity", "rotate90", "rotate180", "rotate270",
    "flip_h", "flip_v", "transpose", "antitranspose",
]


# ─── COLOR TRANSFORMS ─────────────────────────────────────────────────────────

def _recolor_factory(src: int, dst: int) -> Primitive:
    def _fn(g: Grid) -> Grid:
        return [[dst if v == src else v for v in row] for row in g]
    _fn.__name__ = f"recolor_{src}_to_{dst}"
    return _fn


def _swap_colors_factory(a: int, b: int) -> Primitive:
    def _fn(g: Grid) -> Grid:
        def _map(v):
            if v == a: return b
            if v == b: return a
            return v
        return [[_map(v) for v in row] for row in g]
    _fn.__name__ = f"swap_colors_{a}_{b}"
    return _fn


def invert_colors(g: Grid) -> Grid:
    return [[9 - v for v in row] for row in g]


# ─── STRUCTURAL / SIZE TRANSFORMS ─────────────────────────────────────────────

def tile_2x2(g: Grid) -> Grid:
    doubled = [row + row for row in g]
    return doubled + doubled


def tile_2h(g: Grid) -> Grid:
    return [row + row for row in g]


def tile_2v(g: Grid) -> Grid:
    return g + [row[:] for row in g]


def scale2x(g: Grid) -> Grid:
    out = []
    for row in g:
        nr = []
        for v in row:
            nr += [v, v]
        out.append(nr)
        out.append(nr[:])
    return out


def scale3x(g: Grid) -> Grid:
    out = []
    for row in g:
        nr = []
        for v in row:
            nr += [v, v, v]
        for _ in range(3):
            out.append(nr[:])
    return out


def crop_border(g: Grid) -> Grid:
    R, C = grid_size(g)
    if R <= 2 or C <= 2:
        return copy_grid(g)
    return [row[1:C-1] for row in g[1:R-1]]


def mirror_right(g: Grid) -> Grid:
    return [row + row[::-1] for row in g]


def mirror_down(g: Grid) -> Grid:
    return g + g[::-1]


def outline_only(g: Grid) -> Grid:
    R, C = grid_size(g)
    out = copy_grid(g)
    for r in range(1, R-1):
        for c in range(1, C-1):
            out[r][c] = 0
    return out


def hollow_fill(g: Grid) -> Grid:
    R, C = grid_size(g)
    out = copy_grid(g)
    visited = [[False]*C for _ in range(R)]
    bg = most_common_color(g)
    stack = [
        (r, c) for r in range(R) for c in range(C)
        if (r == 0 or r == R-1 or c == 0 or c == C-1) and g[r][c] == bg
    ]
    while stack:
        r, c = stack.pop()
        if r < 0 or r >= R or c < 0 or c >= C: continue
        if visited[r][c] or g[r][c] != bg: continue
        visited[r][c] = True
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            stack.append((r+dr, c+dc))
    border_color = least_common_color(g) if least_common_color(g) != bg else 1
    for r in range(R):
        for c in range(C):
            if not visited[r][c] and g[r][c] == bg:
                out[r][c] = border_color
    return out


def _gravity_factory(direction: str) -> Primitive:
    def _fn(g: Grid) -> Grid:
        R, C = grid_size(g)
        out = [[0]*C for _ in range(R)]
        if direction in ("down", "up"):
            for c in range(C):
                col = [g[r][c] for r in range(R)]
                nonzero = [v for v in col if v != 0]
                zeros = [0] * (R - len(nonzero))
                filled = zeros + nonzero if direction == "down" else nonzero + zeros
                for r in range(R):
                    out[r][c] = filled[r]
        else:
            for r in range(R):
                row = g[r][:]
                nonzero = [v for v in row if v != 0]
                zeros = [0] * (C - len(nonzero))
                out[r] = zeros + nonzero if direction == "right" else nonzero + zeros
        return out
    _fn.__name__ = f"gravity_{direction}"
    return _fn


gravity_down  = _gravity_factory("down")
gravity_up    = _gravity_factory("up")
gravity_left  = _gravity_factory("left")
gravity_right = _gravity_factory("right")


# ─── COMPOSED ────────────────────────────────────────────────────────────────

def compose(*fns: Primitive) -> Primitive:
    def _fn(g: Grid) -> Grid:
        result = g
        for f in fns:
            result = f(result)
        return result
    _fn.__name__ = " → ".join(getattr(f, "__name__", "?") for f in fns)
    return _fn


# ─── PRIMITIVE CATALOGUES ────────────────────────────────────────────────────

GEOMETRIC_PRIMITIVES: List[Primitive] = [
    identity, rotate90, rotate180, rotate270,
    flip_h, flip_v, transpose, antitranspose,
]

STRUCTURAL_PRIMITIVES: List[Primitive] = [
    tile_2x2, tile_2h, tile_2v,
    scale2x, scale3x,
    crop_border, crop_to_content,
    mirror_right, mirror_down,
    outline_only, hollow_fill,
    gravity_down, gravity_up, gravity_left, gravity_right,
]

OBJECT_PRIMITIVES: List[Primitive] = [
    keep_largest_object,
    keep_smallest_object,
    fill_objects_with_color,
    crop_to_content,
    crop_largest_object,
    crop_smallest_object,
    duplicate_objects_h,
    duplicate_objects_v,
    count_objects_to_row,
    object_count_to_grid,
    sort_objects_by_size,
]

COLOR_PRIMITIVES: List[Primitive] = [
    invert_colors,
]

ALL_SINGLE_PRIMITIVES: List[Primitive] = (
    GEOMETRIC_PRIMITIVES + STRUCTURAL_PRIMITIVES + OBJECT_PRIMITIVES + COLOR_PRIMITIVES
)


# ─── TASK-SCOPED PARAMETER FACTORIES ─────────────────────────────────────────

def build_color_primitives_for_task(
    demo_inputs: List[Grid], demo_outputs: List[Grid]
) -> List[Primitive]:
    """Build parameterised color transforms scoped to colors in this task."""
    all_grids = demo_inputs + demo_outputs
    used: set = set()
    for g in all_grids:
        for row in g:
            used.update(row)
    colors = sorted(used)
    prims: List[Primitive] = []
    for src in colors:
        for dst in colors:
            if src != dst:
                prims.append(_recolor_factory(src, dst))
    for i, a in enumerate(colors):
        for b in colors[i+1:]:
            prims.append(_swap_colors_factory(a, b))
    return prims


def build_conditional_primitives_for_task(
    demo_inputs: List[Grid], demo_outputs: List[Grid]
) -> List[Primitive]:
    """
    Build conditional-coloring primitives scoped to the colors in this task.

    Includes:
      - neighbor_recolor(trigger, target, new): if adjacent to trigger, recolor
      - flood_recolor(seed, new): flood from seed outward through background
      - color_touching(border, fill): single-step border expansion
      - enclosed_fill(wall, fill): fill cells enclosed by wall color
      - diagonal_recolor(trigger, target, new): 8-connectivity version
    """
    all_grids = demo_inputs + demo_outputs
    used: set = set()
    for g in all_grids:
        for row in g:
            used.update(row)
    colors = sorted(used)
    prims: List[Primitive] = []

    for trigger in colors:
        for target in colors:
            for new in colors:
                if len({trigger, target, new}) < 2:
                    continue
                # Only generate if trigger != target (meaningful condition)
                if trigger == target:
                    continue
                prims.append(_neighbor_recolor_factory(trigger, target, new))
                prims.append(_diagonal_recolor_factory(trigger, target, new))

    for seed in colors:
        for new in colors:
            if seed != new:
                prims.append(_flood_recolor_factory(seed, new))

    for border in colors:
        for fill in colors:
            if border != fill:
                prims.append(_color_touching_factory(border, fill))
                prims.append(_enclosed_fill_factory(border, fill))

    return prims


def build_all_task_primitives(
    demo_inputs: List[Grid], demo_outputs: List[Grid]
) -> List[Primitive]:
    """Full set of task-scoped primitives: color + conditional + object."""
    return (
        build_color_primitives_for_task(demo_inputs, demo_outputs)
        + build_conditional_primitives_for_task(demo_inputs, demo_outputs)
    )
