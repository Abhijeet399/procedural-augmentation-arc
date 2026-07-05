"""
augmentations.py — D8 dihedral + color permutation augmentation utilities.

All functions operate on raw numpy grids (H×W int arrays with values 0-9).
No model dependency. Used by both the DFS decoder and PoE scorer.
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# D8 dihedral group (8 isometries of the square)
# ---------------------------------------------------------------------------

def d8_transform(grid: np.ndarray, idx: int) -> np.ndarray:
    """
    Apply one of the 8 D8 isometries to a 2-D grid.

    Convention (matches mdlARC training augmentation):
      0 = identity
      1 = 90° CCW rotation
      2 = 180° rotation
      3 = 270° CCW rotation (= 90° CW)
      4 = horizontal flip  (left ↔ right)
      5 = vertical flip    (top ↔ bottom)
      6 = main-diagonal transpose
      7 = anti-diagonal transpose
    """
    g = np.asarray(grid)
    if idx == 0: return g.copy()
    if idx == 1: return np.rot90(g, k=1)
    if idx == 2: return np.rot90(g, k=2)
    if idx == 3: return np.rot90(g, k=3)
    if idx == 4: return np.fliplr(g)
    if idx == 5: return np.flipud(g)
    if idx == 6: return g.T
    if idx == 7: return np.rot90(g.T, k=2)
    raise ValueError(f"d8 index must be 0-7, got {idx}")


def d8_inverse(idx: int) -> int:
    """Return the index of the inverse D8 transform."""
    # Inverses: 0→0, 1→3, 2→2, 3→1, 4→4, 5→5, 6→6, 7→7
    return [0, 3, 2, 1, 4, 5, 6, 7][idx]


def d8_transform_inverse(grid: np.ndarray, idx: int) -> np.ndarray:
    """Apply the INVERSE of D8 transform idx (used for AAIVR un-augmentation)."""
    return d8_transform(grid, d8_inverse(idx))


# ---------------------------------------------------------------------------
# Color permutation
# ---------------------------------------------------------------------------

def apply_color_perm(grid: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """
    Permute grid cell values according to perm[old_value] = new_value.

    Only permutes values 0-9. The perm array must be length 10.
    Tokens ≥ 10 (special tokens) are left unchanged.
    """
    out = grid.copy()
    mask = out < 10
    out[mask] = perm[out[mask]]
    return out


def random_color_perm(rng: np.random.Generator, n_colors: int = 10) -> np.ndarray:
    """Return a random permutation of [0, n_colors)."""
    return rng.permutation(n_colors).astype(np.int64)


def identity_color_perm(n_colors: int = 10) -> np.ndarray:
    return np.arange(n_colors, dtype=np.int64)


# ---------------------------------------------------------------------------
# Augmentation configuration
# ---------------------------------------------------------------------------

@dataclass
class AugParams:
    """A single augmentation: D8 transform + color permutation + example order."""
    d8_idx: int = 0
    color_perm: np.ndarray = field(
        default_factory=lambda: np.arange(10, dtype=np.int64)
    )
    example_order: Optional[list[int]] = None   # None = keep original order

    def apply_to_grid(self, grid: np.ndarray) -> np.ndarray:
        """Apply both D8 and color permutation to a single grid."""
        g = d8_transform(grid, self.d8_idx)
        return apply_color_perm(g, self.color_perm)

    def apply_inverse_to_grid(self, grid: np.ndarray) -> np.ndarray:
        """Inverse: undo color perm then undo D8 (for recovering canonical output)."""
        inv_perm = np.argsort(self.color_perm).astype(np.int64)
        g = apply_color_perm(grid, inv_perm)
        return d8_transform_inverse(g, self.d8_idx)


def identity_aug() -> AugParams:
    return AugParams(d8_idx=0, color_perm=identity_color_perm(), example_order=None)


def random_aug(
    rng: np.random.Generator,
    d8_idx: Optional[int] = None,
    permute_colors: bool = True,
    n_train: Optional[int] = None,
    permute_examples: bool = False,
) -> AugParams:
    """
    Sample a random augmentation.

    Args:
        rng:               NumPy random generator.
        d8_idx:            Fix the D8 index (None = random 0-7).
        permute_colors:    Whether to include a random color permutation.
        n_train:           Number of training examples (needed if permute_examples=True).
        permute_examples:  Whether to permute the order of training examples.
    """
    d = int(rng.integers(0, 8)) if d8_idx is None else d8_idx
    cp = random_color_perm(rng) if permute_colors else identity_color_perm()
    order = list(rng.permutation(n_train)) if permute_examples and n_train else None
    return AugParams(d8_idx=d, color_perm=cp, example_order=order)


def all_d8_augs(permute_colors: bool = False) -> list[AugParams]:
    """Return all 8 pure D8 augmentations (optionally with identity color perm)."""
    return [AugParams(d8_idx=i) for i in range(8)]


def sample_poe_augs(
    n: int,
    rng: np.random.Generator,
    n_train: int,
    permute_examples: bool = True,
) -> list[AugParams]:
    """
    Sample n augmentations for PoE scoring.

    Guarantees coverage of all 8 D8 transforms; any remaining are random.
    """
    augs = []
    # First 8: one per D8 index
    for d8 in range(min(8, n)):
        augs.append(random_aug(rng, d8_idx=d8, permute_colors=True,
                               n_train=n_train, permute_examples=permute_examples))
    # Fill remainder with random augs
    for _ in range(n - 8):
        augs.append(random_aug(rng, permute_colors=True,
                               n_train=n_train, permute_examples=permute_examples))
    return augs


# ---------------------------------------------------------------------------
# Grid-level helpers
# ---------------------------------------------------------------------------

def grid_palette(grid: np.ndarray) -> set[int]:
    """Return the set of color values (0-9) that appear in a grid."""
    return set(int(v) for v in np.unique(grid) if 0 <= v <= 9)


def task_palette(train_pairs: list[tuple[np.ndarray, np.ndarray]],
                 test_input: np.ndarray) -> set[int]:
    """Return the union of all colors appearing in the task's train/test inputs."""
    colors: set[int] = set()
    for inp, out in train_pairs:
        colors |= grid_palette(inp)
        colors |= grid_palette(out)
    colors |= grid_palette(test_input)
    return colors


def grids_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape != b.shape:
        return False
    return bool(np.all(a == b))


def hash_grid(grid: np.ndarray) -> bytes:
    return grid.astype(np.int8).tobytes()
