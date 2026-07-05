# Leak audit summary

Full methodology and reproduction script: `experiments/05_leak_audit/`.

Task `070dd51e` is present in both `assets/challenges.json` (1307-task training set) and
`assets/solutions.json` (400-task ARC-1 public eval set) — a genuine train/eval collision
in the underlying ARC-1 asset files. Every checkpoint trained on re-ARC-augmented data
solves it; the baseline (no re-ARC) checkpoint does not.

| Run | Raw score | `070dd51e` solved? | Leak-adjusted |
|---|---|---|---|
| 01 baseline (no re-ARC), epoch 650 | 37.625% | No | 37.625% |
| 02 re-ARC swap, epoch 650 | 59.375% | Yes | 59.125% |
| 03 cap=300 ablation, epoch 650 | 56.75% | Yes | 56.5% |
| 04 schedule extension, epoch 750 | 60.625% | Yes | 60.375% |
| 04 schedule extension, epoch 1000 (best) | 61.5% | Yes | **61.25%** |

The baseline not solving it, while every re-ARC checkpoint does, indicates the re-ARC
augmentation pipeline amplifies the leak's effect (more synthetic exposure to a training
task's rule makes its eval instance easier to solve by familiarity) rather than merely
inheriting a pre-existing issue. All headline numbers in the top-level README use the
leak-adjusted score.
