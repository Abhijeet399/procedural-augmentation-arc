# 05 — Data-leakage audit

The training set (`assets/challenges.json`, 1307 ARC-1 training tasks) and the evaluation
set (`assets/solutions.json`, 400 ARC-1 public eval tasks) are supposed to be disjoint.
Running `leak_check.py` against them found one collision:

```
$ python experiments/05_leak_audit/leak_check.py \
    --train-challenges assets/challenges.json \
    --eval-solutions assets/solutions.json

Training tasks: 1307
Eval tasks:      400
Collisions:      1
  LEAK: 070dd51e appears in both the training set and the eval set
```

Task `070dd51e` is present in both files. This is a genuine collision in the underlying
ARC-1 asset files used by this project (not something introduced by the re-ARC
augmentation pipeline — `dataset_building_scripts/prepare_dataset_rearc.py` in fact has
an explicit leakage guard that refuses to augment any task ID present in
`solutions.json`, as defense-in-depth against exactly this class of issue, though it
can't remove a collision that already exists in the base training file).

## Correction

Any checkpoint that solves `070dd51e` is getting credit for a task it had direct training
exposure to, which is not a genuine generalization result. The correction is to subtract
credit for that one task from the raw eval score before reporting a percentage. Re-scoring
each run's submission directly against `assets/solutions.json`:

| Run | Raw score | `070dd51e` solved? | Leak-adjusted |
|---|---|---|---|
| 01 baseline (no re-ARC), epoch 650 | 37.625% | No | 37.625% (no change) |
| 02 re-ARC swap, epoch 650 | 59.375% | Yes | 59.125% |
| 03 cap=300 ablation, epoch 650 | 56.75% | Yes | 56.5% |
| 04 schedule extension, epoch 750 | 60.625% | Yes | 60.375% |
| 04 schedule extension, epoch 1000 (best) | 61.5% | Yes | **61.25%** |

Notably, the **baseline (no re-ARC) checkpoint does not solve `070dd51e`** — only
re-ARC-trained checkpoints do. This suggests the re-ARC augmentation pipeline, by
generating many synthetic pairs for every training task ID (including the leaked one),
amplifies the leak's effect rather than merely inheriting it: more synthetic exposure to
a training task's rule makes that task's eval instance easier to solve by familiarity,
not generalization.

All results reported in this repository's top-level README and `results/score_table.md`
use the leak-adjusted number for any checkpoint where `070dd51e` was solved.
