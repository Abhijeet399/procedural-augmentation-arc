# Checkpoints

Checkpoints are **not committed to this repository** (`.gitignore` excludes `*.pt`; each
is ~590 MB, model weights + optimizer state). This file tracks what exists and where each
one came from. See `REPRODUCE.md` for the exact commands to regenerate any of them, and
`results/score_table.md` for the scoring methodology.

## Best checkpoint

**`BEST_rearc_epochs1000_epoch1000_6150.pt`** — epoch 1000 of the schedule-extension run
(`experiments/04_schedule_extension`), trained on the re-ARC-augmented dataset
(`assets/combined_rearc_challenges.json`).

- Raw score: 246/400 = **61.5%**
- Leak-adjusted score (see `results/leak_audit.md`): 245/400 = **61.25%**
- This is the best-performing checkpoint in this project.

## All tracked checkpoints

| File | Run | Raw score | Leak-adjusted |
|---|---|---|---|
| `BEST_rearc_epochs1000_epoch1000_6150.pt` | 04 schedule extension, epoch 1000 | 61.5% | **61.25%** |
| `BEST_rearc_epochs750_epoch750_6062.pt` | 04 schedule extension, epoch 750 | 60.625% | 60.375% |
| `BEST_rearc_v1_epoch650_5937.pt` | 02 re-ARC swap, epoch 650 | 59.375% | 59.125% |
| `BEST_rearc_v1_epoch645_5812.pt` | 02 re-ARC swap, epoch 645 | 58.125% | n/a (not re-scored) |
| `BEST_rearc_cap300_epoch650_5675.pt` | 03 cap=300 ablation, epoch 650 | 56.75% | 56.5% |
| `BEST_v2_epoch650_37625.pt` | 01 baseline, v2, epoch 650 | 37.625% | 37.625% (no leak) |
| `BEST_v3_epoch740_37625.pt` | 01 baseline, v3, epoch 740 | 37.625% | 37.625% (no leak) |
| `BEST_epoch645_36875.pt` | earliest baseline sweep, epoch 645 | 36.875% | n/a (not re-scored) |

Filenames encode the score as `_<score×100, truncated>` (e.g. `6150` = 61.50%, `5675` =
56.75%) — this convention was cross-checked against the sweep logs and holds for every
entry above.
