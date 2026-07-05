# Score table

All scores are on the ARC-1 public evaluation set (400 tasks, 2 attempts/task, partial
credit per test pair). Source: `runs/*_sweep_log.txt` in the working training directory,
re-verified directly against `assets/solutions.json` for the checkpoints in the summary
table. Leak-adjusted numbers subtract credit for task `070dd51e`, a confirmed train/eval
collision — see `leak_audit.md`.

## Summary

| Experiment | Checkpoint | Raw score | `070dd51e` solved? | Leak-adjusted | Solved-task list |
|---|---|---|---|---|---|
| 01 baseline (combined, no re-ARC) | epoch 650 | 150.5/400 = 37.625% | No | 37.625% | `fully_solved/01_baseline_combined_epoch650.json` (149 tasks) |
| 02 re-ARC swap | epoch 650 | 237.5/400 = 59.375% | Yes | 59.125% | `fully_solved/02_rearc_swap_epoch650.json` (236 tasks) |
| 03 cap=300 ablation (regression) | epoch 650 | 227.0/400 = 56.75% | Yes | 56.5% | `fully_solved/03_cap300_ablation_epoch650.json` (225 tasks) |
| 04 schedule extension | epoch 750 | 242.5/400 = 60.625% | Yes | 60.375% | `fully_solved/04_schedule_extension_epoch750.json` (240 tasks) |
| 04 schedule extension (best) | epoch 1000 | 246.0/400 = 61.5% | Yes | **61.25%** | `fully_solved/04_schedule_extension_epoch1000_BEST.json` (243 tasks) |

## Full per-checkpoint sweeps

### 01 — Baseline (combined ARC-1 + ConceptARC, no re-ARC)

Two independent training runs (v2, v3) swept over the tail of the WSD decay schedule:

| Run | epoch560/680 | 590/710 | 610/725 | 625/735 | 635/740 | 645/745 | 650/750 |
|---|---|---|---|---|---|---|---|
| v2 | 30.0% | 29.75% | 30.5% | 36.125% | 35.875% | 37.0% | 37.625% |
| v3 | 31.0% | 30.0% | 35.875% | 37.125% | 37.625% | 37.375% | 37.5% |

Best: v2 epoch 650, 37.625% (rounded to 37.6% in the top-level README).

### 02 — re-ARC generator swap

| epoch | 560 | 590 | 610 | 625 | 635 | 645 | 650 |
|---|---|---|---|---|---|---|---|
| score | 50.25% | 54.625% | 56.625% | 57.375% | 57.125% | 58.125% | 59.375% |

Best: epoch 650, 59.375% (rounded to 59.4%).

### 03 — cap=300 ablation

| epoch | 560 | 590 | 610 | 625 | 635 | 645 | 650 |
|---|---|---|---|---|---|---|---|
| score | 42.625% | 43.875% | 45.125% | 49.75% | 52.0% | 55.0% | 56.75% |

Best: epoch 650, 56.75% — below the 59.375% uncapped result (see run 02). Negative result.

### 04 — Schedule extension (epochs 1000)

| epoch | 900 | 940 | 965 | 980 | 990 | 995 | 1000 |
|---|---|---|---|---|---|---|---|
| score | 52.25% | 56.125% | 59.25% | 59.625% | 59.625% | 61.0% | 61.5% |

Best: epoch 1000, 61.5% raw / 61.25% leak-adjusted.

The epoch-750 run in the same experiment used the schedule 680/710/725/735/740/745/750;
only the final checkpoint (epoch 750, 60.625%) was independently re-scored for this table.
