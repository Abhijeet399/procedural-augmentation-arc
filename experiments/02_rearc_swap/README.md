# 02 — re-ARC generator swap

Same training setup as the baseline, but swaps the data source to
`assets/combined_rearc_challenges.json` — the original ARC-1 training tasks with
synthetic re-ARC generator pairs appended per task (see
`dataset_building_scripts/prepare_dataset_rearc.py`). This is the single biggest lever
found in this project.

**Result: 59.4%** on the ARC-1 public eval set (`scratch_rearc_v1.epoch650`, 237.5/400),
up from the 37.6% baseline. See `../../results/score_table.md` for the full sweep.

```bash
python dataset_building_scripts/prepare_dataset_rearc.py \
    --orig-challenges assets/challenges.json \
    --rearc-tasks-dir assets/re_arc/tasks \
    --eval-solutions assets/solutions.json \
    --max-rearc-pairs-per-task 12 \
    --seed 42 \
    --out-challenges assets/combined_rearc_challenges.json

python experiments/02_rearc_swap/run_rearc_swap.py high
```
