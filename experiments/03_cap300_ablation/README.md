# 03 — cap=300 ablation (negative result)

Tests whether raising the re-ARC pairs-per-task cap from 12 to 300
(`assets/combined_rearc_cap300_challenges.json`, ~2.4x more synthetic data per task than
the run 02 dataset) improves on the 59.4% re-ARC swap result.

**Result: 56.75%** (`scratch_rearc_cap300.epoch650`, 227/400) — a **regression**, not an
improvement, versus the 59.4% uncapped re-ARC swap. Over-saturating a task with synthetic
variations of the same rule appears to dilute signal rather than reinforce it. This cap
was not adopted in later experiments.

```bash
python dataset_building_scripts/prepare_dataset_rearc.py \
    --orig-challenges assets/challenges.json \
    --rearc-tasks-dir assets/re_arc/tasks \
    --eval-solutions assets/solutions.json \
    --max-rearc-pairs-per-task 300 \
    --seed 42 \
    --out-challenges assets/combined_rearc_cap300_challenges.json

python experiments/03_cap300_ablation/run_cap300_ablation.py high
```
