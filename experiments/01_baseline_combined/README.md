# 01 — Baseline (combined ARC-1 + ConceptARC, no re-ARC)

Trains on `assets/combined_challenges.json` (original ARC-1 training tasks + ConceptARC,
no synthetic re-ARC data). This is the pre-procedural-augmentation baseline that every
later experiment is measured against.

**Result: 37.6%** on the ARC-1 public eval set (`scratch_combined_v2.epoch650`, 150.5/400).
See `../../results/score_table.md` for the full per-checkpoint sweep.

```bash
python experiments/01_baseline_combined/run_baseline_combined.py high
```
