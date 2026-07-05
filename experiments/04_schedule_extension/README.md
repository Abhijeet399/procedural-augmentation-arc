# 04 — Schedule extension (650 → 750 → 1000 epochs)

Keeps the run 02 re-ARC dataset (uncapped, 12 pairs/task) and extends training beyond the
650-epoch schedule to see whether the WSD decay schedule was cutting training short.

| Epochs | Score (raw) | Checkpoint |
|---|---|---|
| 650 (run 02) | 59.4% | `BEST_rearc_v1_epoch650_5937.pt` |
| 750 | 60.6% | `BEST_rearc_epochs750_epoch750_6062.pt` |
| 1000 | 61.5% | `BEST_rearc_epochs1000_epoch1000_6150.pt` |

The 1000-epoch checkpoint is the best-performing model in this project. Its raw score
includes credit for task `070dd51e`, which is a confirmed train/eval collision (see
`../05_leak_audit/`) — the **leak-adjusted score is 61.25%** (245/400).

```bash
python experiments/04_schedule_extension/run_epochs750.py high
python experiments/04_schedule_extension/run_epochs1000.py high
```
