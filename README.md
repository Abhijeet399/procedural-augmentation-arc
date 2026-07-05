# procedural-augmentation-arc

Controlled, audited application of procedural data generation (re-ARC generators) to a
transductive ARC-AGI-1 solver — from a 37.6% baseline to 61.25% (leak-adjusted) on the
ARC-1 public evaluation set, plus test-time training, candidate rescoring, and
augmentation-invariant embedding experiments built on top of that base.

This repository documents the experiment sequence, the data-leakage audit that produced
the leak-adjusted numbers, and exact commands to reproduce every result below.

## Results

| Stage | Score (ARC-1 public eval) | Run |
|---|---|---|
| Baseline (ARC-1 + ConceptARC, no re-ARC) | 37.6% | `experiments/01_baseline_combined` |
| re-ARC generator swap | 59.4% | `experiments/02_rearc_swap` |
| cap=300 ablation (per-task pair cap raised to 300) | 56.75% — **regression, not adopted** | `experiments/03_cap300_ablation` |
| Schedule extension (650 → 750 → 1000 epochs) | 60.6% → 61.5% (raw) | `experiments/04_schedule_extension` |
| Schedule extension, leak-adjusted | **61.25%** | after correcting the `070dd51e` train/eval collision (see below) |

All numbers are re-verified directly against `assets/solutions.json`; see
`results/score_table.md` for the full per-checkpoint sweep and `results/fully_solved/` for
the exact solved-task lists behind each score.

### The cap=300 ablation is a negative result

Raising the re-ARC pairs-per-task cap from 12 to 300 (`assets/combined_rearc_cap300_challenges.json`,
~2.4x more synthetic data per task) was tested as a way to push past 59.4%. It underperformed
the uncapped re-ARC swap (56.75% vs 59.4%) — over-saturating a task with synthetic variations
of the same rule diluted the signal rather than reinforcing it. Documented here because it's
a useful negative finding, not because it improved the score.

### Data-leakage audit

Task `070dd51e` is present in both the 1307-task training set (`assets/challenges.json`) and
the 400-task evaluation set (`assets/solutions.json`) — a genuine train/eval collision in the
underlying ARC-1 asset files. Every re-ARC-trained checkpoint solves it; the baseline
(no re-ARC) checkpoint does not — indicating the re-ARC augmentation pipeline amplifies the
leak's effect rather than merely inheriting it. The epoch-1000 checkpoint's raw score
(246/400 = 61.5%) includes this task; the leak-adjusted score (245/400 = 61.25%) removes
credit for it. See `results/leak_audit.md` and `experiments/05_leak_audit/` for the audit
methodology and a reusable `leak_check.py`.

## What's in this repo

- **`src/`** — base training/architecture/evaluation pipeline (`build.py`, `common.py`,
  `evaluate.py`, `tinytransformer.py`, `train.py`, `utils.py`), plus original additions:
  `canonicalize.py` (orientation canonicalization), `rcos.py` (Rule-Consistency candidate
  Scoring), `lora_ttt.py` (LoRA test-time training), and `meta_encoder.py`
  (augmentation-invariant task embeddings). See `NOTICE.md` for which files are which.
- **`experiments/`** — the numbered experiment sequence behind the results table above
  (`01`–`05`), plus the prototype exploration in `06_prototypes/` (orientation voting, DFS+PoE
  candidate rescoring, TTT, meta-encoder, checkpoint ensembling).
- **`results/`** — score tables and fully-solved-task lists, re-verified against
  `assets/solutions.json`.
- **`models/`** — checkpoint tracker (`models/README.md`); `.pt` files are not committed.
- **`dataset_building_scripts/`** — dataset construction, including the re-ARC combination
  script (`prepare_dataset_rearc.py`).

## Setup

```bash
git clone https://github.com/Abhijeet399/procedural-augmentation-arc.git
cd procedural-augmentation-arc
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires CUDA ≥ 12.8 for `flash-attn`; a single 5090 or equivalent trains the largest run
(1000 epochs) in a few hours.

See `REPRODUCE.md` for the exact commands used for every row in the results table, including
dataset build steps and checkpoint locations.

## Checkpoints

Trained checkpoints are not committed to this repository (see `models/README.md` for the
best-checkpoint record). `.gitignore` excludes `*.pt`, the large `assets/*.json` dataset
files, `runs/`, and `__pycache__`.

## Background

This project builds on [mdlARC](https://github.com/mvakde/mdlARC) by Mithil Vakde, used here
under its MIT license, as the base transformer training/inference pipeline. The work in this
repository is the procedural-augmentation research program built on top of that base: the
re-ARC generator swap and cap ablation, the epoch-schedule extension, the data-leakage audit
and correction, and the candidate-rescoring / test-time-training / meta-encoder prototypes in
`experiments/06_prototypes/` and `src/`. See `NOTICE.md` for the file-level breakdown.

## Citation

See `CITATION.cff`. 

## License

MIT — see `LICENSE` (original copyright retained per license terms) and `NOTICE.md` for
the attribution details.
