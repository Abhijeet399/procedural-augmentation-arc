# Reproduce

Exact commands for every row in the results table (`README.md`, `results/score_table.md`).
Run everything from the repo root after `pip install -r requirements.txt`.

## 0. Get the base data

```bash
cd dataset_building_scripts
python download_and_group.py
python build_datasets.py arc1 --add-conceptarc --with-filtered
cd ..
```

This produces `assets/challenges.json` (1307 training tasks), `assets/solutions.json`
(400 eval tasks), and `assets/combined_challenges.json` (ARC-1 + ConceptARC).

You'll also need re-ARC's generators + verifiers at `assets/re_arc/` — see
[re-ARC](https://github.com/michaelhodel/re-arc) for the source, or place your own
generated `re_arc/tasks/*.json` there (one file per ARC-1 training task ID).

## 1. Baseline (37.6%)

```bash
python experiments/01_baseline_combined/run_baseline_combined.py high
```

## 2. re-ARC swap (59.4%)

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

## 3. cap=300 ablation (56.75%, negative result)

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

## 4. Schedule extension (60.6% → 61.5% raw, 61.25% leak-adjusted)

Uses the same `assets/combined_rearc_challenges.json` from step 2.

```bash
python experiments/04_schedule_extension/run_epochs750.py high
python experiments/04_schedule_extension/run_epochs1000.py high
```

## 5. Leak audit

```bash
python experiments/05_leak_audit/leak_check.py \
    --train-challenges assets/challenges.json \
    --eval-solutions assets/solutions.json
```

To compute a leak-adjusted percentage for any checkpoint, subtract 1 point (one task) from
its raw score out of 400 if `070dd51e` appears in its solved-task list, then recompute the
percentage.

## 6. Prototypes

Each prototype has its own usage instructions:

- `experiments/06_prototypes/run_prototype_a.py` — orientation-voting evaluation
- `experiments/06_prototypes/run_prototype_c.py`, `run_prototype_c_v9.py` — beam+sample
  candidate generation with transition ranking
- `experiments/06_prototypes/prototype_d/` — program synthesis + neural fallback
- `experiments/06_prototypes/prototype_e/` — transition-ranking ablations
- `experiments/06_prototypes/prototype_poe/` — DFS decoder + Product-of-Experts rescoring
  (see its own `README.md`)
- `experiments/06_prototypes/prototype_ttt/` — LoRA test-time training
- `experiments/06_prototypes/prototype_more_data/` — NVARC synthetic data fine-tuning
- `experiments/06_prototypes/train_meta_encoder.py`, `collect_embeddings.py` — meta-encoder
  training for augmentation-invariant task embeddings
- `experiments/06_prototypes/merge_checkpoints.py`, `merge_ensemble.py` — checkpoint/ensemble
  merging

All of the above expect to be run from the repo root so their internal path resolution
finds `src/` and `assets/` correctly.
