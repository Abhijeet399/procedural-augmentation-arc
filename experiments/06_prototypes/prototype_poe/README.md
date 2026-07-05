# prototype_poe — Phase A: DFS Decoder + Product-of-Experts Scoring

## What this does

Replaces the `beam=14 / n_sample=30 / λ=0.3 transition ranking` pipeline in
EXP C with:

1. **DFS candidate generation** — threshold-filtered depth-first search over
   output tokens (each token branch kept only if P ≥ threshold ≈ 9%).  
   Generates more diverse candidates than beam search.

2. **Product-of-Experts (PoE) scoring** — each candidate is scored under 16
   augmentations (D8 × random color permutations × example orderings).  
   Score = Σ log P_model(aug(candidate) | aug(prompt)).  
   Directly closes the 35-task ranking gap that EXP C leaves on the table.

**Expected gain vs EXP C baseline (30.50%): +5 to +8 points** (targeting 35–39%).

---

## Files

```
prototype_poe/
├── run_poe_eval.py      ← main script — run this
├── augmentations.py     ← D8 transforms + color permutations
├── seq_builder.py       ← converts grids → model input tensors
├── poe_scorer.py        ← PoE scoring across augmentations
├── dfs_decoder.py       ← threshold-filtered DFS generation
└── README.md            ← this file
```

---

## Quick start

```bash
conda activate ARC
cd procedural-augmentation-arc   # repo root

# STEP 1 — Validate format assumptions (do this FIRST, once)
python experiments/06_prototypes/prototype_poe/run_poe_eval.py --validate-format \
    --checkpoint runs/tiny.pt \
    --data-path  assets/challenges.json \
    --solutions  assets/solutions.json

# STEP 2 — Quick test on 10 tasks to verify everything works
python experiments/06_prototypes/prototype_poe/run_poe_eval.py \
    --checkpoint  runs/tiny.pt \
    --data-path   assets/challenges.json \
    --solutions   assets/solutions.json \
    --output-dir  runs/poe_eval \
    --decoder     dfs \
    --threshold   0.09 \
    --max-live    64 \
    --n-aug-score 16 \
    --max-tasks   10

# STEP 3 — Full eval (all 400 tasks)
python experiments/06_prototypes/prototype_poe/run_poe_eval.py \
    --checkpoint  runs/tiny.pt \
    --data-path   assets/challenges.json \
    --solutions   assets/solutions.json \
    --output-dir  runs/poe_eval \
    --decoder     dfs \
    --threshold   0.09 \
    --max-live    64 \
    --n-aug-score 16
```

---

## Key parameters

| Param | Default | Meaning |
|---|---|---|
| `--decoder` | `dfs` | `dfs` = threshold DFS; `sample` = greedy+temp (EXP C style) |
| `--threshold` | `0.09` | DFS branch threshold. Start at 0.09; try 0.05 if oracle is low |
| `--max-live` | `64` | Max active DFS branches. Higher = more candidates, slower |
| `--n-aug-generate` | `8` | Augmentations used during DFS generation (8 = full D8) |
| `--n-aug-score` | `16` | Augmentations used for PoE scoring. 16 = recommended minimum |

---

## What to look for in output

Each task prints:

```
[12/400] 1a2b3c4d
  [pair 0] oracle=✓  e_rank=✓  shape_rej=3  palette_rej=0  survivors=28  (gen=2.1s score=4.3s)
  [pair 0] a1=✓  a2=✗
  task done in 6.5s
```

Key metrics printed at the end:

```
Oracle : 162/419 = 38.7%   (EXP C ref: 157/419 = 37.5%)
a2 hits: 148/419 = 35.3%
ARC    : 148/400 = 37.00%  (EXP C ref: 30.50%)
Delta vs EXP C: +6.50pp
```

**If oracle barely moves** (≤ 38%): the DFS threshold may be too high. Try
`--threshold 0.05` or increase `--max-live` to 128.

**If ARC% improves but oracle is the same**: PoE scoring is working (ranking
gap closed). 

**If oracle improves but ARC% doesn't**: PoE scoring needs tuning. Increase
`--n-aug-score` to 32.

---

## Troubleshooting

### "KeyError: positions_3d" or wrong tensor shapes

Run `--validate-format` and compare the batch keys from the dataloader to what
`seq_builder.py` produces. The most likely issue: `positions_3d` uses a
different plane encoding (e.g., 0/1/2 instead of 1/2/3), or the separator
token ID is not 10.

**Fix:** Edit `seq_builder.py` constants:
```python
SEP_TOKEN    = 10   # ← change if needed
PLANE_INPUT  = 1    # ← change if needed  
PLANE_SEP    = 2    # ← change if needed
PLANE_OUTPUT = 3    # ← change if needed
```

### "RuntimeError: CUDA out of memory"

Reduce `--max-live` from 64 → 32, and/or `--n-aug-score` from 16 → 8.

### All candidates are shape-rejected

The `_infer_output_shape` function might be returning the wrong shape. Check
if it matches the solution shape for a few tasks. You can hard-code the
expected shape for debugging:

```python
# In run_poe_eval.py, replace:
raw_candidates = generate_candidates_dfs(..., output_shape=expected_shape, ...)
# With explicit solution shape:
raw_candidates = generate_candidates_dfs(..., output_shape=solution.shape, ...)
```

### Very slow (> 60s per task)

DFS with max_live=64 on large output grids (e.g., 10×10 = 100 tokens) can
be slow. Options:
- Reduce `--max-live` to 32
- Reduce `--n-aug-generate` to 4
- For long-sequence tasks, fall back to `--decoder sample`

### PoE scoring returns all equal scores

The log-prob computation may be getting logits from the wrong positions.
The key assumption: `logits[t]` predicts `token[t+1]` (autoregressive shift).
If the model is not causal or uses a different indexing, the scores will be
meaningless. Check this with:

```python
# Quick sanity check — PoE score for the CORRECT answer should be highest
# among all candidates for at least 60% of tasks where oracle=✓
```

---

## Connecting to Phase C (multi-seed ensemble)

Once Phase A is confirmed working, Phase C trains 3 seeds with different
augmentation orderings and pools their DFS candidates:

```
seed_0_candidates + seed_1_candidates + seed_2_candidates
    ↓
48-way PoE scoring (3 models × 16 augmentations)
    ↓
submit top-2
```

Mithil's union-of-runs reaches 55%. Phase C is the biggest single
remaining win after Phase A.

---

## Format assumptions (document any changes here)

These are the assumptions made in `seq_builder.py`. If validation reveals they
are wrong, update this table and the constants in `seq_builder.py`.

| Assumption | Value | Verified? |
|---|---|---|
| Grid token range | 0–9 | ☐ |
| Separator token ID | 10 | ☐ |
| Input plane | 1 | ☐ |
| Separator plane | 2 | ☐ |
| Output plane | 3 | ☐ |
| example_ids encoding | global pair offset | ☐ |
| dihedral_ids range | 0–7 | ☐ |
| Autoregressive shift | logits[t] → token[t+1] | ☐ |

Run `--validate-format` and check the ☐ boxes above.
