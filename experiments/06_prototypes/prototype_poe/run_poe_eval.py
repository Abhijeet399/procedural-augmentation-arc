#!/usr/bin/env python3
"""
run_poe_eval.py — Phase A: candidate generation + Product-of-Experts scoring.

Decoder modes:
  rcos    EXP C beam+sample via rcos.generate_all_candidates (DEFAULT)
  dfs     DFS threshold-filtered generation
  sample  Simple greedy + temperature sampling fallback

Usage (from repo root):
    python experiments/06_prototypes/prototype_poe/run_poe_eval.py \\
        --checkpoint  runs/tiny.pt \\
        --data-path   assets/challenges.json \\
        --solutions   assets/solutions.json \\
        --output-dir  runs/poe_rcos \\
        --decoder rcos --beam-width 14 --n-sample 30 --n-aug-score 16
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import importlib.util as ilu
from pathlib import Path

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Bootstrap: paths
# ---------------------------------------------------------------------------
THIS_DIR   = Path(__file__).resolve().parent
MDLARC_ROOT = str(THIS_DIR.parent.parent.parent)   # repo root
sys.path.insert(0, str(THIS_DIR))
for _p in [MDLARC_ROOT, f'{MDLARC_ROOT}/src']:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from augmentations import task_palette, grids_equal
from seq_builder import validate_format
from poe_scorer import score_candidates_poe, select_top2
from dfs_decoder import generate_candidates_dfs, generate_candidates_sample

# ---------------------------------------------------------------------------
# Lazy rcos loader — MUST register in sys.modules BEFORE exec_module
# so that @dataclass can resolve the module reference.
# ---------------------------------------------------------------------------
_rcos_mod = None

def _get_rcos():
    global _rcos_mod
    if _rcos_mod is None:
        spec = ilu.spec_from_file_location('rcos', f'{MDLARC_ROOT}/src/rcos.py')
        mod  = ilu.module_from_spec(spec)
        sys.modules['rcos'] = mod   # register first
        spec.loader.exec_module(mod)
        _rcos_mod = mod
    return _rcos_mod

# ---------------------------------------------------------------------------
# Lazy mdlARC infrastructure loader
# ---------------------------------------------------------------------------
def _load_mod(name, rel_path):
    spec = ilu.spec_from_file_location(name, f'{MDLARC_ROOT}/{rel_path}')
    mod  = ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def load_infrastructure():
    build = _load_mod('mdlarc_build', 'src/build.py')
    return build.load_checkpoint, build.build_model_and_data

# ---------------------------------------------------------------------------
# Utility: ensure everything is np.ndarray int64
# ---------------------------------------------------------------------------
def _to_array(c) -> np.ndarray:
    if isinstance(c, np.ndarray):
        return c.astype(np.int64)
    return np.array(c, dtype=np.int64)

# ---------------------------------------------------------------------------
# Candidate filter — shape + palette
# ---------------------------------------------------------------------------
def filter_candidates(candidates, expected_shape, allowed_colors):
    """Filter by shape and palette. Returns (survivors, n_shape_rej, n_palette_rej).
    All survivors are np.ndarray int64."""
    shape_rej = palette_rej = 0
    survivors = []
    for raw_c in candidates:
        c = _to_array(raw_c)          # safe regardless of input type
        if c.shape != expected_shape:
            shape_rej += 1
            continue
        colors = set(int(v) for v in np.unique(c) if 0 <= int(v) <= 9)
        if not colors.issubset(allowed_colors):
            palette_rej += 1
            continue
        survivors.append(c)
    return survivors, shape_rej, palette_rej

# ---------------------------------------------------------------------------
# Oracle / correctness helpers
# ---------------------------------------------------------------------------
def check_oracle(candidates, solution):
    sol = _to_array(solution)
    return any(grids_equal(_to_array(c), sol) for c in candidates)

def check_a2(attempt1, attempt2, solution):
    sol = _to_array(solution)
    a1  = attempt1 is not None and grids_equal(_to_array(attempt1), sol)
    a2  = attempt2 is not None and grids_equal(_to_array(attempt2), sol)
    return a1, a2

# ---------------------------------------------------------------------------
# Submission builder
# ---------------------------------------------------------------------------
def build_submission_entry(task_id, test_pair_idx, attempt1, attempt2):
    def to_list(g):
        return _to_array(g).tolist() if g is not None else [[0]]
    return {
        'task_id':       task_id,
        'test_pair_idx': test_pair_idx,
        'attempt_1':     to_list(attempt1),
        'attempt_2':     to_list(attempt2),
    }

# ---------------------------------------------------------------------------
# RCOS candidate generation — EXP C pipeline
# ---------------------------------------------------------------------------
def generate_candidates_rcos(model, dataset, task_id, pair_idx,
                              task_example_id, device,
                              beam_width, n_sample, temps, top_k, verbose):
    """
    EXP C pipeline: orientation selection → beam+sample generation.
    device MUST be torch.device (not a string).
    Returns list of np.ndarray int64 in the original (non-canonical) orientation.
    """
    assert isinstance(device, torch.device), f"device must be torch.device, got {type(device)}"

    rcos = _get_rcos()
    from canonicalize import score_task_orientations
    from common import split_grids_from_tokens
    from common import apply_inverse_dihedral_transform as _inv_d

    demo_exs = [ex for ex in dataset.iter_examples(split='train')
                if ex.task_id == task_id]
    test_exs  = [ex for ex in dataset.iter_examples(split='test')
                 if ex.task_id == task_id]
    test_ex   = next((e for e in test_exs if e.pair_index == pair_idx), None)

    if not demo_exs or test_ex is None:
        return []

    # orientation selection — needs torch.device
    best_d, _ = score_task_orientations(model, demo_exs, device)

    def _gp(ex, d):
        toks = (ex.tokens_by_dihedral[d]
                if ex.tokens_by_dihedral else ex.tokens).tolist()
        gs = split_grids_from_tokens(toks)
        return rcos.GridPair(
            input  = gs[0] if gs else [],
            output = gs[1] if len(gs) > 1 else None,
        )

    demo_pairs = [_gp(e, best_d) for e in demo_exs]
    demo_pairs = [gp for gp in demo_pairs if gp.output is not None]
    test_gp    = _gp(test_ex, best_d)

    # generate candidates in canonical (best_d) space
    raw_cands = rcos.generate_all_candidates(
        model, demo_pairs, test_gp.input,
        task_example_id, best_d, device,    # torch.device required
        use_greedy           = True,
        use_beam             = (beam_width > 0),
        beam_width           = beam_width,
        use_sample           = (n_sample > 0),
        n_per_temperature    = n_sample,
        temperatures         = tuple(temps),
        top_k                = top_k,
        test_seq_ex          = test_ex,
        demo_seq_exs         = demo_exs,
    )

    # invert back to original orientation; _inv_d returns list-of-lists → ndarray
    candidates = []
    for c in raw_cands:
        if c is None:
            continue
        inv = _inv_d(c, best_d)     # returns list-of-lists
        if not inv or len(inv) == 0:
            continue
        candidates.append(_to_array(inv))

    if verbose:
        print(f'  [rcos] d{best_d}, {len(candidates)} candidates')

    return candidates   # list[np.ndarray int64]

# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------
def evaluate(args):
    # torch.device from the very start — never a plain string inside evaluate()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    if args.validate_format:
        print('\n=== Running format validation ===')
        validate_format(checkpoint_path=args.checkpoint,
                        data_path=args.data_path, mdlarc_root=MDLARC_ROOT)
        return

    print('Loading checkpoint and building model...')
    load_checkpoint, build_model_and_data = load_infrastructure()
    ckpt = load_checkpoint(Path(args.checkpoint))

    import types
    build_args = types.SimpleNamespace(
        data_path=args.data_path, batch_size=1, num_workers=0, seed=42)
    _ret    = build_model_and_data(build_args, ckpt, is_eval=True)
    model   = _ret[0]
    dataset = _ret[1] if len(_ret) > 1 else None
    model.eval()
    model.to(device)
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params')

    # poe_scorer expects a plain device string ('cuda' / 'cpu')
    device_str = device.type

    with open(args.data_path) as f:
        challenges: dict = json.load(f)
    with open(args.solutions) as f:
        solutions: dict  = json.load(f)
    print(f'Loaded solutions for {len(solutions)} tasks.')

    task_ids = ckpt['task_ids']
    eval_task_ids = [
        tid for tid in task_ids
        if tid in challenges and 'test' in challenges[tid] and tid in solutions
    ]
    if args.max_tasks:
        eval_task_ids = eval_task_ids[:args.max_tasks]
    print(f'Evaluating {len(eval_task_ids)} tasks.')

    exp_name = (f'{args.decoder.upper()}_PoE'
                f'_beam{args.beam_width}_sample{args.n_sample}'
                f'_naug{args.n_aug_score}')
    out_dir = Path(args.output_dir) / exp_name
    out_dir.mkdir(parents=True, exist_ok=True)

    n_oracle = n_a2 = n_total_pairs = 0
    submission_entries = []
    t0_eval = time.time()
    rng = np.random.default_rng(args.seed)

    bar = '=' * 70
    print(f'\n{bar}')
    print(f'Phase A: {args.decoder}+PoE  beam={args.beam_width}'
          f'  n_sample={args.n_sample}  n_aug_score={args.n_aug_score}')
    print(f'  Reference: EXP C = 30.50% ARC, 37.5% oracle')
    print(bar)

    for task_num, task_id in enumerate(eval_task_ids):
        t_task = time.time()
        task_data       = challenges[task_id]
        train_pairs     = [(_to_array(p['input']), _to_array(p['output']))
                           for p in task_data['train']]
        task_solutions  = solutions[task_id]
        task_example_id = int(dataset.task_id_to_example_id.get(task_id, 0))
        print(f'\n[{task_num+1}/{len(eval_task_ids)}] {task_id}')

        for pair_idx, test_pair in enumerate(task_data['test']):
            test_input     = _to_array(test_pair['input'])
            solution       = _to_array(task_solutions[pair_idx])
            expected_shape = solution.shape
            allowed_colors = task_palette(train_pairs, test_input)

            # ---- Generation ----
            t_gen = time.time()

            if args.decoder == 'rcos':
                raw_candidates = generate_candidates_rcos(
                    model=model, dataset=dataset,
                    task_id=task_id, pair_idx=pair_idx,
                    task_example_id=task_example_id,
                    device=device,              # torch.device
                    beam_width=args.beam_width,
                    n_sample=args.n_sample,
                    temps=args.temps,
                    top_k=args.top_k,
                    verbose=args.verbose,
                )

            elif args.decoder == 'dfs':
                raw_candidates = generate_candidates_dfs(
                    model=model, train_pairs=train_pairs,
                    test_input=test_input,
                    task_example_id=task_example_id,
                    output_shape=expected_shape,
                    n_aug_generate=args.n_aug_generate,
                    threshold=args.threshold,
                    max_live=args.max_live,
                    seed=int(rng.integers(0, 2**32)),
                    device=device_str,          # dfs_decoder wants str
                    verbose=args.verbose,
                )
                if not raw_candidates:
                    raw_candidates = generate_candidates_sample(
                        model=model, train_pairs=train_pairs,
                        test_input=test_input,
                        task_example_id=task_example_id,
                        output_shape=expected_shape,
                        n_samples=30, temperatures=(0.7, 0.8, 1.0, 1.2),
                        seed=int(rng.integers(0, 2**32)), device=device_str,
                    )

            else:  # 'sample'
                raw_candidates = generate_candidates_sample(
                    model=model, train_pairs=train_pairs,
                    test_input=test_input,
                    task_example_id=task_example_id,
                    output_shape=expected_shape,
                    n_samples=30, temperatures=(0.7, 0.8, 1.0, 1.2),
                    seed=int(rng.integers(0, 2**32)), device=device_str,
                )

            gen_time = time.time() - t_gen

            # ---- Filter ----
            survivors, shape_rej, palette_rej = filter_candidates(
                raw_candidates, expected_shape, allowed_colors)

            fallback_note = ''
            if not survivors and raw_candidates:
                survivors = [_to_array(c) for c in raw_candidates]
                fallback_note = '  fallback=all_filtered'

            # ---- Oracle ----
            oracle_hit = check_oracle(survivors, solution)
            n_oracle  += int(oracle_hit)

            # ---- PoE scoring ----
            MAX_POE = 64
            if len(survivors) > MAX_POE:
                import random as _rnd
                survivors = _rnd.Random(int(rng.integers(0, 2**32))).sample(survivors, MAX_POE)
                if args.verbose:
                    print(f'  [subsample] {MAX_POE} candidates for PoE')

            t_score = time.time()
            try:
                scored = score_candidates_poe(
                    model=model, train_pairs=train_pairs,
                    test_input=test_input, candidates=survivors,
                    task_example_id=task_example_id,
                    n_aug=args.n_aug_score,
                    seed=int(rng.integers(0, 2**32)),
                    device=device_str,          # poe_scorer wants plain string
                    verbose=args.verbose,
                )
            except Exception as e:
                print(f'  [PoE] failed: {e}. Using first-best fallback.')
                scored = [(_to_array(c), 0.0) for c in survivors]
            score_time = time.time() - t_score

            attempt1, attempt2 = select_top2(scored)
            a1_ok, a2_ok = check_a2(attempt1, attempt2, solution)
            n_a2          += int(a1_ok or a2_ok)
            n_total_pairs += 1

            print(
                f"  [pair {pair_idx}]  oracle={'✓' if oracle_hit else '✗'}  "
                f"e_rank={'✓' if a1_ok else '✗'}  "
                f"shape_rej={shape_rej}  palette_rej={palette_rej}  "
                f"survivors={len(survivors)}{fallback_note}  "
                f"(gen={gen_time:.1f}s score={score_time:.1f}s)"
            )
            print(f"  [pair {pair_idx}] a1={'✓' if a1_ok else '✗'}  "
                  f"a2={'✓' if a2_ok else '✗'}")

            submission_entries.append(
                build_submission_entry(task_id, pair_idx, attempt1, attempt2))

        print(f'  task done in {time.time() - t_task:.1f}s')

    # ---- Final metrics ----
    elapsed    = time.time() - t0_eval
    arc_pct    = n_a2    / len(eval_task_ids) * 100 if eval_task_ids  else 0.0
    oracle_pct = n_oracle / n_total_pairs      * 100 if n_total_pairs else 0.0
    a2_pct     = n_a2    / n_total_pairs       * 100 if n_total_pairs else 0.0

    print(f'\n  Oracle : {n_oracle}/{n_total_pairs} = {oracle_pct:.1f}%  (EXP C: 37.5%)')
    print(f'  a2 hits: {n_a2}/{n_total_pairs} = {a2_pct:.1f}%')
    print(f'  ARC    : {n_a2}/{len(eval_task_ids)} = {arc_pct:.2f}%  (EXP C: 30.50%)')
    print(f'  Time   : {elapsed:.0f}s')

    sub_path = out_dir / 'submission.json'
    with open(sub_path, 'w') as f:
        json.dump(submission_entries, f)
    print(f'\nSubmission → {sub_path}')

    comp_path = out_dir / 'comparison.json'
    with open(comp_path, 'w') as f:
        json.dump({
            'decoder':      args.decoder,
            'beam_width':   args.beam_width,
            'n_sample':     args.n_sample,
            'n_aug_score':  args.n_aug_score,
            'oracle_pct':   oracle_pct,
            'arc_pct':      arc_pct,
            'a2_pct':       a2_pct,
            'n_oracle':     n_oracle,
            'n_a2':         n_a2,
            'n_total_pairs': n_total_pairs,
            'n_tasks':      len(eval_task_ids),
            'time_seconds': elapsed,
            'reference_expc': {'oracle_pct': 37.5, 'arc_pct': 30.50},
        }, f, indent=2)
    print(f'Comparison → {comp_path}')

    print(f'\n{bar}')
    print(f'RESULT: {arc_pct:.2f}% ARC  |  {oracle_pct:.1f}% oracle')
    print(f'Delta vs EXP C: {arc_pct - 30.50:+.2f}pp')
    print(bar)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='Phase A: RCOS/DFS+PoE evaluation')
    p.add_argument('--checkpoint',      required=True)
    p.add_argument('--data-path',       required=True)
    p.add_argument('--solutions',       required=True)
    p.add_argument('--output-dir',      default='runs/poe_eval')
    p.add_argument('--decoder',         choices=['rcos', 'dfs', 'sample'], default='rcos')
    # rcos args
    p.add_argument('--beam-width',      type=int,   default=14)
    p.add_argument('--n-sample',        type=int,   default=30)
    p.add_argument('--temps',           type=float, nargs='+', default=[0.7, 0.8, 1.0, 1.2])
    p.add_argument('--top-k',           type=int,   default=None)
    # dfs args
    p.add_argument('--threshold',       type=float, default=0.09)
    p.add_argument('--max-live',        type=int,   default=64)
    p.add_argument('--n-aug-generate',  type=int,   default=8)
    # shared
    p.add_argument('--n-aug-score',     type=int,   default=16)
    p.add_argument('--max-tasks',       type=int,   default=None)
    p.add_argument('--seed',            type=int,   default=42)
    p.add_argument('--verbose',         action='store_true')
    p.add_argument('--validate-format', action='store_true')
    return p.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate(args)
