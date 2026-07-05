"""
run_prototype_d.py — Prototype D: Program Synthesis + Neural Fallback
=======================================================================

Strategy:
    attempt_1 = synthesized rule applied to test_input  (if rule found)
               OR neural greedy decode                  (if not found)
    attempt_2 = neural greedy decode (always — safety net)

Key design decisions:
    - Rule search runs on RAW (unrotated) demo pairs.
      Our primitive library includes all 8 dihedral transforms, so orientation
      is handled by the rule itself. This avoids the garbage-orientation problem
      from CE≈0 (memorised embeddings).
    - Neural fallback uses run_canonical_inference_for_task from canonicalize.py
      which handles orientation, colour mapping, and inverse transforms correctly.

Folder layout (relative to the repo root):
    prototype_d/
        run_prototype_d.py          <- this file
        src/
            primitives.py
            rule_search.py
            llm_synthesizer.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Tuple

# ── project path setup ────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent   # experiments/06_prototypes/prototype_d/
_ROOT = _HERE.parent.parent.parent        # repo root
_SRC  = _ROOT / "src"
_DSRC = _HERE / "src"                     # prototype_d/src/

for _p in [str(_ROOT), str(_SRC), str(_DSRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from build import build_model_and_data, load_checkpoint
from canonicalize import (
    score_task_orientations,
    run_canonical_inference_for_task,
    build_color_canon_mapping,
    build_color_inverse_mapping,
)

from rule_search import find_rule, DemoPair
from primitives import grid_equal


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Prototype D: Program Synthesis + Neural Fallback"
    )
    p.add_argument("--checkpoint",       required=True)
    p.add_argument("--data-path",        required=True)
    p.add_argument("--solutions",        default=None)
    p.add_argument("--output-dir",       default="runs/prototype_d_v1")
    p.add_argument("--device",           default="cuda")
    p.add_argument("--max-depth",        type=int, default=2,
                   help="Max rule depth (1=single, 2=pairs, 3=triples)")
    p.add_argument("--use-llm-fallback", action="store_true")
    p.add_argument("--max-tasks",        type=int, default=None)
    p.add_argument("--task-id",          default=None)
    p.add_argument("--verbose-search",   action="store_true")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--batch-size",       type=int, default=1)
    return p.parse_args()


def _build_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        checkpoint_path=Path(args.checkpoint),
        data_path=Path(args.data_path),
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        enable_aug=False,
        enable_color_aug=False,
        enable_dihedral_aug=False,
        max_augments=0,
        color_apply_to_test=False,
        dihedral_apply_to_test=False,
    )


# ─── DATA HELPERS ─────────────────────────────────────────────────────────────

def load_solutions(path: Optional[str]) -> Dict[str, List]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        raw = json.load(f)
    result = {}
    for tid, grids in raw.items():
        if grids and not isinstance(grids[0][0], list):
            grids = [grids]
        result[tid] = grids
    return result


def get_demo_pairs(challenges: dict, task_id: str) -> List[DemoPair]:
    """Raw (unrotated) (input, output) demo pairs."""
    pairs = []
    for ex in challenges[task_id].get("train", []):
        inp = ex["input"]
        out = ex.get("output")
        if out is not None:
            pairs.append((inp, out))
    return pairs


def get_test_inputs(challenges: dict, task_id: str) -> List[List]:
    return [ex["input"] for ex in challenges[task_id].get("test", [])]


def grids_equal(a, b) -> bool:
    if a is None or b is None:
        return False
    try:
        return grid_equal(a, b)
    except Exception:
        return False


# ─── NEURAL GENERATION ────────────────────────────────────────────────────────

def neural_greedy(
    model,
    dataset,
    task_id: str,
    pair_idx: int,
    best_d: int,
    device: torch.device,
) -> Optional[List[List[int]]]:
    """
    Greedy decode for one test pair using run_canonical_inference_for_task.
    Returns the predicted grid in the original orientation, or None on failure.
    """
    demo_exs = [
        ex for ex in dataset.iter_examples(split="train")
        if ex.task_id == task_id
    ]
    test_exs = [
        ex for ex in dataset.iter_examples(split="test")
        if ex.task_id == task_id and ex.pair_index == pair_idx
    ]
    if not test_exs:
        return None

    # Build colour mapping from demo input grids in canonical orientation
    demo_input_grids = []
    for ex in demo_exs:
        try:
            toks = (ex.tokens_by_dihedral[best_d]
                    if ex.tokens_by_dihedral else ex.tokens).tolist()
            from common import tokens_to_grid, IO_SEPARATOR_TOKEN_ID
            sep = next((i for i, t in enumerate(toks) if t == IO_SEPARATOR_TOKEN_ID), None)
            if sep is not None:
                demo_input_grids.append(tokens_to_grid(toks[:sep]))
        except Exception:
            pass

    color_map = build_color_canon_mapping(demo_input_grids) if demo_input_grids else None
    color_map_inv = build_color_inverse_mapping(color_map) if color_map else None

    try:
        preds = run_canonical_inference_for_task(
            model=model,
            demo_examples=demo_exs,
            test_examples=test_exs,
            device=device,
            best_dihedral=best_d,        # ← correct kwarg name
            color_mapping=color_map,
            color_mapping_inv=color_map_inv,
        )
        return preds[0] if preds else None
    except Exception as e:
        return None


# ─── SCORING ──────────────────────────────────────────────────────────────────

def arc_score_submission(submission: dict, solutions_path: str) -> Tuple[int, int]:
    solutions = load_solutions(solutions_path)
    correct = total = 0
    for task_id, gt_pairs in solutions.items():
        for pair_idx, gt_grid in enumerate(gt_pairs):
            total += 1
            task_sub = submission.get(task_id, [])
            if pair_idx >= len(task_sub):
                continue
            pair_sub = task_sub[pair_idx]
            if not pair_sub:
                continue
            # pair_sub = [attempt_1, attempt_2]
            for attempt in pair_sub:
                if attempt and grids_equal(attempt, gt_grid):
                    correct += 1
                    break
    return correct, total


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("PROTOTYPE D — Program Synthesis + Neural Fallback")
    print(f"  Checkpoint    : {args.checkpoint}")
    print(f"  Data          : {args.data_path}")
    print(f"  Max depth     : {args.max_depth}")
    print(f"  LLM fallback  : {'ENABLED' if args.use_llm_fallback else 'disabled'}")
    print(f"  Rule search   : on RAW demo pairs (orientation handled by primitives)")
    print("=" * 70 + "\n")

    with open(args.data_path) as f:
        challenges = json.load(f)

    solutions = load_solutions(args.solutions)

    ckpt = load_checkpoint(Path(args.checkpoint))
    model, dataset, _dl, device, _dp = build_model_and_data(
        _build_args(args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"Dataset: {len(dataset.task_ids)} tasks\n")

    test_task_ids = sorted({
        ex.task_id for ex in dataset.iter_examples(split="test")
    })
    if args.task_id:
        test_task_ids = [args.task_id]
    elif args.max_tasks:
        test_task_ids = test_task_ids[:args.max_tasks]

    submission: Dict = {}
    diagnostics: List[dict] = []
    stats = defaultdict(int)
    t_total = perf_counter()

    for task_num, task_id in enumerate(test_task_ids, 1):
        t_task = perf_counter()
        print(f"[{task_num}/{len(test_task_ids)}] {task_id}")

        test_inputs = get_test_inputs(challenges, task_id)
        demo_pairs  = get_demo_pairs(challenges, task_id)
        gt_grids    = solutions.get(task_id, [])
        print(f"  demo={len(demo_pairs)} test={len(test_inputs)}")

        # Orientation scoring for neural fallback
        demo_exs = [
            ex for ex in dataset.iter_examples(split="train")
            if ex.task_id == task_id
        ]
        t_orient = perf_counter()
        best_d, orient_losses = score_task_orientations(model, demo_exs, device)
        t_orient = perf_counter() - t_orient
        print(f"  orientation: d{best_d}  CE={orient_losses[best_d]:.4f}  ({t_orient:.1f}s)")

        # Rule search on RAW demo pairs
        t_search = perf_counter()
        rule, rule_name, depth = find_rule(
            demo_pairs,
            max_depth=args.max_depth,
            verbose=args.verbose_search,
        )
        t_search = perf_counter() - t_search

        if rule is not None:
            print(f"  [synthesis] ✓ rule: '{rule_name}'  depth={depth}  ({t_search:.2f}s)")
            stats["rule_found"] += 1
        else:
            print(f"  [synthesis] ✗ no rule found  ({t_search:.2f}s)", end="")
            if args.use_llm_fallback:
                print(" → LLM ...", end="", flush=True)
                from llm_synthesizer import llm_synthesize_rule
                rule_fn_llm, llm_desc = llm_synthesize_rule(
                    demo_pairs, test_inputs[0],
                    verbose=args.verbose_search,
                )
                if rule_fn_llm is not None:
                    rule, rule_name, depth = rule_fn_llm, llm_desc, 0
                    print(f" ✓")
                    stats["llm_found"] += 1
                else:
                    print(" ✗")
                    stats["rule_not_found"] += 1
            else:
                print()
                stats["rule_not_found"] += 1

        task_submission = []

        for pair_idx, test_inp in enumerate(test_inputs):
            gt = gt_grids[pair_idx] if pair_idx < len(gt_grids) else None

            # Synthesis on raw test input
            synth_attempt = None
            if rule is not None:
                try:
                    synth_attempt = rule(test_inp)
                except Exception as e:
                    print(f"  [pair {pair_idx}] rule error: {e}")

            synth_correct = grids_equal(synth_attempt, gt)

            # Neural greedy
            t_gen = perf_counter()
            neural = neural_greedy(model, dataset, task_id, pair_idx, best_d, device)
            t_gen = perf_counter() - t_gen
            neural_correct = grids_equal(neural, gt)

            attempt_1 = synth_attempt if synth_attempt is not None else (neural or [])
            attempt_2 = neural or []
            if not attempt_1:
                attempt_1 = attempt_2

            task_submission.append([attempt_1, attempt_2])

            src = "synthesis" if synth_attempt is not None else "neural"
            a1_ok = grids_equal(attempt_1, gt)
            print(f"  [pair {pair_idx}] src={src}  "
                  f"a1={'✓' if a1_ok else '✗'}  "
                  f"neural={'✓' if neural_correct else '✗'}  ({t_gen:.1f}s gen)")

            if synth_correct:  stats["synth_correct"] += 1
            if neural_correct: stats["neural_correct"] += 1
            stats["total_pairs"] += 1

            diagnostics.append({
                "task_id": task_id, "pair_idx": pair_idx,
                "rule_found": rule is not None, "rule_name": rule_name,
                "rule_depth": depth, "synth_correct": synth_correct,
                "neural_correct": neural_correct, "attempt1_src": src,
            })

        submission[task_id] = task_submission
        print(f"  task done in {perf_counter()-t_task:.1f}s\n")

    # Save
    out_dir = Path(args.output_dir)
    (out_dir / "submission").mkdir(parents=True, exist_ok=True)
    sub_path = out_dir / "submission" / "submission.json"
    with open(sub_path, "w") as f:
        json.dump(submission, f)
    diag_path = out_dir / "diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(diagnostics, f, indent=2)
    print(f"Submission  → {sub_path}")
    print(f"Diagnostics → {diag_path}\n")

    # Summary
    total_tasks = len(test_task_ids)
    total_pairs = stats["total_pairs"]
    rule_found  = stats["rule_found"]
    synth_ok    = stats["synth_correct"]
    neural_ok   = stats["neural_correct"]

    print("=" * 70)
    print("PROTOTYPE D — SUMMARY")
    print("=" * 70)
    print(f"  Tasks evaluated  : {total_tasks}")
    print(f"  Test pairs       : {total_pairs}")
    print(f"  Rules found      : {rule_found}/{total_tasks} = {rule_found/max(total_tasks,1)*100:.1f}%")
    if args.use_llm_fallback:
        print(f"  LLM rules found  : {stats.get('llm_found', 0)}")
    print(f"  Synthesis correct: {synth_ok}/{total_pairs} = {synth_ok/max(total_pairs,1)*100:.1f}%")
    print(f"  Neural correct   : {neural_ok}/{total_pairs} = {neural_ok/max(total_pairs,1)*100:.1f}%")

    if args.solutions:
        correct, total = arc_score_submission(submission, args.solutions)
        score_pct = correct / max(total, 1) * 100
        print(f"\n  Official ARC score : {correct}/{total} = {score_pct:.2f}%")
        print(f"\n  Comparison:")
        print(f"    Prototype A (greedy)   : 27.8%")
        print(f"    Prototype C v9e        : 29.0%")
        print(f"    Prototype D (this run) : {score_pct:.2f}%")
        if score_pct >= 44.0:
            print(f"    ✓ BEATS Mithil target (44%)")
        else:
            print(f"    Still {44.0-score_pct:.1f}pp below Mithil (44%)")

    print(f"\nTotal wall time : {perf_counter()-t_total:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
