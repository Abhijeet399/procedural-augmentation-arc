"""
postmortem_missed_oracle.py — Deep diagnosis of missed oracle pairs

For EXP C (beam=14, n_sample=30, λ=0.3, ARC=30.50%), there are 25 pairs
where the correct answer existed in the raw candidate pool (oracle_hit=True)
but neither attempt_1 nor attempt_2 selected it.

This script re-runs those 25 pairs with full instrumentation to classify
each failure into one of four categories:

  FILTER_SHAPE   — oracle rejected by shape filter (never reaches ranker)
  FILTER_PALETTE — oracle survived shape, rejected by palette filter
  RANK_FAIL      — oracle survived both filters but ranked below the winner
  SINGLE_SURV    — only one survivor (same-as-greedy edge case, oracle squeezed out)

For RANK_FAIL cases, it also reports:
  - oracle transition score vs winner score
  - oracle dissimilarity from greedy vs winner dissimilarity
  - oracle combined score (transition + λ*dissimilarity) vs winner
  - oracle rank position in the survivor list

Usage:
    python prototype_e/postmortem_missed_oracle.py \\
        --diag-path  runs/ablation_v4/C_diversity_+_bigger_gen/diagnostics.json \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --solutions  assets/solutions.json \\
        --output-dir runs/postmortem_c \\
        --lambda     0.3
"""

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import List, Tuple, Optional

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "src"))
sys.path.insert(0, str(_HERE.parent.parent.parent))

from run_prototype_e    import load_solutions, load_checkpoint, build_model_and_data, _args_for_build
from ablation_v4        import (
    pixel_dissimilarity, pick_best_non_greedy,
    _grids_from_seq_ex, grids_equal, invert_d
)
from candidate_filters  import hard_filter
from transition_ranker  import rank_by_transition, _grids_equal


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--diag-path",   required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data-path",   required=True)
    p.add_argument("--solutions",   required=True)
    p.add_argument("--output-dir",  required=True)
    p.add_argument("--lambda",      type=float, default=0.3, dest="lam")
    p.add_argument("--task-id",     default=None)
    return p.parse_args()


# =============================================================================
# Find missed oracle pairs from diagnostics JSON
# =============================================================================
def find_missed_oracle_pairs(diag_path: str):
    """
    Return list of (task_id, pair_idx) for pairs where:
      oracle_hit=True, greedy_correct=False, a2_correct=False
    """
    with open(diag_path) as f:
        diags = json.load(f)
    missed = []
    for d in diags:
        if (d.get("oracle_hit") is True
                and not d.get("greedy_correct")
                and not d.get("a2_correct")):
            missed.append((d["task_id"], d["pair_idx"]))
    return missed


# =============================================================================
# Per-pair deep diagnosis
# =============================================================================
def diagnose_pair(
    task_id:          str,
    pair_idx:         int,
    model,
    dataset,
    solutions:        dict,
    device,
    base_args,
    lam:              float,
) -> dict:
    """
    Re-run one task/pair with full instrumentation.
    Returns a rich diagnosis dict.
    """
    from run_prototype_e import (
        generate_all_candidates,
        score_task_orientations,
        apply_dihedral_transform,
    )

    # ── Load task ────────────────────────────────────────────────────────────
    task_examples = [ex for ex in dataset.iter_examples(split="test")
                     if ex.task_id == task_id]
    if not task_examples:
        return {"task_id": task_id, "pair_idx": pair_idx, "error": "task_not_found"}

    example_id = task_examples[0].example_id
    demo_seq_exs = [ex for ex in dataset.iter_examples(split="train")
                    if ex.task_id == task_id]
    test_seq_exs  = task_examples

    if pair_idx >= len(test_seq_exs):
        return {"task_id": task_id, "pair_idx": pair_idx, "error": "pair_idx_oob"}

    best_d, _ = score_task_orientations(model, demo_seq_exs, device)
    demo_pairs_raw = [
        (_grids_from_seq_ex(ex, best_d).input, _grids_from_seq_ex(ex, best_d).output)
        for ex in demo_seq_exs
        if _grids_from_seq_ex(ex, best_d).output is not None
    ]

    test_seq_ex    = test_seq_exs[pair_idx]
    test_gp        = _grids_from_seq_ex(test_seq_ex, best_d)
    canonical_inp  = test_gp.input

    sol_grids      = solutions.get(task_id, [])
    gt_original    = sol_grids[pair_idx] if pair_idx < len(sol_grids) else None
    gt_canonical   = apply_dihedral_transform(gt_original, best_d) if gt_original else None

    if gt_canonical is None:
        return {"task_id": task_id, "pair_idx": pair_idx, "error": "no_gt"}

    # ── Generate ─────────────────────────────────────────────────────────────
    candidates = generate_all_candidates(
        model, demo_pairs_raw, canonical_inp,
        example_id, best_d, device,
        use_greedy        = True,
        use_beam          = base_args.beam_width > 0,
        beam_width        = base_args.beam_width,
        use_sample        = base_args.n_sample > 0,
        n_per_temperature = base_args.n_sample,
        temperatures      = tuple(base_args.temps),
        top_k             = base_args.top_k,
        test_seq_ex       = test_seq_ex,
        demo_seq_exs      = demo_seq_exs,
    )
    greedy_cand = candidates[0] if candidates else []
    n_total     = len(candidates)

    # ── Verify oracle is truly in raw pool ───────────────────────────────────
    oracle_in_raw = any(grids_equal(c, gt_canonical) for c in candidates)
    if not oracle_in_raw:
        return {
            "task_id":      task_id,
            "pair_idx":     pair_idx,
            "failure_mode": "ORACLE_NOT_IN_RAW",  # diagnostics.json was stale/wrong
            "n_candidates": n_total,
        }

    # ── Shape filter ─────────────────────────────────────────────────────────
    from candidate_filters import filter_by_shape, infer_output_shape_rule, \
                                   infer_allowed_colors, filter_by_palette
    shape_rule, fixed_shape = infer_output_shape_rule(demo_pairs_raw)
    after_shape, n_shape_rej = filter_by_shape(candidates, canonical_inp, demo_pairs_raw)
    oracle_survives_shape = any(grids_equal(c, gt_canonical) for c in after_shape)

    if not oracle_survives_shape:
        # Find oracle's shape vs expected
        oracle_cand = next(c for c in candidates if grids_equal(c, gt_canonical))
        oracle_shape = (len(oracle_cand), len(oracle_cand[0]) if oracle_cand else 0)
        inp_shape    = (len(canonical_inp), len(canonical_inp[0]) if canonical_inp else 0)
        return {
            "task_id":       task_id,
            "pair_idx":      pair_idx,
            "failure_mode":  "FILTER_SHAPE",
            "shape_rule":    shape_rule,
            "oracle_shape":  oracle_shape,
            "input_shape":   inp_shape,
            "fixed_shape":   fixed_shape,
            "n_candidates":  n_total,
            "n_shape_rej":   n_shape_rej,
        }

    # ── Palette filter ────────────────────────────────────────────────────────
    allowed = infer_allowed_colors(demo_pairs_raw, canonical_inp)
    after_palette, n_palette_rej = filter_by_palette(after_shape, allowed)
    oracle_survives_palette = any(grids_equal(c, gt_canonical) for c in after_palette)

    if not oracle_survives_palette:
        oracle_cand   = next(c for c in after_shape if grids_equal(c, gt_canonical))
        oracle_colors = set(v for row in oracle_cand for v in row)
        bad_colors    = sorted(oracle_colors - set(allowed) - {0})
        return {
            "task_id":        task_id,
            "pair_idx":       pair_idx,
            "failure_mode":   "FILTER_PALETTE",
            "oracle_colors":  sorted(oracle_colors),
            "allowed_colors": sorted(allowed),
            "bad_colors":     bad_colors,
            "n_candidates":   n_total,
            "n_shape_rej":    n_shape_rej,
            "n_palette_rej":  n_palette_rej,
            "n_survivors":    len(after_palette),
        }

    # ── Transition ranker ─────────────────────────────────────────────────────
    survivors = after_palette if after_palette else candidates
    fallback  = (after_palette == [])

    scored_list, _table = rank_by_transition(
        survivors, canonical_inp, demo_pairs_raw,
        greedy_cand   = greedy_cand,
        greedy_margin = 0.5,
    )
    ranked_grids = [g for g, _ in scored_list]
    scores_map   = {id(g): s for g, s in scored_list}

    # Find oracle's position in ranked list
    oracle_rank_pos   = next((i for i, g in enumerate(ranked_grids)
                              if grids_equal(g, gt_canonical)), None)
    oracle_trans_score = next((s for g, s in scored_list
                               if grids_equal(g, gt_canonical)), None)

    # Compute combined scores (transition + λ*dissimilarity from greedy)
    def combined(cand, trans_score):
        dis = pixel_dissimilarity(cand, greedy_cand) if greedy_cand else 0.0
        return trans_score + lam * dis, dis

    oracle_combined, oracle_dis = combined(gt_canonical, oracle_trans_score)

    # What pick_best_non_greedy actually picks
    best_ng = pick_best_non_greedy(ranked_grids, scored_list, greedy_cand, lam)
    winner_trans_score = next((s for g, s in scored_list
                               if best_ng is not None and grids_equal(g, best_ng)), None)
    winner_combined, winner_dis = combined(best_ng, winner_trans_score) if best_ng else (None, None)
    winner_is_correct = grids_equal(best_ng, gt_canonical) if best_ng else False

    # Greedy score
    greedy_trans_score = next((s for g, s in scored_list
                               if grids_equal(g, greedy_cand)), None)
    if greedy_trans_score is None:
        # greedy might not be in survivors if filtered, look in scored_list
        # actually if fallback=False greedy might have been filtered
        pass

    n_survivors = len(survivors)
    # Score margin by which oracle lost
    score_gap = (winner_combined - oracle_combined) if (winner_combined is not None and oracle_combined is not None) else None

    return {
        "task_id":              task_id,
        "pair_idx":             pair_idx,
        "failure_mode":         "RANK_FAIL",
        "n_candidates":         n_total,
        "n_shape_rej":          n_shape_rej,
        "n_palette_rej":        n_palette_rej,
        "n_survivors":          n_survivors,
        "fallback":             fallback,
        "oracle_rank_pos":      oracle_rank_pos,           # 0-indexed; 0=top
        "oracle_trans_score":   round(oracle_trans_score, 4) if oracle_trans_score else None,
        "oracle_dissimilarity": round(oracle_dis, 4),
        "oracle_combined":      round(oracle_combined, 4) if oracle_combined else None,
        "winner_trans_score":   round(winner_trans_score, 4) if winner_trans_score else None,
        "winner_dissimilarity": round(winner_dis, 4) if winner_dis else None,
        "winner_combined":      round(winner_combined, 4) if winner_combined else None,
        "score_gap":            round(score_gap, 4) if score_gap else None,
        "winner_is_correct":    winner_is_correct,
        # Sub-classify rank fail
        "rank_fail_subtype":   _rank_fail_subtype(
            oracle_rank_pos, oracle_trans_score, winner_trans_score,
            oracle_dis, winner_dis, oracle_combined, winner_combined, lam
        ),
    }


def _rank_fail_subtype(
    oracle_rank_pos, oracle_ts, winner_ts,
    oracle_dis, winner_dis, oracle_comb, winner_comb, lam
) -> str:
    """Classify what specifically caused the rank failure."""
    if oracle_rank_pos == 0:
        # Ranker picked oracle as #1 but diversity picked something else?
        return "ORACLE_TOP1_BUT_DIVERSITY_PICKED_OTHER"
    if oracle_ts is None or winner_ts is None:
        return "UNKNOWN"
    ts_gap   = winner_ts  - oracle_ts
    comb_gap = winner_comb - oracle_comb if (winner_comb and oracle_comb) else None

    if ts_gap <= 0 and comb_gap and comb_gap > 0:
        # Transition score: oracle wins; combined: winner wins → diversity bonus tipped it
        return "DIVERSITY_BONUS_HURT"  # λ amplified a wrong candidate's dissimilarity
    if ts_gap > 2.0:
        return "LARGE_TRANSITION_GAP"   # winner dominates in transition signal (>2 nats)
    if ts_gap > 0.5:
        return "MEDIUM_TRANSITION_GAP"  # winner leads by 0.5–2 nats
    return "SMALL_TRANSITION_GAP"       # winner leads by <0.5 nats — near-tie


# =============================================================================
# Main
# =============================================================================
def main():
    args = parse_args()

    print("=" * 70)
    print("POST-MORTEM: Missed Oracle Pairs")
    print(f"  Diagnostics : {args.diag_path}")
    print(f"  Checkpoint  : {args.checkpoint}")
    print(f"  Lambda      : {args.lam}")
    print("=" * 70)

    # ── Find missed pairs ─────────────────────────────────────────────────────
    missed = find_missed_oracle_pairs(args.diag_path)
    if args.task_id:
        missed = [(t, p) for t, p in missed if t == args.task_id]
    print(f"\nMissed oracle pairs: {len(missed)}")
    for t, p in missed:
        print(f"  {t}  pair {p}")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    solutions = load_solutions(args.solutions)
    ckpt      = load_checkpoint(Path(args.checkpoint))

    # Build a fake args namespace for build_model_and_data
    import torch
    _device = "cuda" if torch.cuda.is_available() else "cpu"

    class InfArgs:
        data_path  = args.data_path
        checkpoint = args.checkpoint
        device     = _device
        beam_width = 14
        n_sample   = 30
        temps      = [0.7, 1.0]
        top_k      = 0
        max_tasks  = None
        task_id    = None

    inf_args = InfArgs()
    model, dataset, _dl, device, _dp = build_model_and_data(
        _args_for_build(inf_args), checkpoint=ckpt, is_eval=True
    )
    model.eval()
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # ── Run diagnosis per pair ────────────────────────────────────────────────
    results = []
    for i, (task_id, pair_idx) in enumerate(missed):
        print(f"\n[{i+1}/{len(missed)}] {task_id}  pair {pair_idx}")
        t0 = perf_counter()
        d  = diagnose_pair(
            task_id, pair_idx, model, dataset, solutions, device, inf_args, args.lam
        )
        elapsed = perf_counter() - t0
        d["elapsed_s"] = round(elapsed, 1)
        results.append(d)

        fm = d.get("failure_mode", "?")
        if fm == "FILTER_SHAPE":
            print(f"  → FILTER_SHAPE  rule={d['shape_rule']}  "
                  f"oracle_shape={d['oracle_shape']}  input_shape={d['input_shape']}")
        elif fm == "FILTER_PALETTE":
            print(f"  → FILTER_PALETTE  bad_colors={d['bad_colors']}  "
                  f"allowed={d['allowed_colors']}")
        elif fm == "RANK_FAIL":
            print(f"  → RANK_FAIL [{d['rank_fail_subtype']}]  "
                  f"oracle_rank={d['oracle_rank_pos']}  "
                  f"oracle_comb={d['oracle_combined']}  "
                  f"winner_comb={d['winner_combined']}  "
                  f"gap={d['score_gap']}")
        else:
            print(f"  → {fm}")
        print(f"  ({elapsed:.1f}s)")

    # ── Summary ───────────────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    by_mode = {}
    for r in results:
        fm = r.get("failure_mode", "UNKNOWN")
        by_mode.setdefault(fm, []).append(r)

    total = len(results)
    for fm, items in sorted(by_mode.items(), key=lambda x: -len(x[1])):
        pct = 100 * len(items) / total
        print(f"\n{fm}: {len(items)}/{total}  ({pct:.0f}%)")
        if fm == "RANK_FAIL":
            subtypes = {}
            for r in items:
                st = r.get("rank_fail_subtype", "?")
                subtypes.setdefault(st, []).append(r)
            for st, sitems in sorted(subtypes.items(), key=lambda x: -len(x[1])):
                print(f"  {st}: {len(sitems)}")
            gaps = [r["score_gap"] for r in items if r.get("score_gap") is not None]
            if gaps:
                print(f"  avg score_gap (winner - oracle): {sum(gaps)/len(gaps):.3f} nats")
            ranks = [r["oracle_rank_pos"] for r in items if r.get("oracle_rank_pos") is not None]
            if ranks:
                print(f"  oracle rank positions: {sorted(ranks)}")
        elif fm == "FILTER_SHAPE":
            rules = [r.get("shape_rule", "?") for r in items]
            from collections import Counter
            print(f"  shape rules: {dict(Counter(rules))}")
        elif fm == "FILTER_PALETTE":
            all_bad = [c for r in items for c in r.get("bad_colors", [])]
            from collections import Counter
            print(f"  bad color frequencies: {dict(Counter(all_bad))}")

    print()
    print(f"ACTIONABLE BREAKDOWN:")
    n_filter  = len(by_mode.get("FILTER_SHAPE",   [])) + len(by_mode.get("FILTER_PALETTE", []))
    n_rank    = len(by_mode.get("RANK_FAIL",       []))
    n_other   = total - n_filter - n_rank
    print(f"  Filter failures (need better generation/filters): {n_filter}/{total}")
    print(f"  Ranking failures (need better signal):            {n_rank}/{total}")
    if n_other:
        print(f"  Other:                                           {n_other}/{total}")

    # ── Save results ──────────────────────────────────────────────────────────
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "postmortem_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results → {out_path}")


if __name__ == "__main__":
    main()
