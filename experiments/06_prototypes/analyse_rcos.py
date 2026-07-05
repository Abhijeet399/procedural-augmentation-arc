"""
analyse_rcos.py — Deep analysis of Prototype C diagnostics.json

Run after run_prototype_c.py to understand precisely where RCOS wins and fails,
and which bottleneck to fix next.

Usage:
    python analyse_rcos.py runs/prototype_c/diagnostics.json

Output sections:
  1. Top-line numbers (oracle, RCOS accuracy, lift)
  2. Failure mode breakdown (generation failure vs scoring failure)
  3. Rank distribution histogram
  4. Score gap statistics
  5. CE analysis (baseline vs augmented)
  6. Orientation distribution
  7. Actionable takeaways
"""

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional


# =============================================================================
# Utilities
# =============================================================================

def load(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)


def pct(n: int, d: int, dec: int = 1) -> str:
    if d == 0:
        return "N/A"
    return f"{n/d*100:.{dec}f}%"


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def percentile(vals: List[float], p: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100.0
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def stats_str(vals: List[float]) -> str:
    if not vals:
        return "  (no data)"
    return (f"  n={len(vals)}  mean={mean(vals):+.4f}  "
            f"p25={percentile(vals,25):+.4f}  median={percentile(vals,50):+.4f}  "
            f"p75={percentile(vals,75):+.4f}")


# =============================================================================
# Analysis
# =============================================================================

def analyse(diag: List[Dict]):
    n = len(diag)
    if n == 0:
        print("No diagnostics found.")
        return

    n_tasks = len({d["task_id"] for d in diag})

    # Partition into failure modes
    cat_c = [d for d in diag if d["rcs_selects_correct"]]            # RCOS correct
    cat_b = [d for d in diag if d["oracle_hit"]
             and not d["rcs_selects_correct"]]                        # in set, wrong rank
    cat_a = [d for d in diag if not d["oracle_hit"]]                  # not in set

    oracle_n  = len(cat_b) + len(cat_c)  # oracle_hit
    rcs_n     = len(cat_c)
    lift      = (rcs_n / oracle_n * 100) if oracle_n > 0 else 0.0

    SEP = "─" * 70

    print()
    print("═" * 70)
    print("  RCOS EXPERIMENT 1 — DEEP ANALYSIS")
    print("═" * 70)

    # ─── 1. Top-line ───────────────────────────────────────────────────────
    print(f"\n  Tasks: {n_tasks}   Test pairs: {n}\n")
    print(f"  {'Oracle accuracy':<38}: {oracle_n}/{n} = {pct(oracle_n,n)}")
    print(f"  {'RCOS accuracy':<38}: {rcs_n}/{n} = {pct(rcs_n,n)}")
    print(f"  {'RCOS-oracle lift':<38}: {lift:.1f}%")
    print(f"    (ideal = 100%: RCOS always picks correct when it's in the pool)")

    # ─── 2. Failure modes ──────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  FAILURE MODE BREAKDOWN")
    print(SEP)
    print(f"  A. Generation failure (correct ∉ candidates)  : "
          f"{len(cat_a):4d} / {n}  = {pct(len(cat_a),n)}")
    print(f"  B. Scoring failure    (in set but mis-ranked) : "
          f"{len(cat_b):4d} / {n}  = {pct(len(cat_b),n)}")
    print(f"  C. RCOS correct                               : "
          f"{len(cat_c):4d} / {n}  = {pct(len(cat_c),n)}")
    print()
    if len(cat_a) > len(cat_b):
        print("  → GENERATION is the dominant bottleneck.")
        print("    Next step: LoRA-TTT to improve the quality of generated candidates.")
    elif len(cat_b) > len(cat_a):
        print("  → SCORING is the dominant bottleneck.")
        print("    Next step: more demos in RCS context, or LoRA-TTT for better CE signal.")
    else:
        print("  → Generation and scoring failures are roughly balanced.")

    # ─── 3. Rank distribution ──────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  RANK OF CORRECT ANSWER (oracle-hit tasks)")
    print(SEP)
    oracle_diag = [d for d in diag if d["oracle_hit"]]
    rank_cnts: Dict[int, int] = Counter(
        d["rcs_rank_of_correct"] for d in oracle_diag
        if d.get("rcs_rank_of_correct") is not None
    )
    cumulative = 0
    for rank in range(1, 16):
        cnt = rank_cnts.get(rank, 0)
        cumulative += cnt
        pct_cum = pct(cumulative, len(oracle_diag))
        bar = "█" * min(cnt, 35)
        print(f"  rank {rank:3d}: {cnt:4d}  {bar:35s}  cumul={pct_cum}")
    remainder = sum(v for k, v in rank_cnts.items() if k is not None and k > 15)
    if remainder:
        print(f"  rank >15: {remainder}")
    print()
    print("  Insight: rank-1 rate among oracle hits = "
          f"{pct(rank_cnts.get(1,0), len(oracle_diag))}")
    print("           rank ≤3 rate among oracle hits = "
          f"{pct(sum(rank_cnts.get(r,0) for r in range(1,4)), len(oracle_diag))}")

    # ─── 4. Score gap ──────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  SCORE GAP  (rcs_correct − rcs_top1)")
    print(SEP)
    print("  Interpretation: >0 = RCOS correctly ranked correct answer above all others")
    print("                  <0 = correct answer was outscored by a wrong candidate\n")
    gaps_c = [d["score_gap"] for d in cat_c if d.get("score_gap") is not None]
    gaps_b = [d["score_gap"] for d in cat_b if d.get("score_gap") is not None]
    print(f"  When RCOS correct  (n={len(gaps_c)}):")
    print(stats_str(gaps_c))
    print(f"\n  When RCOS wrong    (n={len(gaps_b)}):")
    print(stats_str(gaps_b))

    # Histogram of gap magnitude for wrong cases
    if gaps_b:
        buckets = [0] * 5   # [0-0.01, 0.01-0.1, 0.1-0.5, 0.5-2, >2]
        for g in gaps_b:
            ag = abs(g)
            if ag < 0.01:   buckets[0] += 1
            elif ag < 0.1:  buckets[1] += 1
            elif ag < 0.5:  buckets[2] += 1
            elif ag < 2.0:  buckets[3] += 1
            else:           buckets[4] += 1
        labels = ["0–0.01", "0.01–0.1", "0.1–0.5", "0.5–2.0", ">2.0"]
        print("\n  Gap-magnitude distribution for mis-ranked tasks:")
        for lab, cnt in zip(labels, buckets):
            bar = "█" * cnt
            print(f"    |gap| {lab:10s}: {cnt:4d}  {bar}")
        print("\n  → Small gaps (<0.1) are near-misses: better CE signal could fix them.")
        print("    Large gaps (>0.5) need fundamentally better candidates or model quality.")

    # ─── 5. CE analysis ────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CE STATISTICS")
    print(SEP)

    def ce_stats(key: str, data: List[Dict], label: str):
        vals = [d[key] for d in data
                if d.get(key) is not None and d[key] != math.inf]
        print(f"  {label} (n={len(vals)}):")
        print(stats_str(vals))

    ce_stats("baseline_ce",   diag,        "Baseline CE  (demos only, no synthetic pair)")
    ce_stats("aug_ce_top1",   diag,        "Augmented CE (top-1 RCOS candidate prepended)")
    ce_stats("aug_ce_correct", oracle_diag, "Augmented CE (correct candidate, oracle hits)")
    ce_stats("rcs_score_top1", diag,        "RCS score    (top-1 candidate)")
    ce_stats("rcs_score_correct", oracle_diag, "RCS score   (correct candidate)")

    # ─── 6. Orientation distribution ───────────────────────────────────────
    print(f"\n{SEP}")
    print("  ORIENTATION SELECTION")
    print(SEP)
    names = ["identity","rot90","rot180","rot270","flip_h","flip_v","flip_main","flip_anti"]
    orient_cnts: Counter = Counter(d.get("best_d", 0) for d in diag)
    for d_idx, name in enumerate(names):
        cnt = orient_cnts.get(d_idx, 0)
        bar = "█" * min(cnt, 40)
        print(f"  d{d_idx} ({name:15s}): {cnt:4d}  {bar}")

    # ─── 7. Candidate count ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  CANDIDATE POOL STATISTICS")
    print(SEP)
    n_cands = [d.get("n_candidates", 0) for d in diag]
    print(f"  Avg: {mean(n_cands):.1f}  Min: {min(n_cands)}  Max: {max(n_cands)}")
    print(f"  p50: {percentile(n_cands,50):.0f}  p90: {percentile(n_cands,90):.0f}")

    # Breakdown: how many tasks had 0 candidates?
    zero_cands = sum(1 for n in n_cands if n == 0)
    if zero_cands:
        print(f"\n  WARNING: {zero_cands} tasks had 0 valid candidates! "
              f"These will always fail.")

    # ─── 8. RCOS accuracy vs candidate pool size ──────────────────────────
    print(f"\n{SEP}")
    print("  RCOS ACCURACY BY POOL SIZE")
    print(SEP)
    buckets_acc: Dict[str, List[int]] = defaultdict(list)
    for d in diag:
        nc = d.get("n_candidates", 0)
        if nc == 0:     key = "0"
        elif nc == 1:   key = "1"
        elif nc <= 5:   key = "2-5"
        elif nc <= 15:  key = "6-15"
        elif nc <= 30:  key = "16-30"
        else:           key = ">30"
        buckets_acc[key].append(int(d["rcs_selects_correct"]))
    for key in ["0","1","2-5","6-15","16-30",">30"]:
        v = buckets_acc.get(key, [])
        if v:
            acc = pct(sum(v), len(v))
            print(f"  n_cands={key:5s}: {len(v):4d} tasks  RCOS_acc={acc}")
    print()
    print("  Insight: if RCOS accuracy drops as pool size grows, the scoring")
    print("           function is struggling to separate many similar candidates.")

    # ─── 9. Per-CE-quartile analysis ──────────────────────────────────────
    print(f"\n{SEP}")
    print("  RCOS ACCURACY BY BASELINE CE QUARTILE")
    print(SEP)
    base_ces = [d.get("baseline_ce", math.inf) for d in diag
                if d.get("baseline_ce") != math.inf]
    if base_ces:
        q1 = percentile(base_ces, 25)
        q2 = percentile(base_ces, 50)
        q3 = percentile(base_ces, 75)

        def quartile(ce):
            if ce == math.inf: return "Q4+"
            if ce <= q1: return "Q1 (low CE ← easy)"
            if ce <= q2: return "Q2"
            if ce <= q3: return "Q3"
            return "Q4 (high CE ← hard)"

        qt: Dict[str, List[int]] = defaultdict(list)
        for d in diag:
            ce = d.get("baseline_ce", math.inf)
            qt[quartile(ce)].append(int(d["rcs_selects_correct"]))

        for label in ["Q1 (low CE ← easy)", "Q2", "Q3", "Q4 (high CE ← hard)", "Q4+"]:
            v = qt.get(label, [])
            if v:
                print(f"  {label}: RCOS_acc={pct(sum(v),len(v))}  (n={len(v)})")

    # ─── 10. Key takeaways ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  KEY TAKEAWAYS")
    print(SEP)

    gen_fail_rate = len(cat_a) / n
    score_fail_rate = len(cat_b) / n

    if oracle_n / n < 0.40:
        print(f"  ● Oracle rate is LOW ({pct(oracle_n,n)}). Generation is the bottleneck.")
        print("    → LoRA-TTT will give the model a better adapted prior, producing")
        print("      candidates closer to the correct answer.")
        print("    → Also try: larger beam width, more temperature samples.")
    elif lift < 60.0:
        print(f"  ● Oracle rate is acceptable ({pct(oracle_n,n)}) but RCOS lift is LOW ({lift:.1f}%).")
        print("    → Scoring is the bottleneck. RCS signal is noisy.")
        print("    → Try: multi-augmented RCS (score under multiple orientations and average),")
        print("      or use the test pair itself as a demo for other pairs (cross-pair RCS).")
    else:
        print(f"  ● Both oracle ({pct(oracle_n,n)}) and RCOS lift ({lift:.1f}%) look promising!")
        print("    → Focus on LoRA-TTT to push oracle rate above 60%.")
        print("    → At oracle=60% and lift=80%: final score ≈ 48% (beating Mithil baseline).")

    print()
    print("  Expected end-game targets:")
    print("    oracle_accuracy ≥ 60%  (achievable with LoRA-TTT)")
    print("    RCOS lift       ≥ 75%  (achievable with good scoring signal)")
    print("    Combined score  ≥ 45%  (beats Mithil baseline of 44%)")
    print()


# =============================================================================
# Main
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    if not Path(path).exists():
        print(f"ERROR: file not found: {path}")
        sys.exit(1)

    diag = load(path)
    analyse(diag)


if __name__ == "__main__":
    main()
