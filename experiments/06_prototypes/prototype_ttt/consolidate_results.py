"""
consolidate_results.py — Reads all ablation_v9 run directories and prints
a single comparison table across all experiments.

Usage (run from the repo root):
    python prototype_ttt/consolidate_results.py \
        runs/ablation_v9_lr_sweep \
        runs/ablation_v9_lr_sweep_5e-5 \
        runs/ablation_v9_lr_sweep_1e-4 \
        runs/ablation_v9_steps_sweep
"""
import sys
import json
from pathlib import Path


def main():
    roots = [Path(p) for p in sys.argv[1:]]
    if not roots:
        print("Usage: python prototype_ttt/consolidate_results.py <run_dir1> [<run_dir2> ...]")
        return

    rows = []
    for root in roots:
        comp_path = root / "comparison.json"
        if not comp_path.exists():
            print(f"[skip] no comparison.json in {root}")
            continue
        summaries = json.loads(comp_path.read_text())
        for s in summaries:
            rows.append({
                "label":      s.get("label", root.name),
                "ttt_steps":  s.get("ttt_steps", 0),
                "ttt_lr":     s.get("ttt_lr", 0.0),
                "n_pairs":    s.get("n_pairs", "?"),
                "n_oracle":   s.get("n_oracle", "?"),
                "n_a2":       s.get("n_a2_correct", "?"),
                "arc_pct":    s.get("arc_pct", 0.0),
                "source_dir": root.name,
            })

    print()
    print("=" * 95)
    print("CONSOLIDATED EXP O RESULTS  (TTT step-count × learning-rate sweep)")
    print("=" * 95)
    hdr = (f"{'Experiment':<38} {'steps':>5} {'lr':>8}  "
           f"{'oracle':>8}  {'a2':>4}  {'ARC%':>6}  source")
    print(hdr)
    print("-" * 95)
    print(f"{'EXP C ref (no-TTT, 400 tasks)':<38} {'0':>5} {'—':>8}  "
          f"{'157/419':>8}  {'15':>4}  {'30.50%':>6}  reference")
    print()

    for r in rows:
        lr_s = f"{r['ttt_lr']:.0e}" if r["ttt_lr"] else "—"
        oracle_s = f"{r['n_oracle']}/{r['n_pairs']}"
        print(f"{r['label']:<38} {r['ttt_steps']:>5} {lr_s:>8}  "
              f"{oracle_s:>8}  {str(r['n_a2']):>4}  {r['arc_pct']:>5.2f}%  {r['source_dir']}")

    print("=" * 95)

    if rows:
        # Exclude K (no TTT) rows when finding best TTT config
        ttt_rows = [r for r in rows if r["ttt_steps"] > 0]
        all_rows = rows
        best_all = max(all_rows, key=lambda r: r["arc_pct"])
        print(f"\nBest overall : {best_all['label']}  ({best_all['arc_pct']:.2f}%)")
        if ttt_rows:
            best_ttt = max(ttt_rows, key=lambda r: r["arc_pct"])
            print(f"Best TTT     : {best_ttt['label']}  "
                  f"steps={best_ttt['ttt_steps']} lr={best_ttt['ttt_lr']:.0e}  "
                  f"({best_ttt['arc_pct']:.2f}%)")
        delta = best_all["arc_pct"] - 30.50
        sign  = "+" if delta >= 0 else ""
        print(f"Δ vs EXP C   : {sign}{delta:.2f}pp")
        print(f"Gap to Mithil: {44.0 - best_all['arc_pct']:.2f}pp")


if __name__ == "__main__":
    main()
