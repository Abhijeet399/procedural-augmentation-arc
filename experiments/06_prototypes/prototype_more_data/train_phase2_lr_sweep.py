"""
prototype_more_data/train_phase2_lr_sweep.py
=============================================
Phase 2: Fine-tune Phase 1 checkpoint on original 1307 tasks, sweeping LR.

After Phase 1 (NVARC training), the model knows NVARC transformation patterns
but may have forgotten original task solutions.  Phase 2 re-specializes on the
original 1307 tasks using a REDUCED NorMuon LR to avoid overwriting NVARC gains.

LR sweep rationale:
  Original training LR: normuon=0.02, adam=3e-4
  Phase 2 candidates  : normuon=0.002, 0.001, 0.0005 (10x-40x reduction)
  Lower LR → less forgetting of NVARC patterns, slower adaptation to original
  Higher LR → faster adaptation, more forgetting of NVARC

For each LR we:
  1. Fine-tune on original 1307 tasks for --epochs epochs
  2. Save checkpoint as runs/phase2_sweep/phase2_lr{LR}.pt
  3. Run oracle ceiling eval on the 400 eval tasks
  4. Print comparison table

Usage (run from the repo root):
    python prototype_more_data/train_phase2_lr_sweep.py \\
        --phase1-checkpoint  runs/phase1_nvarc.pt \\
        --data-path          assets/challenges.json \\
        --eval-solutions     assets/solutions.json \\
        --out-dir            runs/phase2_sweep \\
        --normuon-lrs        0.002 0.001 0.0005 \\
        --epochs             5

    # Quick test with one LR:
        ... --normuon-lrs 0.001 --epochs 1 --eval-max-tasks 20
"""

import sys
import json
import argparse
from pathlib import Path
from time import perf_counter
from typing import Dict, List

_HERE = Path(__file__).parent.resolve()
_REPO_ROOT = _HERE.parent.parent.parent   # repo root
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(_HERE.parent / "prototype_e"))
sys.path.insert(0, str(_HERE.parent / "prototype_e" / "src"))
sys.path.insert(0, str(_REPO_ROOT))

import importlib.util as _ilu

_spec = _ilu.spec_from_file_location("mdlarc_build", str(_REPO_ROOT / "src" / "build.py"))
_build = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_build)
build_model_and_data = _build.build_model_and_data
load_checkpoint      = _build.load_checkpoint

_spec2 = _ilu.spec_from_file_location("mdlarc_train", str(_REPO_ROOT / "src" / "train.py"))
_train = _ilu.module_from_spec(_spec2); _spec2.loader.exec_module(_train)
train_model = _train.train_model


def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 — LR sweep on original tasks")
    p.add_argument("--phase1-checkpoint", required=True,
                   help="Phase 1 checkpoint (runs/phase1_nvarc.pt)")
    p.add_argument("--data-path",         required=True,
                   help="Original challenges.json (assets/challenges.json)")
    p.add_argument("--eval-solutions",    default="assets/solutions.json")
    p.add_argument("--out-dir",           default="runs/phase2_sweep")
    p.add_argument("--device",            default="cuda")
    p.add_argument("--seed",              type=int, default=42)
    # LR sweep
    p.add_argument("--normuon-lrs",       type=float, nargs="+",
                   default=[0.002, 0.001, 0.0005],
                   help="NorMuon LRs to sweep (default: 0.002 0.001 0.0005)")
    p.add_argument("--adamw-lr-ratio",    type=float, default=0.015,
                   help="adamw_lr = normuon_lr * ratio (default: 0.015 → adam=3e-4 when muon=0.02)")
    # Training
    p.add_argument("--epochs",            type=int,   default=5)
    p.add_argument("--weight-decay",      type=float, default=0.1)
    p.add_argument("--grad-clip",         type=float, default=1.0)
    p.add_argument("--warmup-pct",        type=float, default=0.05,
                   help="Warmup fraction (default: 5% — slightly longer than original 2%)")
    p.add_argument("--wsd-decay-start-pct", type=float, default=0.7)
    p.add_argument("--lr-floor",          type=float, default=0.01)
    # Eval
    p.add_argument("--eval-max-tasks",    type=int, default=400)
    p.add_argument("--eval-beam",         type=int, default=5)
    p.add_argument("--eval-n-sample",     type=int, default=10)
    p.add_argument("--skip-eval",         action="store_true",
                   help="Skip oracle eval (faster, just compare val loss)")
    return p.parse_args()


def _build_args_for_data(args, data_path: Path, checkpoint_path: Path):
    """Namespace for build_model_and_data."""
    import argparse as _ap
    return _ap.Namespace(
        checkpoint_path      = checkpoint_path,
        data_path            = data_path,
        seed                 = args.seed,
        batch_size           = 1,
        device               = args.device,
        enable_aug           = True,
        enable_color_aug     = True,
        enable_dihedral_aug  = True,
        max_augments         = 2,
        color_apply_to_test  = True,
        dihedral_apply_to_test = True,
    )


def eval_oracle(model, eval_dataset, eval_solutions, device, args) -> float:
    """Quick oracle ceiling: count tasks where correct answer appears in candidates."""
    import torch
    from rcos import generate_all_candidates, grids_equal
    from canonicalize import score_task_orientations
    from run_prototype_e import _grids_from_seq_ex
    from common import apply_dihedral_transform

    model.eval()
    task_ids = sorted({ex.task_id for ex in eval_dataset.iter_examples(split="test")})
    if args.eval_max_tasks:
        task_ids = task_ids[:args.eval_max_tasks]

    n_oracle = 0
    with torch.no_grad():
        for task_id in task_ids:
            demo_seq_exs = [ex for ex in eval_dataset.iter_examples(split="train")
                            if ex.task_id == task_id]
            test_seq_exs = [ex for ex in eval_dataset.iter_examples(split="test")
                            if ex.task_id == task_id]
            if not demo_seq_exs or not test_seq_exs:
                continue
            example_id = eval_dataset.task_id_to_example_id[task_id]
            try:
                best_d, _ = score_task_orientations(model, demo_seq_exs, device)
            except Exception:
                continue
            from rcos import GridPair
            demo_pairs = [_grids_from_seq_ex(ex, best_d) for ex in demo_seq_exs]
            demo_pairs = [gp for gp in demo_pairs if gp.output is not None]
            sol_grids  = eval_solutions.get(task_id, [])
            task_ok = True
            for pi, test_ex in enumerate(test_seq_exs):
                test_gp = _grids_from_seq_ex(test_ex, best_d)
                gt_orig = sol_grids[pi] if pi < len(sol_grids) else None
                if gt_orig is None:
                    continue
                gt = apply_dihedral_transform(gt_orig, best_d)
                try:
                    cands = generate_all_candidates(
                        model, demo_pairs, test_gp.input, example_id, best_d, device,
                        use_greedy=True,
                        use_beam=args.eval_beam > 0, beam_width=args.eval_beam,
                        use_sample=args.eval_n_sample > 0,
                        n_per_temperature=args.eval_n_sample,
                        temperatures=(0.7, 1.0), top_k=None,
                        test_seq_ex=test_ex, demo_seq_exs=demo_seq_exs,
                    )
                except Exception:
                    task_ok = False; break
                if not any(grids_equal(c, gt) for c in cands):
                    task_ok = False; break
            if task_ok:
                n_oracle += 1

    model.train()
    return n_oracle / max(len(task_ids), 1) * 100


def run_one_lr(
    lr: float,
    args,
    eval_solutions: Dict,
    out_dir: Path,
) -> Dict:
    """Train one Phase 2 run at a given LR, return summary dict."""
    import argparse as _ap, torch

    adamw_lr = lr * args.adamw_lr_ratio
    out_ckpt = out_dir / f"phase2_normuon{lr:.0e}.pt"

    print()
    print("=" * 70)
    print(f"Phase 2 — normuon_lr={lr:.0e}  adamw_lr={adamw_lr:.2e}")
    print(f"  epochs={args.epochs}  out={out_ckpt}")
    print("=" * 70)

    # Load Phase 1 checkpoint, then reset epoch/step counters so
    # train_model treats this as a fresh run (not "already past target").
    import copy as _copy
    phase1_ckpt_raw = load_checkpoint(Path(args.phase1_checkpoint))
    old_epoch = int(phase1_ckpt_raw.get("epoch", 0))
    phase1_ckpt = _copy.deepcopy(phase1_ckpt_raw)
    phase1_ckpt["epoch"] = 0
    phase1_ckpt["global_step"] = 0
    print(f"  (Phase 1 checkpoint was epoch={old_epoch}; "
          f"resetting counters for {args.epochs} fresh epochs)")

    # Build model on original 1307 tasks
    build_a = _build_args_for_data(args, Path(args.data_path),
                                   Path(args.phase1_checkpoint))
    model, dataset, dataloader, device, data_path = build_model_and_data(
        build_a, checkpoint=phase1_ckpt_raw, is_eval=False  # weights from phase1
    )
    print(f"  Model  : {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Tasks  : {len(dataset.task_ids)}")
    print(f"  Steps/epoch: {len(dataloader)}")

    train_a = _ap.Namespace(
        epochs                      = args.epochs,
        optimizer                   = "normuon",
        normuon_lr                  = lr,
        adamw_lr                    = adamw_lr,
        weight_decay                = args.weight_decay,
        attention_weight_decay      = args.weight_decay,
        token_embedding_weight_decay= 0.0,
        task_embedding_weight_decay = 0.0,
        grad_clip                   = args.grad_clip,
        gradient_accumulation_steps = 1,
        warmup_pct                  = args.warmup_pct,
        wsd_decay_start_pct         = args.wsd_decay_start_pct,
        lr_floor                    = args.lr_floor,
        save_path                   = out_ckpt,
        checkpoint_epochs           = None,
        do_validate                 = False,
        train_log_mode              = "10_steps",
        log_location                = "terminal",
        train_log_file              = None,
        batch_size                  = 1,
        val_batch_size              = 1,
    )

    t0 = perf_counter()
    train_model(
        args=train_a, model=model, dataloader=dataloader,
        dataset=dataset, device=device, data_path=data_path,
        checkpoint=phase1_ckpt,  # counters reset to 0
    )
    t_train = perf_counter() - t0
    print(f"  Training done in {t_train/60:.0f}m")

    # Oracle eval
    oracle_pct = None
    if not args.skip_eval and eval_solutions:
        print(f"  Evaluating oracle ceiling...")
        # Load eval dataset (is_eval=True to get test split)
        eval_build_a = _ap.Namespace(
            checkpoint_path      = out_ckpt,
            data_path            = Path(args.data_path),
            seed                 = args.seed,
            batch_size           = 1,
            device               = args.device,
            enable_aug           = False,
            enable_color_aug     = False,
            enable_dihedral_aug  = False,
            max_augments         = 0,
            color_apply_to_test  = False,
            dihedral_apply_to_test = False,
        )
        eval_ckpt = load_checkpoint(out_ckpt)
        _, eval_ds, _, eval_device, _ = build_model_and_data(
            eval_build_a, checkpoint=eval_ckpt, is_eval=True
        )
        # Load the fine-tuned model for eval
        eval_model, _, _, _, _ = build_model_and_data(
            eval_build_a, checkpoint=eval_ckpt, is_eval=True
        )
        t_eval0 = perf_counter()
        oracle_pct = eval_oracle(eval_model, eval_ds, eval_solutions, eval_device, args)
        t_eval = perf_counter() - t_eval0
        print(f"  Oracle : {oracle_pct:.1f}%  ({t_eval:.0f}s)")

    return {
        "normuon_lr":  lr,
        "adamw_lr":    adamw_lr,
        "epochs":      args.epochs,
        "oracle_pct":  oracle_pct,
        "train_time_m": round(t_train / 60, 1),
        "checkpoint":  str(out_ckpt),
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PHASE 2 — LR SWEEP on Original 1307 Tasks")
    print(f"  Phase 1 checkpoint : {args.phase1_checkpoint}")
    print(f"  Original data      : {args.data_path}")
    print(f"  NorMuon LRs to try : {args.normuon_lrs}")
    print(f"  Epochs per run     : {args.epochs}")
    print("=" * 70)

    # Load eval solutions once
    eval_solutions = {}
    if not args.skip_eval and Path(args.eval_solutions).exists():
        with open(args.eval_solutions) as f:
            raw = json.load(f)
        for tid, grids in raw.items():
            if grids and not isinstance(grids[0][0], list):
                grids = [grids]
            eval_solutions[tid] = grids
        print(f"Loaded solutions for {len(eval_solutions)} tasks.")

    summaries = []
    for lr in args.normuon_lrs:
        s = run_one_lr(lr, args, eval_solutions, out_dir)
        summaries.append(s)

    # Save comparison
    comp_path = out_dir / "comparison.json"
    with open(comp_path, "w") as f:
        json.dump(summaries, f, indent=2)

    # Print table
    print()
    print("=" * 70)
    print("PHASE 2 LR SWEEP RESULTS")
    print(f"  Reference: EXP C base model = 37.5% oracle, 30.50% ARC%")
    print("=" * 70)
    hdr = f"{'normuon_lr':>12}  {'adamw_lr':>10}  {'oracle':>8}  {'time':>6}"
    print(hdr)
    print("-" * 45)
    for s in summaries:
        oracle = f"{s['oracle_pct']:.1f}%" if s['oracle_pct'] is not None else "—"
        print(f"  {s['normuon_lr']:.0e}       {s['adamw_lr']:.2e}    {oracle:>8}  {s['train_time_m']:>5.0f}m")
    print("=" * 70)

    if summaries and any(s["oracle_pct"] for s in summaries):
        best = max((s for s in summaries if s["oracle_pct"]), key=lambda s: s["oracle_pct"])
        print(f"\nBest LR  : normuon={best['normuon_lr']:.0e}  oracle={best['oracle_pct']:.1f}%")
        print(f"Checkpoint: {best['checkpoint']}")
        print()
        print("Next — run full eval with TTT:")
        print(f"  python prototype_ttt/ablation_v9_ttt.py \\")
        print(f"      --checkpoint {best['checkpoint']} \\")
        print(f"      --data-path  assets/challenges.json \\")
        print(f"      --solutions  assets/solutions.json \\")
        print(f"      --output-dir runs/phase2_final_eval \\")
        print(f"      --exp K,M --ttt-lr 1e-5 --ttt-layers 4")

    print(f"\nComparison saved → {comp_path}")


if __name__ == "__main__":
    main()
