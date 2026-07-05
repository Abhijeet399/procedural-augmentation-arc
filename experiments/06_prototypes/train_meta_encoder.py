"""
train_meta_encoder.py — Train the MetaEncoder on converged per-task embeddings.

This is Step 2 of the Prototype B pipeline.

Pipeline
--------
1. Load converged embeddings (from collect_embeddings.py).
2. Load main model checkpoint (frozen — used only to read tokenised demo pairs).
3. Build MetaEncoder.
4. Train with:
       L = MSE(e_hat, e_target) + λ * L_aug_consistency
   where:
       e_hat          = meta_encoder(demo_pairs_of_task)
       e_target       = converged per-task embedding
       L_aug_consist. = ||meta_enc(aug_view_a) - meta_enc(aug_view_b)||^2
5. Save the trained MetaEncoder checkpoint.

Usage
-----
    python train_meta_encoder.py \\
        --embeddings   runs/embeddings/task_embeddings.pt \\
        --checkpoint   runs/tiny.pt \\
        --data-path    assets/challenges.json \\
        --output       runs/meta_encoder/meta_enc.pt \\
        --epochs       40 \\
        --lr           3e-4 \\
        --lambda-aug   0.1 \\
        --val-fraction 0.15

Training set / validation split
---------------------------------
--val-fraction (default 0.15): hold out this fraction of training tasks to
validate meta-encoder generalisation.  Report MSE on held-out tasks each epoch.
This is the most important diagnostic — if held-out MSE is much worse than
train MSE, the meta-encoder is memorising task IDs rather than learning to
encode rules from demo pairs.
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the LCTE MetaEncoder.")
    p.add_argument("--embeddings", type=Path, required=True,
                   help="Path to task_embeddings.pt from collect_embeddings.py")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Main model checkpoint (to load dataset / tokenised sequences)")
    p.add_argument("--data-path", type=Path, default=Path("assets/challenges.json"))
    p.add_argument("--output", type=Path, default=Path("runs/meta_encoder/meta_enc.pt"))
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--lambda-aug", type=float, default=0.1,
                   help="Weight for augmentation-consistency loss")
    p.add_argument("--batch-size", type=int, default=32,
                   help="Number of tasks per training step")
    p.add_argument("--val-fraction", type=float, default=0.15,
                   help="Fraction of tasks held out for validation")
    p.add_argument("--meta-d-model", type=int, default=256,
                   help="Internal dimension of the meta-encoder")
    p.add_argument("--meta-n-layers", type=int, default=2)
    p.add_argument("--meta-n-heads", type=int, default=4)
    p.add_argument("--meta-d-ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--log-every", type=int, default=5,
                   help="Print loss every N epochs")
    p.add_argument("--copy-token-embedding", action="store_true", default=True,
                   help="Initialise meta-encoder token embedding from main model")
    p.add_argument("--freeze-token-embedding", action="store_true", default=False,
                   help="Freeze meta-encoder token embedding during training")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _get_demo_sequences(
    dataset: "ARCExampleDataset",  # noqa: F821
    task_id: str,
    dihedral: Optional[int] = None,
) -> List[torch.Tensor]:
    """Return token tensors for all demo pairs of task_id at the given dihedral."""
    seqs = []
    for ex in dataset.examples:
        if ex.task_id != task_id or ex.split != "train" or not ex.has_output:
            continue
        if dihedral is not None and ex.tokens_by_dihedral is not None:
            seqs.append(ex.tokens_by_dihedral[dihedral])
        else:
            seqs.append(ex.tokens)
    return seqs


def _sample_aug_pair(
    dataset: "ARCExampleDataset",  # noqa: F821
    task_id: str,
    rng: random.Random,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Draw two independently augmented views of the same task's demo pairs.
    Each view uses a randomly chosen dihedral index.
    """
    d_a = rng.randint(0, 7)
    d_b = rng.randint(0, 7)
    # Ensure the two views are different (unless only 1 augmentation is possible)
    tries = 0
    while d_b == d_a and tries < 10:
        d_b = rng.randint(0, 7)
        tries += 1
    seqs_a = _get_demo_sequences(dataset, task_id, dihedral=d_a)
    seqs_b = _get_demo_sequences(dataset, task_id, dihedral=d_b)
    return seqs_a, seqs_b


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_meta_encoder(args: argparse.Namespace) -> None:
    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root / "src"))

    import argparse as _argparse
    from build import build_model_and_data, load_checkpoint
    from meta_encoder import (
        AugConsistencyLoss,
        MetaEncoder,
        MetaEncoderConfig,
        build_meta_encoder_from_main_model,
    )

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    rng = random.Random(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # -----------------------------------------------------------------------
    # Load converged embeddings (supervision targets)
    # -----------------------------------------------------------------------
    print(f"\nLoading converged embeddings from {args.embeddings} …")
    emb_data = torch.load(args.embeddings, map_location="cpu", weights_only=False)
    target_embeddings: torch.Tensor = emb_data["embeddings"]  # [N, d_model]
    task_ids_all: List[str] = emb_data["task_ids"]
    task_id_to_idx: Dict[str, int] = {tid: i for i, tid in enumerate(task_ids_all)}
    target_d_model: int = int(emb_data["d_model"])
    n_tasks_total = len(task_ids_all)
    print(f"  {n_tasks_total} tasks | d_model={target_d_model}")

    # -----------------------------------------------------------------------
    # Load main model (frozen, used only to access dataset)
    # -----------------------------------------------------------------------
    print(f"\nLoading main model from {args.checkpoint} …")
    cfg = _argparse.Namespace(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        d_model=target_d_model, n_heads=12,
        d_ff=target_d_model * 4, n_layers=8,
        dropout=0.0, attention_dropout=None, seed=args.seed,
        enable_aug=False, max_augments=0,
        enable_color_aug=False, color_apply_to_test=False,
        enable_dihedral_aug=False, dihedral_apply_to_test=False,
        batch_size=1, device=str(device), eval_only=True,
    )
    checkpoint = load_checkpoint(args.checkpoint)
    main_model, dataset, _, _, _ = build_model_and_data(cfg, checkpoint=checkpoint, is_eval=True)
    main_model.eval()
    # We won't use main_model's forward during training — only the dataset.
    del checkpoint

    # -----------------------------------------------------------------------
    # Train / validation split
    # -----------------------------------------------------------------------
    # Only keep task IDs that appear in the dataset AND have converged embeddings
    common_tasks = [t for t in dataset.task_ids if t in task_id_to_idx]
    # Hold out val_fraction for generalisation check
    n_val = max(1, int(len(common_tasks) * args.val_fraction))
    rng.shuffle(common_tasks)
    val_task_ids = set(common_tasks[:n_val])
    train_task_ids = [t for t in common_tasks if t not in val_task_ids]

    print(f"\nTrain tasks: {len(train_task_ids)} | Val tasks: {len(val_task_ids)}")

    # -----------------------------------------------------------------------
    # Build MetaEncoder
    # -----------------------------------------------------------------------
    meta_cfg = MetaEncoderConfig(
        d_model=args.meta_d_model,
        target_d_model=target_d_model,
        n_heads=args.meta_n_heads,
        n_layers=args.meta_n_layers,
        d_ff=args.meta_d_ff,
        dropout=args.dropout,
    )
    if args.copy_token_embedding:
        meta_encoder = build_meta_encoder_from_main_model(main_model, meta_cfg)
    else:
        meta_encoder = MetaEncoder(meta_cfg)

    meta_encoder = meta_encoder.to(device)

    if args.freeze_token_embedding:
        meta_encoder.token_embedding.requires_grad_(False)
        print("MetaEncoder token embedding: FROZEN")

    n_params = sum(p.numel() for p in meta_encoder.parameters() if p.requires_grad)
    print(f"MetaEncoder trainable parameters: {n_params:,}")

    # -----------------------------------------------------------------------
    # Optimizer
    # -----------------------------------------------------------------------
    optimizer = AdamW(
        [p for p in meta_encoder.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    consistency_loss_fn = AugConsistencyLoss(reduction="mean")
    use_amp = device.type == "cuda"

    # Pre-cache demo sequences (saves re-fetching every batch)
    print("\nPre-caching demo sequences …")
    demo_seqs_by_task: Dict[str, Dict[int, List[torch.Tensor]]] = {}
    for task_id in common_tasks:
        demo_seqs_by_task[task_id] = {}
        for d in range(8):
            seqs = _get_demo_sequences(dataset, task_id, dihedral=d)
            demo_seqs_by_task[task_id][d] = seqs

    print(f"Demo sequence cache built for {len(demo_seqs_by_task)} tasks.")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_mse = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        meta_encoder.train()
        rng.shuffle(train_task_ids)

        epoch_mse = 0.0
        epoch_aug = 0.0
        n_batches = 0

        for batch_start in range(0, len(train_task_ids), args.batch_size):
            batch_tasks = train_task_ids[batch_start: batch_start + args.batch_size]
            if not batch_tasks:
                continue

            mse_losses: List[torch.Tensor] = []
            aug_losses: List[torch.Tensor] = []

            for task_id in batch_tasks:
                emb_idx = task_id_to_idx[task_id]
                target = target_embeddings[emb_idx].to(device)  # [d_model]

                # Forward pass on identity orientation (d=0)
                seqs_d0 = [s.to(device) for s in demo_seqs_by_task[task_id][0]]
                if not seqs_d0:
                    continue

                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                ):
                    e_hat = meta_encoder.forward(seqs_d0)

                mse_losses.append(F.mse_loss(e_hat.float(), target.float()))

                # Augmentation-consistency loss: two random orientations
                if args.lambda_aug > 0:
                    d_a = rng.randint(0, 7)
                    d_b = rng.randint(0, 7)
                    while d_b == d_a:
                        d_b = rng.randint(0, 7)
                    seqs_a = [s.to(device) for s in demo_seqs_by_task[task_id][d_a]]
                    seqs_b = [s.to(device) for s in demo_seqs_by_task[task_id][d_b]]
                    if seqs_a and seqs_b:
                        with torch.autocast(
                            device_type=device.type, dtype=torch.bfloat16, enabled=use_amp
                        ):
                            e_a = meta_encoder.forward(seqs_a)
                            e_b = meta_encoder.forward(seqs_b)
                        aug_losses.append(
                            consistency_loss_fn(
                                e_a.float().unsqueeze(0), e_b.float().unsqueeze(0)
                            )
                        )

            if not mse_losses:
                continue

            mse_loss = torch.stack(mse_losses).mean()
            total_loss = mse_loss
            if aug_losses:
                aug_loss = torch.stack(aug_losses).mean()
                total_loss = mse_loss + args.lambda_aug * aug_loss
                epoch_aug += aug_loss.item()
            else:
                aug_loss = torch.tensor(0.0)

            optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            nn.utils.clip_grad_norm_(meta_encoder.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_loss.item()
            n_batches += 1

        scheduler.step()

        if epoch % args.log_every == 0 or epoch == 1 or epoch == args.epochs:
            avg_mse = epoch_mse / max(1, n_batches)
            avg_aug = epoch_aug / max(1, n_batches)

            # Validation MSE
            meta_encoder.eval()
            val_mse = 0.0
            n_val_tasks = 0
            with torch.no_grad():
                for task_id in val_task_ids:
                    emb_idx = task_id_to_idx.get(task_id)
                    if emb_idx is None:
                        continue
                    target = target_embeddings[emb_idx].to(device)
                    seqs = [s.to(device) for s in demo_seqs_by_task.get(task_id, {}).get(0, [])]
                    if not seqs:
                        continue
                    e_hat = meta_encoder.forward(seqs).float()
                    val_mse += F.mse_loss(e_hat, target.float()).item()
                    n_val_tasks += 1

            avg_val_mse = val_mse / max(1, n_val_tasks)
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{args.epochs} | "
                f"lr={lr_now:.2e} | "
                f"train MSE={avg_mse:.4f} | "
                f"aug loss={avg_aug:.4f} | "
                f"val MSE={avg_val_mse:.4f}"
            )

            if avg_val_mse < best_val_mse:
                best_val_mse = avg_val_mse
                best_epoch = epoch
                # Save best checkpoint
                args.output.parent.mkdir(parents=True, exist_ok=True)
                torch.save({
                    "meta_encoder_state": meta_encoder.state_dict(),
                    "meta_config": {
                        "d_model": meta_cfg.d_model,
                        "target_d_model": meta_cfg.target_d_model,
                        "n_heads": meta_cfg.n_heads,
                        "n_layers": meta_cfg.n_layers,
                        "d_ff": meta_cfg.d_ff,
                        "dropout": meta_cfg.dropout,
                        "lambda_consistency": meta_cfg.lambda_consistency,
                    },
                    "epoch": epoch,
                    "val_mse": avg_val_mse,
                    "train_mse": avg_mse,
                    "main_checkpoint": str(args.checkpoint),
                }, args.output)
                print(f"  ✓ New best val MSE={avg_val_mse:.4f} — saved to {args.output}")

    print(f"\nTraining done. Best val MSE={best_val_mse:.4f} at epoch {best_epoch}.")
    print(f"Meta-encoder checkpoint: {args.output}")


if __name__ == "__main__":
    args = parse_args()
    train_meta_encoder(args)
