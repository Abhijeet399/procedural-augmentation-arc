"""
collect_embeddings.py — Extract converged per-task embeddings from a trained checkpoint.

This is Step 1 of the Prototype B pipeline.

After training Mithil's model (run_script.py), the example_embedding.weight
matrix contains one learned vector per task.  These vectors serve as
supervision targets for training the meta-encoder.

Usage
-----
    python collect_embeddings.py \\
        --checkpoint runs/tiny.pt \\
        --data-path  assets/challenges.json \\
        --output     runs/embeddings/task_embeddings.pt

Output format
-------------
A dict saved with torch.save:
{
  "embeddings": Tensor[n_train_tasks, d_model],   # converged embeddings
  "task_ids":   List[str],                         # ordering matches dim 0
  "example_ids": List[int],                        # example_id per task
  "d_model":    int,
  "checkpoint": str,
}

Notes on stability
------------------
Run training multiple times and average the embeddings to reduce variance
from random initialisation:

    python collect_embeddings.py --checkpoint runs/tiny.pt   --output runs/embs/run1.pt
    python collect_embeddings.py --checkpoint runs/tiny2.pt  --output runs/embs/run2.pt
    python average_embeddings.py runs/embs/run1.pt runs/embs/run2.pt \\
                                  --output runs/embs/averaged.pt

Then train the meta-encoder on the averaged embeddings.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Extract per-task embeddings from a trained mdlARC checkpoint."
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-path", type=Path, default=Path("assets/challenges.json"))
    p.add_argument("--output", type=Path, default=Path("runs/embeddings/task_embeddings.pt"))
    p.add_argument("--splits", nargs="+", default=["train"], choices=["train", "test"],
                   help="Which dataset splits to collect embeddings for.")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device for loading the model (cpu is fine, embeddings are small).")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    _repo_root = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(_repo_root / "src"))

    import argparse as _argparse
    from build import build_model_and_data, load_checkpoint

    cfg = _argparse.Namespace(
        data_path=args.data_path,
        checkpoint_path=args.checkpoint,
        d_model=768, n_heads=12, d_ff=3072, n_layers=8,
        dropout=0.0, attention_dropout=None,
        seed=42,
        enable_aug=False, max_augments=0,
        enable_color_aug=False, color_apply_to_test=False,
        enable_dihedral_aug=False, dihedral_apply_to_test=False,
        batch_size=1, device=args.device, eval_only=True,
    )

    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = load_checkpoint(args.checkpoint)

    # We only need the embedding table — no need for the full dataset/dataloader
    model, dataset, _, device, _ = build_model_and_data(
        cfg, checkpoint=checkpoint, is_eval=True
    )
    model.eval()

    # The example_embedding.weight[example_id] is the converged per-task embedding.
    emb_weight = model.example_embedding.weight.detach().cpu()  # [n_tasks, d_model]

    # Build mapping: task_id → example_id (from dataset)
    task_ids: List[str] = []
    example_ids: List[int] = []

    for task_id in dataset.task_ids:
        example_id = dataset.task_id_to_example_id[task_id]
        task_ids.append(task_id)
        example_ids.append(example_id)

    # Collect embeddings in task_id order
    emb_tensors = torch.stack([emb_weight[eid] for eid in example_ids], dim=0)
    # Shape: [n_tasks, d_model]

    d_model = int(emb_tensors.size(1))
    print(f"Collected {len(task_ids)} task embeddings | d_model={d_model}")

    # Compute embedding statistics for sanity check
    norms = emb_tensors.norm(dim=-1)
    print(f"Embedding norm: mean={norms.mean():.4f}  std={norms.std():.4f}  "
          f"min={norms.min():.4f}  max={norms.max():.4f}")

    # Save
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "embeddings": emb_tensors,
        "task_ids": task_ids,
        "example_ids": example_ids,
        "d_model": d_model,
        "checkpoint": str(args.checkpoint),
        "n_tasks": len(task_ids),
    }
    torch.save(payload, args.output)
    print(f"Saved embeddings to {args.output}")


# ---------------------------------------------------------------------------
# Bonus: average multiple embedding files (call from CLI)
# ---------------------------------------------------------------------------

def average_embeddings_cli() -> None:
    """
    python collect_embeddings.py --average file1.pt file2.pt ... --output averaged.pt
    """
    p = argparse.ArgumentParser(description="Average multiple embedding files.")
    p.add_argument("files", nargs="+", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    all_embs = []
    reference = None
    for f in args.files:
        data = torch.load(f, map_location="cpu", weights_only=False)
        if reference is None:
            reference = data
        all_embs.append(data["embeddings"])
        print(f"Loaded {f} | {data['n_tasks']} tasks")

    averaged = torch.stack(all_embs, dim=0).mean(dim=0)
    reference["embeddings"] = averaged
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(reference, args.output)
    print(f"Saved averaged embeddings ({len(args.files)} runs) to {args.output}")


if __name__ == "__main__":
    import sys as _sys
    if "--average" in _sys.argv:
        _sys.argv.remove("--average")
        average_embeddings_cli()
    else:
        main()
