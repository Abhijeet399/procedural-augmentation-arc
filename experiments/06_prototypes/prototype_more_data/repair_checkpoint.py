"""
repair_checkpoint.py — Merge fine-tuned weights into original checkpoint structure.

Fixes a checkpoint saved with the wrong key ('model') by copying the weights
into a fresh copy of the base checkpoint under the correct key ('model_state').

Usage:
    python prototype_more_data/repair_checkpoint.py \
        --base    runs/tiny.pt \
        --finetuned runs/q_a_finetuned.pt \
        --out     runs/q_a_finetuned_fixed.pt
"""
import sys, torch, copy, argparse
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--base",      required=True, help="Original base checkpoint (tiny.pt)")
    p.add_argument("--finetuned", required=True, help="Broken fine-tuned checkpoint")
    p.add_argument("--out",       required=True, help="Output path for repaired checkpoint")
    return p.parse_args()

def main():
    args = parse_args()
    print(f"Loading base      : {args.base}")
    base = torch.load(args.base, map_location="cpu", weights_only=False)
    print(f"Loading fine-tuned: {args.finetuned}")
    ft   = torch.load(args.finetuned, map_location="cpu", weights_only=False)

    # Find the model weights in the fine-tuned checkpoint
    model_state = None
    for key in ("model_state", "model", "model_state_dict"):
        if key in ft and isinstance(ft[key], dict):
            sample = next(iter(ft[key].values()), None)
            if hasattr(sample, "shape"):
                model_state = ft[key]
                print(f"  Found weights under key '{key}' ({len(model_state)} tensors)")
                if "dihedral_embedding.weight" in model_state:
                    print(f"  ✓ dihedral_embedding.weight present")
                else:
                    print(f"  ✗ dihedral_embedding.weight MISSING — weights may be corrupt")
                break

    if model_state is None:
        print("ERROR: could not find model weights in fine-tuned checkpoint")
        sys.exit(1)

    # Build repaired checkpoint: base structure + fine-tuned weights
    repaired = copy.deepcopy(base)
    repaired["model_state"] = copy.deepcopy(model_state)
    repaired["finetune_metadata"] = ft.get("finetune_metadata", {})

    print(f"\nSaving repaired checkpoint → {args.out}")
    torch.save(repaired, args.out)
    size = Path(args.out).stat().st_size / 1e6
    print(f"  Size: {size:.1f} MB")
    print(f"\nVerify with:")
    print(f"  python prototype_more_data/inspect_ckpt.py {args.out}")

if __name__ == "__main__":
    main()
