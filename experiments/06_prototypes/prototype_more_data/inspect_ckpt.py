"""
inspect_ckpt.py — Print the top-level structure of a checkpoint.
Run from the repo root:
    python prototype_more_data/inspect_ckpt.py runs/tiny.pt
"""
import sys, torch
from pathlib import Path

path = sys.argv[1] if len(sys.argv) > 1 else "runs/tiny.pt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)

print(f"Checkpoint: {path}")
print(f"Type: {type(ckpt).__name__}")

if isinstance(ckpt, dict):
    print(f"\nTop-level keys ({len(ckpt)}):")
    for k, v in ckpt.items():
        if isinstance(v, dict):
            # Check if it looks like a state dict (values are tensors)
            sample = next(iter(v.values()), None)
            is_state = hasattr(sample, "shape") if sample is not None else False
            has_dihedral = "dihedral_embedding.weight" in v
            print(f"  '{k}': dict[{len(v)}]  state_dict={is_state}  has_dihedral={has_dihedral}")
            if is_state and has_dihedral:
                print(f"    *** THIS IS THE MODEL STATE DICT KEY ***")
        elif isinstance(v, torch.Tensor):
            print(f"  '{k}': Tensor {tuple(v.shape)}")
        else:
            print(f"  '{k}': {type(v).__name__} = {str(v)[:60]}")
elif isinstance(ckpt, (list, tuple)):
    print(f"Length: {len(ckpt)}")
    for i, v in enumerate(ckpt[:3]):
        print(f"  [{i}]: {type(v).__name__}")
