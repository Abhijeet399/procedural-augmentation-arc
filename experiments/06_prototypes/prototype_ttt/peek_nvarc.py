"""
peek_nvarc.py — Deep format inspection of NVARC JSON files.
Run from the repo root:
    python prototype_ttt/peek_nvarc.py
"""
import json
from pathlib import Path

nvarc_root = Path("assets/nvarc_synthetic")

for split in ("nvarc_training", "nvarc_full"):
    split_dir = nvarc_root / split
    if not split_dir.exists():
        print(f"Skipping {split} (not found)")
        continue

    sample = next(split_dir.rglob("*.json"))
    data = json.load(open(sample))

    print(f"\n{'='*60}")
    print(f"SPLIT: {split}")
    print(f"File:  {sample.name}")
    print(f"Type:  {type(data).__name__}")

    if isinstance(data, list):
        print(f"List length: {len(data)}")
        elem = data[0]
        print(f"\nElement[0] type: {type(elem).__name__}")
        if isinstance(elem, dict):
            print(f"Element[0] keys: {list(elem.keys())}")
            for k, v in elem.items():
                if isinstance(v, list) and v:
                    inner = v[0]
                    print(f"\n  '{k}': list[{len(v)}]")
                    if isinstance(inner, dict):
                        print(f"    item keys: {list(inner.keys())}")
                        for sk, sv in inner.items():
                            if isinstance(sv, list):
                                if sv and isinstance(sv[0], list):
                                    print(f"      '{sk}' = 2D grid {len(sv)}×{len(sv[0])}, row[0]={sv[0][:8]}")
                                elif sv and isinstance(sv[0], (int, float)):
                                    print(f"      '{sk}' = flat list len={len(sv)}, first few={sv[:5]}")
                                else:
                                    print(f"      '{sk}' = list of {type(sv[0]).__name__}")
                            else:
                                print(f"      '{sk}' = {type(sv).__name__}: {sv!r}")
                    elif isinstance(inner, list):
                        print(f"    inner row: {inner[:8]}")
                    else:
                        print(f"    inner: {inner!r}")
                elif isinstance(v, list):
                    print(f"\n  '{k}': empty list")
                else:
                    print(f"\n  '{k}' = {type(v).__name__}: {v!r}")
        else:
            print(f"Element[0]: {str(elem)[:200]}")

    elif isinstance(data, dict):
        print(f"Top-level keys (first 5): {list(data.keys())[:5]}")
        first_val = next(iter(data.values()))
        print(f"First value type: {type(first_val).__name__}")
        if isinstance(first_val, dict):
            print(f"First value keys: {list(first_val.keys())}")

print("\n\n--- CHECKING: do any files have 'train'/'test' at top level? ---")
for split in ("nvarc_training", "nvarc_full"):
    split_dir = nvarc_root / split
    if not split_dir.exists():
        continue
    found = 0
    for f in list(split_dir.rglob("*.json"))[:20]:
        d = json.load(open(f))
        if isinstance(d, dict) and "train" in d:
            found += 1
    print(f"{split}: {found}/20 sampled files have 'train' key at top level")
