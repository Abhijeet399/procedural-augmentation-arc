"""
read_build.py — Print the training-relevant parts of mdlARC's build.py.
Run from the repo root:
    python prototype_more_data/read_build.py
"""
import sys
from pathlib import Path

build_path = Path("src/build.py")
if not build_path.exists():
    print("ERROR: src/build.py not found")
    sys.exit(1)

content = build_path.read_text()
lines = content.split('\n')

print(f"=== src/build.py ({len(lines)} lines) ===\n")

# Print function signatures and key lines
in_func = False
for i, line in enumerate(lines):
    stripped = line.strip()
    # Print all def lines
    if stripped.startswith('def ') or stripped.startswith('class '):
        print(f"{i+1:4d}: {line}")
        in_func = True
    # Print lines with key training concepts
    elif any(kw in line.lower() for kw in [
        'muon', 'normuon', 'optimizer', 'scheduler', 'lr', 'learning_rate',
        'task_id', 'embedding', 'is_eval', 'build_model', 'checkpoint',
        'train_script', 'train.py', 'main()'
    ]):
        print(f"{i+1:4d}: {line}")

print("\n\n=== First 80 lines of build.py ===")
for i, line in enumerate(lines[:80]):
    print(f"{i+1:4d}: {line}")

# Also check if there's a train.py
train_path = Path("train.py")
if train_path.exists():
    train_lines = train_path.read_text().split('\n')
    print(f"\n\n=== train.py exists! ({len(train_lines)} lines) ===")
    print("First 60 lines:")
    for i, line in enumerate(train_lines[:60]):
        print(f"{i+1:4d}: {line}")
else:
    print("\n\ntrain.py does NOT exist in the repo root")

# Check for any other training scripts
import os
for f in os.listdir('.'):
    if 'train' in f.lower() and f.endswith('.py'):
        print(f"Found: {f}")
for f in os.listdir('src') if os.path.exists('src') else []:
    if 'train' in f.lower() and f.endswith('.py'):
        print(f"Found: src/{f}")
