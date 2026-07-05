"""
read_train.py — Print src/train.py in full.
Run from the repo root:
    python prototype_more_data/read_train.py 2>&1 | head -300
"""
from pathlib import Path
content = Path("src/train.py").read_text()
print(content)
