# Notice

This repository contains two categories of code:

**Base training/inference pipeline** — `src/build.py`, `src/common.py`, `src/evaluate.py`,
`src/tinytransformer.py`, `src/train.py`, `src/utils.py`, and `dataset_building_scripts/`.
This is taken from [mdlARC](https://github.com/mvakde/mdlARC) by Mithil Vakde, used here
under the MIT license reproduced in `LICENSE`. It provides the transformer architecture,
augmentation-aware dataset builder, training loop, and evaluation harness that the rest of
this repository builds on.

**Original contributions** — everything in `experiments/`, `results/`, `models/`, and
`src/canonicalize.py`, `src/rcos.py`, `src/lora_ttt.py`, `src/meta_encoder.py`, plus
`prepare_dataset_rearc.py`. This is the procedural-augmentation research program described
in the top-level README: the re-ARC generator swap, the cap=300 ablation, the
epoch-schedule extension, the data-leakage audit and correction, and the candidate-rescoring
/ test-time-training / meta-encoder prototypes.

If you use the base pipeline itself, please cite mdlARC directly. If you use the
procedural-augmentation methodology, results, or the additional modules listed above,
please cite this repository — see `CITATION.cff`.
