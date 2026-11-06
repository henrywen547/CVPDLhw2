# CVPDL 2026 HW2

This repository tracks the official starter and validation experiments. The original assignment files, `data/`, and `source_resnet50_160.pth` are one directory above this repository. Do not read `data/` directly; use the official `starter/run.py` and `starter/harness.py`.

## Environment

- Host: Ubuntu 24.04, Python 3.12.3, NVIDIA GeForce GTX 1080 Ti (11 GiB), driver 580.178.04.
- Runtime: PyTorch 2.6.0+cu124, torchvision 0.21.0+cu124, NumPy 2.4.4, pandas 2.3.3, Pillow 11.3.0. CUDA is available, GPU compute capability 6.1.
- The local virtual environment is `../.venv`, as requested. It currently uses a `.pth` link to packages already installed at `/home/weal222c/71hw/part1/myvenv/lib/python3.12/site-packages`. This is local reuse, not a portable installation. For a new machine, install the versions in `requirements.txt` from a compatible PyTorch CUDA wheel index and verify `torch.cuda.is_available()`.
- Re-run timing and VRAM checks on an RTX 3090 before submission. The present GPU is different from the grading GPU.

## Baseline

From this repository root:

```bash
cd starter
../../.venv/bin/python run.py --method source --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/source_val --seed 0
```

`source` runs in evaluation mode and uses `torch.no_grad()`. The official harness writes `val_pred.csv`, `val_run_log.json`, and prints validation accuracy. The run uses the official batch size of 64 and ImageNet mean/std normalization. `harness.py` remains unchanged.

On the GTX 1080 Ti, the baseline achieved **0.5454545 Top-1 (1,152/2,112)** with 2.7 seconds of harness stream time and 706 MiB sampled peak VRAM. These values came from starter commit `0b26fb09acd2749098eb20222d0cc947f9e1034b`. The supplied checkpoint SHA-256 is `8ec867598014a03c00237f61145d311ff245d8abeb650c06e070866efa48363f`.

## Validation diagnostics

The optional `source_diag` method receives the same normalized batches from the official harness and writes prediction and image summaries without labels or IDs. It is for validation exploration only; it adds logging overhead and should not be used for final timing or test submission.

```bash
cd starter
CVPDL_DIAG_JSONL=../runs/source_diag/image_summaries.jsonl ../../.venv/bin/python run.py --method source_diag --split val --data ../../data --ckpt ../../source_resnet50_160.pth --out ../runs/source_diag --seed 0
cd ..
../.venv/bin/python scripts/analyze_diagnostics.py --diagnostics runs/source_diag/image_summaries.jsonl --baseline-pred runs/source_val/val_pred.csv --diagnostic-pred runs/source_diag/val_pred.csv --out runs/source_diag/summary.json
```

The analysis verifies that diagnostic predictions exactly match the baseline. The findings and limitations are in [docs/data_exploration.md](docs/data_exploration.md).

## Rules

Only the supplied checkpoint may be used. Adaptation may only use the current and previously served unlabeled batches. Validation labels are reserved for official evaluation and method selection. Never modify the harness, manually read `data/`, or view test labels.
