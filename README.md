# language-guided-medical-segmentation

Research project on language-guided medical image segmentation. The work is organised in phases; the
current phase is **Phase 0 — image-only segmentation benchmark** (U-Net, TransUNet, nnU-Net 2D on
QaTa-COV19 and MosMedData+, LViT split, seeds 1001/1002/1003). Plan: [`reports/PHASE0_PLAN.md`](reports/PHASE0_PLAN.md).

## Status

| Task | Status | Output |
|---|---|---|
| 0.1 Project and data | done | [`reports/PHASE0_DATA_SUMMARY.md`](reports/PHASE0_DATA_SUMMARY.md), `results/phase0_data_check.json`, `reports/figures/data_examples.png` |
| 0.2 Common 224×224 data preparation | planned | |
| 0.3 Model setup (U-Net, TransUNet, nnU-Net) | planned | |
| 0.4–0.7 18 runs, evaluation, comparison | not started | |

## Layout

| Folder | Contents |
|---|---|
| `external/` | LViT and TransUNet clones (not committed; pinned commits in [`external/VERSIONS.md`](external/VERSIONS.md)) |
| `src/` | project code (data preparation, checks, later training / evaluation) |
| `configs/` | per-method configuration files |
| `data/` | local data links only — datasets live in `~/datasets/` and are never committed |
| `runs/` | training runs (logs, checkpoints, predictions; large files are not committed) |
| `results/` | small result tables / JSON |
| `reports/` | plans, reports and figures |

## Environment

- Cluster: DGX Spark, Slurm partition `gpu` (NVIDIA GB10)
- Conda environment `~/envs/lgms` (cloned from the research_training environment; `torch 2.14.0+cu130`),
  Jupyter kernel `lgms`
- Datasets: `~/datasets/QaTa-COV19/`, `~/datasets/MosMedData+/` (LViT layout), raw archives in `~/datasets/raw/`

## Data

```bash
python src/prepare_lvit_datasets.py --raw ~/datasets/raw --out ~/datasets   # LViT layout from the Kaggle archives
python src/inspect_datasets.py --data ~/datasets                            # format check + example figure
```
See [`reports/PHASE0_DATA_SUMMARY.md`](reports/PHASE0_DATA_SUMMARY.md) for sources and download commands.
