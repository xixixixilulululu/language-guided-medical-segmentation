# language-guided-medical-segmentation

Research project on language-guided medical image segmentation. The work is organised in phases; the
current phase is **Phase 0 — image-only segmentation benchmark** (U-Net, TransUNet, nnU-Net 2D on
QaTa-COV19 and MosMedData+, LViT split, seeds 1001/1002/1003). Plan: [`reports/PHASE0_PLAN.md`](reports/PHASE0_PLAN.md).

## Status

| Task | Status | Output |
|---|---|---|
| 0.1 Project and data | done | [`reports/PHASE0_DATA_SUMMARY.md`](reports/PHASE0_DATA_SUMMARY.md), `results/phase0_data_check.json`, `reports/figures/data_examples.png` |
| 0.2 Common 224×224 data preparation | done | [`reports/PHASE0_MODEL_SETUP.md`](reports/PHASE0_MODEL_SETUP.md) §1, `results/phase0_common224_summary.json` |
| 0.3 Model setup (U-Net, TransUNet, nnU-Net) | done, quick checks passed | [`reports/PHASE0_MODEL_SETUP.md`](reports/PHASE0_MODEL_SETUP.md), `configs/*.yaml`, `results/phase0_quick_check.json`, `results/phase0_metric_check.txt` |
| 0.4 18 main runs | running (Slurm jobs 212–229, submitted 2026-10-11) | `runs/` (not committed) |
| 0.5–0.7 Evaluation, summary, pairwise comparison | analysis plan fixed in code, tested on synthetic / validation data; to be run once after the 18 runs | `src/analyze_phase0.py`, `src/test_analyze_phase0.py`, `reports/templates/PHASE0_RESULTS_methods.md` |

## Layout

| Folder | Contents |
|---|---|
| `external/` | LViT and TransUNet clones (not committed; pinned commits in [`external/VERSIONS.md`](external/VERSIONS.md)) |
| `src/` | project code: data preparation, training scripts, common evaluator (`src/lgms/`, `src/evaluate.py`), nnU-Net trainer (`src/nnunet_ext/`), notifications |
| `slurm/` | Slurm job files (all training runs on the `gpu` partition) |
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

## Phone notifications

Slurm jobs and Claude Code send short ntfy notifications through `src/notify.sh`. The channel name is
read from `~/.ntfy_topic`, which is not part of the repository; see `CLAUDE.md`.

## Data

```bash
python src/prepare_lvit_datasets.py --raw ~/datasets/raw --out ~/datasets   # LViT layout from the Kaggle archives
python src/inspect_datasets.py --data ~/datasets                            # format check + example figure
```
See [`reports/PHASE0_DATA_SUMMARY.md`](reports/PHASE0_DATA_SUMMARY.md) for sources and download commands.
