# language-guided-medical-segmentation

Research project on language-guided medical image segmentation. The work is organised in phases.
Phase 0 — the **image-only segmentation benchmark** — is **complete** (U-Net, TransUNet, nnU-Net 2D on
QaTa-COV19 and MosMedData+, LViT split, seeds 1001/1002/1003). Plan: [`reports/PHASE0_PLAN.md`](reports/PHASE0_PLAN.md).
Phase 1 — **reproduction of three language-guided methods** (LViT-T, RecLMIS, DD-CMD) on the same data,
split, evaluator and seeds, and comparison with the Phase 0 baselines — is **complete**. Plan:
[`reports/PHASE1_PLAN.md`](reports/PHASE1_PLAN.md); summary: **[`reports/PHASE1_SUMMARY.md`](reports/PHASE1_SUMMARY.md)**.
Phase 2 — the **counterfactual text reliability benchmark** — has its **validation stage complete**; the
test stage waits for the Phase 3 plan and the freeze. Plan: [`reports/PHASE2_PLAN.md`](reports/PHASE2_PLAN.md);
validation summary: **[`reports/PHASE2_VALIDATION_SUMMARY.md`](reports/PHASE2_VALIDATION_SUMMARY.md)**.

## Status

| Task | Status | Output |
|---|---|---|
| 0.1 Project and data | done | [`reports/PHASE0_DATA_SUMMARY.md`](reports/PHASE0_DATA_SUMMARY.md), `results/phase0_data_check.json`, `reports/figures/data_examples.png` |
| 0.2 Common 224×224 data preparation | done | [`reports/PHASE0_MODEL_SETUP.md`](reports/PHASE0_MODEL_SETUP.md) §1, `results/phase0_common224_summary.json` |
| 0.3 Model setup (U-Net, TransUNet, nnU-Net) | done, quick checks passed | [`reports/PHASE0_MODEL_SETUP.md`](reports/PHASE0_MODEL_SETUP.md), `configs/*.yaml`, `results/phase0_quick_check.json`, `results/phase0_metric_check.txt` |
| 0.4 18 main runs | done, 18/18 complete (submitted by hand, one job at a time, under the cluster fair-use policy) | [`reports/PHASE0_SUBMISSION_PLAN.md`](reports/PHASE0_SUBMISSION_PLAN.md), `slurm/submit_next.sh`, `runs/` (not committed) |
| 0.5–0.7 Evaluation, summary, pairwise comparison | done; fixed analysis (commit 6ed8aa7) run once, conclusions final | **[`reports/PHASE0_SUMMARY.md`](reports/PHASE0_SUMMARY.md)** (short summary), **[`reports/PHASE0_RESULTS.md`](reports/PHASE0_RESULTS.md)** (full results), `src/analyze_phase0.py`, `results/phase0_*.csv` |
| 1.1–1.6 Setup of LViT-T, RecLMIS, DD-CMD | done, quick checks passed, configurations frozen | [`reports/PHASE1_SETUP_PLAN.md`](reports/PHASE1_SETUP_PLAN.md), [`reports/PHASE1_MODEL_SETUP.md`](reports/PHASE1_MODEL_SETUP.md), `configs/{lvit,reclmis,ddcmd}.yaml`, `configs/phase1_frozen/`, `results/phase1_quick_check.json` |
| 1.7 18 main runs | done, 18/18 complete (submitted by hand, one job at a time); test evaluations logged | `slurm/submit_next_p1.sh`, `results/phase1_test_eval_log.tsv`, `runs/<method>/<dataset>/seed<k>/` (per-case results committed; weights and predictions not) |
| 1.8–1.11 Six-method results, pairwise comparison, summary | done; fixed analysis (commit ba53ba5) run once, results re-computed independently, conclusions final | **[`reports/PHASE1_SUMMARY.md`](reports/PHASE1_SUMMARY.md)** (short summary), **[`reports/PHASE1_RESULTS.md`](reports/PHASE1_RESULTS.md)** (full results), `src/analyze_phase1.py`, `results/final_six_method_*.csv`, `results/phase1_main_results.csv` |
| 2.0 Setup checks and plan | done | [`reports/PHASE2_SETUP_PLAN.md`](reports/PHASE2_SETUP_PLAN.md), `results/phase2_checkpoint_sha256.tsv` |
| 2.1–2.2 Text attributes, nine conditions, Phase 2 lock | done on train/val; protocol draft, not frozen | [`reports/PHASE2_PROTOCOL.md`](reports/PHASE2_PROTOCOL.md), `src/phase2/{texts,conditions,lock}.py` |
| 2.3 Text-path check and validation benchmark | done (jobs 368, 369, 370; 18 checkpoints, 252 condition passes) | `src/phase2/infer.py`, `results/phase2_textpath_summary.csv`, `results/phase2_val_per_case.csv.gz` |
| 2.4–2.7 Validation analysis and Phase 3 rule | done on validation; summary for the supervisor | **[`reports/PHASE2_VALIDATION_SUMMARY.md`](reports/PHASE2_VALIDATION_SUMMARY.md)**, `src/phase2/analyze_phase2*.py`, `results/phase2_*_val.csv` |
| 2.8 Freeze and test evaluation | waiting for the Phase 3 plan (Phase 2 lock closed) | – |

## Layout

| Folder | Contents |
|---|---|
| `external/` | LViT, TransUNet, RecLMIS and DD-CMD clones (not committed; pinned commits in [`external/VERSIONS.md`](external/VERSIONS.md)) |
| `src/` | project code: data preparation, training scripts, common evaluator (`src/lgms/`, `src/evaluate.py`), nnU-Net trainer (`src/nnunet_ext/`), notifications |
| `slurm/` | Slurm job files (all training runs on the `gpu` partition) |
| `configs/` | per-method configuration files |
| `data/` | local data links only — datasets live in `~/datasets/` and are never committed |
| `runs/` | training runs (logs, checkpoints, predictions; large files are not committed) |
| `results/` | small result tables / JSON |
| `reports/` | plans, reports and figures |

## Environment

- Cluster: DGX Spark, Slurm partition `gpu` (2 × NVIDIA GB10). Fair-use policy: max 1 running + 1 waiting
  job per user, 12 h per job, manual submission (`slurm/submit_next.sh`; see CLAUDE.md)
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
