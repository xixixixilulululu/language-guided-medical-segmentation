## Methods (fixed analysis plan)

The analysis follows `reports/PHASE0_PLAN.md`, Tasks 0.5–0.7, and adds or changes no method. The choices
below were fixed on 2026-10-12, before any new test-set result was summarised. They are implemented in
`src/analyze_phase0.py`, and the commit that fixed them is named at the top of this report.

**Task 0.5: common evaluation.** Every run is scored by the same evaluator (`src/lgms/metrics.py`):
- 224×224 space, lesion probability > 0.5;
- Dice and per-case IoU (mIoU = mean over cases);
- HD95 and ASSD in pixels (medpy 0.5.2);
- an empty prediction counts as Dice = IoU = 0 and HD95 = ASSD = 316.78 px.

Checkpoints are selected on the validation set only (PHASE0_MODEL_SETUP.md).

**Main table (Task 0.6).**
- **Mean ± SD.** Each seed gives one value per metric: the test-set mean over cases of Dice, mIoU, HD95
  or ASSD. "Mean ± SD" is the mean and the **sample** SD (ddof = 1) of the 3 per-seed values. This is
  done for all four metrics.
- **Dice 95% CI**, a per-case bootstrap:
  1. Each test case's Dice is averaged over the 3 seeds.
  2. The cases are resampled with replacement 10,000 times, and the mean is taken each time.
  3. The CI is the 2.5% / 97.5% percentile interval of these means.
  4. The random generator is numpy `default_rng(0)`, created anew for every dataset × method.

**Pairwise comparisons (Task 0.7)**, within each dataset, for U-Net vs TransUNet, U-Net vs nnU-Net and
TransUNet vs nnU-Net:
1. Each test case's Dice is averaged over the 3 seeds within each method.
2. The same test cases are aligned.
3. The paired differences d = Dice_A − Dice_B are computed.
4. ΔDice = mean(d).

- **Primary analysis:** two-sided paired t-test, with the 95% CI of the mean difference from the t
  distribution (mean(d) ± t(0.975, n−1)·SD(d)/√n). Holm correction is applied across the 3 comparisons
  of each dataset.
- **Supplementary sensitivity analysis** (specified in advance, not used for the main conclusions):
  two-sided Wilcoxon signed-rank test on the same paired differences (scipy defaults), with its own Holm
  correction within each dataset.
- Significance is never tested on the three seed means alone.

**Case figures.** Cases are chosen by rule, never by hand:
1. Each test case's Dice is averaged over the 3 seeds and then over the 3 methods.
2. The cases are sorted by this score (ties broken by sample id).
3. The cases at the 10/30/50/70/90% quantiles are shown (position round(q·(n−1))).

Each figure shows the **seed-1001 predictions**, and every panel's Dice / HD95 label is also the
**seed-1001** value for that case.

**Limitation.** The case bootstrap and the paired tests treat every test image as an independent sample.
In MosMedData+, neighbouring slices of the same CT volume are correlated. The MosMedData+ confidence
intervals may therefore be too narrow and the p-values too small.

### Test-set scores seen before the analysis was fixed

All test-set scores that were seen before the analysis plan was fixed are listed below. Times are
America/New_York. The list covers every test-set summary written by any Phase 0 job up to the fixing
commit.

| Model | Dataset | Seed | Test Dice (%) | Written | Seen | Purpose |
|---|---|---|---:|---|---|---|
| U-Net, main run (job 212) | QaTa-COV19 | 1001 | 79.12 (mIoU 69.68, HD95 30.69, ASSD 9.63) | 2026-10-11 20:06 | 2026-10-12 00:28 | agreed pipeline check against LViT Table II |
| U-Net, main run (job 213) | MosMedData+ | 1001 | 73.92 (mIoU 60.86, HD95 24.29, ASSD 8.97) | 2026-10-11 18:22 | 2026-10-12 00:28 | agreed pipeline check against LViT Table II |
| U-Net, main run | QaTa-COV19, MosMedData+ | 1001 | the same predictions re-scored with the LViT algorithm, pooled Dice and 512×512 Dice | — | 2026-10-12 11:16 | investigation of the MosMed gap (PHASE0_MODEL_SETUP.md §7) |
| U-Net, quick-check subset model (batch 2, 32 training images, job 206) | QaTa-COV19 | 1001 | 57.09 | 2026-10-11 14:56 | 2026-10-11 14:56 | pipeline test of the final evaluation |
| TransUNet, quick-check subset model (48 training images, job 206) | QaTa-COV19 | 1001 | 65.53 | 2026-10-11 15:01 | 2026-10-12, while compiling this list | pipeline test |
| U-Net, quick-check subset model (batch 2, job 206) | MosMedData+ | 1001 | 10.29 | 2026-10-11 15:04 | 2026-10-12, while compiling this list | pipeline test |
| TransUNet, quick-check subset model (job 206) | MosMedData+ | 1001 | 14.13 | 2026-10-11 15:11 | 2026-10-12, while compiling this list | pipeline test |
| U-Net, quick-check subset model (final config, 48 training images, job 211) | QaTa-COV19 | 1001 | 57.24 | 2026-10-11 15:42 | 2026-10-12, while compiling this list | pipeline test |
| U-Net, quick-check subset model (final config, job 211) | MosMedData+ | 1001 | 13.54 | 2026-10-11 15:44 | 2026-10-12, while compiling this list | pipeline test |
| nnU-Net, quick-check model (3 epochs, job 210) | QaTa-COV19 | 1001 | 71.66 | 2026-10-11 15:52 | 2026-10-11 15:52 | pipeline test of selection + final evaluation |
| nnU-Net, quick-check model (3 epochs, job 210) | MosMedData+ | 1001 | 62.58 | 2026-10-11 16:00 | 2026-10-11 16:01 | pipeline test of selection + final evaluation |

How these scores were used:
- The quick-check models are pipeline tests (subsets of the training data or 3 epochs). They are not
  part of the 18 main runs, and their scores were not used for anything else.
- The two U-Net seed-1001 scores were used only for the agreed comparison with the LViT paper and for
  checking the pipeline.
- After these scores were seen, **no training setting, no checkpoint-selection rule and no analysis method
  was changed**:
  - all 18 runs had been submitted before the first main-run score was seen (2026-10-11), with the
    settings in `configs/` (commit 97a2330);
  - the analysis methods above were fixed afterwards, but without reference to any test score. The
    choices that the plan left open were decided by the project lead from the options listed in advance.

### Stored probabilities (`test_probs.npz`, `val_probs.npz`)

These archives store the lesion probabilities as **float16**. They are a backup only and take part in no
scoring. Every per-case metric (`per_case.csv`) and every predicted mask was computed inside the run
from the float32 probabilities. Masks re-derived from the float16 archive can differ in boundary pixels
whose probability lies within one float16 step of the 0.5 threshold. To keep the code identical for all
18 runs, the format is not changed during Phase 0; it will become float32 in Phase 1.

The investigation of the MosMed gap (`src/analysis/lvit_metric_check.py`, PHASE0_MODEL_SETUP.md §7)
re-scored U-Net seed-1001 predictions from these float16 archives. It is an analysis outside the
statistics of this report, and its two-decimal values agreed with `per_case.csv`.

### nnU-Net runs trained in segments (12-hour job limit)

*This subsection was added on 2026-10-12, after the analysis plan was fixed (commit 6ed8aa7). It documents
how some runs were executed. It changes no training setting, no checkpoint-selection rule and no analysis
method.*

**Why the runs were split.**
- All jobs were submitted without an explicit time limit, so each job received the `gpu` partition's
  MaxTime at the moment it started. That value was changed at run time by the cluster administrators:
  - at submission on 2026-10-11 it was "infinite";
  - for jobs starting from 2026-10-11 evening it was 24 h;
  - for jobs starting from 2026-10-12 evening it was 12 h.

  `/etc/slurm/slurm.conf` still lists `MaxTime=INFINITE`.
- One nnU-Net run needs about 23 h, and a user cannot request more than the partition MaxTime. The runs
  that started under the 12-hour limit were therefore trained in segments.

**How segmentation works** (`slurm/train_nnunet.slurm`):
- Each segment resumes with nnU-Net's own `nnUNetv2_train --c`, starting from `checkpoint_latest.pth`.
  That file is written every 50 epochs and restores the network, optimiser, grad-scaler and logger state
  and the epoch counter.
- Up to 50 epochs (≈ 68 min) are recomputed after each interruption.
- Snapshot selection on the full validation set and the final evaluation run after epoch 1000, as for
  the unsegmented runs.

**Effect on reproducibility.** On every resume the data-augmentation worker processes restart with their
fixed seeds, so their random stream starts again from the beginning. Segmented runs are therefore not
bit-identical to an uninterrupted run with the same seed. The training recipe is unchanged, and nnU-Net
runs are seeded but not bit-reproducible in any case (multi-process data loading).

| Run | Segments (Slurm jobs) | Resumed after epoch | Continuity check at the first resume |
|---|---|---|---|
| nnU-Net QaTa-COV19 seed 1001 | 1 (job 214, 24 h limit) — training not segmented | — | — |
| nnU-Net MosMedData+ seed 1001 | 1 (job 215, 24 h limit) — not segmented | — | — |
| nnU-Net QaTa-COV19 seed 1002 | *[placeholder: from runs/nnunet/QaTa-COV19/seed1002/segments.tsv; first segment = job 216]* | *[placeholder]* | *[placeholder]* |
| nnU-Net MosMedData+ seed 1002 | *[placeholder; first segment = job 217]* | *[placeholder]* | *[placeholder]* |
| nnU-Net QaTa-COV19 seed 1003 | *[placeholder; first segment = job 218]* | *[placeholder]* | *[placeholder]* |
| nnU-Net MosMedData+ seed 1003 | *[placeholder; first segment = job 219]* | *[placeholder]* | *[placeholder]* |

The continuity check, done on the training log at the first resume, verifies four things:
- the epoch numbers continue without a gap after the resume point;
- the learning rate continues on the original poly schedule, lr = 0.01 · (1 − epoch/1000)^0.9;
- the training loss shows no jump;
- the 50-epoch snapshots continue to be written.

**Job 214 (nnU-Net QaTa-COV19 seed 1001).**
- Training (1000 epochs), snapshot selection (epoch 250 selected on the validation set) and the
  per-case test metrics (`per_case.csv`, `summary.json`) were complete when the 24-hour limit stopped
  the job, while it was writing the predicted test masks (1217 of 2113 written, the last one truncated
  to 0 bytes).
- The missing masks were then regenerated from the stored `test_probs.npz`. A check against
  `per_case.csv` found 221 of these 896 masks differing from the scored predictions by 1–21 boundary
  pixels. The reason is that `test_probs.npz` stores the probabilities as float16 (`runner.save_probs`),
  and pixels within one float16 step of the 0.5 threshold flip. The masks written by job 214 itself
  (1217) all match.
- Repair, decided on 2026-10-12 and run exactly once (job 237, `src/analysis/repredict_214_masks.py`):
  1. Before and after the repair, the sha256 of `per_case.csv` and `summary.json` is recorded and compared.
  2. The test set is predicted once with the already selected checkpoint (epoch 250), with exactly the
     original inference settings (including mirroring TTA). The new masks are written to a separate
     folder; nothing is scored or re-selected.
  3. Check 1: the new masks of the 1217 original cases vs. the masks job 214 wrote, pixel by pixel.
  4. Check 2: the new masks of the 896 missing cases vs. `per_case.csv` (Dice difference < 1e-6).
  5. Only if both checks show no mismatch are those 896 masks copied into the official folder.
     Otherwise the float16-derived masks remain, with the 221 cases that differ by 1–21 boundary pixels.
- *[placeholder: outcome of job 237 — mismatching cases in checks 1 and 2, whether the masks were copied,
  sha256 unchanged]*
- `per_case.csv` — the input of every statistic in this report — was computed by the job from float32
  probabilities and is not affected.
