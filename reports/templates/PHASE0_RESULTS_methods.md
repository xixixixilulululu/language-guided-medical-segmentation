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
