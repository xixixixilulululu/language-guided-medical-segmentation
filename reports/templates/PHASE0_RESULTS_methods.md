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
| nnU-Net QaTa-COV19 seed 1002 | 2: job 216 (epochs 0–510, stopped by the 12 h limit), job 249 (epochs 500–1000, snapshot selection and final evaluation; stopped by the 12 h limit 12 s before its last log line — all outputs verified complete and consistent with `per_case.csv`, DONE marker written by hand) | 500 | passed (see below) |
| nnU-Net MosMedData+ seed 1002 | 2: job 217 (epochs 0–506, stopped by the 12 h limit), job 250 (epochs 500–1000, snapshot selection and final evaluation, completed) | 500 | passed (see below) |
| nnU-Net QaTa-COV19 seed 1003 | 3: job 282 (epochs 0–515, stopped by the 12 h limit), job 290 (epochs 500–1000; stopped by the 12 h limit during nnU-Net's built-in validation after epoch 1000, before snapshot selection), job 305 (snapshot selection and final evaluation only, completed in 30 min). The originally planned jobs 218 / 219 for seed 1003 were removed from the queue (see below). | 500 | passed (see below) |
| nnU-Net MosMedData+ seed 1003 | 2: job 309 (epochs 0–508, stopped by the 12 h limit), job 314 (epochs 500–1000, snapshot selection and final evaluation, completed after 11 h 40 min) | 500 | passed (see below) |

**Queue removal and resubmission.**
- Between 2026-10-12 19:50 and 2026-10-13 01:16 (EDT), all 14 jobs of this project that were waiting
  in the queue (218–229, 233–237) were removed by someone else before they ran. Cluster accounting is
  disabled, so who removed them and why could not be determined.
- On the project lead's decision, the remaining work was resubmitted unchanged on 2026-10-13 as jobs
  248–262.
- A watchdog (`src/queue_watchdog.sh`) then raised an alert, and never resubmitted, if jobs disappeared
  from the queue. It was retired on 2026-10-13 under the fair-use policy (no background processes on the
  login node).
- On 2026-10-13 between 09:25 and 09:30 (EDT), all 14 jobs of this project waiting in the queue
  (251–264) were again removed before they ran. The watchdog reported it, and nothing was resubmitted.
  On the project lead's decision, no jobs were kept waiting in the queue from then on.

**Cluster fair-use policy (from 2026-10-13).** The supervisor then communicated the rules under which
the queued jobs had been removed:
- at most 2 jobs per user (1 running + 1 waiting);
- at most 12 h per job;
- jobs submitted by hand, no batch submission.

How the project complies:
- All `.slurm` files request `--time=12:00:00`.
- The cluster enforces the policy with an admission filter. It chains a user's second job behind the
  first (`afterany`) and refuses user-supplied dependencies.
- nnU-Net segments no longer submit their successor. Each segment is submitted by hand with
  `slurm/submit_next.sh`, which submits exactly one job per call from `slurm/phase0_plan.tsv` and
  refuses when the limits would be exceeded.
- The remaining 12 runs are executed one at a time, with ~12 h nnU-Net segments placed overnight
  (`reports/PHASE0_SUBMISSION_PLAN.md`). This moves the estimated completion from 2026-10-15 to about
  2026-10-17.
- The rules change only how and when jobs run. They change no training setting, no checkpoint-selection
  rule and no analysis method.
- Submission log: `runs/phase0_submit_log.tsv` (job ids, times). Slurm jobs that produced the 18 runs
  (Slurm logs in `runs/slurm_logs/`):

  | Method | QaTa-COV19 s1001 | QaTa-COV19 s1002 | QaTa-COV19 s1003 | MosMedData+ s1001 | MosMedData+ s1002 | MosMedData+ s1003 |
  |---|---|---|---|---|---|---|
  | U-Net | 212 | 288 | 308 | 213 | 285 | 303 |
  | TransUNet | 289 | 311 | 312 | 287 | 307 | 310 |
  | nnU-Net | 214 (+ repair job 248) | 216, 249 | 282, 290, 305 | 215 | 217, 250 | 309, 314 |

**12 h limit for U-Net and TransUNet** (no resume function):
- TransUNet always ends within the limit (150 epochs ≤ 6.8 h).
- A U-Net run that has not met the early-stopping rule after 12 h (about 375 QaTa-COV19 epochs; seed 1001
  stopped at epoch 119) is stopped by the limit and is not handled automatically. **This did not happen:**
  all six U-Net runs met the early-stopping rule well within the limit (QaTa-COV19: epochs 119 / 120 / 130,
  3.8–4.2 h; MosMedData+: epochs 179 / 209 / 160, 2.0–2.5 h, for seeds 1001 / 1002 / 1003). All six
  TransUNet runs completed their 150 epochs.

The continuity check, done on the training log at the first resume, verifies four things:
- the epoch numbers continue without a gap after the resume point;
- the learning rate continues on the original poly schedule, lr = 0.01 · (1 − epoch/1000)^0.9;
- the training loss shows no jump;
- the 50-epoch snapshots continue to be written.

Result at the first resume, seed 1002 (checked on 2026-10-13; the resumed segments 249 / 250 had run 68 / 39
epochs). Both runs passed:

| Check | QaTa-COV19 seed 1002 | MosMedData+ seed 1002 |
|---|---|---|
| Epoch numbers | resumed exactly at epoch 500, no gap | resumed exactly at epoch 500, no gap |
| Learning rate | equal to the poly schedule at every epoch (deviation 0 at the printed 5 decimals); identical to segment 1 on the re-run epochs 500–510 | same; identical on the re-run epochs 500–506 |
| Training loss | re-run epochs 500–510 within 0.0024 of segment 1 (mean difference +0.0002); mean of epochs 480–499 −0.8944 vs. first 20 resumed epochs −0.8955 | re-run epochs within 0.0009 (mean −0.0003); −0.9363 vs. −0.9376 |
| Snapshots | `checkpoint_ep0550.pth` written in the resumed segment | next one due at epoch 550 (not yet reached at the check) |

Result at the first resume, seed 1003 (checked on 2026-10-18 on the complete training logs). Both runs passed:

| Check | QaTa-COV19 seed 1003 | MosMedData+ seed 1003 |
|---|---|---|
| Epoch numbers | resumed exactly at epoch 500, epochs 500–999 without a gap | resumed exactly at epoch 500, epochs 500–999 without a gap |
| Learning rate | equal to the poly schedule at every resumed epoch (deviation 0 at the printed 5 decimals); identical to segment 1 on the re-run epochs 500–515 | same; identical on the re-run epochs 500–508 |
| Training loss | re-run epochs 500–515 within 0.0045 of segment 1 (mean difference +0.0000); mean of epochs 480–499 −0.8945 vs. first 20 resumed epochs −0.8950 | re-run epochs within 0.0019 (mean −0.0004); −0.9356 vs. −0.9367 |
| Snapshots | written up to `checkpoint_ep1000.pth` | written up to `checkpoint_ep1000.pth` |

The resume checkpoint was verified before each resumed segment started (`current_epoch` 500 in all four
runs).

**Job 214 (nnU-Net QaTa-COV19 seed 1001).**
- Training (1000 epochs), snapshot selection (epoch 250 selected on the validation set) and the
  per-case test metrics (`per_case.csv`, `summary.json`) were complete when the 24-hour limit stopped
  the job, while it was writing the predicted test masks (1217 of 2113 written, the last one truncated
  to 0 bytes).
- The missing masks were then regenerated from the stored `test_probs.npz`. A check against
  `per_case.csv` found 221 of these 896 masks differing from the scored predictions. The reason is that
  `test_probs.npz` stores the probabilities as float16 (`runner.save_probs`). A float32 probability just
  above 0.5 can round to exactly 0.5 in float16 and then fails `prob > 0.5`; no other pixel can flip
  (a probability ≤ 0.5 never rounds to a value above 0.5). The float16-derived mask therefore misses a few
  scored foreground pixels and never adds one.
  *Corrected on 2026-10-18 (an earlier version said "1–21 boundary pixels").* The exact number of missing
  pixels per case follows from the scored Dice in `per_case.csv` and the pixels stored as exactly 0.5
  (it was uniquely determined in all 221 cases): **1–6 pixels, median 1**. The masks written by job 214 itself
  (1217) all match.
- Repair, decided on 2026-10-12 and run exactly once (job 237, `src/analysis/repredict_214_masks.py`):
  1. Before and after the repair, the sha256 of `per_case.csv` and `summary.json` is recorded and compared.
  2. The test set is predicted once with the already selected checkpoint (epoch 250), with exactly the
     original inference settings (including mirroring TTA). The new masks are written to a separate
     folder; nothing is scored or re-selected.
  3. Check 1: the new masks of the 1217 original cases vs. the masks job 214 wrote, pixel by pixel.
  4. Check 2: the new masks of the 896 missing cases vs. `per_case.csv` (Dice difference < 1e-6).
  5. Only if both checks show no mismatch are those 896 masks copied into the official folder.
     Otherwise the float16-derived masks remain, with the 221 cases that differ from the scored predictions.
- Outcome:
  - Job 237 was removed from the queue by someone else before it ran (see below), so the repair ran
    once as job 248 on 2026-10-13.
  - The sha256 of `per_case.csv` and `summary.json` matched the recorded values before the job and was
    unchanged after it.
  - **Check 1: 389 of the 1217 original cases differ.** A second GPU inference with the same checkpoint
    and the same settings is not bit-identical to the first one.
  - **Check 2: 249 of the 896 missing cases** do not reproduce the scored Dice.
  - As decided in advance, nothing was copied, and the inference was not repeated.
  - Size of the GPU re-inference differences (prediction vs. prediction; no test score involved): in the
    389 differing original cases, the two inferences differ by **at most 10 pixels (median 1)**, and the
    lowest Dice between the two predictions of one case is **0.9935**.
  - The 896 masks in `eval_test/masks/` are therefore the float16-derived ones (option 2): 221 of them
    miss 1–6 scored foreground pixels (median 1) whose float32 probability was just above 0.5.
  - These masks are only drawn in the case figures. All statistics, and the Dice / HD95 labels in the
    figures, come from `per_case.csv`.
- `per_case.csv` — the input of every statistic in this report — was computed by the job from float32
  probabilities and is not affected.
