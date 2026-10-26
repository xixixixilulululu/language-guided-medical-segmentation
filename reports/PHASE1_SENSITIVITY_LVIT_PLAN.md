# Phase 1 — Plan of a post-hoc sensitivity check: LViT-T on MosMedData+ with batch size 24

Written 2026-10-26 and pushed **before any run of this check was submitted**.

## 1. Why, and what it is not

- **Observation.** In the Phase 1 main runs, LViT-T on MosMedData+ reaches a test Dice of 61.26 ± 1.10. That
  is 13.31 points below the LViT paper (Table II: 74.57), and every seed has 38–41 empty test predictions out
  of 273 ([PHASE1_RESULTS.md](PHASE1_RESULTS.md), sections 6 and 8).
- **The setting in question.** The main runs combine batch size 2 (from the official code, a comment specific
  to LViT-T) with the MosMedData+ learning rate 1e-3 (from the paper). Neither the paper nor the code
  explicitly uses this combination.
- **Question.** Does the main-run LViT-T result on MosMedData+ change when the batch size is the paper's
  default, 24, with everything else unchanged?
- **This check was decided after the main results were seen.** It is a post-hoc sensitivity analysis.
- **Whatever its outcome, the Phase 1 main results and conclusions are not changed.**
  `PHASE1_RESULTS.md` and `PHASE1_SUMMARY.md` keep the main-run values. This check is reported separately
  and does not enter `src/analyze_phase1.py` or its statistics.

## 2. Design (fixed now)

| Item | Sensitivity check | Main run |
|---|---|---|
| Method, dataset | LViT-T, MosMedData+ | same |
| **Batch size** | **24** (LViT paper, Sec. IV-C: "The default batch size is 24 for the QaTa-COV19 dataset and the MosMedData+ dataset") | 2 |
| Learning rate | 1e-3 | same |
| Everything else | unchanged: model, text features, loss, optimizer, schedule, augmentation, max. epochs 2000, early stopping (patience 50), checkpoint selection on validation Dice only, data and split, common evaluator (probability > 0.5), `deterministic: true` | – |
| Seeds | 1001, 1002, 1003 | same |

The batch size is the only change: `configs/lvit_sensitivity_b24.yaml` is `configs/lvit.yaml` with
`batch_size: 24`. Loading both YAML files and comparing them gives exactly one difference, `batch_size` 2 → 24.
Its sha256 is recorded in `configs/phase1_frozen/lvit_b24.sha256`.

**Code.** The frozen main-run driver `src/phase1/train_lvit.py` is used unchanged, with
`--config configs/lvit_sensitivity_b24.yaml`. The Slurm file `slurm/phase1_lvit_sensitivity_b24.slurm`
checks before training:
- the frozen LViT-T record (`configs/phase1_frozen/lvit.sha256`: driver, shared code, text features);
- the pinned LViT commit;
- the sha256 of the sensitivity configuration.

**Separation from the main runs.**
- Output goes to `runs/lvit_b24/MosMedData+/seed<k>/`. No main-run file is written or overwritten.
- `src/analyze_phase1.py` reads only `runs/{unet,transunet,nnunet,lvit,reclmis,ddcmd}/` and never sees
  these runs.
- The driver logs its test evaluation in `results/phase1_test_eval_log.tsv` like a main run, with method
  `lvit`. These entries are identified by their Slurm job IDs, which the report lists.

## 3. What will be reported (fixed now)

`reports/PHASE1_SENSITIVITY_LVIT.md`, with the table produced by `src/phase1/summarize_sensitivity_lvit.py`
(written and pushed with this plan; output `results/phase1_sensitivity_lvit_b24.csv`):

- For **batch 24** and for the **main run (batch 2)**, side by side, from each run's `eval_test/summary.json`:
  - per seed: test Dice, mIoU, HD95 and ASSD, and the number of empty test predictions (of 273);
  - the 3-seed mean ± sample SD of these per-seed values (the same summary as the Phase 1 main table, so the
    batch-2 mean Dice is 61.26).
- The training record per seed: best epoch and its validation Dice, last epoch, and how training ended.
- The paper value (LViT Table II, Dice 74.57).
- The differences of the mean Dice: batch 24 − batch 2, paper − batch 2, paper − batch 24.
- **Descriptive only:**
  - no significance test, no confidence interval and no claim beyond the numbers;
  - with 3 seeds and one changed setting, the report states what changed, not why.
- **Non-finite losses:** the LViT-T driver has no per-step check, so they are checked as in the main runs
  (`epochs.csv`: training loss and validation Dice of every epoch). If a run ends with a non-finite loss,
  this is reported as observed.
- The test set is evaluated once per run, after training, on the checkpoint selected on validation.
- `per_case.csv`, `summary.json` and `training_summary.json` of the three runs are committed. Weights,
  masks and probabilities are not committed.
- A Chinese reading copy `PHASE1_SENSITIVITY_LVIT_zh_CN.md` is kept locally only.

## 4. Running

- **Submission.** Same rules as the main runs: submitted by hand with `slurm/submit_next_p1.sh --day`, one job
  at a time, 12 h per job.
- **Plan rows.** The three runs are rows 25–27 of `slurm/phase1_plan.tsv` (method `lvit_b24`). The submit
  tool picks them in this order, continues a run in a new segment if the wall-clock guard ended it, and never
  submits a run twice.
- **Time.** Unknown in advance. A main-run epoch (batch 2) takes about 89 s, and the main runs stopped
  after 88–208 epochs. Each job is limited to 12 h (wall-clock guard 11.5 h).
- **Technical failures.** If a job fails for a technical reason (e.g. out of GPU memory with batch 24), it is
  reported. No setting is changed without the project lead's approval.
