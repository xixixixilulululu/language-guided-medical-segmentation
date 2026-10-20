# Phase 1 — Progress report for the supervisor (interim)

Status: 2026-10-19, 20:30 EDT. **This is an interim report.** It contains no comparison between methods yet.
The final results are produced after all 18 runs by the analysis program, which was fixed before any
Phase 1 test result existed. This report will then be replaced by the final summary. Details:
[PHASE1_MODEL_SETUP.md](PHASE1_MODEL_SETUP.md) (setup, deviations, quick checks) and
[PHASE1_SETUP_PLAN.md](PHASE1_SETUP_PLAN.md) (checks and decisions).

## 1. Summary

1. **All three language-guided methods are reproduced and their configurations are frozen.** The methods
   are LViT-T, RecLMIS and DD-CMD. Each passed a quick check on the validation set: image–mask–text
   pairing, the text reaching the language branch, falling loss, rising validation Dice, valid masks,
   resume after a 12 h segment, and timing.
2. **The test set is protected.** The statistical analysis (`src/analyze_phase1.py`; 6 methods, 15 pairs
   per dataset, Holm correction across the 15, α = 0.05) was reviewed, fixed and pushed before the first
   Phase 1 test evaluation:
   - its frozen record reached GitHub at 2026-10-19 04:02:14 UTC (GitHub server time, from GitHub's event
     log);
   - the first Phase 1 test evaluation started at 16:20:18 UTC (12:20:18 EDT, cluster clock,
     `results/phase1_test_eval_log.tsv`).

   Two independent locks enforce this order, and every test evaluation is logged.
3. **Progress: 1 of 18 runs is complete.** One more is finishing after a numerical problem (item 4),
   one is running and one is queued. At the measured speed, the remaining runs need about 5–7 days.
4. **Numerical problem in RecLMIS (QaTa-COV19, seed 1001).** In epoch 152 its contrastive loss became
   NaN. We stopped the run before the NaN reached the model weights. We then fixed a rule for all runs
   **before seeing further results**: end training and evaluate the best validation checkpoint. On the
   same training trajectory, this gives the same checkpoint that the official code would finally select
   (section 4).
5. **Deviations from the official code are few and documented.** They cover the text encoder of LViT-T,
   checkpoint selection for DD-CMD, and compatibility fixes (section 3).

## 2. Methods and setup

All methods use the Phase 0 data: the same LViT split, the same 224×224 images and masks, and the same
common evaluator (threshold 0.5; Dice, mIoU, HD95, ASSD). They use seeds 1001/1002/1003 and choose the
checkpoint **on the validation set only** (mean per-case Dice). All three use the text at training and at
test time.

| Method | Official code | Text encoder | Cross-modal mechanism |
|---|---|---|---|
| LViT-T | HUANGLIZI/LViT | BERT-base uncased (frozen), 10 word vectors | text features fused into a hybrid CNN–ViT at every level |
| RecLMIS | ShawnHuang497/RecLMIS | CLIP ViT-B/32 text tower (frozen) | conditioned reconstruction of masked text / image tokens + contrastive loss |
| DD-CMD | maklachur/DD-CMD | BiomedBERT (frozen) | gated text–image cross-attention + text-conditioned frequency (DCT) decoding |

"Frozen" means that the text encoder is not trained:
- **LViT-T:** its word vectors are computed outside the model.
- **RecLMIS:** the official config sets `frozen_clip = True`.
- **DD-CMD:** the official `FrozenTextEncoder` disables the gradients.

This was verified on trained checkpoints: the CLIP text tower (RecLMIS run, epoch 129) and BiomedBERT
(DD-CMD quick-check run) are bit-identical to the pretrained weights, while DD-CMD's image encoder has
changed.

Training settings are the official ones: loss, optimizer, learning rate, schedule, augmentation, epochs
and early stopping. Where the paper and the code disagree, the more specific source was used. For
LViT-T this means a learning rate of 3e-4 for QaTa and 1e-3 for MosMed (paper), and a batch size of 2 (a
code comment about LViT-T; the paper's general default is 24).

## 3. Deviations from the official code

| Item | What we did | Why |
|---|---|---|
| LViT-T text encoder | The official `bert_embedding` package (MXNet) cannot be installed on this cluster (aarch64). Its computation was re-implemented in PyTorch with the same BERT weights and precomputed. | Closest possible substitute. **It could not be checked against the original MXNet output**, because MXNet does not run here. |
| Random seeds | 1001 / 1002 / 1003 for every method (official: LViT 666, RecLMIS 3407, DD-CMD 42). | Phase 1 plan. |
| DD-CMD checkpoint selection | Mean per-case validation Dice instead of the official pixel-pooled validation mIoU. | The same rule for all six methods. |
| DD-CMD early stopping | Also monitors the mean per-case validation Dice (official: validation mIoU); patience 50 as official. | Consistent with the checkpoint selection. |
| DD-CMD data loading | 0 data-loader workers, the official value. We had planned 8, but with 8 the training hung before the first batch (diagnosed in a dedicated job). | No deviation remains. |
| Bugs in the official code | Fixed in our drivers, and checked at every step: RecLMIS silently switched off its reconstruction losses when a dependency was missing, and loaded the QaTa config for MosMed; LViT's released loader pairs files by the MoNuSeg naming, so QaTa images and masks do not match; text padding was lost under the current pandas. | Without these fixes, the methods would not run as published. |

## 4. The RecLMIS numerical problem and the rule fixed for it

**What happened.** RecLMIS on QaTa-COV19, seed 1001:
- epochs 1–151 trained normally, with the best validation Dice 0.831 at epoch 129;
- in epoch 152 one loss term, `loss_ccl` (the conditional contrastive loss), became NaN; the other three
  loss terms were finite;
- our driver checks every loss in every step, so it stopped **before that step changed the model**. The
  epoch-129 checkpoint and the epoch-151 state are intact.

**What the official code would do.** It does not check this loss:
- the NaN would spread into all weights, so the validation Dice could not improve any more;
- after the patience of 100 epochs, early stopping would end the run with the same epoch-129 checkpoint.

This NaN occurred in our environment (torch 2.14, aarch64). Whether it would also occur in the authors'
original environment (torch 1.8) cannot be determined.

**Rule, fixed before any further result** (2026-10-19, `PHASE1_MODEL_SETUP.md` section 3.4).
- For every run of every method, a NaN or infinite loss ends training before that step is applied.
- The best validation checkpoint is then evaluated.
- The epoch, the step and the loss term are reported.
- This reproduces the official outcome and only skips the useless epochs after the event.
- For every run we also count how often a loss was non-finite. This includes the cases in which the
  official RecLMIS code itself silently replaces a NaN reconstruction loss by 0.

**Status.** The run will be resumed from epoch 151 under this rule. Its outcome, and any further
occurrence in other runs, will be reported.

## 5. An observation from the quick checks (no interpretation)

These are small models (64 training images, one seed), not the main runs. Each was evaluated on 32
validation images twice: with each image's own text, and with every text replaced by one fixed text.

| Dataset | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| QaTa-COV19: Dice own text → fixed text | 0.649 → 0.609 | 0.607 → 0.593 | 0.760 → 0.668 |
| MosMedData+: Dice own text → fixed text | 0.783 → 0.783 | 0.763 → 0.763 | 0.790 → 0.790 |

On QaTa-COV19 the Dice drops for all three methods. On MosMedData+ it is unchanged to the third decimal
for all three, although 26 of the 32 images received a different text. Whether the trained main models
depend on the text is the subject of Phase 2. No conclusion is drawn here.

## 6. Progress of the 18 runs

| Method | QaTa-COV19 (seeds 1001 / 1002 / 1003) | MosMedData+ (seeds 1001 / 1002 / 1003) |
|---|---|---|
| LViT-T | **done** (early stopping at epoch 161; best validation Dice 0.810 at epoch 110) / – / – | – / – / – |
| RecLMIS | **finishing** (NaN rule, section 4; best validation Dice 0.831, epoch 129) / – / – | – / – / – |
| DD-CMD | queued / – / – | **running** / – / – |

- Test-set numbers of single runs are not reported here, except for the reproduction check below. All
  test results are reported together, by the fixed analysis program, once the three seeds of a method
  are complete.

**Reproduction check (not a method comparison).** LViT-T on QaTa-COV19, seed 1001, using the checkpoint
selected on validation (epoch 110):

| | Test Dice (%) |
|---|---:|
| Ours, seed 1001 (2113 test images) | 82.47 |
| LViT paper (LViT-T, QaTa-COV19): mean of three runs, Table IV 83.66 ± 0.8 (= Table II) | 83.66 |
| Difference | −1.19 |

- This is **a single seed**. The final value is the mean over three seeds.
- **The paper's value is the mean of three runs.** Table IV of the LViT paper gives LViT-T as
  mean ± std, with the std over three runs (QaTa-COV19 Dice 83.66 ± 0.8; also MosMedData+ mIoU
  61.33 ± 0.5, as in Table II). The seeds are not stated. This is also recorded in Phase 0
  (`PHASE0_MODEL_SETUP.md`, section 7). Our seed 1001 is 1.19 points below that mean.
- **For LViT-T the metric definition is in effect the same.** In Phase 0 the LViT official test
  algorithm (per-image Dice with a 1e-5 smoothing term, threshold 0.5, mean over the test images) was
  applied to our saved predictions. It gave exactly the values of our common evaluator: QaTa-COV19 79.12
  vs. 79.12, MosMedData+ 73.92 vs. 73.92 (`src/analysis/lvit_metric_check.py`,
  `results/phase0_lvit_metric_check.json`). Among the three methods, only DD-CMD's paper uses a different
  definition (pixel-pooled Dice); it is not involved here.
- The paper does not state whether its LViT-T training used batch size 2 or 24.
- The value is shown only to check that the reproduction is in the expected range. It is **not used to
  compare methods**.
- Measured time per epoch: QaTa 171–247 s, MosMed 63–92 s. Estimated total: 110–175 GPU hours; by the
  cluster rules only one job runs at a time.
- Jobs follow the cluster rules: at most 12 h each (longer runs are resumed in segments) and submitted
  by hand. Long QaTa runs are scheduled at night where possible.

## 7. Next steps

1. Complete the remaining runs (about 5–7 days).
2. Run the fixed analysis once: six-method table, 15 pairwise comparisons per dataset, figures.
3. Replace this interim report by the final Phase 1 summary, answering the eight questions of the plan.
