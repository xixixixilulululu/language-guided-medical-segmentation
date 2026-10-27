# Phase 1 — Summary for the supervisor

Three language-guided methods (LViT-T, RecLMIS, DD-CMD) reproduced on QaTa-COV19 and MosMedData+ (LViT
split), seeds 1001 / 1002 / 1003, and compared with the Phase 0 image-only baselines (U-Net, TransUNet,
nnU-Net). All numbers below are the final values of [PHASE1_RESULTS.md](PHASE1_RESULTS.md), which also
holds the full methods, all 30 pairwise comparisons and the per-run records. This summary adds no analysis.
It replaces the interim report `PHASE1_PROGRESS.md` (kept in the git history).

## 1. Summary

1. **All 18 runs are complete** (3 methods × 2 datasets × 3 seeds). Every method uses the same data, split,
   224×224 images, common evaluator (probability > 0.5) and seeds as Phase 0; checkpoints were selected on
   the validation set only.
2. **The test set was protected.** The statistical analysis was fixed and pushed (commit ba53ba5) before
   the first Phase 1 test evaluation: all 36 test-evaluation log entries are later than the fixing time. It
   was run once, unchanged. The results were re-computed independently from the per-case files with
   separate code, and matched exactly; the conclusions were reviewed by the project lead.
3. **QaTa-COV19: RecLMIS (84.77) and DD-CMD (84.64) are level and best.** Both are significantly better
   than LViT-T (81.71), which is significantly better than the three image-only methods (79.18–79.72).
4. **MosMedData+: DD-CMD (74.85), RecLMIS (74.44) and U-Net (74.01) are level and best.** LViT-T (61.26)
   is significantly below every other method, including all three image-only ones.
   **Note (post-hoc sensitivity check):** that LViT-T is lowest on MosMedData+ holds only under our pre-fixed
   setting (batch size 2, learning rate 1e-3). With batch size 24 in the sensitivity check, its mean Dice
   (74.01) is at the level of the three best methods (74.01–74.85). The sensitivity check has no significance
   test, so this is descriptive only ([PHASE1_SENSITIVITY_LVIT.md](PHASE1_SENSITIVITY_LVIT.md)).
5. **Language-guided methods are therefore not consistently better than image-only ones.** These are
   comparisons of different architectures; they do not show whether the text itself causes any
   difference. That is the question of Phase 2.

## 2. Main results

Test set. Mean ± sample SD over the 3 seeds; 95% CI = case bootstrap of the seed-averaged per-case Dice.
Dice and mIoU in %, HD95 and ASSD in pixels. The image-only rows are the Phase 0 results, unchanged.

| Dataset | Method | Text | Dice | Dice 95% CI | mIoU | HD95 | ASSD |
|---|---|---|---|---|---:|---:|---:|
| QaTa-COV19 | U-Net | No | 79.18 ± 0.22 | [78.30, 80.04] | 69.73 ± 0.15 | 31.36 ± 0.70 | 9.95 ± 0.51 |
| QaTa-COV19 | TransUNet | No | 79.47 ± 0.41 | [78.61, 80.30] | 70.07 ± 0.64 | 29.81 ± 0.56 | 9.46 ± 0.76 |
| QaTa-COV19 | nnU-Net | No | 79.72 ± 0.16 | [78.87, 80.56] | 70.39 ± 0.17 | 29.23 ± 1.35 | 8.73 ± 0.79 |
| QaTa-COV19 | LViT-T | Yes | 81.71 ± 0.69 | [80.89, 82.51] | 72.90 ± 0.79 | 20.96 ± 1.48 | 8.61 ± 1.24 |
| QaTa-COV19 | RecLMIS | Yes | **84.77 ± 0.33** | [84.06, 85.45] | 76.46 ± 0.42 | 13.96 ± 0.65 | 4.31 ± 0.38 |
| QaTa-COV19 | DD-CMD | Yes | **84.64 ± 0.10** | [83.96, 85.30] | 76.16 ± 0.10 | 13.73 ± 0.30 | 3.99 ± 0.27 |
| MosMedData+ | U-Net | No | **74.01 ± 0.86** | [72.02, 75.90] | 60.98 ± 0.84 | 23.02 ± 2.10 | 8.43 ± 1.46 |
| MosMedData+ | TransUNet | No | 69.43 ± 0.53 | [66.98, 71.73] | 56.32 ± 0.51 | 24.02 ± 0.62 | 10.91 ± 0.73 |
| MosMedData+ | nnU-Net | No | 72.06 ± 0.84 | [69.59, 74.34] | 59.61 ± 0.86 | 22.54 ± 2.96 | 8.90 ± 2.54 |
| MosMedData+ | LViT-T | Yes | 61.26 ± 1.10 | [57.62, 64.69] | 49.91 ± 1.24 | 62.04 ± 3.60 | 50.29 ± 2.01 |
| MosMedData+ | RecLMIS | Yes | **74.44 ± 1.00** | [72.59, 76.18] | 61.35 ± 0.96 | 16.01 ± 1.79 | 3.79 ± 0.58 |
| MosMedData+ | DD-CMD | Yes | **74.85 ± 0.53** | [72.85, 76.72] | 62.12 ± 0.59 | 16.86 ± 0.50 | 5.46 ± 0.83 |

Bold: the methods that are best and not significantly different from each other (section 3).

QaTa-COV19: the CIs of RecLMIS and DD-CMD overlap each other and lie above that of LViT-T, which lies above
the three image-only methods.

![QaTa-COV19 Dice with 95% CI](figures/phase1_dice_ci_QaTa-COV19.png)

MosMedData+: the CIs of U-Net, RecLMIS and DD-CMD overlap; LViT-T lies below every other method.

![MosMedData+ Dice with 95% CI](figures/phase1_dice_ci_MosMedDataplus.png)

Across seeds, DD-CMD has the smallest SD on both datasets (0.10 and 0.53) and LViT-T the largest (0.69 and
1.10); three seeds give only a rough estimate.

## 3. Key pairwise comparisons

Primary analysis: paired t-test on the seed-averaged per-case Dice, Holm correction across the 15
comparisons of each dataset, two-sided α = 0.05. ΔDice = Dice(A) − Dice(B) in Dice points. Significant
pairs: 11 of 15 on QaTa-COV19, 12 of 15 on MosMedData+ (all 30 in PHASE1_RESULTS.md, section 2).

| Dataset | A vs. B | ΔDice | 95% CI | Holm p | |
|---|---|---:|---|---:|---|
| QaTa-COV19 | RecLMIS vs. DD-CMD | +0.13 | [−0.13, +0.38] | 0.428 | level |
| QaTa-COV19 | LViT-T vs. RecLMIS | −3.06 | [−3.42, −2.69] | 1.10e-55 | RecLMIS better |
| QaTa-COV19 | LViT-T vs. DD-CMD | −2.93 | [−3.31, −2.54] | 1.91e-46 | DD-CMD better |
| QaTa-COV19 | nnU-Net vs. LViT-T | −1.99 | [−2.52, −1.46] | 1.58e-12 | LViT-T better |
| QaTa-COV19 | U-Net vs. nnU-Net | −0.54 | [−0.97, −0.11] | 0.057 | level |
| MosMedData+ | U-Net vs. RecLMIS | −0.43 | [−1.60, +0.74] | 0.637 | level |
| MosMedData+ | U-Net vs. DD-CMD | −0.84 | [−2.04, +0.37] | 0.517 | level |
| MosMedData+ | RecLMIS vs. DD-CMD | −0.41 | [−1.22, +0.40] | 0.637 | level |
| MosMedData+ | nnU-Net vs. RecLMIS | −2.38 | [−3.59, −1.17] | 6.58e-04 | RecLMIS better |
| MosMedData+ | nnU-Net vs. DD-CMD | −2.79 | [−3.85, −1.73] | 3.49e-06 | DD-CMD better |
| MosMedData+ | U-Net vs. LViT-T | +12.75 | [+9.40, +16.11] | 1.17e-11 | U-Net better |
| MosMedData+ | nnU-Net vs. LViT-T | +10.80 | [+7.21, +14.40] | 8.75e-08 | nnU-Net better |

- **On QaTa-COV19** all 9 language-guided vs. image-only pairs favour the language-guided method. Among the
  image-only methods no pair is significant. U-Net vs. nnU-Net was significant in Phase 0 (Holm across 3
  pairs, p = 0.043) and is not here (Holm across 15 pairs, p = 0.057): same raw p (0.014), stricter
  correction; the two results do not contradict each other.
- **On MosMedData+** only 4 of the 9 language-guided vs. image-only pairs favour the language-guided method
  (RecLMIS and DD-CMD vs. TransUNet and nnU-Net); 2 are level (vs. U-Net) and 3 favour the image-only method
  (all three vs. LViT-T). The last 3 depend on LViT-T's pre-fixed setting (see the note in section 1, item 4).
- **Boundary metrics** separate methods that Dice does not: on MosMedData+ U-Net's HD95 is 7.01 px
  [3.35, 10.67] above RecLMIS and 6.17 px [3.29, 9.05] above DD-CMD.
- **The supplementary Wilcoxon test** (not used for the conclusions) disagrees with the t-test on 7 pairs,
  e.g. QaTa-COV19 RecLMIS vs. DD-CMD (Wilcoxon Holm p = 5.28e-04); the list is in PHASE1_RESULTS.md.
- **MosMedData+ p-values may be too small:** neighbouring slices of one CT volume are correlated, but the
  tests treat each slice as independent.

## 4. Real cases

Selection rule: each test case's Dice is averaged over the 3 seeds and the 6 methods, and the cases at the
10/30/50/70/90% quantiles of this score are shown. The drawn predictions and the Dice / HD95 labels are
those of **seed 1001**. Ground truth: yellow dashed contour.

On QaTa-COV19 the methods differ most on the harder cases: at q10 nnU-Net reaches Dice 26.1 (HD95 91.0)
while LViT-T and RecLMIS reach 73.3 (HD95 40.2 and 17.2); at q70 nnU-Net covers only part of one lung region
(Dice 78.6, HD95 31.3) while the other four reach 95.2–96.8. At q90 all five reach 96.6–97.3.

![QaTa-COV19 cases](figures/phase1_cases_QaTa-COV19.png)

On MosMedData+ no method is best in every row: at q10 U-Net has the highest Dice (71.8, HD95 3.2) and
LViT-T predicts additional regions away from the lesion (HD95 92.6); at q70 DD-CMD has the highest Dice
(85.6, HD95 2.2).

![MosMedData+ cases](figures/phase1_cases_MosMedDataplus.png)

## 5. Comparison with the papers

Ours: mean over 3 seeds, common evaluator. Papers: each paper's own setting and metric definition. Dice /
mIoU in %.

| Dataset | Method | Ours Dice | Paper Dice | Δ | Ours mIoU | Paper mIoU | Δ | Paper source |
|---|---|---:|---:|---:|---:|---:|---:|---|
| QaTa-COV19 | LViT-T | 81.71 | 83.66 | −1.95 | 72.90 | 75.11 | −2.21 | LViT Table II (mean of 3 runs, Table IV) |
| QaTa-COV19 | RecLMIS | 84.77 | 85.22 | −0.45 | 76.46 | 77.00 | −0.54 | RecLMIS Table I (arXiv 2404.02845) |
| QaTa-COV19 | DD-CMD | 84.64 | 91.46 | −6.82 | 76.16 | 84.26 | −8.10 | DD-CMD (arXiv 2608.11335), pixel-pooled Dice |
| MosMedData+ | LViT-T | 61.26 | 74.57 | −13.31 | 49.91 | 61.33 | −11.42 | LViT Table II (mean of 3 runs, Table IV) |
| MosMedData+ | RecLMIS | 74.44 | 77.48 | −3.04 | 61.35 | 65.07 | −3.72 | RecLMIS Table I (arXiv 2404.02845) |
| MosMedData+ | DD-CMD | 74.85 | 81.95 | −7.10 | 62.12 | 69.42 | −7.30 | DD-CMD (arXiv 2608.11335), pixel-pooled Dice |

- **LViT-T:** our metric matches LViT's own test code exactly (checked in Phase 0). On QaTa-COV19 we are
  1.95 points below the paper's mean of three runs.
- **RecLMIS:** within 0.5 points on QaTa-COV19 and 3.0 points below on MosMedData+. The paper does not state
  whether its values come from one or several runs, and its metric definition has not been checked.
- **DD-CMD:** the paper uses a pixel-pooled Dice, not the per-case mean used here, so the values are not
  directly comparable.
- **Limitation — LViT-T on MosMedData+ is about 13 points below the paper.** Every seed has 38–41 empty test
  predictions out of 273 (every other method: 0–6). Our LViT-T uses batch size 2 (from the official code, a
  comment specific to LViT-T) with the MosMedData+ learning rate 1e-3 (from the paper); neither the paper
  nor the code explicitly uses this combination. No setting of the main runs was changed because of it.
  In a sensitivity check decided after the main results had been seen, with a plan written and pushed
  beforehand, only the batch size was changed from 2 to 24. LViT-T on MosMedData+ then reached a test Dice of
  74.01 ± 1.11 (main runs 61.26 ± 1.10; paper 74.57), and the empty predictions fell from 38–41 to 0–2 per
  seed ([PHASE1_SENSITIVITY_LVIT.md](PHASE1_SENSITIVITY_LVIT.md); descriptive, no significance test). The main
  results above are unchanged.
  A post-hoc supplementary MosMedData+ table (decided after the results had been seen; it does not replace the
  main results) repeats the fixed six-method analysis with these batch-24 runs in place of the main LViT-T runs
  ([PHASE1_SENSITIVITY_LVIT.md, section 5](PHASE1_SENSITIVITY_LVIT.md#5-post-hoc-supplementary-analysis-mosmeddata-six-method-comparison-with-lvit-t-batch-24)).
  In it, U-Net, LViT-T, RecLMIS and DD-CMD (Dice 74.01–74.85) are not significantly different from each other
  (Holm p ≥ 0.249), and all four are significantly better than nnU-Net and TransUNet.

## 6. Main issues found during the reproduction and how they were handled

- **RecLMIS could silently switch off its core losses.** A missing dependency (fairseq, which cannot be
  built here) makes the official bare `except` drop the contrastive loss and both reconstruction losses
  without an error; we provide fairseq's `softmax` function and stop the run if any of these losses is
  missing.
- **RecLMIS loads the QaTa config for MosMedData+** (a bug in the official `train_model.py`); our driver
  loads the correct config for each dataset and checks every value.
- **One NaN in 18 runs.** RecLMIS, QaTa-COV19, seed 1001: `loss_ccl` became NaN at epoch 152. Under a rule
  fixed before any further result, training ended and the best validation checkpoint (epoch 129) was
  evaluated, the same one the official code would end with. No other run had a non-finite loss.
- **DD-CMD hung with 8 data-loader workers** before the first training batch; a diagnostic job showed that
  the forked workers hung, so we use the official 0 workers.
- **DD-CMD needs the alias `np.Inf`**, which NumPy 2 removed; it is restored with the identical value.
- **DD-CMD checkpoint selection and early stopping use the per-case validation Dice** instead of the
  official pixel-pooled validation mIoU, so all six methods are selected by the same rule.
- **LViT's released data loader cannot pair QaTa-COV19 images and masks** (it uses MoNuSeg file naming);
  we use its own commented-out QaTa lines.
- **LViT-T's text encoder (MXNet `bert_embedding`) does not run on this cluster;** it was re-implemented in
  PyTorch with the same BERT weights. It could not be checked against the original MXNet output.

Details: [PHASE1_MODEL_SETUP.md](PHASE1_MODEL_SETUP.md).

## 7. Next step: Phase 2

Phase 1 compares different architectures, so it cannot say whether these models actually use the text.
Phase 2 tests text dependence within the same frozen model, for example by evaluating each trained model
with its own text and with altered texts. A quick check on small models already showed a difference between
datasets: replacing every text by one fixed text lowered the validation Dice of all three methods on
QaTa-COV19 but left it unchanged to the third decimal on MosMedData+ (PHASE1_MODEL_SETUP.md, section 5).
