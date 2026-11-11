# Phase 5 Part 2 — U-Net backbone sensitivity (post-hoc)

Plan: [PHASE5_PART2_PLAN.md](PHASE5_PART2_PLAN.md). Setup: [PHASE5_PART2_SETUP.md](PHASE5_PART2_SETUP.md).
Freeze: [PHASE5_PART2_PROTOCOL_FREEZE.md](PHASE5_PART2_PROTOCOL_FREEZE.md) (commit a66c80a). Test computation: Slurm job
481 (HEAD a66c80a), outputs committed in c3451e2 (`results/phase5_part2_*.csv`).

**Status.** Post-hoc sensitivity analysis on QaTa-COV19 test; this split was already examined in Phases 2, 3 and 5, so
the results are descriptive and not an independent confirmation. Language-guided models (LANG): LViT-T, RecLMIS,
DD-CMD, seeds 1001–1003, each seed matched with the frozen Phase 0 U-Net of the same seed. U-Net Fixed Average (FA):
P = 0.5 · P_lang + 0.5 · P_img_unet, mask P > 0.5, no tuning. Differences are paired group-weighted Dice differences
in **Dice points (Dice × 100)** with 95% t-intervals over verified groups (472 groups for C4–C6, 474 for C0/C2); no
p-values. Image-weighted means of all metrics are in `results/phase5_part2_unet_test_results.csv`. No new figure.

## 1. Were the frozen U-Net artifacts reproduced correctly?

Yes. Inference only with the unchanged program reproduced the Phase 0 archive exactly for all three seeds
(2113 images each): 0 images with a mask difference, maximum pixel difference 0, mean and maximum |ΔDice| 0, and the
stored masks equal packbits(P > 0.5) (SETUP section 4).

## 2. Under C4–C6, does U-Net Fixed Average improve over LANG?

Yes, in all 9 model × condition cells; every 95% CI lies above 0 (smallest lower bound +2.99).

U-Net FA − LANG, Dice points, mean [95% CI]:

| Model | C4 | C5 | C6 |
|---|---:|---:|---:|
| LViT-T | +9.89 [+9.07, +10.72] | +7.19 [+6.50, +7.87] | +15.03 [+14.27, +15.80] |
| RecLMIS | +8.31 [+7.56, +9.05] | +7.04 [+6.43, +7.64] | +13.17 [+12.35, +13.99] |
| DD-CMD | +3.21 [+2.99, +3.43] | +3.89 [+3.64, +4.13] | +5.39 [+5.07, +5.71] |

## 3. How does U-Net Fixed Average compare with U-Net IMG under C4–C6?

U-Net FA is lower than U-Net IMG in 8 of 9 cells, with the CI entirely below 0; the exception is RecLMIS C5
(+0.27 [-0.32, +0.86], CI includes 0). With mismatched text, U-Net IMG alone is therefore
higher in these 8 cells. FA is a compromise between correct and mismatched text, not the best strategy under mismatched text.

U-Net FA − U-Net IMG, Dice points, mean [95% CI]:

| Model | C4 | C5 | C6 |
|---|---:|---:|---:|
| LViT-T | -2.83 [-3.67, -2.00] | -0.75 [-1.29, -0.21] | -5.12 [-5.83, -4.40] |
| RecLMIS | -1.23 [-1.98, -0.47] | +0.27 [-0.32, +0.86] | -3.76 [-4.47, -3.05] |
| DD-CMD | -3.96 [-4.97, -2.94] | -2.33 [-3.14, -1.52] | -10.98 [-12.03, -9.93] |

## 4. What are the C0/C2 clean-text effects?

With correct text (C0, C2), U-Net FA − LANG is positive with the CI above 0 for LViT-T and DD-CMD;
for RecLMIS it is negative under C0 (CI below 0) and its CI includes 0 under C2. Across the six cells it ranges
from -0.67 to +1.10. U-Net IMG alone is below LANG in all six cells (CI below 0). These intervals describe the
trade-off only; they are not used to claim equivalence or absence of harm.

U-Net FA − LANG and U-Net IMG − LANG, Dice points, mean [95% CI]:

| Model | FA − LANG C0 | FA − LANG C2 | IMG − LANG C0 | IMG − LANG C2 |
|---|---:|---:|---:|---:|
| LViT-T | +1.10 [+0.50, +1.70] | +0.98 [+0.39, +1.57] | -3.07 [-4.09, -2.05] | -2.98 [-3.99, -1.98] |
| RecLMIS | -0.67 [-1.12, -0.22] | -0.06 [-0.56, +0.45] | -7.37 [-8.43, -6.32] | -6.13 [-7.18, -5.07] |
| DD-CMD | +0.42 [+0.27, +0.57] | +0.41 [+0.26, +0.56] | -7.30 [-8.30, -6.31] | -7.30 [-8.29, -6.30] |

## 5. What mixed-reliability range favors U-Net Fixed Average over both LANG and U-Net IMG?

On the common C4/C5/C6 targets (2081 images, 472 groups), with E(p) = (1 − p) D(C0) + p D_wrong and
D_wrong = [D(C4) + D(C5) + D(C6)] / 3, U-Net FA is better than both LANG and U-Net IMG for p in
LViT-T 0.000–0.593, RecLMIS 0.068–0.812 and DD-CMD 0.000–0.575. "none" for p* vs LANG means FA is
above LANG at every p (it is above LANG on correct text as well). p is a scenario parameter of this benchmark
mixture, not an estimate of the real prevalence of report errors, and a crossover is not a decision threshold for practice.

Equal C4/C5/C6 mixture (`results/phase5_part2_unet_crossovers.csv`):

| Model | p* vs LANG | p* vs U-Net IMG | U-Net FA better than both |
|---|---:|---:|---|
| LViT-T | none | 0.593 | 0.000–0.593 (width 0.593) |
| RecLMIS | 0.068 | 0.812 | 0.068–0.812 (width 0.744) |
| DD-CMD | none | 0.575 | 0.000–0.575 (width 0.575) |

Each mismatch condition alone (sensitivity). For RecLMIS C5, p* vs U-Net IMG is "none" because FA is above U-Net IMG
at every p in this mixture (on the common targets FA is above U-Net IMG under both C0 and C5):

| Model | Mixture | p* vs LANG | p* vs U-Net IMG | U-Net FA better than both |
|---|---|---:|---:|---|
| LViT-T | C4 only | none | 0.599 | 0.000–0.599 (width 0.599) |
| LViT-T | C5 only | none | 0.850 | 0.000–0.850 (width 0.850) |
| LViT-T | C6 only | none | 0.452 | 0.000–0.452 (width 0.452) |
| RecLMIS | C4 only | 0.077 | 0.847 | 0.077–0.847 (width 0.770) |
| RecLMIS | C5 only | 0.090 | none | 0.090–1.000 (width 0.910) |
| RecLMIS | C6 only | 0.050 | 0.643 | 0.050–0.643 (width 0.593) |
| DD-CMD | C4 only | none | 0.663 | 0.000–0.663 (width 0.663) |
| DD-CMD | C5 only | none | 0.770 | 0.000–0.770 (width 0.770) |
| DD-CMD | C6 only | none | 0.415 | 0.000–0.415 (width 0.415) |

## 6. Is the simple-fusion finding qualitatively consistent between nnU-Net and U-Net?

Yes, by the judging rule frozen before the test computation (freeze section 6: all three C4–C6 CI lower bounds of
FA − LANG above 0 and a p range of width > 0 where FA beats both LANG and IMG): all three U-Net rows and all three
nnU-Net rows are "consistent". The table's conclusion, verbatim:

> Conclusion: the simple-fusion mitigation finding is not specific to nnU-Net.

`results/phase5_part2_backbone_consistency.csv` (Dice points; FA − LANG as mean [95% CI]; nnU-Net values from the
committed Phase 5 results):

| Model | Backbone | IMG clean Dice (C0) | FA − LANG C4 | FA − LANG C5 | FA − LANG C6 | p range better than both (width) | Row |
|---|---|---:|---:|---:|---:|---|---|
| LViT-T | nnU-Net | 79.72 | +12.04 [+11.08, +13.01] | +8.90 [+8.11, +9.69] | +17.35 [+16.47, +18.23] | 0.000–0.689 (0.689) | consistent |
| LViT-T | U-Net | 79.18 | +9.89 [+9.07, +10.72] | +7.19 [+6.50, +7.87] | +15.03 [+14.27, +15.80] | 0.000–0.593 (0.593) | consistent |
| RecLMIS | nnU-Net | 79.72 | +9.69 [+8.84, +10.54] | +8.10 [+7.38, +8.82] | +15.02 [+14.08, +15.95] | 0.117–0.841 (0.724) | consistent |
| RecLMIS | U-Net | 79.18 | +8.31 [+7.56, +9.05] | +7.04 [+6.43, +7.64] | +13.17 [+12.35, +13.99] | 0.068–0.812 (0.744) | consistent |
| DD-CMD | nnU-Net | 79.72 | +5.04 [+4.67, +5.40] | +6.03 [+5.60, +6.45] | +8.17 [+7.66, +8.69] | 0.000–0.606 (0.606) | consistent |
| DD-CMD | U-Net | 79.18 | +3.21 [+2.99, +3.43] | +3.89 [+3.64, +4.13] | +5.39 [+5.07, +5.71] | 0.000–0.575 (0.575) | consistent |

The two backbones differ in size, not in direction. The U-Net FA gain over LANG is smaller than the nnU-Net gain in
all 9 model × condition cells (U-Net +3.21 to +15.03; nnU-Net +5.04 to +17.35). The p range is
narrower with U-Net for LViT-T (0.593 vs 0.689) and DD-CMD
(0.575 vs 0.606); for RecLMIS it is shifted to lower p
(0.068–0.812 vs 0.117–0.841) and slightly wider (0.744 vs 0.724).

## 7. What exact claim should be carried into the EMBC manuscript?

> In a post-hoc analysis on QaTa-COV19 with benchmark-constructed mismatched text (C4–C6), fixed equal-weight
> averaging of the language-guided and image-only probability maps gave higher Dice than the language-guided model
> alone for all three language-guided models with both image-only backbones tested (nnU-Net and U-Net; all 95% CIs
> above zero), whereas under mismatched text the U-Net image-only model alone gave higher Dice than the average in 8
> of 9 model–condition comparisons, and the effect on correct text depended on the model (U-Net: -0.67 to +1.10 Dice points under C0/C2).
