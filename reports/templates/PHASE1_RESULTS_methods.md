## Methods (fixed analysis plan)

This section is fixed together with `src/analyze_phase1.py`. Both are protected by the program lock
(`src/phase1/analysis_lock.py`): the test split becomes readable, and the analysis can be run on it, only
after both are committed and pushed and their sha256 are recorded. The plan is `reports/PHASE1_PLAN.md`,
Tasks 1.7–1.11.

- **Methods (6), in this fixed order:** U-Net, TransUNet, nnU-Net (image-only, Phase 0, unchanged) and
  LViT-T, RecLMIS, DD-CMD (language-guided, Phase 1). Each method has 3 seeds (1001, 1002, 1003). The
  checkpoint is selected on validation Dice only.
- **Common evaluation:** 224 × 224, binary threshold 0.5, the same test cases and the same target masks
  for every method (`src/evaluate.py`). Metrics: Dice, mIoU (%), HD95 and ASSD (pixels of the 224 × 224
  image). An empty prediction gets HD95 = ASSD = 316.78 px, the image diagonal.
- **Main tables** (`results/phase1_main_results.csv`: the 3 language-guided methods;
  `results/final_six_method_results.csv`: all 6, with "Text at Train / Test"):
  - Mean ± SD: the mean and the sample SD (ddof = 1) of the 3 per-seed test means, for every metric.
  - Dice 95% CI: case bootstrap of the seed-averaged per-case Dice. Each case's Dice is averaged over the
    3 seeds; the cases are resampled 10,000 times with numpy `default_rng(0)`, created anew for every
    dataset × method; the CI is the 2.5% / 97.5% percentile interval.
- **Pairwise comparisons** (`results/final_six_method_pairwise.csv`):
  - The 15 pairs of the 6 methods, within each dataset. Δ = A − B, where A comes earlier in the fixed
    order above.
  - Steps:
    1. Average every test case over the 3 seeds within each method.
    2. Align the cases by sample id.
    3. Compute the per-case paired differences d.
  - **Dice (primary):**
    - ΔDice = mean(d), in Dice points.
    - 95% CI = mean(d) ± t(0.975, n − 1) · SD(d) / √n.
    - Two-sided paired t-test.
    - **Holm correction across the 15 comparisons of the dataset.**
    - **Significance level: two-sided α = 0.05 after Holm, fixed before any Phase 1 test result
      existed.**
  - **Supplementary sensitivity analysis, specified in advance:** two-sided Wilcoxon signed-rank test on
    the same differences, with its own Holm correction across the same 15 comparisons. It is not used for
    the main conclusions.
  - **mIoU, HD95 and ASSD:** paired mean difference with the same t-based 95% CI, and no p-values. Because
    HD95 and ASSD are heavy-tailed, a case-bootstrap percentile CI of the mean difference (10,000
    resamples, `default_rng(0)`) is reported next to the t-CI.
  - Significance is never tested on the three seed means alone.
- **Limitation:** the case bootstrap and the paired tests treat every image as an independent sample.
  Neighbouring MosMedData+ slices of one CT volume are correlated, so the MosMed CIs may be too narrow and
  the p-values too small.
- **Causality:** comparing different architectures cannot show that the text itself causes a difference.
  Text dependence within one frozen model is tested in Phase 2.
- **Figures** (`reports/figures/`):
  - `phase1_dice_ci_<dataset>.png`: mean Dice with its 95% CI, 6 methods (required figures 1 and 2).
  - `phase1_seed_stability.png`: the 3 per-seed test Dice values with mean ± SD, for the 3
    language-guided methods, with the 3 image-only methods for reference (required figure 3).
  - `phase1_cases_<dataset>.png`: real cases, selected by rule.
    - Selection score of a case = its Dice averaged over the 3 seeds and the 6 methods.
    - Cases are sorted by this score (ascending; ties broken by sample id).
    - The cases at the 10/30/50/70/90% quantiles are shown (position round(q × (n − 1))).
    - Each row shows: image, ground truth (yellow dashed contour), and the seed-1001 prediction of U-Net,
      nnU-Net, LViT-T, RecLMIS and DD-CMD. Every panel is labelled with that method's seed-1001 Dice and
      HD95 for the case.
