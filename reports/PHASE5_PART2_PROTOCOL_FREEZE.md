# Phase 5 Part 2 — Protocol freeze (U-Net backbone sensitivity, post-hoc)

Status: 2026-11-11, **frozen** when this file and `configs/phase5_part2/artifact_manifest.tsv` are committed
together.
Plan: [PHASE5_PART2_PLAN.md](PHASE5_PART2_PLAN.md), commit c8854eb, sha256
`157c67db89bc0e1a34ad6cfc9a49ceafc3d061590f359b2d85471932c67b8120`.
Rules: `configs/phase5_part2/spec.json`, commit 90a7bd2, sha256
`c04cc7a139d973ebea54f3872003764ec7db679f6ad5508c026dfb447fbfe0f9`.
Setup: [PHASE5_PART2_SETUP.md](PHASE5_PART2_SETUP.md), commit 761b335.

**Status of this analysis.** Phase 5 Part 2 is a **post-hoc sensitivity analysis**. The QaTa-COV19 test split was
already evaluated and its results seen in Phases 2, 3 and 5; this analysis is therefore not an independent
confirmation. It asks only whether the Phase 5 Fixed Average finding depends on nnU-Net being the image-only
reference, by replacing it with the frozen Phase 0 U-Net. No segmentation model is trained, no gate or SDC rule is
fitted, no weight or threshold is searched, no text is generated, TransUNet and MosMedData+ are not run. All
outputs are descriptive; the results and pairwise tables carry the label "post-hoc sensitivity analysis,
descriptive (test split seen in Phases 2, 3 and 5)".

## 1. Reconciled U-Net artifacts (plan item 1)

The U-Net test probabilities were produced by inference only with the unchanged `src/phase3/infer_img.py` (job 479,
HEAD 90a7bd2) from the Phase 0 checkpoints (`results/phase3_checkpoints.tsv` rows 20, 22, 24), and reproduce the
Phase 0 archive exactly for every seed: 2113 images per seed, 0 images with a mask difference, 0 differing pixels,
|ΔDice| = 0 on the stored per-case files, and masks.npz = packbits(P > 0.5) (details and hashes in
[PHASE5_PART2_SETUP.md](PHASE5_PART2_SETUP.md)). For each seed k, `probs.npz` (float32 P_img_unet), `masks.npz`
(B_img_unet = P > 0.5), `per_case.csv` and `reconcile_phase0.csv` under `runs/phase3/test/img/unet/QaTa-COV19/seed<k>/`
are frozen by their sha256 in the manifest (section 8), together with
`results/phase5_part2_unet_img_test_per_case.csv.gz`.

## 2. Fixed fusion rule (item 2)

P_avg_unet = 0.5 · P_lang + 0.5 · P_img_unet from the stored float32 maps, **computed in float64**
(`src/phase5/core.py` `fixed_average`, as in Phase 5); B_avg_unet = P_avg_unet > 0.5 (a pixel exactly 0.5 is
background). Scored anew with `core.score_mask` = `lgms.metrics.case_metrics(mask, gt > 0)` plus empty = not
B_avg_unet.any(). The weight 0.5 and the threshold 0.5 are not tuned. The LANG maps are the same stored files as in
Phase 5. Matched seeds: LANG seed k with U-Net seed k (k = 1001, 1002, 1003).

## 3. Eligible conditions, rows and donor manifest (item 3)

QaTa-COV19 test; LViT-T, RecLMIS, DD-CMD × seeds 1001–1003; conditions **C0, C2, C4, C5, C6**. The rows, keys
(including the donor of every mismatched row) and verified groups are those of
`results/phase3_feature_labels_test.csv.gz`, i.e. the exact Phase 2 donor manifest and the Phase 3 grouping:
3 × 3 × (2113 + 2113 + 3 × 2081 × 3) = **206,595** rows; C0 and C2 2113 images / 474 groups each, C4, C5, C6 2081
images / 472 groups each. The program asserts these counts before writing any output. Every LANG and U-Net archived
mask is checked to equal packbits(P > 0.5) row by row. LANG and U-Net IMG per-case metrics are the archived ones
(`results/phase2_test_per_case.csv.gz`; U-Net `per_case.csv`); ground truth is read with
`lgms.data.load_split_arrays('QaTa-COV19', 'test')` under `slurm/env_phase1.sh` in each worker.

## 4. Aggregation rules (item 4)

- **Tables** (`src/phase3/evaluate_gates.py` `image_weighted`): per seed, donors averaged within image, then images;
  mean ± SD over the three seeds. Strategies LANG, U-Net IMG, U-Net Fixed Average; Dice, mIoU, HD95, ASSD, empty
  rate; all five conditions.
- **Paired differences** (`src/phase3/stats_phase3.py` `group_differences`, `t_over_groups`): donor differences
  within image and seed → three seeds within image → images within group → equal-weight groups; mean and 95%
  t-interval over groups (ddof 1). **No p-values are output.**
  - Main (C4, C5, C6, per model): U-Net Fixed Average − LANG; U-Net Fixed Average − U-Net IMG.
  - Clean-text context (C0, C2, per model): U-Net Fixed Average − LANG; U-Net IMG − LANG. This describes the
    robustness / clean-performance trade-off only; no equivalence or safety is claimed from an interval.

## 5. Mixed-reliability formula (item 5)

As in Phase 5. Population: the common targets eligible for C4, C5 and C6 (2081 images, 472 groups; asserted), C0
on the same targets. D(C) = group-weighted mean Dice (`core.group_weighted_mean`: donors → seeds → images →
groups). D_wrong = [D(C4) + D(C5) + D(C6)] / 3 (equal weight; a benchmark stress-test mixture, not a real error
distribution). E(p) = (1 − p) D(C0) + p D_wrong, tabulated at p = 0.00, 0.01, …, 1.00 for summary values only.
Exact crossovers (`core.crossover`): U-Net Fixed Average vs LANG and vs U-Net IMG, p* = −Δ_clean / (Δ_wrong −
Δ_clean), reported when 0 ≤ p* ≤ 1, otherwise "none". The range of p where U-Net Fixed Average is better than both
(`core.better_interval` for each comparison, then `core.intersect`; an empty or single-point intersection is
"none") is written as plain floats with its width. Sensitivity: the same with C4, C5 or C6 alone. No other mixture
is introduced. p is a scenario parameter, not an estimate of report-error prevalence, and a crossover is not a
clinical threshold.

## 6. Backbone-consistency table and judging rule (item 6, Task 5.2.5)

One row per model (LViT-T, RecLMIS, DD-CMD), columns:

- **IMG clean Dice**: image-weighted Dice of IMG under C0 (mean of 3 seeds). nnU-Net read from
  `results/phase5_test_results.csv` (strategy IMG, C0); U-Net from this analysis.
- **Fixed Average gain over LANG under C4, C5, C6**: paired group-weighted difference and 95% CI (section 4) for
  both backbones. The nnU-Net values are computed in this run from the committed Phase 5 per-row results
  (`results/phase5_test_per_case.csv.gz`, Fixed Average) and `results/phase2_test_per_case.csv.gz` (LANG), a
  descriptive statistic with no new inference. Self-check: the image-weighted Dice difference Fixed Average − LANG
  from these per-row data must equal `results/phase5_test_results.csv` within 1e-6 for every model × condition
  (C0, C2, C4, C5, C6), otherwise the program stops. These nnU-Net rows are also written to
  `results/phase5_part2_unet_pairwise.csv` with kind "nnU-Net gain".
- **p range better than both LANG and IMG** (equal C4/C5/C6 mixture): lower end, upper end and width. nnU-Net
  parsed from `results/phase5_crossover_summary.csv` (column `better_than_both_p_range`, whose text contains
  `np.float64(...)`; the numbers are parsed from it); U-Net from section 5.
- **Row verdict** for each backbone and the **qualitative conclusion**.

**Judging rule** (fixed here before the test computation; `src/phase5_part2/compute.py` `row_consistent` and
`overall`, sha256 in section 7):

1. A model row is **consistent** for a backbone if (a) the 95% CI lower bound of Fixed Average − LANG is > 0 under
   **each** of C4, C5 and C6, **and** (b) the range of p where Fixed Average is better than both LANG and IMG
   (equal C4/C5/C6 mixture) exists and has width > 0. Otherwise it is **not consistent**.
2. The same rule is applied to the nnU-Net half (for reference) and to the U-Net half.
3. The overall conclusion uses the **U-Net rows only**: if all three U-Net rows are consistent, it is the plan's
   sentence "Conclusion: the simple-fusion mitigation finding is not specific to nnU-Net."; if any U-Net row is not
   consistent, it is "Conclusion: the mitigation effect is backbone-dependent and the manuscript claim must be
   limited accordingly."
4. Numerical equality across backbones is not required. The widths of both p ranges and all CIs are reported next
   to the verdicts, so that a narrow range is visible even when the rule is met.

## 7. Required outputs and analysis code (item 6)

Outputs (written only by the test computation; absent beforehand, never overwritten):
`results/phase5_part2_unet_test_results.csv`, `results/phase5_part2_unet_pairwise.csv`,
`results/phase5_part2_unet_mixed_reliability.csv`, `results/phase5_part2_unet_crossovers.csv`,
`results/phase5_part2_backbone_consistency.csv`; for independent checking
`results/phase5_part2_unet_test_per_case.csv.gz` (keys and U-Net Fixed Average Dice / mIoU / HD95 / ASSD / empty,
`%.17g`); the append-only run log `results/phase5_part2_test_run_log.tsv`. Report:
`reports/PHASE5_PART2_UNET_SENSITIVITY.md` (the plan's seven questions only). No figure.

Code commit **761b335** (HEAD when the manifest was made, job 480; the code itself was last changed in 90a7bd2).
The test computation is `src/phase5_part2/compute.py` (job `slurm/phase5_part2_test.slurm`). The code files below —
the Part 2 programs, every in-repository module they import (collected by importing them) and the job scripts —
are hashed in the manifest and checked again before the test computation:

| File | sha256 |
|---|---|
| `src/analyze_phase0.py` | `026a6908e3097b5d2d4c7bbe4fd8af84d3c0683e407398175d1807e6509eac55` |
| `src/lgms/__init__.py` | `240a87f493dc274340a97b06b334bb471dffdb8df2d578baf87a48b75bc21593` |
| `src/lgms/data.py` | `6c8a87efe261f66ec1bcdf941e22c7e8cb81d557a4cc29a616ec129c03861ff6` |
| `src/lgms/metrics.py` | `a1bb39144081810b9c6f81e6ad33120046b961b544ab33866fa7d00542dfb6ce` |
| `src/phase1/__init__.py` | `a0e520eeb011f9154fe231bce8f930e1c8613a54a69a3599f415d2564cca50c0` |
| `src/phase1/analysis_lock.py` | `8774c05e5ae3a2f873bf200f393edaf49331cd97df758012105685c8b673cf24` |
| `src/phase2/infer.py` | `e7f17195caa9f714a1f23fdf4543116e011274cda26445accb1150ecd041f3ad` |
| `src/phase2/lock.py` | `70fca4bd7806f533a0061dd1054e4f57ca65fbbabeef3e9a19f4399fdea65a64` |
| `src/phase3/evaluate_gates.py` | `503b59bb38a0af6f2437f7714bb0302ebd3f8f7933f708006b62abca36fe6674` |
| `src/phase3/features.py` | `ef725248800431ca00f188270327b539db4526ed8e4c5c12da1f7280ce34ad4e` |
| `src/phase3/infer_img.py` | `bc4a7193359316b536ff46f896e45c0b611f64a449042097fc3b3434e6a07a00` |
| `src/phase3/stats_phase3.py` | `11fd0150797a441020d33e6bf216eeccd53db8b153664172a7d2c4af16525ed5` |
| `src/phase5/core.py` | `b463f589cc4e8db661315d4e45f6f7c4b5623fada78be3afc32750e53128fc68` |
| `src/phase5/guards.py` | `8721ae234528631cbd3dfbd4206218901cc000f8dbdacbcf7414cdc815072a3c` |
| `src/phase5_part2/compute.py` | `75018c15aff6b9cf45793f1d47b16474037d614c7fda1fb4cd53ff54ab54245d` |
| `src/phase5_part2/make_manifest.py` | `b665139239d52bd4101f6b6eb87dcdada2684c20a740a15b92f4e93847b92331` |
| `src/phase5_part2/reconcile_unet.py` | `f6adb46b5ef51809061e5fa7bdfc25c5067e5c41cd38ba1b0375536c0fe71aab` |
| `src/phase5_part2/test_phase5_part2.py` | `0df09009e48bde9c47951aab0cae404a2407691b87f3c17e1e3112f83d3842d3` |
| `slurm/phase5_part2_test.slurm` | `182ef692b07d1edc4e8ff6cee2a464d54ccd086f3c85cf4c1210ba0d11bee2a4` |
| `slurm/phase5_part2_unet_infer.slurm` | `7d0ac2b7ab892381ff10267eb52e6972d0f99a87a7e6d85b6267bd704df43f35` |
| `slurm/env_phase1.sh` | `7e6d3d6cfcf3ca78ba660584d7ca2e1ba3d89d9e2066eec679b08ef53d72aa25` |
| `slurm/env.sh` | `4a380a68eec9a148be5917211b514a9b360e255aca348c2d6ef42e7b6268e704` |
| `slurm/notify_trap.sh` | `62a768772156bc7ccb25bd645a62560cacb035242e2a6a8a7b4ee04ea6966e93` |

The spec and this document are not in the manifest; the program checks that each was committed exactly once,
is unchanged and is an ancestor of HEAD.

## 8. Artifact hashes

`configs/phase5_part2/artifact_manifest.tsv`: **2452 files, 39,602,025,988 bytes (36.9 GiB)**, made by
`src/phase5_part2/make_manifest.py` in job 480 (2026-11-11, HEAD 761b335); sha256 of the manifest file
`69683bbc225526817faa32b8c14ffa1a8e74506424d706e6116d1681aadd260c`. It lists every input of the test computation:
the test LANG probability maps, archived masks and per-case files of C0, C2, C4–C6 (297 files), the U-Net
probability maps, masks, per-case and reconciliation files (12), the QaTa test ground-truth masks and id list
(2114), the existing tables (6: `phase3_feature_labels_test.csv.gz`, `phase2_test_per_case.csv.gz`,
`phase5_test_per_case.csv.gz`, `phase5_test_results.csv`, `phase5_crossover_summary.csv`,
`phase5_part2_unet_img_test_per_case.csv.gz`), and the code and job scripts (23). The nnU-Net probability maps are
not inputs (the nnU-Net half is read from the committed Phase 5 tables). The code and table hashes were checked
against HEAD 761b335 (0 differences).

## 9. Running the test computation

- It runs **once**, as Slurm job `slurm/phase5_part2_test.slurm`, submitted by hand after this freeze is committed
  and checked. Before reading any test data the program checks: Phase 2 lock open; spec, this document and the
  manifest each committed once and unchanged; clean tree; HEAD on origin/main; every manifest file with its
  recorded sha256; no result file present.
- If the job stops for a cluster reason before any result file is written: the run log
  `results/phase5_part2_test_run_log.tsv` written by that attempt is an untracked file, which the clean-tree check
  reports, and `results/` is not ignored by git. The run log is therefore first committed and pushed unchanged (a
  public record of the failed attempt); then the same Slurm job is resubmitted unchanged. Code and tests are not
  changed. The log records every attempt.
- If the job stops after any result file has been written, or the nnU-Net self-check (section 6) or a count check
  fails, nothing is rerun; the work stops and the case is reported.
- If any code or rule has to change, the work stops first; the change is written down as an amendment, and only
  then is it decided whether to run.
- The fusion rule, the judging rule and the IMG backbones are not changed after the result is seen; no other
  backbone (e.g. TransUNet) and no other averaging weight is tried to change the conclusion.
