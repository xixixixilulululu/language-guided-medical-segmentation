# Phase 5 — Protocol freeze (post-hoc supplementary comparison)

Status: 2026-11-10, **frozen** when this file and `configs/phase5/artifact_manifest.tsv` are committed together.
Plan: [PHASE5_PLAN.md](PHASE5_PLAN.md), commit b97d07f, sha256
`8c415df813cefa7177b7324bded1c69e48edaed0054571a5e0f71b6cc39ab2ff`.
Rules: `configs/phase5/spec.json`, commit 5adeb25, sha256
`6a60c11cf4c13e51a07e148a8ad86f7d9db34aa02ea0ebd7259636f433c5669a`.

**Status of Phase 5.** Phase 5 is a **post-hoc supplementary comparison**. The QaTa-COV19 test split was already
evaluated and its results seen in Phases 2 and 3; Phase 5 is therefore not an independent confirmation. Its
purpose is to compare the frozen Phase 3 LR-Gate with two simpler alternatives (SDC-Fallback, Fixed Average) and
with always-LANG / always-IMG when text reliability is unknown. Phase 4 remains STOP (38eb77d); no spatial
experiment is run. No segmentation model is trained, no split is changed, and D-Gate and LR-Gate are unchanged.

## 1. SDC and the SDC-Fallback rule (Task 5.2 item 1)

- **Formula** (Borges, Pacheco, Silva, *Machine Learning* 2026, DOI 10.1007/s10994-026-07096-w; arXiv 2402.10665
  v6 eq. 22): SDC = 2 Σ(P_lang · B_lang) / [Σ P_lang + Σ B_lang], computed in float64 per row from the stored
  float32 LANG probability map.
- **Empty case:** SDC = 0 when the denominator is 0 (P_lang all zero, hence B_lang empty), as SDC(0, 0) = 0 in the
  paper.
- **Binary mask:** B_lang = P_lang > 0.5, the scored LANG mask of the common evaluator (`src/lgms/metrics.py`,
  strict). The paper's example binarisation is p ≥ γ; the plan requires the evaluator's mask, so the two differ
  only at pixels exactly equal to 0.5.
- **Rule:** fall back to the IMG mask if SDC < τ(model, seed), otherwise keep the LANG mask. The per-case metrics
  of SDC-Fallback are those of the selected LANG or IMG row (same masks, same evaluator).

## 2. Threshold selection and the frozen thresholds (item 2)

- One τ per language model × seed, chosen on QaTa **GATE_CAL** only (validation), with the rows of
  `src/phase3/gates.py` `main()` (features and labels merged one-to-one on the keys; QaTa; C0, C2, C4, C5, C6).
- `gates.calibrate(cal, "neg_sdc")` unchanged, with neg_sdc = −SDC: candidates {−inf, +inf} and every distinct
  neg_sdc; Δ(c) by donors → images → groups, C0 / C2 on all eligible GATE_CAL images and C4–C6 on S_CAL; objective
  mean Δ(C4, C5, C6); feasible if all five Δ ≥ −0.005; ties (≤ 1e-12) to the higher t (fewer fallbacks); +inf
  (never fall back) if the best feasible objective is ≤ 0. Reported as τ = −t.
- Result: job 469 (2026-11-10, HEAD 3d76801), commit 9110861; `results/phase5_sdc_calibration.csv` (sha256
  `a7a35a3c0f7573b41afdab408833f0797de2173daa23542e91a763cb73d386f9`); per-row SDC of the GATE_CAL rows in
  `results/phase5_sdc_val.csv.gz` (sha256 `885fe3bf4ff8357f57467ca92e0d747dd418dd00d41275253457385cdb0d60f0`). The
  numbers below are read from the CSV text: τ is the stored string (full precision; the test computation reads
  it as a string and converts it with float(), because pandas' default float parser was found to be 1 ulp off for
  four of the nine values); Δ (Dice units) and fallback rates (switched rows / rows) are float(stored string)
  rounded to 6 decimals. They were recomputed independently with separate code that does not import the
  repository code (maximum absolute difference 1.2e-16).

| Model | Seed | τ (fall back if SDC < τ) | Choice | Δ C0 | Δ C2 | Δ C4 | Δ C5 | Δ C6 | Fallback rate C0 / C2 / C4 / C5 / C6 |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| LViT-T | 1001 | `0.76623150762105452` | feasible maximum | -0.001131 | -0.004079 | +0.043978 | +0.028429 | +0.061985 | 0.075439 / 0.071930 / 0.128070 / 0.167251 / 0.179532 |
| LViT-T | 1002 | `0.80686778269080872` | feasible maximum | -0.004346 | -0.003332 | +0.046403 | +0.022934 | +0.130955 | 0.142105 / 0.142105 / 0.203509 / 0.236842 / 0.380117 |
| LViT-T | 1003 | `0.76610056565580087` | feasible maximum | -0.004644 | -0.004960 | +0.054180 | +0.018220 | +0.100947 | 0.124561 / 0.105263 / 0.181871 / 0.209942 / 0.288304 |
| RecLMIS | 1001 | `0.79282069500849839` | feasible maximum | -0.004894 | -0.004853 | +0.031054 | +0.022806 | +0.056086 | 0.043860 / 0.049123 / 0.125146 / 0.145029 / 0.176023 |
| RecLMIS | 1002 | `0.6821835250294308` | feasible maximum | -0.004550 | -0.004201 | +0.022912 | +0.015382 | +0.036841 | 0.015789 / 0.021053 / 0.081287 / 0.081287 / 0.109357 |
| RecLMIS | 1003 | `0.74428603387047898` | feasible maximum | -0.004588 | -0.002501 | +0.032045 | +0.015407 | +0.049571 | 0.045614 / 0.073684 / 0.118713 / 0.132749 / 0.164912 |
| DD-CMD | 1001 | `0.96497024557188282` | feasible maximum | -0.004551 | -0.004802 | +0.026176 | +0.022189 | +0.079214 | 0.064912 / 0.066667 / 0.269591 / 0.356725 / 0.373099 |
| DD-CMD | 1002 | `0.94450605392916853` | feasible maximum | -0.004942 | -0.004931 | +0.019447 | +0.017512 | +0.065581 | 0.043860 / 0.043860 / 0.134503 / 0.166667 / 0.216959 |
| DD-CMD | 1003 | `0.95831055514840158` | feasible maximum | -0.004429 | -0.004901 | +0.025993 | +0.022280 | +0.068978 | 0.066667 / 0.071930 / 0.244444 / 0.338596 / 0.319883 |

Never-fallback was a candidate for every model × seed and was not chosen (all nine are "feasible maximum").

## 3. Fixed Average (item 3)

P_avg = 0.5 · P_lang + 0.5 · P_img from the stored float32 maps, **computed in float64** (a float32 sum would
round values just above 0.5 down to 0.5); B_avg = P_avg > 0.5 (a pixel exactly 0.5 is background). Scored anew
with `metrics.case_metrics(metrics.binarize(P_avg), gt > 0)` as `src/evaluate.py` `_score`; empty = not
(P_avg > 0.5).any(). No weight or threshold is tuned.

## 4. Conditions and rows (item 4)

QaTa-COV19 test, three language models (LViT-T, RecLMIS, DD-CMD) × seeds 1001–1003, conditions **C0, C2, C4, C5,
C6** (not C1, C3a–c, MosMedData+). Rows: 3 × 3 × (2113 + 2113 + 3 × 2081 × 3) = **206,595**; the program
asserts this number before writing any output. Rows, keys and probability maps as `src/phase3/features.py`;
every LANG and IMG archived mask is checked to equal packbits(P > 0.5) row by row. Ground truth read as in the
Phase 2 test inference (`lgms.data.load_split_arrays` under `slurm/env_phase1.sh`).

## 5. Aggregation and statistics (item 5)

- **Tables** (`evaluate_gates.image_weighted`): per seed, donors averaged within image, then images; mean ± SD
  over the three seeds; fallback rate image-weighted the same way. Six strategies: LANG, IMG, D-Gate, LR-Gate,
  SDC-Fallback, Fixed Average; Dice, mIoU, HD95, ASSD, empty rate and fallback rate.
- **Paired differences** (`stats_phase3.group_differences`): donor differences within image and seed → seeds
  within image → images within group → equal-weight groups; mean and 95% t-interval over groups
  (`t_over_groups`, ddof 1). **No p-values are output.** Main post-hoc comparisons: LR-Gate − SDC-Fallback and
  LR-Gate − Fixed Average under C4, C5, C6 per model. Clean-text cost: strategy − LANG under C0 and C2 for IMG,
  D-Gate, LR-Gate, SDC-Fallback and Fixed Average; the −0.005 development margin is shown as a reference line
  only; no equivalence or safety is claimed from an interval.
- **Mixed text reliability** (Task 5.4): on the common targets eligible for C4, C5 and C6 (2081 images, 472
  groups; asserted), D(C) = group-weighted mean Dice (donors → seeds → images → groups); D_wrong =
  [D(C4) + D(C5) + D(C6)] / 3 (an equal-weight benchmark stress-test mixture, not a real error distribution);
  E(p) = (1 − p) D(C0) + p D_wrong, plotted at p = 0.00, 0.01, …, 1.00. Exact crossovers for A vs B:
  p* = −Δ_clean / (Δ_wrong − Δ_clean), reported when 0 ≤ p* ≤ 1, otherwise (or Δ_wrong = Δ_clean) "none"; the
  range of p where a strategy beats both always-LANG and always-IMG; sensitivity: crossovers with C4, C5 or C6
  alone. A crossover is not a clinical threshold.
- **D-Gate is not part of Task 5.4**: the plan's Task 5.4 report items and Figure 2 list LANG, IMG, LR-Gate,
  SDC-Fallback and Fixed Average only. D-Gate stays in the results table and in the C0 / C2 cost comparison.
- **Compute** (approximation): job 391 mean_ms (validation, 64 QaTa images, seed 1001, C0) of "LANG only" per
  model and "IMG only"; LR-Gate, D-Gate, Fixed Average = LANG + IMG; SDC-Fallback = LANG + fallback rate × IMG,
  with the image-weighted test SDC fallback rate per model × condition; score overhead ignored.

## 6. Outputs and figures (item 6)

`results/phase5_test_results.csv`, `results/phase5_pairwise_results.csv`, `results/phase5_compute_summary.csv`,
`results/phase5_mixed_reliability.csv`, `results/phase5_crossover_summary.csv`; for independent checking
`results/phase5_test_per_case.csv.gz` (keys, SDC, SDC-Fallback switch, Fixed Average Dice / mIoU / HD95 / ASSD /
empty; full precision) and `results/phase5_sdc_val.csv.gz` (above); the append-only run log
`results/phase5_test_run_log.tsv`. Figures (`src/phase5/make_figures.py`, from the output files only):
`reports/figures/phase5/fig1_strategies.png` (Dice under C0, C4, C5, C6 for LANG, IMG, LR-Gate, SDC-Fallback,
Fixed Average) and `reports/figures/phase5/fig2_mixed_reliability.png` (expected Dice vs p, LR-Gate crossovers
with always-LANG and always-IMG marked when they exist). Report: `reports/PHASE5_RESULTS.md`.

## 7. Analysis code (item 7)

Code commit **7159e4a** (HEAD when the manifest was made, job 475). The test computation is
`src/phase5/compute_test_split.py` (job `slurm/phase5_test.slurm`). The code files below — the Phase 5 programs,
every in-repository module they import (collected by importing them) and the job scripts — are hashed in the
manifest and checked again before the test computation:

| File | sha256 |
|---|---|
| `src/analyze_phase0.py` | `026a6908e3097b5d2d4c7bbe4fd8af84d3c0683e407398175d1807e6509eac55` |
| `src/lgms/__init__.py` | `240a87f493dc274340a97b06b334bb471dffdb8df2d578baf87a48b75bc21593` |
| `src/lgms/data.py` | `6c8a87efe261f66ec1bcdf941e22c7e8cb81d557a4cc29a616ec129c03861ff6` |
| `src/lgms/metrics.py` | `a1bb39144081810b9c6f81e6ad33120046b961b544ab33866fa7d00542dfb6ce` |
| `src/phase1/__init__.py` | `a0e520eeb011f9154fe231bce8f930e1c8613a54a69a3599f415d2564cca50c0` |
| `src/phase1/analysis_lock.py` | `8774c05e5ae3a2f873bf200f393edaf49331cd97df758012105685c8b673cf24` |
| `src/phase2/lock.py` | `70fca4bd7806f533a0061dd1054e4f57ca65fbbabeef3e9a19f4399fdea65a64` |
| `src/phase3/evaluate_gates.py` | `503b59bb38a0af6f2437f7714bb0302ebd3f8f7933f708006b62abca36fe6674` |
| `src/phase3/features.py` | `ef725248800431ca00f188270327b539db4526ed8e4c5c12da1f7280ce34ad4e` |
| `src/phase3/gates.py` | `5ab292335def427e300bb836211be5622a2a45c0a81c64e07be4b72bc9b66e41` |
| `src/phase3/stats_phase3.py` | `11fd0150797a441020d33e6bf216eeccd53db8b153664172a7d2c4af16525ed5` |
| `src/phase4/__init__.py` | `9924fa906b250751f11e44ad7990b5d4403de00b1f7296ff18a148e76964c7dd` |
| `src/phase4/guard.py` | `28fbc03c31517233f80e93ad37182ef36cb4027dac741184093dc074015f9cf1` |
| `src/phase5/calibrate_sdc.py` | `0644678107ec7aff4378c1077871dae8a12f001d6188a0ffabdbf712adc3e5b9` |
| `src/phase5/compute_test_split.py` | `c894aeb4e9cce91267faae17c4eb2ddcb8bcae0fbe8ebe592f0bbf855f9c104d` |
| `src/phase5/core.py` | `b463f589cc4e8db661315d4e45f6f7c4b5623fada78be3afc32750e53128fc68` |
| `src/phase5/guards.py` | `8721ae234528631cbd3dfbd4206218901cc000f8dbdacbcf7414cdc815072a3c` |
| `src/phase5/make_figures.py` | `cc6dd68bcc6cfbe057ba0c2552073db6074a1fb4b9da750d076efbf2b3639196` |
| `src/phase5/make_manifest.py` | `c1f24cb1b98f6697daa171d873cce4078002fcc48973f4257143ba3e5befc5bb` |
| `src/phase5/test_phase5.py` | `bd7f15ad21248691ee18f9e94961108bbc94816434e0e51c9c35e0c4b0ebbc78` |
| `slurm/phase5_test.slurm` | `e52e084dd60cb92a89d245694350cc77513c5358398e0a6930023e170ce59b5f` |
| `slurm/env_phase1.sh` | `7e6d3d6cfcf3ca78ba660584d7ca2e1ba3d89d9e2066eec679b08ef53d72aa25` |
| `slurm/env.sh` | `4a380a68eec9a148be5917211b514a9b360e255aca348c2d6ef42e7b6268e704` |
| `slurm/notify_trap.sh` | `62a768772156bc7ccb25bd645a62560cacb035242e2a6a8a7b4ee04ea6966e93` |

D-Gate and LR-Gate decisions are computed by the unchanged Phase 3 code (`evaluate_gates.load_rows`,
`apply_gates`), which reads F1–F5 with pandas' default float parser, as in the Phase 3 test run (job 394); they are
therefore expected to reproduce the frozen Phase 3 decisions, and this is checked after the run by comparing the
D-Gate and LR-Gate rows with `results/phase3_test_results.csv`.

## 8. Artifact hashes (item 8)

`configs/phase5/artifact_manifest.tsv`: **2763 files, 65,853,402,117 bytes (61.3 GiB)**, made by
`src/phase5/make_manifest.py` in job 475 (2026-11-10, HEAD 7159e4a); sha256 of the manifest file
`685276401d9cb7ad5dccce2e78b8db555a2c838eabba5a796744c4b8c447de4d`. It lists every input of the test computation:
validation and test LANG probability maps, archived masks and per-case files of C0, C2, C4–C6 (297 + 297 files),
the IMG nnU-Net maps, masks and per-case files (18), the QaTa test ground-truth masks and id list (2114), the
existing tables and the SDC calibration outputs (13), and the code and job scripts (24). An earlier
manifest (job 470, HEAD b4d500c) was discarded before being committed, because the threshold reading in
`src/phase5/compute_test_split.py` was then changed (commit 7159e4a).

## 9. Running the test computation

- It runs **once**, as Slurm job `slurm/phase5_test.slurm`, submitted by hand after this freeze is committed and
  checked. Before reading any test data the program checks: Phase 2 lock open; spec, this document, the manifest
  and the SDC calibration each committed once and unchanged; clean tree; HEAD on origin/main; every manifest file
  with its recorded sha256; no result file present.
- If the job stops for a cluster reason before any result file is written: the run log
  `results/phase5_test_run_log.tsv` written by that attempt is an untracked file, which the clean-tree check
  (`git status --porcelain`) reports, and `results/` is not ignored by git. The run log is therefore first
  committed and pushed unchanged (a public record of the failed attempt); then the same Slurm job is resubmitted
  unchanged. Code and tests are not changed. The log records every attempt.
- If the job stops after any result file has been written, nothing is rerun; the work stops and the case is
  reported.
- If any code or rule has to change, the work stops first; the change is written down as an amendment, and only
  then is it decided whether to run.
- No rule, threshold or gate is changed after the Phase 5 test comparison is seen.
