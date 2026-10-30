# Phase 3 — Protocol freeze (FROZEN)

Status: 2026-10-30, **frozen** together with PHASE2_PROTOCOL_FREEZE.md (record
`configs/phase2_frozen/phase2_lock.json`).
- At the freeze, no test inference, feature or gate decision on test existed.
- **Decision** (`configs/phase2_frozen/phase3_decision.json`): Phase 3 continues; gates as fixed at b58cdf6, no
  amendment. The supervisor was informed on 2026-10-28 that the C0 / C2 losses on GATE_FIT lie beyond the −0.005
  margin (default: keep the gates as planned) and had not asked for a change by 2026-10-30.
- The gates are not changed because of test results. This file and every file of the record are not changed
  after the freeze; test results and their explanations go to `reports/PHASE3_RESULTS.md`.
- **Scope.** PHASE3_PLAN Task 3.6, items 2–4: IMG architecture, checkpoint identifiers, feature definitions,
  evaluation rules; partitions, LR artifacts, all thresholds (including never-fallback), comparison families;
  exposure history and reproducibility. Development record: [PHASE3_SETUP.md](PHASE3_SETUP.md).
- **Companion.** [PHASE2_PROTOCOL_FREEZE.md](PHASE2_PROTOCOL_FREEZE.md); both are frozen in one commit.
- The tables of numbers in this file are generated from the result files named next to them.

## 1. Strategies

Per row (dataset × language model × seed × image × condition × donor):

| Strategy | Prediction |
|---|---|
| LANG | the language-guided model under the supplied text (Phase 2 test benchmark) |
| IMG | nnU-Net, same dataset and seed |
| D-Gate | IMG if F1 > t_D(model, seed), else LANG |
| LR-Gate | IMG if s_LR > t_LR(model, seed), else LANG; s_LR = decision function of the fitted LR |
| NULL (C1) | reference: the C1 prediction of the same image and seed |
| ORACLE | reference, not deployable: IMG if Dice(IMG) > Dice(LANG), else LANG |

A gate's metrics are those of the selected prediction (no re-scoring). The rule is strict (`>`): a score equal
to its threshold stays with LANG.

## 2. IMG architecture and checkpoint identifiers

- **IMG = nnU-Net** (Task 3.1C: higher clean group-weighted Dice than U-Net on QaTa GATE_FIT;
  `results/phase3_fallback_selection.csv`), seed matched to the language model.
- **Inference** (`src/phase3/infer_img.py`, the Phase 0 settings): the selected snapshot of
  `runs/nnunet/<ds>/seed<k>/selection.json`; sliding window, Gaussian weighting, mirroring TTA; softmax
  probability of the lesion class, float32; mask = probability > 0.5; test images from `imagesTs`.
- **IMG checkpoints** (`results/phase3_checkpoints.tsv`):

<!-- BEGIN IMGCKPT -->
| dataset | seed | checkpoint | sha256 |
|---|---|---|---|
| QaTa-COV19 | 1001 | `runs/nnunet/nnUNet_results_seed1001/Dataset501_QaTaCOV19/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0250.pth` | `7a7ee608d7cb65275e3ef0c68b0a7bd8773c38791f73a4d10c0718db7f8784cd` |
| QaTa-COV19 | 1002 | `runs/nnunet/nnUNet_results_seed1002/Dataset501_QaTaCOV19/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0150.pth` | `30e2c874f1801d46c629a3c778de09425e58d2c34b8506f48401afeb71a5300b` |
| QaTa-COV19 | 1003 | `runs/nnunet/nnUNet_results_seed1003/Dataset501_QaTaCOV19/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0800.pth` | `aaef054b8bab66e1776875c037f771ae246c1144188af451488487a86ed3c786` |
| MosMedData+ | 1001 | `runs/nnunet/nnUNet_results_seed1001/Dataset502_MosMedDataPlus/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0250.pth` | `7421070ccad317d31d2ee2b03d0f4b261f14fea016f7391e1d7d4840409736f8` |
| MosMedData+ | 1002 | `runs/nnunet/nnUNet_results_seed1002/Dataset502_MosMedDataPlus/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0400.pth` | `642de6d3733fecce39c2a360df74b99fce8d29985ffef808ddc4b0e0edc09b52` |
| MosMedData+ | 1003 | `runs/nnunet/nnUNet_results_seed1003/Dataset502_MosMedDataPlus/nnUNetTrainerLGMS__nnUNetPlans__2d/fold_0/checkpoint_ep0150.pth` | `54f4ade5b5413f85099e1331143d6a9cb6f0e5068f6a0440303621a43b8a267a` |
<!-- END IMGCKPT -->

- **LANG checkpoints.** The 18 Phase 1 checkpoints of `results/phase2_checkpoint_sha256.tsv` (also in
  `results/phase3_checkpoints.tsv`, role LANG). LANG test predictions are the Phase 2 test benchmark outputs,
  probabilities kept by amendment A1; no separate Phase 3 LANG run on test.

## 3. Feature definitions (Task 3.2; `src/phase3/features.py`)

Per row, float64 from the float32 maps, before any averaging; B = P > 0.5 (the scored masks):
- **F1** = mean |P_lang − P_img|
- **F2** = 1 − Dice(B_lang, B_img); both empty → Dice 1, exactly one empty → Dice 0
- **F3** = |mean(B_lang) − mean(B_img)|
- **F4** = mean H(P_lang), **F5** = mean H(P_img); H(p) = −p ln p − (1 − p) ln(1 − p), p clipped to
  [1e-6, 1 − 1e-6] for the entropy only.

The gates read F1–F5 only. Labels (Dice, gain, y) are stored in a separate file and are never a gate input.

## 4. Partitions and grouping

- **Validation (development).** QaTa validation split by groups into GATE_FIT and GATE_CAL
  (`src/phase3/make_partitions.py`, seed 3001; `results/phase3_gate_partitions.csv`):

<!-- BEGIN PARTITIONS -->
| partition | groups | images |
|---|---|---|
| GATE_CAL | 478 | 570 |
| GATE_FIT | 716 | 859 |
<!-- END PARTITIONS -->

- **Test.** All QaTa test images (part "test"); none is used for fitting or calibration. MosMedData+ test:
  exploratory only.
- **Groups.** QaTa: Phase 2 patient units (`sub-S`); `covid_*` images (validation only) are their own groups,
  which limits inference for them. MosMed: CT volumes, `bjorke` pooled (9 test units).

## 5. Gates: fitted artifacts and all thresholds (as of commit b58cdf6)

**Fitting and calibration** (validation only; PHASE3_SETUP.md section 5, items 2–7 and 18; `src/phase3/gates.py`):
- Rows: QaTa, conditions C0, C2, C4, C5, C6. C1, C3a–C3c and MosMed never enter fitting or calibration.
- LR: GATE_FIT rows; y = 1 if Dice(IMG) − Dice(LANG) > 0.005; weights: groups balanced within condition,
  images within group, donor rows within image, conditions × 1/5, rescaled to mean 1; weighted
  `StandardScaler`, then `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)` (L2, scikit-learn 1.9.0).
- Calibration (both gates): GATE_CAL; Δ(c, t) = group-weighted mean of gain × 1{score > t} (donors → images →
  groups); objective = mean of Δ(C4), Δ(C5), Δ(C6) on S_CAL (targets eligible for all three); feasible if
  Δ ≥ −0.005 for C0, C2, C4, C5, C6; candidates {−∞, +∞} ∪ every distinct GATE_CAL score; ties (≤ 1e-12) to
  the higher threshold; if the best feasible objective is ≤ 0, t = +∞ (**never fall back**).
- **Outcome.** Never-fallback was chosen for **none** of the 18 gates (choice "feasible maximum" for all; no
  LR-Gate unsupported).

**D-Gate thresholds** (full precision, `results/phase3_dgate_thresholds.json`):

<!-- BEGIN DGATE -->
| model × seed | t_D (F1 >) | choice |
|---|---|---|
| LViT-T 1001 | 0.140166322 | feasible maximum |
| LViT-T 1002 | 0.133617733 | feasible maximum |
| LViT-T 1003 | 0.141213606 | feasible maximum |
| RecLMIS 1001 | 0.126640495 | feasible maximum |
| RecLMIS 1002 | 0.117704556 | feasible maximum |
| RecLMIS 1003 | 0.135673481 | feasible maximum |
| DD-CMD 1001 | 0.126029371 | feasible maximum |
| DD-CMD 1002 | 0.131128038 | feasible maximum |
| DD-CMD 1003 | 0.142629902 | feasible maximum |
<!-- END DGATE -->

**LR-Gates** (`results/phase3_lr_artifacts.json`; features in the order F1–F5; score =
((x − mean) / scale) · coef + intercept):

<!-- BEGIN LR -->
| model × seed | t_LR (score >) | choice | intercept | coef F1–F5 | scaler mean | scaler scale |
|---|---|---|---|---|---|---|
| LViT-T 1001 | 1.2621671011295992 | feasible maximum | 0.494525 | 0.216749, 0.10069, 0.0111551, 0.231908, -0.384723 | 0.0483011, 0.272374, 0.0383485, 0.0564738, 0.0272306 | 0.0456784, 0.247584, 0.0485156, 0.0343676, 0.0161569 |
| LViT-T 1002 | 1.2878790772096826 | feasible maximum | 0.618796 | 0.242842, 0.146057, 0.0315353, 0.0948286, -0.378943 | 0.0505324, 0.28408, 0.0392715, 0.0627286, 0.0263479 | 0.0453513, 0.249102, 0.0485568, 0.0359086, 0.0156566 |
| LViT-T 1003 | 0.9145965368794482 | feasible maximum | 0.524099 | -0.172234, 0.122177, 0.349987, 0.289886, -0.415602 | 0.0537268, 0.291365, 0.0407913, 0.0676976, 0.0197877 | 0.0467315, 0.255595, 0.0507666, 0.0389135, 0.0126054 |
| RecLMIS 1001 | 1.241038348516753 | feasible maximum | 0.273855 | 0.992568, 0.201819, -0.683666, 0.29835, -0.434021 | 0.0431997, 0.256703, 0.0339367, 0.0487493, 0.0272306 | 0.0432383, 0.232382, 0.0463682, 0.0314884, 0.0161569 |
| RecLMIS 1002 | 1.2367712102579849 | feasible maximum | 0.213724 | -0.102141, 0.197283, 0.297773, 0.452756, -0.415246 | 0.044099, 0.248182, 0.0339381, 0.0617064, 0.0263479 | 0.0405617, 0.224958, 0.0439979, 0.0370867, 0.0156566 |
| RecLMIS 1003 | 1.1066551942245177 | feasible maximum | 0.302738 | 0.0617794, 0.164183, 0.186829, 0.422577, -0.418306 | 0.0476657, 0.268188, 0.0367465, 0.0570674, 0.0197877 | 0.0442314, 0.233945, 0.0476308, 0.0346208, 0.0126054 |
| DD-CMD 1001 | 1.067103324061239 | feasible maximum | 0.202 | 1.2135, -0.0829851, -0.688638, 0.213558, -0.542317 | 0.0408287, 0.239581, 0.0328646, 0.0162142, 0.0272306 | 0.0414333, 0.206723, 0.0434931, 0.01402, 0.0161569 |
| DD-CMD 1002 | 1.3009355657499453 | feasible maximum | 0.203202 | 0.689354, 0.152559, -0.111843, 0.145831, -0.474353 | 0.0434799, 0.259397, 0.0357869, 0.0185219, 0.0263479 | 0.0456241, 0.220747, 0.0476028, 0.0162878, 0.0156566 |
| DD-CMD 1003 | 0.8274833843179241 | feasible maximum | 0.251303 | 0.693286, -0.084325, -0.311639, 0.249917, -0.441708 | 0.042451, 0.247557, 0.034412, 0.0188673, 0.0197877 | 0.0429929, 0.2076, 0.0449077, 0.0157678, 0.0126054 |

Coefficients and scaler values are shown to 6 significant digits; the JSON file holds them at full precision and is the one the gates read.
<!-- END LR -->

**MosMed transfer.** The QaTa gate of model × seed k is applied unchanged to the MosMed checkpoint of the same
model and seed, with the MosMed nnU-Net of seed k.

## 6. Evaluation rules on test (Task 3.5; `src/phase3/evaluate_gates.py --split test`)

- All nine conditions, every eligible image, donor and seed; all strategies of a condition on the same rows.
- Metrics: Dice, mIoU, HD95, ASSD and empty rate of the selected prediction; the original empty-mask policy of
  the common evaluator for HD95 / ASSD is kept.
- Tables: image-weighted mean ± SD over the 3 per-seed means (donors averaged within image and seed);
  fallback rate image-weighted; routing quality (switched rows, beneficial / harmful / neutral with
  |ΔDice| ≤ 1e-6, net gain); MosMed exploratory table.
- Outputs: `results/phase3_test_results.csv` and `results/phase3_test_per_case.csv.gz` (every row × strategy;
  compressed, otherwise the `phase3_test_per_case.csv` of the plan), plus `phase3_test_paired_differences.csv`,
  `phase3_test_fallback_rates.csv`, `phase3_test_routing_quality.csv`, `phase3_test_mosmed_exploratory.csv`,
  `phase3_test_gate_decisions.csv.gz`.

## 7. Comparison families (Task 3.7; `src/phase3/stats_phase3.py --split test`)

Paired Dice differences on identical rows; aggregation: donors within image and seed → seeds within image →
images within group → groups equally; mean, 95% t-CI and two-sided one-sample t-test over groups; image set =
all images eligible for the condition.

| Family | Comparisons | Correction |
|---|---|---|
| **Primary** (QaTa) | LR-Gate − LANG under C4, C5, C6 × LViT-T, RecLMIS, DD-CMD (9) | Holm across the 9, α = 0.05 |
| **Secondary** (QaTa) | LR-Gate − D-Gate under C4, C5, C6 × the 3 methods (9) | Holm across the 9, α = 0.05 |
| Descriptive (QaTa) | D-Gate − LANG (all conditions); LR-Gate − LANG and LR-Gate − D-Gate under C0, C1, C2, C3a–c | mean and CI, no p |
| Preservation (QaTa) | gate − LANG under C0 and C2, both gates, with the −0.005 margin and the flag "CI lower bound ≥ −0.005" | none; a nonsignificant difference does not prove preservation |
| Exploratory | MosMedData+, every comparison | mean and CI, no p |

Output `results/phase3_pairwise_results.csv`. The program was tested on synthetic data (16 checks) and run
once on validation, descriptively (`results/phase3_pairwise_results_val.csv`; section 7 below).

## 8. Test pipeline (after the freeze only)

1. Phase 2 steps 1–4 of PHASE2_PROTOCOL_FREEZE.md section 7 (freeze, link-test, Phase 2 test benchmark with
   the C0 reconciliation, Phase 2 test analysis).
2. Job `slurm/phase3_test.slurm` (by hand): lock check first (exit 5 if closed); Phase 2 test outputs present;
   nnU-Net test inference (6 runs); `export_per_case.py`, `features.py`, `evaluate_gates.py`,
   `stats_phase3.py`, each with `--split test` and each checking the lock again.
3. Nothing is fitted, calibrated or selected on test; the gates are not changed because of test results.

## 9. Validation / test exposure history (Phase 3)

**Test:** nothing of Phase 3 has touched the test split (no IMG rerun, feature, gate decision or result). The
clean (C0) test results of Phases 0 and 1 were seen earlier (PHASE2_PROTOCOL_FREEZE.md section 6).

**Validation, in order (2026-10-28):**
1. Task 3.1: inventory, float32 reruns (IMG; LANG C2 / C3), the fixed group split, IMG choice on GATE_FIT
   (nnU-Net), oracle diagnostics on GATE_FIT.
2. Setup items 1–18 accepted before any gate result; weight normalisation (item 18) fixed before any LR fit.
3. Tasks 3.2–3.4: features; D-Gate and LR-Gate fitted on GATE_FIT and calibrated on GATE_CAL (commit b58cdf6).
   The GATE_CAL calibration values (C0 / C2 losses, C4–C6 gains, fallback rates) were seen.
4. Gates fixed as of b58cdf6 after they were checked by an independent recomputation with separate code
   (written without reading the project code): 0 differences, including a refit of the nine LRs.
5. Task 3.5 on validation with the fixed gates: GATE_FIT, GATE_CAL, all nine conditions and MosMed were seen
   (C1, C3a–C3c and MosMed for the first time), after the gates were fixed. Notably, on GATE_FIT the C0 / C2
   losses lie beyond the −0.005 margin; this was seen and is not used to change the gates.
6. Task 3.7 program run once on validation (descriptive; it reproduces the Task 3.5 group-weighted values).
7. The D-Gate thresholds are read at full precision from `phase3_dgate_thresholds.json` (record precision, not
   a change; PHASE3_SETUP.md section 9).

## 10. Frozen files and reproducibility

- **Record.** The Phase 3 files below are in `LOCKED_FILES` of `src/phase2/lock.py`; the freeze record
  `configs/phase2_frozen/phase2_lock.json` (commit + sha256 of every file) is authoritative. The table is a
  sha256 of the files in the frozen commit (`python src/phase2/lock.py manifest`, generated just before the
  freeze). This file itself is covered by the record only.
- **Code commit.** Gates fixed as of b58cdf6; the frozen commit is the one in the record.
- **Environment.** `~/envs/lgms_p1`: torch 2.14.0+cu130, scikit-learn 1.9.0, nnunetv2.

<!-- BEGIN MANIFEST-PHASE3 -->
| file | sha256 |
|---|---|
| `src/phase3/infer_img.py` | `bc4a7193359316b536ff46f896e45c0b611f64a449042097fc3b3434e6a07a00` |
| `src/phase3/features.py` | `ef725248800431ca00f188270327b539db4526ed8e4c5c12da1f7280ce34ad4e` |
| `src/phase3/gates.py` | `5ab292335def427e300bb836211be5622a2a45c0a81c64e07be4b72bc9b66e41` |
| `src/phase3/evaluate_gates.py` | `503b59bb38a0af6f2437f7714bb0302ebd3f8f7933f708006b62abca36fe6674` |
| `src/phase3/export_per_case.py` | `c8341c0a456d36be9bd07d5aff0c69e78754dc2081bcddcc37f7f390536e2fb7` |
| `src/phase3/stats_phase3.py` | `11fd0150797a441020d33e6bf216eeccd53db8b153664172a7d2c4af16525ed5` |
| `src/phase3/make_partitions.py` | `acb535faeb7a027c33f17847f67ce92cb7763e89ee9741bf6bfd2f5dd65c87cd` |
| `results/phase3_checkpoints.tsv` | `a075167c9bd8df54a90ab7a1049df24a9985a450aee1791ae381d20eb3c2ea91` |
| `results/phase3_fallback_selection.csv` | `bce80a171ce1ad50f5e5f47c837c689d231eb068cd4b17724a7e4130f1399064` |
| `results/phase3_gate_partitions.csv` | `77c254e2eb00868668a737b523656ac3075985966b204eedb5a8f3d61d14574d` |
| `results/phase3_dgate_thresholds.json` | `df671c1900c82649191175141d73aa4cc7655f77b475c700c8607fa8f1df55bd` |
| `results/phase3_lr_artifacts.json` | `cecaab140e5f225464e539e8b5076128238155165f3c6533eb5caafabf4260ab` |
| `slurm/phase3_test.slurm` | `d66fc693265eff2af5473f4b4b50466aeef280bffdd56ccc884339793730c48f` |
| `reports/PHASE3_SETUP.md` | `a826693abc149637dbe854f5d0b66b7117e072396faf4ab038217404f75b1f5d` |
<!-- END MANIFEST-PHASE3 -->

## 11. Completed before the freeze

- Task 3.8 cost (job 391, PHASE3_SETUP.md section 10); it does not change the gates.
- The supervisor's reply: no change of the gate constraints; the Phase 3 decision record.
