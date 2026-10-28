# Phase 3 — Setup (Task 3.1 preparation; DRAFT)

Status: 2026-10-28, **draft** (section 5 accepted by the project lead, items 16–18 added). Covers Task 3.1A (inventory, rerun plan) and Task 3.1B (the fixed QaTa group split)
of [PHASE3_PLAN.md](PHASE3_PLAN.md), plus the implementation details that the plan leaves open and that must be
fixed before any Phase 3 result is seen (section 5).

- **Validation only.** The Phase 2 lock is closed. No test file was read and no test inference was run, also for
  U-Net and nnU-Net (Task 3.6).
- **Nothing retrained or changed.** No segmentation model was retrained. The Phase 2 conditions, donors and
  checkpoints are unchanged.
- **No job submitted.** The rerun job of section 3 is prepared, not submitted.
- **No Phase 3 result exists yet.** No gate, feature or oracle number has been computed.

Code for this step: `src/phase3/inventory.py`, `src/phase3/make_partitions.py` (+ `test_make_partitions.py`,
5 checks passed), `src/phase3/infer_img.py`, `src/phase3/infer_lang.py`, `slurm/phase3_infer_val.slurm`.

## 1. Checkpoints

The full table, with paths and sha256 values, is `results/phase3_checkpoints.tsv`.

| Role | Models | Checkpoint | sha256 |
|---|---|---|---|
| LANG | LViT-T, RecLMIS, DD-CMD; QaTa and MosMed; seeds 1001–1003 | the 18 Phase 2 checkpoints. MosMed LViT-T = the **batch-24** runs (`runs/lvit_b24/…`; training batch size; post-hoc Phase 1 sensitivity study) | re-hashed now; all 18 equal `results/phase2_checkpoint_sha256.tsv` |
| IMG | U-Net | `runs/unet/<ds>/seed<k>/best_model.pt` (Phase 0, validation-selected) | recorded in the table |
| IMG | nnU-Net | the snapshot selected in Phase 0 (`runs/nnunet/<ds>/seed<k>/selection.json`): QaTa epochs 250 / 150 / 800, MosMed 250 / 400 / 150 (seeds 1001 / 1002 / 1003), in `runs/nnunet/nnUNet_results_seed<k>/…/fold_0/` | recorded in the table |

**IMG inference settings (Phase 0, unchanged):**
- **U-Net:** eval mode, inference batch 32, one sigmoid output = foreground probability, no TTA.
- **nnU-Net:** nnU-Net's own predictor (`src/nnunet_select.py`): sliding window (step 0.5), Gaussian weighting,
  **mirroring TTA on** (the nnU-Net default, also used for checkpoint selection and the Phase 0 test).
  Probability = softmax probability of the lesion class; `> 0.5` equals argmax for two classes.
- **Both:** common 224 × 224 targets, `probability > 0.5`, common evaluator, empty prediction → HD95 = ASSD =
  316.78 px.

## 2. Inventory of the validation artifacts (Task 3.1A)

`src/phase3/inventory.py` → `results/phase3_inventory_val.csv`: one row per method × dataset × seed × condition
× donor variant.

**LANG** (Phase 2 benchmark, `runs/phase2/val/<method>/<dataset>/seed<k>/`). The table is identical for the
three methods, both datasets and the three seeds:

| Condition | Passes (method × dataset × seed × donor) | Per-case metrics | Scored masks | float32 probabilities | Keys match (dataset, id, model, seed, condition, donor) | Mask = probability > 0.5 (first 50 cases of each pass) |
|---|---:|---|---|---|---|---|
| C0 | 18 | yes | yes | **yes** | yes | yes |
| C1 | 18 | yes | yes | **yes** | yes | yes |
| C2 | 18 | yes | yes | **no** | yes | – |
| C3a | 18 | yes | yes | **no** | yes | – |
| C3b | 18 | yes | yes | **no** | yes | – |
| C3c | 18 | yes | yes | **no** | yes | – |
| C4 (d1–d3) | 54 | yes | yes | **yes** | yes (donor id in every row) | yes |
| C5 (d1–d3) | 54 | yes | yes | **yes** | yes | yes |
| C6 (d1–d3) | 54 | yes | yes | **yes** | yes | yes |

- **Confirmed (D13):** only C0, C1 and C4–C6 have probability maps.
- **Match.** For those passes, the stored masks equal `probability > 0.5` in all 198 checked passes.
- **Checkpoints.** The LANG checkpoints are unchanged.

**IMG** (Phase 0, `runs/{unet,nnunet}/<ds>/seed<k>/`):

| | Per-case metrics | Scored masks (PNG) | Probabilities | Mask = probability > 0.5 |
|---|---|---|---|---|
| U-Net, 2 datasets × 3 seeds | yes (`eval_val/per_case.csv`) | yes | **float16** (`val_probs.npz`) | **no** (11 of 12 runs differ in the first 50 cases) |
| nnU-Net, 2 datasets × 3 seeds | yes | yes | **float16** | **no** |

The float16 archives do not reproduce the scored masks. Seed 1001, full validation:

| Run | Cases with differing pixels | Max pixels per case |
|---|---|---|
| U-Net QaTa | 693 / 1429 | 11 |
| U-Net MosMed | 28 / 273 | 3 |
| nnU-Net QaTa | 413 / 1429 | 4 |
| nnU-Net MosMed | 6 / 273 | 1 |

Following the plan ("do not attach an old score to a numerically different mask"), the IMG probabilities, masks
and scores are re-created together in float32.

**Missing, therefore:**
- IMG float32 probabilities + masks + scores: 12 runs.
- LANG probabilities for C2, C3a, C3b, C3c: 18 checkpoints × 4 conditions.

## 3. Inference-only rerun (validation)

**One job:** `slurm/phase3_infer_val.slurm` (submitted by hand; 12 h limit; resumable).

1. **IMG** (`src/phase3/infer_img.py`):
   - Same frozen checkpoint (sha256 checked against `results/phase3_checkpoints.tsv`) and the Phase 0 settings of
     section 1.
   - float32 probabilities, packed scored masks and per-case scores are written from **one** pass to
     `runs/phase3/val/img/<arch>/<ds>/seed<k>/`.
   - Reconciliation with the Phase 0 archive (per-case |ΔDice| and mask pixels) goes to `reconcile_phase0.csv`.
     It is a reproducibility record only; Phase 3 uses the new scores.
2. **LANG C2, C3a, C3b, C3c** (`src/phase3/infer_lang.py`):
   - The Phase 2 manifest (same texts and eligible cases), the Phase 2 inference path (`src/phase2/infer.py`,
     imported unchanged), the same checkpoints and the per-pass input check (model input changed exactly where
     the text changed).
   - float32 probabilities + masks + scores go to `runs/phase3/val/lang/<method>/<ds>/seed<k>/`.
   - **Phase 3 uses these new scores for C2 / C3.** The pixel differences to the Phase 2 masks of the same pass
     are recorded (`phase2_mask_diff_<cond>.csv`, reproducibility only). LViT-T and RecLMIS are expected to be
     identical; DD-CMD may differ by a few pixels (cross-process non-determinism, Phase 2 protocol 4.8).
   - **Phase 2 files are not touched.**
3. **C0, C1, C4–C6 are not rerun.** Their Phase 2 probabilities, masks and scores come from one pass and match
   (section 2).

**Time:** about 30–60 min.
- IMG: U-Net is about 1 min per run. For nnU-Net with TTA, Phase 0 measured 49–59 s per full QaTa validation
  pass and 13–31 s for MosMed.
- LANG: 72 passes, at about 13 s per pass in the Phase 2 benchmark, plus float32 writing.
- Latest end: submission + 12 h (wall-clock guard 11.5 h).

**Storage.** Measured on the Phase 2 archives: a compressed float32 map takes about 182–185 KB (float32
probabilities barely compress), and a packed mask about 0.4–0.6 KB.

| Item | Maps | Size |
|---|---:|---:|
| Validation now: LANG C2 + C3a–C3c (QaTa 5710 + MosMed 1084 per method × seed, × 9) | 61,146 | ≈ 11.2 GB |
| Validation now: IMG (2 archs × 3 seeds × 1702 images) | 10,212 | ≈ 1.9 GB |
| **Validation total now** | 71,358 | **≈ 13 GB** |
| Later, test (upper bound): LANG, all 15 passes per image (C0, C1, C2, C3a–c, 9 donor passes) × 9 checkpoints × (2113 + 273) | 322,110 | ≈ 59 GB |
| Later, test: IMG, both archs (or only the chosen one: ≈ 1.3 GB) | 14,316 | ≈ 2.6 GB |
| **Test total (upper bound)** | | **≈ 62 GB** |

**Disk.**
- `/mnt/cv_data` is a shared 37 TB file system with 16 TB free.
- No `quota` tool is available on the cluster, so a per-user quota could not be checked.
- The project now uses 111 GB in `runs/` (Phase 2 validation: 30 GB).

## 4. Group split of QaTa validation (Task 3.1B)

`src/phase3/make_partitions.py` → `results/phase3_gate_partitions.csv` (dataset, sample_id, group_id, group_type,
partition). Generated once, on 2026-10-28; the script refuses to overwrite it, and no other split was tried.

**Groups.** The Phase 2 units: BIDS subject `sub-S…` for patient images. Every `covid_*` image has no patient id
and is its own provisional group (limitation: images of one unknown patient may fall into both parts).

**Algorithm:**
1. G = the distinct group ids, sorted as strings.
2. order = `numpy.random.default_rng(3001).permutation(len(G))`.
3. n_fit = `floor(0.6 · len(G) + 0.5)`.
4. The groups `G[order[0..n_fit−1]]` form GATE_FIT; the others form GATE_CAL.

Every image goes with its group, together with all its conditions, donors and seeds.

| Part | Groups | of which patients (`sub-S`) | of which `covid_*` images | Images |
|---|---:|---:|---:|---:|
| GATE_FIT | **716** | 352 | 364 | **859** |
| GATE_CAL | **478** | 245 | 233 | **570** |
| Total | 1194 | 597 | 597 | 1429 |

The validation set was already used to select the segmentation checkpoints and to study Phase 2 behaviour.
Neither part is an independent cohort.

## 5. Implementation details the plan does not fix (to decide before any Phase 3 result)

Each item gives a proposal and the reason.

1. **Source of each prediction.**
   - LANG C0, C1, C4–C6: Phase 2 (probabilities, masks and scores from one pass).
   - LANG C2, C3a–C3c: the Phase 3 rerun.
   - IMG: the Phase 3 rerun.
   - All features and all Dice values are computed from the same artifact (probability → mask → score).
   - *Reason:* the plan's consistency rule; no old score next to a different mask.
2. **Eligible sets.**
   - Each condition keeps its Phase 2 eligibility (manifest unchanged). IMG exists for every image.
   - All strategies under a condition use the same images, donors and seeds.
   - The C4–C6 calibration objective uses S_CAL = GATE_CAL targets eligible for C4, C5 **and** C6.
   - C0 and C2 constraints use all eligible GATE_CAL images (570 each). The C4/C5/C6 constraints use S_CAL.
   - *Reason:* the plan's "same target-image intersection" for the objective and "full eligible cases" for C0/C2.
     Using S_CAL for the C4–C6 constraints keeps objective and constraints on one set.
3. **Aggregation for calibration.**
   - One threshold per language model × seed; no seed averaging.
   - Δ per row = Dice(gate) − Dice(LANG).
   - Average the donors within an image, then the images within a group, then the groups equally.
   - *Reason:* the plan's rule applied per model × seed, since thresholds are per model × seed.
4. **Threshold candidates.**
   - For each model × seed: {−∞, +∞} ∪ {every distinct score value on the GATE_CAL rows of C0, C2, C4, C5, C6,
     all donors}, with the rule `score > threshold`.
   - Every candidate is evaluated. Feasible = all five constraints Δ ≥ −0.005.
   - Choose the feasible candidate with the highest objective; ties (|difference| ≤ 1e-12) go to the higher
     threshold.
   - If the best feasible objective is ≤ 0, choose +∞ (never fall back). +∞ is always feasible (Δ = 0).
   - C1 and C3 rows never enter the candidates or the choice.
   - *Reason:* "enumerate threshold candidates from the calibration scores, including both infinite endpoints";
     thresholding at each observed score reaches every distinct switching set.
5. **LR benefit score.**
   - `decision_function` (log-odds) of the fitted LR, thresholded as above.
   - *Reason:* same ranking as `predict_proba`, but no ties caused by probabilities saturating at 1.0. It is not
     described as a probability.
6. **LR sample weights** (per model × seed, GATE_FIT rows of C0, C2, C4, C5, C6):
   - For condition c with groups G_c: each group gets 1/|G_c|.
   - Within a group, the weight is split equally across its images; within an image, equally across its donor
     rows.
   - Every condition then sums to 1, and each condition is multiplied by 1/5.
   - Finally all weights are rescaled so that their mean is 1 (sum = number of rows).
   - The same weights go into `StandardScaler.fit(sample_weight=…)` and `LogisticRegression.fit(sample_weight=…)`.
   - *Reason:* the plan's "balance groups within each condition, then equal condition weight". The rescaling
     keeps `C = 1.0` on its usual scale.
7. **LR labels.**
   - y = 1 if Dice(IMG) − Dice(LANG) > 0.005, on the same pass and seed.
   - If GATE_FIT has only one class for a model × seed: **LR-Gate unsupported** there, with no other split.
8. **IMG choice** (Task 3.1C).
   - Clean IMG Dice from the Phase 3 rerun, GATE_FIT images only.
   - Seed mean per image, then group means, then the mean of the group means.
   - The higher wins; if |difference| < 1e-6, nnU-Net.
9. **Oracle diagnostics** (Task 3.1D).
   - Per row on GATE_FIT, C4–C6: actual_gain and oracle_gain; when the two Dice values are equal the oracle keeps
     LANG.
   - Reported image-weighted and group-weighted (donors → images → groups), per model × seed and as the mean over
     seeds.
   - **"Insufficient headroom"** for a language model: the group-weighted mean oracle gain over C4–C6 on
     S_FIT, averaged over the three seeds, is < 0.005. In that case the gate is marked not pursued for that
     model.
10. **"Neutral" switch.** |Dice(IMG) − Dice(LANG)| ≤ 1e-6 (the per-case files store Dice to 6 decimals; the plan
    uses 1e-6 for the IMG tie).
11. **Features.**
    - Computed in float64 from the float32 maps, per row, before any averaging.
    - F2: Dice of the two scored masks (both empty → 1, exactly one empty → 0).
    - F4 / F5: entropy with clipping to [1e-6, 1 − 1e-6] for the entropy only.
12. **Feature storage.**
    - `results/phase3_reliability_features.csv.gz`: keys (dataset, method, seed, sample_id, condition,
      variant_id, donor_sample_id) + F1–F5 only.
    - `results/phase3_feature_labels.csv.gz`: the same keys + Dice(LANG), Dice(IMG), gain, y, group, partition.
    - The gate code reads F1–F5 from the first file only.
13. **AUROC / average precision on GATE_CAL** (diagnostic). Unweighted over the GATE_CAL rows of C0, C2, C4–C6
    (sklearn), reported only when both classes occur.
14. **MosMed transfer.**
    - The frozen QaTa scaler, coefficients and thresholds of model × seed k are applied to the MosMed
      checkpoint of the same model and seed k, with the same IMG architecture and seed.
    - Units: CT volumes + pooled `bjorke` (11), as in Phase 2.
15. **Strategy metrics.**
    - A gate's per-case Dice, mIoU, HD95 and ASSD are those of the selected prediction (IMG or LANG row of the same
      pass). No re-scoring.
    - NULL: the C1 mask of the same image and seed.
    - Fallback rate: the share of rows switched to IMG.

16. **Reproducibility expectations for the rerun** (checks, not thresholds; added 2026-10-28):
    - **LViT-T and RecLMIS** are deterministic. Their new C2/C3 masks must equal the Phase 2 masks (0 pixels).
      Any difference stops the work until its cause is found and reported to the project lead.
    - **DD-CMD** is expected to differ by about 1–4 pixels per case. The differences are recorded as they are.
    - **IMG** vs. the Phase 0 archive: the result is reported as it is (pixels and Dice of any difference). It
      does not block the next steps; Phase 3 uses the new scores.
17. **Task 3.5 validation results.** GATE_FIT and GATE_CAL are reported separately, never pooled, and both are
    labelled development data. GATE_FIT is used to fit LR and GATE_CAL to choose the thresholds, so neither is an
    independent evaluation.
18. **Weight normalisation (item 6)** (revised 2026-10-28, before any LR result).
    - **Check of whether the weight scale matters.** `src/phase3/check_weight_scale.py` →
      `results/phase3_weight_scale_check.json`, run with the cluster's scikit-learn 1.9.0.
      - Setup: the same synthetic data fitted with weights w and 10·w: weighted `StandardScaler`, then
        `LogisticRegression(C=1.0, solver="lbfgs", max_iter=2000)`.
      - The scaler (mean, scale) is identical. The **coefficients differ** (largest difference 0.050): in
        scikit-learn the weight scale acts like a change of C.
    - **Decision.** Weights are rescaled to **mean 1**, so that C = 1.0 keeps its usual scale. This was fixed
      before any LR result. If the supervisor means another normalisation (e.g. sum 1), the LRs are refitted
      before the freeze.
    - **Task 3.4 does not wait for the supervisor's reply.**

**Decisions recorded.** On 2026-10-28 the project lead accepted items 1–15 and added items 16–18.

## 7. Results of the rerun and of Tasks 3.1C / 3.1D (validation, QaTa GATE_FIT; development data)

**Rerun** (job 371, 2026-10-27 22:25–22:49 EDT, 0 failures). Checks: `src/phase3/check_rerun.py` →
`results/phase3_rerun_checks.csv`, over every case of every new file.
- **All complete.** In every file the mask equals probability > 0.5, and the probabilities are float32.
- **LViT-T and RecLMIS C2 / C3a–c:** identical to the Phase 2 masks (0 pixels), as expected (item 16).
- **DD-CMD C2 / C3a–c:**
  - differs from Phase 2 in 39 of 17,130 QaTa rows (≤ 1 pixel) and 73 of 3252 MosMed rows (≤ 2 pixels);
  - max per-case |ΔDice| 0.0157, on a small MosMed lesion;
  - recorded as it is; Phase 3 uses the new scores.
- **IMG vs. the Phase 0 archive:**
  - U-Net: identical (0 pixels, all 6 runs).
  - nnU-Net: 382–444 of 1429 QaTa cases differ by ≤ 12 pixels, and 13–18 of 273 MosMed cases by ≤ 2 pixels.
    Max per-case |ΔDice| 0.0064; mean Dice per run within 0.00006 of Phase 0. This is the same GPU
    non-determinism Phase 0 recorded for nnU-Net. Phase 3 uses the new scores.

**Task 3.1C, IMG choice** (`src/phase3/select_img_oracle.py`, 10 synthetic tests passed →
`results/phase3_fallback_selection.csv`). Clean Dice on GATE_FIT (859 images, 716 groups), group-weighted:

| Architecture | Dice (group-weighted) |
|---|---|
| **nnU-Net** | **0.7613** |
| U-Net | 0.7592 |

- Difference (U-Net − nnU-Net): −0.0022; image-weighted: 0.7701 vs. 0.7686.
- **nnU-Net is chosen** for all three language models and all conditions, with seeds matched (LANG seed k ↔ IMG
  seed k).

**Task 3.1D, oracle diagnostics** (`results/phase3_oracle_diagnostics.csv`).
- **Set.** S_FIT has 853 images in 710 groups.
- **Sources.** LANG C4–C6 come from Phase 2; IMG is nnU-Net from the rerun.
- **Values.** Group-weighted means over the three seeds; Dice on the [0, 1] scale.

| Method | Condition | Mean oracle gain | Mean actual gain (always IMG) | Rows IMG better | Rows LANG better |
|---|---|---:|---:|---:|---:|
| LViT-T | C4 / C5 / C6 | 0.145 / 0.098 / 0.237 | 0.116 / 0.064 / 0.221 | 75.0 / 69.7 / 86.2% | 24.4 / 29.9 / 13.6% |
| RecLMIS | C4 / C5 / C6 | 0.116 / 0.102 / 0.199 | 0.082 / 0.067 / 0.181 | 70.9 / 68.7 / 84.7% | 28.6 / 30.9 / 15.2% |
| DD-CMD | C4 / C5 / C6 | 0.102 / 0.083 / 0.196 | 0.068 / 0.049 / 0.178 | 70.9 / 68.2 / 83.6% | 28.7 / 31.6 / 16.3% |

- **Headroom** (mean of C4–C6, mean over seeds): LViT-T 0.160, RecLMIS 0.139, DD-CMD 0.127. All are far above
  0.005, so **no model is "insufficient headroom"**.
- **Single rows.** The actual gain ranges from −0.85 to +0.97. The per-row quantiles are in the CSV.
- **What this is.** Diagnostics on development data; the oracle is not deployable.

## 6. Next steps (after confirmation)

1. You submit `slurm/phase3_infer_val.slurm`.
2. Checks after the job:
   - reconciliation of IMG with Phase 0;
   - LANG C2/C3 differences to Phase 2;
   - the mask = probability > 0.5 check on all new files.
3. Task 3.1C (IMG choice on GATE_FIT) and 3.1D (oracle diagnostics), with the details of section 5 fixed in
   advance.
