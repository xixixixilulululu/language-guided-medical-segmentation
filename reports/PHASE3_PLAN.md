# Phase 3: Reliability-Aware Selective Fallback

## Goal

Develop a simple, output-based mechanism that decides when to keep a **language-guided segmentation** or use an existing **image-only segmentation** instead.

The research question is:

> Can selective fallback recover segmentation performance under mismatched text without sacrificing the advantage of the original paired text?

For the image and the supplied report, produce two predictions:

- **LANG:** the frozen language-guided model's prediction using that report.
- **IMG:** a frozen image-only model's prediction of the same image.

A gate selects one **complete** prediction. It does not edit text, combine mask pixels, or train another segmentation network.

### Models and data

| Component | Models / setting |
|---|---|
| Language-guided models | LViT-T, RecLMIS, DD-CMD |
| Image-only fallback candidates | U-Net, nnU-Net (already trained in Phase 0) |
| Datasets | QaTa-COV19 and MosMedData+ |
| Segmentation seeds | 1001, 1002, 1003 |
| Main development dataset | **QaTa-COV19 validation** |
| Secondary dataset | **MosMedData+, exploratory application of the frozen gate** |
| New training | Only small logistic-regression gates, **no segmentation-model retraining** |

Two gate methods are required:

1. **D-Gate:** one threshold on prediction disagreement.
2. **LR-Gate:** logistic regression on five output features, trained to predict when IMG is likely to give a better segmentation than LANG.

## Starting point: Phase 2 validation findings

Phase 2 evaluated the three frozen language-guided models under nine text conditions. Its **validation** experiments are complete. Counterfactual **test** evaluation remains locked.

- On **QaTa-COV19**, the three-attribute mismatch (C6) reduces Dice relative to the original paired text (C0) by approximately **24–26 Dice points** across the three models. Missing location wording (C3c) also causes large losses.
- On **MosMedData+**, the mismatch effects are much smaller and the validation analysis has only **11 grouped units**. Conclusions there remain exploratory.
- Equivalent wording (C2) generally preserves mean Dice, although some models and individual cases are sensitive.
- A mismatched text does not always cause a worse prediction. Recognizing a mismatched report is **not** the same as knowing when IMG is better.

Therefore, **QaTa is the main setting for developing and evaluating the gate**. MosMed is used to check whether an unchanged selection rule transfers or causes unnecessary fallback.

Use the existing `PHASE2_VALIDATION_SUMMARY.md` and its saved outputs. Keep the established text conditions:

| Condition | Meaning |
|---|---|
| C0 | Original paired benchmark text |
| C1 | Empty-string input |
| C2 | Meaning-preserving word substitution |
| C3a / C3b / C3c | Scope / count / location wording removed |
| C4 / C5 / C6 | Real donor report differing in one / two / three defined attributes |

Do not regenerate donor assignments, reinterpret C0 as independently verified clinical truth, or treat C4–C6 as a validated clinical severity scale. C1 is an empty-input control, **not** a separately trained image-only model.

---

# Task 3.1: Prepare Matched Predictions and Measure Fallback Potential

**Goal:** Before training gates, confirm that IMG can actually improve some LANG predictions on the **same validation images**.

### A. Check existing artifacts

Reuse Phase 0–2 checkpoints, validation predictions, scores, and the frozen Phase 2 condition/donor manifest. Record the paths and hashes in `reports/PHASE3_SETUP.md`.

For MosMed LViT-T, use the **batch-size-24** checkpoints recorded in Phase 1 and used in Phase 2. The batch size refers to **training**. Keep the recorded inference settings for each model.

Check that the required **probability maps, exact scored binary masks, and per-case metrics** are available and can be matched by dataset, image ID, model, seed, condition, and donor ID. Use the same 224 × 224 targets and `probability > 0.5` scoring rule. If a required probability map is missing, rerun **inference only** with the same frozen checkpoint and settings. Save new predictions and matching scores together. Do not attach an old score to a numerically different mask.

### B. Split QaTa validation for gate development

Create one fixed group split of the existing QaTa validation cases:

```text
GATE_FIT: approximately 60% of groups
GATE_CAL: approximately 40% of groups
random seed: 3001
```

Use verified patient/subject IDs as groups. For `covid_*` images without a reliable patient ID, retain the existing image-level provisional group and disclose that limitation. Keep every image's conditions, donors, and three seeds in the same partition. Do not try different splits to obtain better performance.

- **GATE_FIT:** choose IMG, construct LR labels, and fit LR.
- **GATE_CAL:** choose operating thresholds and check the trade-off between recovery and clean-text loss.

The original validation set was already used to select segmentation checkpoints and study Phase 2 behavior. Neither gate partition is a new independent cohort for segmentation evaluation.

### C. Choose one IMG architecture

Choose between the already trained **U-Net and nnU-Net** using **clean image-only Dice on QaTa GATE_FIT only**:

1. Average the three seed Dice values per image.
2. Average images within each group.
3. Average group means equally.

Select the higher mean; if tied within `1e-6` Dice, select nnU-Net. This single architecture is used for all three language models and text conditions. Match the seeds:

```text
LANG seed 1001 -> IMG seed 1001
LANG seed 1002 -> IMG seed 1002
LANG seed 1003 -> IMG seed 1003
```

Do not select the image-only model using test results or separately for different text conditions.

### D. Measure the available improvement

On GATE_FIT, match LANG and IMG predictions on the same C4/C5/C6 cases. For each image, seed, condition and donor calculate:

```text
actual_gain = Dice(IMG) - Dice(LANG)
oracle_gain = max(Dice(IMG), Dice(LANG)) - Dice(LANG)
```

The **oracle** is an offline diagnostic that uses ground-truth masks to select the better of the two predictions. It is **not deployable**, and its GATE_CAL/test decisions must never be used to fit a gate.

Report the mean oracle gain, fraction of cases where IMG is better, and the size of possible gains/losses. If the mean oracle gain under C4–C6 is negligible (for example, below 0.005 Dice), explain that this fallback has little potential. In that situation, gate fitting may be marked **not pursued (insufficient headroom)** for that model, rather than forced. Do not change the chosen IMG or train additional models just to force improvement. A no-fallback result is acceptable.

**Deliverables:**

- `reports/PHASE3_SETUP.md`
- `results/phase3_gate_partitions.csv`
- `results/phase3_fallback_selection.csv`
- `results/phase3_oracle_diagnostics.csv`

---

# Task 3.2: Calculate Five Reliability Features

For each image, language model, seed, text condition and donor variant, obtain the frozen probability maps `P_lang` and `P_img` and binary masks `B_lang` and `B_img`.

| Feature | Definition |
|---|---|
| **F1** Probability disagreement | `mean(abs(P_lang - P_img))` |
| **F2** Binary-mask disagreement | `1 - Dice(B_lang, B_img)` |
| **F3** Predicted-area difference | `abs(mean(B_lang) - mean(B_img))` |
| **F4** LANG prediction entropy | `mean(H(P_lang))` |
| **F5** IMG prediction entropy | `mean(H(P_img))` |

For F2 only, treat two empty masks as Dice = 1, and exactly one empty mask as Dice = 0. For entropy use natural logarithms and clip probabilities to `[1e-6, 1-1e-6]` **only for computing entropy**:

```python
H(p) = -p * log(p) - (1 - p) * log(1 - p)
```

Calculate features **before** averaging across seeds or donors. Use consistent prediction artifacts throughout.

**Gate inputs are restricted to F1–F5.** Do not provide ground-truth masks, segmentation Dice, condition labels, donor IDs, patient IDs, or the unavailable C0 prediction at inference. Ground truth may be used to prepare the GATE_FIT training labels and to evaluate/calibrate gates, but never to make the prediction-time switching decision.

**Deliverable:** `results/phase3_reliability_features.csv` (metadata and ground-truth-derived labels stored separately from F1–F5).

---

# Task 3.3: Implement D-Gate

D-Gate uses only F1, without classifier training:

```python
if F1 > threshold:
    prediction = B_img
else:
    prediction = B_lang
```

For each language model and seed, select one threshold on **QaTa GATE_CAL**, using the common calibration rule below. Do not use separate thresholds for C4, C5, and C6.

### Common calibration rule for D-Gate and LR-Gate

For each threshold compute `Delta(C) = Dice(gate) - Dice(LANG)` using paired cases. Average donor differences within images, images within groups, then groups equally. For the C4–C6 objective, use **the same target-image intersection** across all three conditions.

Maximize:

```text
objective = [Delta(C4) + Delta(C5) + Delta(C6)] / 3
```

Subject to:

```text
Delta(C0) >= -0.005
Delta(C2) >= -0.005
Delta(C4) >= -0.005
Delta(C5) >= -0.005
Delta(C6) >= -0.005
```

Dice is on a 0–1 scale: `0.005` means **0.5 Dice percentage points**. The first two constraints protect original and equivalent wording. The other constraints prevent a large loss in one mismatch condition from being hidden by gains in another.

Evaluate C0/C2 constraints using their full eligible GATE_CAL cases. Include a **never-fallback** candidate (`threshold = +infinity`); this is the chosen rule if no feasible threshold yields positive mean improvement. Use `score > threshold` consistently and break equal objective scores by choosing the **higher threshold** (fewer fallbacks).

Enumerate threshold candidates from the calibration scores, including both infinite endpoints. Do not choose thresholds after inspecting test performance or the withheld C1/C3 gate outcomes. These calibration limits are research targets, not clinical guarantees.

**Deliverable:** `results/phase3_dgate_calibration.csv` (threshold, C0/C2 losses, C4–C6 gains, and fallback rates).

---

# Task 3.4: Implement LR-Gate

**Goal:** Predict whether switching to IMG is likely to improve Dice, rather than merely classify which artificial text condition was used.

Train **one logistic regression per language model × seed** using QaTa **GATE_FIT only**, up to **nine small fits**.

### Training labels

For each available C0, C2, C4, C5 and C6 prediction on GATE_FIT, calculate:

```python
gain = Dice(IMG) - Dice(LANG)
y = 1 if gain > 0.005 else 0
```

`y = 1` means that IMG improves Dice by more than 0.5 percentage points under the benchmark mask. `y = 0` means that this improvement threshold is not met. These are **relative segmentation-benefit labels**, not verified clinical truth or text-correctness labels.

Do **not** use C1 or C3a/C3b/C3c for LR fitting. Keep all donors and seeds for the same target image within GATE_FIT or GATE_CAL as assigned in Task 3.1. **Fit the classifier only from GATE_FIT labels.** GATE_CAL ground truth is permitted for threshold calibration and diagnostics, and test ground truth is permitted only for final evaluation, never for fitting or selecting a gate.

### Fixed model and training rules

```text
Inputs: F1, F2, F3, F4, F5
Scaler: StandardScaler fitted on GATE_FIT only
Classifier: logistic regression, L2 penalty, C=1.0
Solver: lbfgs
Max iterations: 2000
```

Use sample weights so that groups with many images or several donor variants do not dominate training: balance groups within each condition, then give the five included conditions equal total weight. Normalize weights before fitting, and use the same training weights for feature standardization. Do not fit an additional model family or tune hyperparameters.

If the training labels contain only one class, report **LR-Gate unsupported** for that model/seed. Keep D-Gate and the other baselines; do not select a favorable new split.

### Set the LR operating threshold

On **GATE_CAL**, compute the fitted model's benefit score and choose a threshold using exactly **Task 3.3's calibration rule**. The score is used for ranking possible beneficial switches; do not describe it as a clinically calibrated probability.

```python
if benefit_score > threshold:
    prediction = B_img
else:
    prediction = B_lang
```

Freeze the scaler, coefficients, intercept, and selected threshold together. Do not refit LR on all validation data after threshold calibration.

As a **secondary diagnostic**, use the withheld GATE_CAL ground truth to report AUROC and average precision for predicting the `gain > 0.005` label when both classes exist. These scores are calibration-partition diagnostics, not an independent final assessment. The main success metric is **segmentation gain**, not classifier AUROC.

**Deliverables:**

- `results/phase3_lr_calibration.csv`
- `runs/phase3/<language_model>/seed<k>/gate_artifacts/`

---

# Task 3.5: Evaluate the Gates on Validation

Compare four deployable prediction strategies:

| Strategy | Prediction |
|---|---|
| **LANG** | Always use the supplied-text language-guided mask |
| **IMG** | Always use the selected image-only mask |
| **D-Gate** | Switch using F1 |
| **LR-Gate** | Switch using the five-feature benefit score |

Also show two diagnostic references:

- **NULL:** the existing C1 mask for the same image/seed, where available. It is not an image-only model.
- **ORACLE:** the higher-Dice mask between LANG and IMG, chosen using ground truth. It is not deployable.

For each text condition report mean Dice, mIoU, HD95, ASSD, empty-prediction rate and fallback rate. All strategies under a condition must use **the same eligible images, donors and seeds**. Report image-weighted results for continuity with Phase 2; retain group-weighted paired differences for statistical comparisons.

### Routing quality

Use ground truth **only after routing**, to classify each switch:

```text
Beneficial: Dice(IMG) > Dice(LANG)
Harmful:    Dice(IMG) < Dice(LANG)
Neutral:    equal within numerical tolerance
```

Report the number of switched cases, beneficial/harmful switch percentages, net Dice gain, and clean-text loss. A high switching rate is not itself an improvement.

### Conditions excluded from optimization

After gate fitting and threshold selection, evaluate **C1, C3a, C3b and C3c** without changing either gate. Pay particular attention to **C3c**, because removing location wording caused large losses in Phase 2. This is a different intervention from donor-text substitution, but it is **not** a previously unknown phenomenon: its Phase 2 validation results are already available.

### MosMedData+ exploratory check

After fixing the QaTa gates, apply the **same architecture choice, feature scaler, LR coefficients and thresholds** to MosMed; use the corresponding **MosMed segmentation checkpoints**, including LViT-T batch 24. Do not fit a MosMed-specific gate or normalization, choose a new IMG model, or adjust thresholds using MosMed results.

Report the transfer results descriptively, including the **11 validation analysis units**, oracle headroom and unnecessary C0/C2 switches. Failure on MosMed is a meaningful limitation, not permission to retune the test protocol.

**GATE_CAL scores reported here are post-calibration summaries, not independent held-out estimates of gate effectiveness.**

**Deliverables:**

- `results/phase3_validation_results.csv`
- `results/phase3_fallback_rates.csv`
- `results/phase3_routing_quality.csv`
- `results/phase3_mosmed_exploratory.csv`

---

# Task 3.6: Freeze Both Phases Before Counterfactual Test Evaluation

**Do not evaluate Phase 2 counterfactual test conditions, run Phase 3 gates on test, or inspect counterfactual test features until both protocols are frozen.**

First finish GATE_FIT/GATE_CAL development and all decisions. Then record and commit:

1. Phase 2 condition and donor-generation rules, eligibility and statistical rules.
2. Phase 3 IMG architecture, segmentation checkpoint identifiers, feature definitions, and exact evaluation rules.
3. Fixed partitions, fitted LR artifacts, all thresholds (including never-fallback), and comparison families.
4. The validation/test exposure history and reproducibility information.

Create:

- `reports/PHASE2_PROTOCOL_FREEZE.md`
- `reports/PHASE3_PROTOCOL_FREEZE.md`

Only **after** this freeze, run the final counterfactual test benchmark with the existing Phase 2 rules and evaluate all Phase 3 strategies using the **same** outputs. Do not change the gate based on test results. Check C0 against the archived Phase 1 predictions using the existing numerical-tolerance policy.

Phase 1 clean test results have already been seen; the protected outcome here is the **new counterfactual test evaluation**, not a completely untouched test cohort.

**Deliverables:**

- `results/phase3_test_results.csv`
- `results/phase3_test_per_case.csv`
- `reports/PHASE2_RESULTS.md` (completed with the final locked Phase 2 test results)

---

# Task 3.7: Statistical Comparison

Use **paired** differences between strategies on identical images, donors and seeds. Within each comparison:

1. Average donor differences within image and seed.
2. Average the three seed differences within each image.
3. Average image differences within verified patient groups (QaTa) or CT volumes (MosMed).
4. Analyze the equally weighted group differences.

Do not count repeated donors, seeds or neighboring CT slices as independent patients. For QaTa `covid_*` images with unknown identity, state that their image-level provisional grouping limits inference.

### Primary comparison (QaTa only)

```text
LR-Gate versus LANG under C4, C5 and C6
for LViT-T, RecLMIS and DD-CMD
```

These are **nine planned comparisons**. Report mean Dice difference, group-level 95% confidence interval, raw paired p-value, and Holm-adjusted p-value across this family of nine. Use two-sided alpha = 0.05.

### Secondary comparison (QaTa only)

```text
LR-Gate versus D-Gate under C4, C5 and C6
for the same three methods
```

Apply Holm separately across these nine comparisons. Also report D-Gate versus LANG descriptively.

Report C0 and C2 gate-minus-LANG differences with group-level confidence intervals and the `-0.005` development margin. A nonsignificant difference from zero **does not prove** preservation within that margin.

Report the C1/C3 effects and MosMed exploratory results descriptively. MosMed has few validation groups and known train/validation/test volume overlap; do not infer equivalence from nonsignificant comparisons. Preserve the original empty-mask policy for HD95/ASSD.

**Deliverable:** `results/phase3_pairwise_results.csv`.

---

# Task 3.8: Computational Cost

On a small, fixed validation subset, report approximate end-to-end latency per image/report and peak GPU memory for:

```text
LANG only
IMG only
LANG + IMG + D-Gate
LANG + IMG + LR-Gate
```

Use consistent hardware, image size, inference settings, and timing. Include the time for **both** segmentation models: the gate needs both predictions before it can choose one. Caching IMG for repeated donor reports saves offline benchmark time, but is not the cost of processing a new patient.

**Deliverable:** `results/phase3_computational_cost.csv`.

---

# Task 3.9: Final Report

Create `reports/PHASE3_RESULTS.md`. Answer:

1. How much improvement is theoretically available from selecting between LANG and IMG (oracle headroom)?
2. Can D-Gate and LR-Gate recover some of that improvement under C4/C5/C6?
3. Does LR-Gate add meaningful value over the simpler D-Gate?
4. How much performance is lost or gained under C0 and C2?
5. Do the frozen gates help under the withheld missing-location condition C3c?
6. How often do gates make beneficial versus harmful switches?
7. What happens when the QaTa gate is applied unchanged to MosMed?
8. Is the extra computation justified by the measured segmentation benefit?

Distinguish **observed results, exploratory findings and limitations**. Do not claim that the gate verifies clinical-report truth. A method that selects never to fall back, or that fails to outperform a simple threshold, is a legitimate result.

### Required figures

1. QaTa Dice under **C0/C2 versus C4/C5/C6** for LANG, IMG, D-Gate and LR-Gate.
2. Fallback rates and beneficial/harmful switches across text conditions.
3. Paired Dice improvements of LR-Gate over LANG and D-Gate, including the C3c diagnostic.
4. A compact MosMed exploratory comparison with group counts and oracle headroom.

Use a documented selection rule for illustrative cases; include harmful as well as successful switches.

## Completion checklist

- [ ] Phase 2 validation conditions, donors and checkpoint versions preserved.
- [ ] QaTa GATE_FIT/GATE_CAL partition fixed and grouped.
- [ ] IMG chosen on GATE_FIT, with matched seeds.
- [ ] Probabilities, masks and metrics are consistently matched.
- [ ] Oracle headroom and F1–F5 calculated.
- [ ] D-Gate implemented and calibrated, or a documented insufficient-headroom outcome.
- [ ] LR-Gate trained on **beneficial-fallback** labels and calibrated separately, or documented as unsupported/not pursued.
- [ ] C0 and C2 performance costs reported.
- [ ] C1/C3 withheld from fitting and threshold choice.
- [ ] MosMed checked without gate retraining or local threshold tuning.
- [ ] Both protocols frozen **before** counterfactual test access.
- [ ] Final Phase 2 and Phase 3 test results, paired comparisons, routing quality and cost reported.

## Project inputs

- `PHASE2_VALIDATION_SUMMARY.md` and its Phase 2 donor manifest, per-case validation results and saved predictions.
- `PHASE1_SENSITIVITY_LVIT.md` for the MosMed LViT batch-size-24 checkpoints.
- `PHASE1_RESULTS.md` and Phase 0 checkpoints for model and evaluator provenance.

The findings motivating this phase are from **validation**. The effectiveness of selective fallback remains to be measured.
