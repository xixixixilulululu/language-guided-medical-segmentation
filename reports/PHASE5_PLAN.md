# Phase 5: EMBC Story Completion

## Goal

Complete only the additional analyses needed to finish the EMBC paper.

The paper story is:

> Frozen language-guided segmentation models perform strongly with their original paired text, but can fail severely when the supplied text is missing or mismatched. A lightweight selective fallback can recover part of this loss without retraining the segmentation models. Phase 5 tests whether that fallback provides a useful trade-off when text reliability is unknown, compared with simpler alternatives.

Phase 5 does **not** develop a new segmentation architecture.

Phase 4 remains:

```text
Stage 1: STOP
```

Do not run any new spatial-grid, location-mapping, spatial-feature, or spatial-gate experiment.

---

## Established Results

Use the frozen Phase 2/3 results as the starting point.

On QaTa-COV19:

1. Removing case-specific text or supplying mismatched text substantially reduces Dice in all three frozen language-guided models.
2. Three-attribute mismatch (C6) causes about **23–24 Dice-point loss** relative to the original paired text.
3. Location-related text changes are especially harmful; the number of changed attributes is not a reliable severity ordering.
4. The Phase 3 LR-Gate improves LANG under C4/C5/C6 for all three language-guided models.
5. LR-Gate preserves most, but not all, clean-text performance and recovers only part of the available oracle headroom.
6. On average, LANG is stronger under C0/C2, while IMG is stronger under C4–C6.

Therefore, the remaining EMBC question is:

> When text reliability is unknown, does selective fallback provide a useful trade-off compared with always-LANG, always-IMG, a simple segmentation-confidence fallback, or simple probability averaging?

---

## Fixed Scope

| Item | Requirement |
|---|---|
| Main dataset | **QaTa-COV19 only** |
| Language-guided models | LViT-T, RecLMIS, DD-CMD |
| Image-only fallback | Frozen nnU-Net from Phase 3 |
| Seeds | 1001, 1002, 1003 |
| Main conditions | C0, C2, C4, C5, C6 |
| Existing strategies | LANG, IMG, D-Gate, LR-Gate |
| New strategies | **SDC-Fallback, Fixed Average** |
| New segmentation training | **None** |
| New dataset split | **None** |
| Phase 4 continuation | **Not allowed** |

Reuse the exact Phase 2 donor assignments, Phase 3 GATE_FIT/GATE_CAL split, frozen checkpoints, probability maps, masks, grouping rules, and evaluator.

Do not change D-Gate or LR-Gate.

No new MosMed experiment is required. Keep the existing MosMed results as exploratory transfer evidence only.

The existing Phase 3 C3c result may be used in the paper as a descriptive condition that was excluded from gate fitting/calibration. Phase 5 does not need to rerun all new baselines on C3c.

---

# Task 5.1: Add Two Necessary Simple Baselines

The purpose is to test whether the Phase 3 LR-Gate provides value beyond simpler alternatives.

Only two new strategies are required.

## A. SDC-Fallback

Implement **Soft Dice Confidence (SDC)** exactly as defined by Borges, Pacheco, and Silva, *Machine Learning*, 2026.

Reference:

```text
Soft Dice Confidence: A Near-Optimal Confidence Estimator
for Selective Prediction in Semantic Segmentation
DOI: 10.1007/s10994-026-07096-w
```

For the frozen LANG probability map `P_lang` and binary mask `B_lang`:

```text
SDC =
2 * sum(P_lang * B_lang)
/
[sum(P_lang) + sum(B_lang)]
```

If both are empty:

```text
SDC = 0
```

Use the same scored LANG binary mask definition as the existing evaluator.

Fallback rule:

```python
if SDC < threshold:
    prediction = B_img
else:
    prediction = B_lang
```

For each language model × seed, choose one threshold on the existing **GATE_CAL** partition.

Use exactly the Phase 3 threshold objective:

```text
maximize mean[Delta(C4), Delta(C5), Delta(C6)]
```

subject to:

```text
Delta(C0) >= -0.005
Delta(C2) >= -0.005
Delta(C4) >= -0.005
Delta(C5) >= -0.005
Delta(C6) >= -0.005
```

where:

```text
Delta(C) = Dice(strategy) - Dice(LANG)
```

Use the same donor → image → group aggregation as Phase 3.

Include a never-fallback candidate.

Do not modify the SDC formula, add another confidence estimator, or tune a separate threshold for each condition.

### Output

```text
results/phase5_sdc_calibration.csv
```

Required fields:

```text
model
seed
threshold
C0_delta
C2_delta
C4_delta
C5_delta
C6_delta
fallback_rate_C0
fallback_rate_C4
fallback_rate_C5
fallback_rate_C6
```

---

## B. Fixed Average

Use a parameter-free probability average:

```python
P_avg = 0.5 * P_lang + 0.5 * P_img
B_avg = P_avg > 0.5
```

Use the stored float32 probability maps used by Phase 3.

Do not tune the averaging weight or binary threshold.

This baseline answers:

> Is selective routing needed, or is simply combining LANG and IMG enough?

No training or calibration is required.

---

# Task 5.2: Freeze Phase 5 Before the New Test Comparison

Phase 2/3 test results have already been examined.

Therefore, Phase 5 is a **post-hoc supplementary comparison**, not a new independent confirmation.

Before computing either new strategy on the test artifacts, freeze and commit:

1. exact SDC formula and empty-case rule;
2. SDC threshold-selection rule;
3. Fixed Average formula;
4. eligible conditions;
5. aggregation/statistical rules;
6. required outputs and figures;
7. analysis code;
8. code commit and relevant artifact hashes.

Create:

```text
reports/PHASE5_PROTOCOL_FREEZE.md
```

After the freeze:

- compute Phase 5 results on the frozen QaTa test artifacts once;
- do not change the rules after seeing the Phase 5 test comparison;
- do not retune LR-Gate, D-Gate, SDC-Fallback, or Fixed Average.

If a required float32 probability map is missing, rerun **inference only** with the same frozen checkpoint and settings, document the rerun, and reconcile its scored mask with the existing archive before using it.

---

# Task 5.3: Main Strategy Comparison

Compare six deployable strategies on QaTa test:

| Strategy | Definition |
|---|---|
| **LANG** | Always use language-guided prediction |
| **IMG** | Always use image-only prediction |
| **D-Gate** | Existing Phase 3 disagreement gate |
| **LR-Gate** | Existing Phase 3 five-feature benefit-aware gate |
| **SDC-Fallback** | Fall back when LANG segmentation confidence is low |
| **Fixed Average** | Average LANG and IMG probabilities |

Evaluate:

```text
C0
C2
C4
C5
C6
```

For every strategy × model × condition report:

```text
Dice
mIoU
HD95
ASSD
empty-prediction rate
fallback rate, when applicable
```

Use identical eligible images, donors, seeds, and groups for every paired comparison.

## Main Phase 5 comparisons

For C4/C5/C6 report:

```text
LR-Gate - SDC-Fallback
LR-Gate - Fixed Average
```

for all three language-guided models.

Use the Phase 3 aggregation order:

1. donor differences within image and seed;
2. seeds within image;
3. images within verified patient/group;
4. equal-weight group differences.

Report:

```text
paired group-weighted Dice difference
95% confidence interval
```

These are **post-hoc supplementary comparisons**.

Do not present them as a new confirmatory hypothesis family.

Formal p-values are not required for the EMBC story. If they are generated for completeness, label them explicitly as post-hoc/descriptive and do not use them as the basis of the paper's main claim.

## Clean-text cost

For C0 and C2 report:

```text
strategy - LANG
```

with group-level 95% confidence intervals.

Keep the Phase 3 development margin visible:

```text
-0.005 Dice
```

Do not claim equivalence or safety from a confidence interval or nonsignificant result.

### Outputs

```text
results/phase5_test_results.csv
results/phase5_pairwise_results.csv
```

---

## Computational Comparison

Do not run a new timing benchmark.

Reuse the Phase 3 measured mean inference times for LANG and IMG.

Report approximate model-forward cost:

```text
LR-Gate / D-Gate / Fixed Average:
    LANG_time + IMG_time

SDC-Fallback:
    LANG_time + fallback_rate * IMG_time
```

This ignores the small score-computation overhead and should be labelled an approximation.

The purpose is to acknowledge that SDC-Fallback can avoid running IMG on cases that do not fall back, while LR-Gate and Fixed Average require both predictions.

### Output

```text
results/phase5_compute_summary.csv
```

---

# Task 5.4: Mixed Text-Reliability Analysis

This task answers the main deployment question:

> Why use selective fallback instead of always-LANG or always-IMG?

No new fitting is allowed.

Use the frozen QaTa test outputs from Task 5.3.

Use the **same common target population** for C0, C4, C5, and C6.

For each model and strategy define:

```text
D_wrong =
[D(C4) + D(C5) + D(C6)] / 3
```

where each `D(C)` is the group-weighted mean Dice on the matched population.

This is an equal-weight **benchmark stress-test mixture**.

It is not an estimate of the real distribution of report errors.

Let:

```text
p = assumed prevalence of unreliable text
```

Define:

```text
Expected_Dice(p) =
(1 - p) * D(C0)
+ p * D_wrong
```

## Plotting

For visualization calculate:

```text
p = 0.00, 0.01, ..., 1.00
```

## Crossover points

Do **not** estimate crossover points from the 0.01 plotting grid.

For strategy A versus strategy B define:

```text
delta_clean = D_A(C0) - D_B(C0)
delta_wrong = D_A(wrong) - D_B(wrong)
```

Then solve exactly:

```text
delta_clean + p * (delta_wrong - delta_clean) = 0
```

If:

```text
delta_wrong != delta_clean
```

the crossover is:

```text
p_star =
-delta_clean
/
(delta_wrong - delta_clean)
```

Report the crossover only if:

```text
0 <= p_star <= 1
```

Otherwise report:

```text
none
```

For each model report:

1. expected Dice curves;
2. exact crossover of LR-Gate versus always-LANG;
3. exact crossover of LR-Gate versus always-IMG;
4. range of `p` where LR-Gate is better than both fixed choices;
5. corresponding results for SDC-Fallback and Fixed Average.

### Sensitivity to mismatch type

Repeat only the crossover calculation separately for:

```text
C4 only
C5 only
C6 only
```

Do not create additional corruption mixtures.

Do not interpret a crossover value as a clinical threshold.

### Outputs

```text
results/phase5_mixed_reliability.csv
results/phase5_crossover_summary.csv
```

---

# Required Figures

Only **two new figures** are required.

## Figure 1: Strategy comparison

Create one compact figure showing the main C0/C4/C5/C6 results for:

```text
LANG
IMG
LR-Gate
SDC-Fallback
Fixed Average
```

for the three language-guided models.

D-Gate may remain in the result table because its comparison with LR-Gate is already established in Phase 3.

## Figure 2: Mixed text reliability

Plot expected Dice versus assumed unreliable-text prevalence `p` for:

```text
LANG
IMG
LR-Gate
SDC-Fallback
Fixed Average
```

using the equal C4/C5/C6 benchmark mixture.

Mark the LR-Gate crossover points with always-LANG and always-IMG when they exist.

Do not create a new failure-case figure in Phase 5. Reuse the existing Phase 3 successful, harmful, and missed-switch examples if a qualitative figure is needed in the paper.

---

# Required Final Report

Create:

```text
reports/PHASE5_RESULTS.md
```

Keep it concise.

It must answer:

1. How does LR-Gate compare with SDC-Fallback under C4–C6?
2. How does LR-Gate compare with Fixed Average under C4–C6?
3. What clean-text cost does each strategy introduce under C0/C2?
4. What is the approximate computational trade-off between LR-Gate and SDC-Fallback?
5. Under what assumed unreliable-text prevalence does LR-Gate become preferable to always-LANG?
6. Under what prevalence does always-IMG become preferable to LR-Gate?
7. Is there a range where LR-Gate is better than both fixed choices?
8. Are the conclusions consistent across C4, C5, and C6?
9. Which conclusions come from the frozen Phase 2/3 study, and which are post-hoc Phase 5 analyses?

---

# EMBC Story to Preserve

The paper should follow this logic.

## 1. Benefit and dependence

Under the original paired text, the frozen language-guided models achieve strong segmentation performance.

Removing case-specific text causes a large drop on QaTa-COV19, showing that these frozen models depend strongly on their text input.

Do not claim that cross-architecture Phase 1 differences alone prove the causal benefit of language.

## 2. Risk

The same models can fail severely when the supplied text is mismatched.

Location-related changes are especially influential on QaTa-COV19.

The number of changed attributes is not a validated clinical severity scale.

## 3. Mitigation

A frozen image-only model provides a useful fallback.

The Phase 3 LR-Gate uses disagreement and uncertainty from two frozen predictions to estimate when switching to IMG is likely to improve segmentation.

No segmentation-model retraining is required.

## 4. Practical trade-off

On average:

```text
reliable paired text -> LANG is stronger
known unreliable text -> IMG is stronger
```

The practical problem is the intermediate setting where text reliability is unknown.

Phase 5 tests whether LR-Gate provides a useful trade-off relative to:

```text
single-model segmentation confidence
simple probability averaging
always-LANG
always-IMG
```

The mixed-reliability analysis is a benchmark scenario analysis, not an estimate of real clinical report-error prevalence.

## 5. Limits

Do not claim:

- that the gate verifies whether a clinical report is true;
- that C4–C6 are clinically validated error-severity levels;
- that Phase 5 is independent confirmation;
- that MosMed establishes generalization;
- that the Phase 4 spatial pilot succeeded;
- that the method guarantees safety.

The contribution is:

> a controlled study of unreliable text in frozen language-guided segmentation models, together with a lightweight selective-fallback analysis that quantifies when fallback can be useful and where it remains limited.

---

# Phase 5 Completion Checklist

- [ ] Phase 0–3 checkpoints and existing results remain unchanged.
- [ ] Phase 4 remains STOP.
- [ ] SDC implemented exactly from the published definition.
- [ ] SDC thresholds selected on GATE_CAL only.
- [ ] Fixed Average uses exactly 0.5/0.5 probability averaging.
- [ ] Phase 5 protocol/code frozen before computing new test comparisons.
- [ ] Phase 5 test artifacts computed once with no result-driven retuning.
- [ ] LR-Gate vs SDC-Fallback completed for C4–C6.
- [ ] LR-Gate vs Fixed Average completed for C4–C6.
- [ ] C0/C2 costs reported.
- [ ] Approximate computational comparison reported.
- [ ] Mixed-reliability curves completed.
- [ ] Exact crossover points completed.
- [ ] C4/C5/C6 crossover sensitivity completed.
- [ ] `reports/PHASE5_RESULTS.md` complete.

---

# Required Outputs

```text
reports/PHASE5_PROTOCOL_FREEZE.md
reports/PHASE5_RESULTS.md

results/phase5_sdc_calibration.csv
results/phase5_test_results.csv
results/phase5_pairwise_results.csv
results/phase5_compute_summary.csv
results/phase5_mixed_reliability.csv
results/phase5_crossover_summary.csv
```

No additional experiment should be added without supervisor approval.
