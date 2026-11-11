# Phase 5 Part 2: U-Net Backbone Sensitivity

## Goal

Test whether the main Phase 5 mitigation finding depends on using **nnU-Net** as the image-only reference.

The question is:

> If the frozen IMG reference is changed from nnU-Net to U-Net, does simple fixed probability fusion still provide the same qualitative robustness benefit?

This is a **targeted sensitivity analysis**, not a new method-development phase.

Do not add a new segmentation model, new dataset, new text condition, new gate, or new fusion weight.

---

## Why This Check Is Necessary

Phase 0 trained three image-only baselines: U-Net, TransUNet, and nnU-Net.

Phase 3 selected one fixed IMG reference for all later mitigation experiments. U-Net and nnU-Net were the predefined candidates, and nnU-Net was selected on QaTa `GATE_FIT`.

Phase 5 then found that, with nnU-Net as IMG, **Fixed Average** strongly mitigated C4-C6 mismatch and provided a favorable mixed-reliability trade-off.

The remaining reviewer-level question is:

> Is this fusion result specific to nnU-Net, or does it remain when a second independently trained image-only backbone is used?

Use **U-Net** as the second IMG reference because it is already trained, was an original Phase 3 fallback candidate, and has similar clean QaTa performance to nnU-Net.

Do not add TransUNet in this phase.

---

## Fixed Scope

| Item | Requirement |
|---|---|
| Dataset | **QaTa-COV19 only** |
| Language-guided models | LViT-T, RecLMIS, DD-CMD |
| Second IMG reference | Frozen **U-Net** from Phase 0 |
| Seeds | 1001, 1002, 1003, seed-matched to LANG |
| Conditions | C0, C2, C4, C5, C6 |
| Strategies to compute | LANG, U-Net IMG, U-Net Fixed Average |
| New segmentation training | **None** |
| New gate fitting/calibration | **None** |
| New fusion-weight search | **None** |
| New text generation | **None** |
| MosMedData+ | **Do not run** |

Reuse the exact Phase 2 donor manifest, Phase 3 grouping rules, frozen LANG predictions, and evaluator.

This analysis is **post-hoc sensitivity evidence**. The QaTa test split has already been examined in earlier phases.

---

# Task 5.2.1: Prepare Frozen U-Net Predictions

Use the existing Phase 0 U-Net checkpoints for QaTa-COV19:

```text
seed 1001
seed 1002
seed 1003
```

Match seeds:

```text
LANG seed 1001 -> U-Net seed 1001
LANG seed 1002 -> U-Net seed 1002
LANG seed 1003 -> U-Net seed 1003
```

Required artifacts:

```text
P_img_unet: float32 lesion probability map
B_img_unet: binary mask using probability > 0.5
```

If exact float32 U-Net probabilities are not already available, rerun **inference only** with the frozen Phase 0 checkpoint and original inference settings.

Do not retrain U-Net.

Before any sensitivity result is computed, reconcile the U-Net scored masks and per-case Dice with the archived Phase 0 evaluation.

Record:

```text
checkpoint/configuration for each seed
inference command and commit
probability/mask paths
number of cases with mask differences versus the Phase 0 archive
maximum pixel difference
mean and maximum |Delta Dice|
artifact hashes
```

If there is a material mismatch that cannot be explained by stored-probability precision, stop and report it before continuing.

### Output

```text
reports/PHASE5_PART2_SETUP.md
```

---

# Task 5.2.2: Freeze the Sensitivity Analysis

Before computing the new test results, freeze and commit:

1. the reconciled U-Net probability/mask artifacts;
2. the fixed fusion rule;
3. eligible conditions and donor manifest;
4. aggregation rules;
5. mixed-reliability formula;
6. required outputs and analysis code.

Create:

```text
reports/PHASE5_PART2_PROTOCOL_FREEZE.md
```

After this freeze, run the test analysis once.

Do not change the fusion rule or add another IMG backbone after inspecting the result.

---

# Task 5.2.3: U-Net Fixed Average

For each language-guided model and matched seed, compute:

```python
P_avg_unet = 0.5 * P_lang + 0.5 * P_img_unet
B_avg_unet = P_avg_unet > 0.5
```

Do not tune the averaging weight.

Do not tune the binary threshold.

Use the same stored float32 LANG probability maps used in Phase 5.

Evaluate:

```text
C0
C2
C4
C5
C6
```

Compare only:

```text
LANG
U-Net IMG
U-Net Fixed Average
```

### Main mismatch comparison

For each language-guided model under C4/C5/C6 report:

```text
U-Net Fixed Average - LANG
U-Net Fixed Average - U-Net IMG
```

Use the same Phase 3/5 aggregation order:

1. average donor differences within image and seed;
2. average the three seeds within image;
3. average images within verified patient/group;
4. analyze equally weighted group differences.

Report:

```text
image-weighted mean Dice for readability
paired group-weighted Dice difference
95% confidence interval
```

No new confirmatory p-value family is required.

### Clean-text context

For C0 and C2 report:

```text
U-Net Fixed Average - LANG
U-Net IMG - LANG
```

with group-level 95% confidence intervals.

The purpose is to show the robustness/clean-performance trade-off, not to claim equivalence or safety.

### Outputs

```text
results/phase5_part2_unet_test_results.csv
results/phase5_part2_unet_pairwise.csv
```

---

# Task 5.2.4: Repeat the Mixed-Reliability Analysis with U-Net

This step is required because the mixed-reliability trade-off is part of the final EMBC story.

Use the same common C0/C4/C5/C6 target population and the same definition used in Phase 5.

For each language-guided model define:

```text
D_wrong = [D(C4) + D(C5) + D(C6)] / 3
```

with equal weight across C4, C5, and C6.

For each strategy:

```text
LANG
U-Net IMG
U-Net Fixed Average
```

calculate:

```text
Expected_Dice(p) = (1 - p) * D(C0) + p * D_wrong
```

where `p` is the assumed prevalence of unreliable text in the benchmark scenario.

Use `p = 0.00, 0.01, ..., 1.00` only for plotting/summary values.

Compute exact crossover points analytically, as in Phase 5:

```text
Fixed Average vs LANG
Fixed Average vs U-Net IMG
```

Also report the range of `p` where U-Net Fixed Average is better than both fixed choices.

Repeat the crossover calculation separately for C4, C5, and C6 as a compact sensitivity table.

Do not introduce additional mismatch mixtures.

`p` is a scenario parameter, not an estimate of clinical report-error prevalence.

### Outputs

```text
results/phase5_part2_unet_mixed_reliability.csv
results/phase5_part2_unet_crossovers.csv
```

---

# Task 5.2.5: Backbone-Consistency Summary

Create one compact table comparing the two IMG backbones.

Required rows:

```text
LViT-T
RecLMIS
DD-CMD
```

Required columns:

```text
nnU-Net IMG clean Dice
U-Net IMG clean Dice
nnU-Net Fixed Average: C4/C5/C6 gain over LANG
U-Net Fixed Average: C4/C5/C6 gain over LANG
nnU-Net Fixed Average: p range better than LANG and IMG
U-Net Fixed Average: p range better than LANG and IMG
qualitative conclusion
```

The main question is:

> Does simple fixed fusion remain a useful mitigation when the image-only backbone changes?

Do not require numerical equality across backbones.

### Interpretation

If U-Net Fixed Average again improves the main C4-C6 mismatch results and shows a useful mixed-reliability region:

```text
Conclusion: the simple-fusion mitigation finding is not specific to nnU-Net.
```

If the benefit becomes inconsistent or disappears:

```text
Conclusion: the mitigation effect is backbone-dependent and the manuscript claim must be limited accordingly.
```

Do not run TransUNet or tune the averaging weight to rescue the result.

### Output

```text
results/phase5_part2_backbone_consistency.csv
```

---

# Required Final Report

Create:

```text
reports/PHASE5_PART2_UNET_SENSITIVITY.md
```

Keep it short.

It must answer only:

1. Were the frozen U-Net artifacts reproduced correctly?
2. Under C4-C6, does U-Net Fixed Average improve over LANG?
3. How does U-Net Fixed Average compare with U-Net IMG under C4-C6?
4. What are the C0/C2 clean-text effects?
5. What mixed-reliability range favors U-Net Fixed Average over both LANG and U-Net IMG?
6. Is the simple-fusion finding qualitatively consistent between nnU-Net and U-Net?
7. What exact claim should be carried into the EMBC manuscript?

No new figure is required unless requested by the supervisor after reviewing the table.

---

# Required Outputs

```text
reports/PHASE5_PART2_SETUP.md
reports/PHASE5_PART2_PROTOCOL_FREEZE.md
reports/PHASE5_PART2_UNET_SENSITIVITY.md

results/phase5_part2_unet_test_results.csv
results/phase5_part2_unet_pairwise.csv
results/phase5_part2_unet_mixed_reliability.csv
results/phase5_part2_unet_crossovers.csv
results/phase5_part2_backbone_consistency.csv
```

After this sensitivity analysis, **do not start another experiment automatically**.

Submit the result for supervisor review and move to manuscript preparation unless the result reveals a clear scientific inconsistency that requires discussion.
