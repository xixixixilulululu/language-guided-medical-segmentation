# Phase 4: Spatial Evidence Feasibility Pilot

## Goal

Determine whether a **simple spatial agreement signal** between a benchmark-provided report and an independent image-only segmentation can improve selective fallback beyond the Phase 3 LR-Gate.

**Research question:** When a report is supplied, does comparing its stated lesion location with visual evidence help us decide whether to keep the language-guided segmentation (LANG) or use the image-only segmentation (IMG)?

This is a **small, staged feasibility study**, not a commitment to developing a new architecture. **Stop as soon as a required gate fails.** A well-documented negative result is a valid completion outcome.

### Fixed scope

| Item | Requirement |
|---|---|
| Initial dataset | **QaTa-COV19, existing validation data only** |
| Initial language-guided model | **DD-CMD, seed 1001 only** |
| Image-only reference | Frozen **nnU-Net, seed 1001**, as used in Phase 3 |
| Initial conditions | C0, C2, C4, C5, C6; focus on C4 location-only and C6 |
| Existing gate features | Phase 3 F1–F5, unchanged |
| Additional signal | One predefined spatial mismatch score |
| Allowed new training | Small logistic-regression fits only, if earlier checks pass |
| Not allowed | New segmentation training, new datasets, hyperparameter search, report rewriting, expert-label correction, or clinical assessment by the student |

The Phase 2/3 test results have already been examined. **Do not use the existing test split to design, tune, choose, or evaluate the new method in this phase.** This pilot uses development data to decide whether more research is justified. Even a successful pilot is **not** independent confirmation; any eventual paper-level new-method claim needs new independent evidence.

### Why this question is worth testing

Phase 2 found large sensitivity to changed or missing location information on QaTa. Phase 3 improved C4–C6 segmentation with output-only LR-Gate, but recovered only a limited share of the oracle headroom. A spatial signal might help, but **its usefulness has not been demonstrated**. In particular, IMG can itself be wrong, and a report-to-image location comparison may merely exploit the structure of the benchmark's text templates.

Do not claim that a spatial mismatch score verifies whether an expert report is clinically true. The original text and masks remain unchanged.

---

## Stage 0: Confirm that a spatial score can be computed

**Purpose:** Resolve feasibility without training any model.

1. Locate and reuse the Phase 2 structured location attributes, fixed condition/donor manifest, and Phase 3 QaTa validation predictions. Confirm that each row can be linked to its **exact scored** LANG and IMG masks and existing F1–F5 features.
2. Reuse Phase 3's patient/group identifiers and the existing `GATE_FIT` and `GATE_CAL` assignments. Do not regroup cases or inspect GATE_CAL performance to design the spatial rule.
3. Using **documented image metadata and the existing preprocessing code**, establish how image coordinates correspond to patient left/right. Do not infer this from an unverified visual assumption. If orientation is inconsistent or cannot be established reliably, **STOP**.
4. Define one deterministic, coarse **2 × 3 image-coordinate grid** in the common 224 × 224 image: split at the image midpoint horizontally and at one-third and two-thirds of image height vertically. Convert image left/right to *patient* left/right using only the verified orientation from Step 3. This is an *image-coordinate proxy*, not a clinically validated lung-zone segmentation. Do not use ground-truth masks to define regions.
5. Map explicitly supported report phrases into grid cells using a written lookup table. For a phrase naming several zones (e.g., `upper middle right lung`), use the **union** of the named cells; `all left lung` covers all three left-side cells. Fix the table before scoring and use it for every case. If orientation, laterality, location, or wording cannot be interpreted mechanically, mark the score **unavailable**, not incorrect. Do not manually reinterpret or correct expert annotations.
6. Compute the spatial score from the **frozen IMG binary mask**, not from the ground-truth mask:

   ```text
   spatial_agreement = IMG lesion pixels in report-allowed grid cells
                       / all IMG lesion pixels
   spatial_mismatch  = 1 - spatial_agreement
   ```

   If IMG predicts no lesion, or the location cannot be mapped, set the score to **unavailable**. These cases retain the original F1–F5 gate decision in later experiments. Do not silently fill them with zero or label the report incorrect.

**Stage 0 decision:** Define the denominator from the frozen manifest: QaTa targets eligible for **C0, C4 location-only, and C6**. A target is spatial-score eligible if IMG is nonempty, C0 is mappable, and **at least one donor** is mappable in each of C4 location-only and C6. Continue only if orientation/mapping is verified without medical judgment and **at least 80% of these target images and 80% of their recorded patient/groups** are eligible. Report the numerator, denominator, and condition/donor coverage; do not change the denominator after looking at outcomes. Otherwise **STOP**.

**Output:** `reports/PHASE4_FEASIBILITY.md`, with the coordinate conventions, deterministic mapping, coverage, unsupported cases, and a clear `GO` or `STOP` decision.

---

## Stage 1: Test whether the score contains a useful signal

**Purpose:** Reject an uninformative feature before fitting a gate.

Use only the **existing Phase 3 `GATE_FIT` groups**. For this stage, examine DD-CMD seed 1001 and the frozen C0, C4 and C6 report assignments. Do not look at the old test split.

1. On the **same Stage 0-eligible target images**, compare the C0 score with the **mean of available C4 location-only donor scores** for each target. Report image/group counts, the paired mean difference, and an image-level AUROC for the two benchmark conditions (one C0 and one donor-averaged C4 score per target). Repeat descriptively for C0 versus donor-averaged C6 on these same targets.
2. Confirm that the paired mean spatial-mismatch score is **lower for C0** than for both C4 location-only and C6. Do not interpret AUROC as clinical error-detection accuracy.
3. Show group-level summaries so an apparent signal is not dominated by one group. Donors and repeated images from one patient are not independent patients.

**Stage 1 decision:** Continue only if (a) C0 versus C4 location-only has AUROC **at least 0.65** on the fixed eligible targets and (b) the paired mean score difference has the expected direction for **both** C4 location-only and C6. These are **engineering screening criteria**, not significance tests. If any condition fails, **STOP**. Do not try alternate grid sizes, location mappings, mask thresholds, or new features to rescue the result.

**Output:** A short Stage 1 table in `reports/PHASE4_FEASIBILITY.md` and the frozen `results/phase4_spatial_scores_val.csv`.

---

## Stage 2: One controlled gate comparison

**Run this stage only after Stages 0 and 1 pass.** The goal is to find out whether the spatial signal improves **segmentation decisions**, not merely separates text-condition labels.

### A. One development/assessment split

Within the existing Phase 3 `GATE_FIT` groups, make a **single** deterministic 75%/25% group split, random seed `4001`:

- `PILOT_TRAIN`: fit the small classifiers.
- `PILOT_TUNE`: select operating thresholds using the fixed objective below.

Reserve the existing Phase 3 `GATE_CAL` groups for **one final pilot assessment** after code, models, and thresholds are frozen. Although this partition is separate from PILOT_TRAIN/TUNE, it was used in Phase 3 development and is **not a completely untouched validation cohort**. Report the result as exploratory.

Keep all conditions and donor variants of one image/patient in the same partition. Do not try alternative splits.

### B. Compare only the necessary strategies

Fit **both logistic-regression models on exactly the same spatial-score-available PILOT_TRAIN rows**; use identical fallback IMG, labels, row weights, software settings, and calibration rule. This isolates the effect of the new spatial feature. Assess all strategies on the same eligible target rows:

| Strategy | Inputs / meaning |
|---|---|
| **Baseline LR** | Original F1–F5 only, refitted for this pilot split |
| **Spatial-only rule** | Switch to IMG when `spatial_mismatch > threshold`; no classifier |
| **Evidence LR** | F1–F5 **plus one** spatial-mismatch score |
| **Shuffled-evidence control** | Same Evidence LR, but spatial evidence comes from a different target image, assigned by a fixed within-partition group shuffle (`seed=4002`) |

For the shuffled control, **permute only available spatial-score values** between different target groups **within the same condition and partition**, using the fixed seed. Preserve every row's spatial-score availability, its F1–F5, training label, and actual IMG mask. This breaks case-specific spatial matching while preserving condition-level score distributions. Never move target images across partitions or use a shuffled IMG prediction for fallback/scoring.

For logistic regression, keep the Phase 3 setup: weighted `StandardScaler`, L2 logistic regression (`C=1`, `lbfgs`, `max_iter=2000`), and **beneficial-fallback labels** formed on PILOT_TRAIN:

```python
gain = Dice(IMG, GT) - Dice(LANG, GT)
y = 1 if gain > 0.005 else 0
```

The ground-truth mask can create training labels and calculate development Dice, but **never enters features or the prediction-time switching rule**. Reuse the Phase 3 group/donor weighting principle; determine label-class weights from PILOT_TRAIN only, with the same weights for both LR models. Do not train a larger classifier or optimize hyperparameters. For rows with no spatial score, **use the fitted Baseline LR decision** for the Spatial-only, Evidence LR, and shuffled-control strategies. This preserves the same evaluation population and missing-score behavior; do not impute missing scores as zero.

### C. Fixed threshold objective

Select a single threshold per strategy on **PILOT_TUNE only**. For C4–C6, maximize the equal-condition mean of group-weighted `(gate Dice − LANG Dice)` on the **common eligible target set**, subject to:

```text
C0: gate − LANG >= −0.005 Dice
C2: gate − LANG >= −0.005 Dice
C4/C5/C6: each gate − LANG >= −0.005 Dice
```

The score threshold must be common across conditions. Include **never fall back** as a candidate. Prefer fewer switches when tied. If no feasible positive gain exists, choose never fall back. Do not inspect `GATE_CAL` to set or revise thresholds.

### D. One-time pilot assessment

Freeze both logistic models, the spatial-only rule, shuffle control, all thresholds, and the code. Evaluate **once** on QaTa `GATE_CAL`, DD-CMD seed 1001, with unchanged C0/C2/C4/C5/C6 conditions and matched eligible targets.

Report only the comparisons needed to decide whether to proceed:

1. **Primary:** Evidence LR minus Baseline LR, in **group-weighted Dice points**, macro-averaged equally over C4, C5, C6 on their common eligible groups.
2. **Clean cost:** Evidence LR minus LANG on C0 and C2, each separately.
3. **Negative control:** Evidence LR minus Shuffled-evidence LR on C4–C6.
4. **Interpretation:** Spatial-only versus Evidence LR; switch rates, harmful switches, and the share of available oracle improvement recovered. Include IMG and LANG as references.

Use the same target groups, donors and weights in all paired comparisons. Report per-condition paired differences and exploratory group-bootstrap 95% intervals; **do not claim confirmatory p-values** from this pilot.

### Stage 2 decision

**GO only if all conditions hold on the single GATE_CAL assessment:**

- Evidence LR improves the **C4–C6 macro-averaged Dice by at least 2.0 points** over the refitted Baseline LR.
- Both C0 and C2 have a gate-minus-LANG mean Dice of at least **−0.5 points**.
- Evidence LR exceeds the shuffled-evidence control by at least **0.5 Dice points** on the same C4–C6 macro-average.
- There is no C4, C5 or C6 condition with an Evidence LR loss greater than **0.5 points** relative to Baseline LR.

These thresholds are fixed project **go/no-go criteria**, not clinically meaningful safety margins or statistical proof. If any fails, **STOP**. Do not retune the gate, change the score, inspect old test cases, or introduce a new architecture. Report the result and end Phase 4.

**Outputs:** `results/phase4_pilot_comparison.csv`, `results/phase4_pilot_decisions.csv`, and an updated `reports/PHASE4_FEASIBILITY.md` containing a `GO` or `STOP` decision.

---

## Stage 3: Expand only after supervisor approval

**Do not start this stage automatically.** Submit Stage 2 results for supervisor review first.

If Stage 2 passes and the supervisor approves:

1. Apply the **same fixed spatial rule, feature definitions, training procedure and decision criteria** to DD-CMD seeds 1002 and 1003. Use their seed-matched frozen IMG predictions. No new model family or grid adjustment.
2. Evaluate the same pilot conditions on the same group partitions. Check whether the C4–C6 macro-average advantage over the matched five-feature baseline is positive for **at least two of the three seeds** and whether C0/C2 costs stay within the 0.5-point development target.
3. Report the seed-level effects and failure cases. The method must not be presented as robust if the benefit is concentrated in one seed.

**Decision:** If this repeatability check fails, **STOP**. If it passes, report `GO FOR INDEPENDENT VALIDATION` and await a new research plan. Do **not** automatically extend to LViT-T, RecLMIS, MosMedData+, a region-level model, or a new dataset in this phase. Those require a separate decision and, for a new publication claim, independent validation evidence.

---

## Required final report

Create one concise standalone file: **`reports/PHASE4_FEASIBILITY.md`**.

It must state:

1. Which stage was reached, and the exact `GO`/`STOP` decision.
2. The fixed coordinate/text mapping and its coverage, or why it was infeasible.
3. The minimal numerical evidence for each completed gate, including matched cases/groups.
4. For a Stage 2/3 result: Baseline LR, spatial-only, Evidence LR, shuffled control, C0/C2 costs, and key C4–C6 gains.
5. Limitations: approximate image-coordinate regions, uncertain anatomical orientation when unsupported, reliance on IMG predictions, benchmark-generated mismatches, and prior exposure to Phase 2/3 test outcomes.
6. Whether there is **enough evidence to justify a separate independent-validation stage**. A positive feasibility result is not a paper-ready confirmation.

**Keep the report short. Do not create separate long reports for stopped stages.** Preserve the existing Phase 0–3 results and checkpoints. The objective of this phase is to make one sound research decision with the fewest necessary experiments.
