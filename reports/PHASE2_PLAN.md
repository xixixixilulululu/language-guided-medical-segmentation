# Phase 2: Counterfactual Text Reliability Benchmark

## Goal

Measure how strongly frozen language-guided medical image segmentation models depend on their accompanying benchmark-provided text, and distinguish missing information, benign wording changes, and report mismatch.

Keep the original scope: **LViT-T, RecLMIS, DD-CMD; QaTa-COV19 and MosMedData+; seeds 1001, 1002, 1003.** No segmentation-model retraining, additional architecture, new dataset split, or LLM-generated text is required.

This revision retains Tasks 2.1–2.7. It updates the checkpoint choice following the Phase 1 LViT sensitivity check and makes the text conditions, eligible samples, aggregation, and validation/test order explicit. These are prospective Phase 2 instructions, not claims that the checks have already passed.

## Prerequisite: Fix the Phase 2 baseline

| Dataset | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| QaTa-COV19 | Original Phase 1 checkpoints | Original Phase 1 checkpoints | Original Phase 1 checkpoints |
| MosMedData+ | **Batch-size-24 sensitivity checkpoints** | Original Phase 1 checkpoints | Original Phase 1 checkpoints |

Use all three seeds for each entry. The LViT batch size above is the **training** batch size; do not automatically change inference batch size or other inference settings.

The MosMed batch-24 LViT runs have reported test Dice 75.03 / 72.82 / 74.19, mean 74.01. They were selected on validation Dice in a post-hoc Phase 1 sensitivity study. Keep that provenance explicit. Do not overwrite the original batch-2 Phase 1 results or use their 61.26 mean as the Phase 2 C0 baseline.

Create `reports/PHASE2_PROTOCOL.md`. Record checkpoint paths and hashes, configuration identifiers, code commit, split-manifest hashes, evaluator, text encoder/tokenizer, inference precision, inference batch size, TTA, threshold, and empty-mask policy. Record the patient/volume grouping rule and any unavailable identifiers. Preserve the common 224 × 224 targets and `probability > 0.5` rule.

### Validation-first order

1. Develop the parser and text transformations using **train/validation text only**.
2. Verify C0 and the actual text-input path on validation, then run the validation benchmark.
3. Decide whether Phase 3 is justified using validation results only.
4. Freeze Phase 2 rules and analysis. If Phase 3 proceeds, also freeze its mechanism, calibration, and selection rules before evaluating any Phase 2/3 counterfactual test predictions.
5. If Phase 3 is explicitly cancelled, freeze that decision and complete the final Phase 2 test evaluation.

Phase 1 clean test results have already been inspected. Do not call this a never-observed test set. The protection here is against using **new counterfactual test outcomes** to design the benchmark or mitigation.

After the final freeze, the program may read test texts to generate same-split donors automatically under the frozen rules. Do not manually choose test donors or change rules after inspecting their segmentation outcomes. Record protocol amendments and software failures; do not silently rerun a result-dependent protocol.

---

# Task 2.1: Organize the Clinical Text

Extract the original three attribute families from text:

1. infection scope/type;
2. infected-area count;
3. location description.

Define exactly what `scope` means for each dataset before making distance labels. Do not mix laterality, infection type, and severity opportunistically in one field. Normalize equivalent expressions and location sets consistently. Preserve unknown or unparseable values as unknown, not as zero or a new mismatch category.

Do not infer missing attributes from target masks, model predictions, or an LLM. These experiments use the provided annotations; they do not establish that each annotation is an independently verified clinical report.

Create `results/phase2_text_attributes.csv` with:

```text
dataset, split, sample_id, patient_id, volume_id,
raw_text, scope, count, location, parse_ok, parse_note
```

Record which grouping IDs are unavailable rather than inventing patient identities from unverified filenames. For MosMed, use the documented source/volume mapping; for QaTa, verify whether multiple images belong to one patient.

In `reports/PHASE2_PROTOCOL.md`, report attribute frequencies, unique-text counts, parsing coverage, and manual review of examples covering each observed category and transformation rule. This describes what the text can support; do not assume every dataset contains all three kinds of attribute variation.

---

# Task 2.2: Create Seven Text-Condition Families

There are seven families but **nine evaluated conditions**, because C3 has three subconditions:

```text
C0, C1, C2, C3a, C3b, C3c, C4, C5, C6
```

## C0: Original paired text

Use the original paired report, without rewriting. C0 is the benchmark reference, not a claim of independently verified clinical truth.

## C1: Null / no case-specific information

Use one fixed missing-information input for each method, documented before the reliability runs:

- Prefer the normal tokenizer/encoder processing of an empty string if it produces a valid input.
- If that fails structurally, use one fixed neutral statement, such as `No case-specific clinical information is available.`, through the ordinary text pathway.

Choose this rule based on input validity, not segmentation performance. Do not use `No infection`, `Normal lungs`, another patient's findings, or an arbitrary all-zero feature tensor as interchangeable null inputs. Do not disable the language branch or alter the architecture.

Record exact strings, tokens, padding/masks, and any model-specific fallback. C1 measures a frozen text-trained model under missing-information input. It is **not** a separately trained image-only baseline and may involve input-distribution shift.

## C2: Equivalent wording

Use a deterministic, fixed-template rewrite that preserves **all substantive information**, not merely the three extracted attributes. Preserve counts, laterality, location, qualifiers, and negation where present. Do not use an LLM.

Review the templates on train/validation examples. If a report cannot be rewritten faithfully by the frozen rules, mark C2 unavailable for that case; do not call a shortened summary equivalent.

## C3: Incomplete wording

Keep separate conditions:

- C3a: remove explicit scope/type wording;
- C3b: remove explicit count wording;
- C3c: remove explicit location wording.

Remove information without inventing or contradicting the remaining findings. These are **explicit-clause deletions**, not guaranteed removal of every semantic cue. For example, left- and right-sided locations can still imply bilateral involvement after a scope clause is removed. Record such residual cues.

If the attribute is not stated, or cannot be removed without an unsupported rewrite, mark that subcondition unavailable. Do not replace unavailable deletions with a different operation.

## C4–C6: Real-report attribute mismatch

Retain the real-donor-report design:

| Condition | Number of differing, known attributes |
|---|---:|
| C4 | 1 |
| C5 | 2 |
| C6 | 3 |

Distance is the number of unequal canonical values among `scope`, `count`, and `location`. Require all three attributes to be known for both target and donor. Unknown values do not count as disagreements.

Use the terms **one-, two-, and three-attribute mismatch** in conclusions. The former mild/moderate/severe names are shorthand only: the number of differing fields is not a validated scale of clinical severity, and these fields may be semantically dependent. Replacing a whole report can also change unmeasured content.

### Donor-selection rule

Use the same dataset and split as the target, but not the target case itself. When verified IDs exist, exclude donors from the target's patient or CT volume. Do not cross splits to increase coverage.

Select up to three distinct donor texts per target and condition, without replacement. Deduplicate identical text after whitespace normalization. Select deterministically by a fixed hash of dataset, split, target ID, condition, donor ID, and a recorded salt. Choose donors before examining their model outputs. Never select the worst-performing donor.

Use the **same donor manifest for every method and every model seed**. Report donor reuse and available donor counts. If no eligible donor exists, mark the condition unavailable. Do not fabricate a report, loosen the distance definition, or force C6 to exist.

### Output and effective-input audit

Create `results/phase2_text_conditions.csv`:

```text
dataset, split, sample_id, condition, variant_id,
original_text, counterfactual_text, donor_sample_id,
donor_group_id, attribute_distance, changed_attributes,
eligible, exclusion_reason, text_hash
```

For each method, inspect the input **after its actual tokenizer, truncation, padding, encoder, and caching**. Record whether the intended change survives preprocessing. Cache keys must distinguish the actual text and encoder settings, not just the image ID.

Keep the frozen text pipeline and context length. A valid raw-text change erased by truncation remains an observed limitation, not a reason to secretly extend context or delete that case from the primary result. Report such no-effective-change cases separately. Fix actual pairing or caching bugs before proceeding.

---

# Task 2.3: Run the Reliability Benchmark

## First: validate the final checkpoints and text path

Use final checkpoints, not the earlier small-subset models. On a fixed validation subset, compare original text, C1, and clearly different eligible donor text. Check:

- image/mask identity and text pairing;
- changed raw text and actual model-input features;
- probability-map change, mask change, and per-image Dice change;
- a repeated same-image/same-text inference as a numerical-noise reference.

Then reproduce full-validation C0 against the archived validation evaluation, with the original inference settings. Fix a numerical-tolerance rule from these repeatability checks before counterfactual test evaluation. Explain discrepancies; do not retrain or change thresholds to make C0 match.

A text substitution does **not** need to reduce Dice to pass. If the inputs truly change but predictions do not, retain and report that behavior. Distinguish valid model insensitivity from a broken input path.

## Full validation, then locked test evaluation

Run all nine conditions for all three methods and all three seeds on their eligible cases. Keep images, preprocessing, model weights, inference mode, TTA, and the evaluator fixed. Use `eval()` and do not update model state or adapt at inference.

At the final test stage, rerun C0 under the same Phase 2 pipeline and reconcile it with the corresponding archived Phase 1 checkpoints, using the tolerance fixed on validation. Test C0 is a reproducibility check, not a new model-selection step.

Save the exact scored binary masks and per-case Dice, mIoU, HD95, and ASSD. Save probabilities as float32 when probabilities are retained; do not reconstruct scored masks from lossy float16 archives. Retain validation probabilities needed by a potential Phase 3, without launching new training.

Every per-case result must identify dataset, split, method, checkpoint/configuration, seed, sample, condition, and donor/variant. Empty predictions remain scored under the frozen policy. An invalid forward pass is an error to investigate, not permission to drop a hard case.

## Aggregate donors and seeds correctly

For each image, method, seed, and condition, average **the metrics** across available donors first. Then average those per-image values across the three seeds for paired analysis. Do not average probability maps into a new ensemble, treat donors/seeds as independent patients, or give images with more donors more weight.

For each comparison, use identical eligible target cases on both sides. In particular, compare C0 and C1 with C6 **on C6-eligible cases**, not their full-dataset averages.

## Quantities

Use Dice in [0,1] in machine-readable per-case files and explicitly labelled percentage points in reports. Positive values below mean benefit of reference text or harm from the intervention:

```text
Paired-text advantage over null = D(C0) - D(C1)
Equivalent-wording signed change = D(C0) - D(C2)
Incomplete-text harm = D(C0) - D(C3a/C3b/C3c)
Mismatch harm = D(C0) - D(C4/C5/C6)
Wrong-vs-null penalty = D(C1) - D(C4/C5/C6)
```

Report absolute per-image C0–C2 changes and mask-change frequency too: opposite signed changes can cancel in the mean. None of these quantities proves the causal benefit of adding language during training.

---

# Task 2.4: Main Result Table

Create `results/phase2_main_results.csv`, with separate `split` values for validation and test. Do not merge their estimates.

| Dataset | Method | C0 | C1 | C2 | C3a | C3b | C3c | C4 | C5 | C6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QaTa-COV19 | LViT-T | | | | | | | | | |
| QaTa-COV19 | RecLMIS | | | | | | | | | |
| QaTa-COV19 | DD-CMD | | | | | | | | | |
| MosMedData+ | LViT-T (batch-24 training) | | | | | | | | | |
| MosMedData+ | RecLMIS | | | | | | | | | |
| MosMedData+ | DD-CMD | | | | | | | | | |

Each cell reports mean Dice ± sample SD across the three per-seed image means after donor averaging. Show eligible image count and independent-unit count beside the table or in an accompanying coverage table. Use NA rather than zero for unavailable conditions.

The main table is image-weighted for continuity with Phase 1. If conditions have different eligible cases, do not interpret its raw column differences as paired effects. The paired table must include the two means on the same eligible cases.

Report mIoU, HD95, ASSD, empty-prediction rate, and all three per-seed results in supplementary tables. Do not hide C3a–C3c behind one unspecified C3 average.

---

# Task 2.5: Statistical Analysis

Freeze this Phase 2 analysis prospectively. It does not alter Phase 1 statistics.

## Eleven planned comparisons per method and dataset

Use these signed differences:

```text
C0-C1, C0-C2, C0-C3a, C0-C3b, C0-C3c,
C0-C4, C0-C5, C0-C6,
C1-C4, C1-C5, C1-C6
```

Within each method × dataset, treat these as one Holm family of 11 planned comparisons; two-sided alpha = 0.05. This is not a single family controlling every claim across all models and datasets. Do not add comparisons after viewing test results.

## Independent unit and estimand

For each comparison:

1. Average donor metrics within image and seed, then seeds within image, as above.
2. Compute the paired image-level difference on the common eligible image set.
3. Average differences within the independent unit: **patient** when verified, otherwise **CT volume** for MosMed. QaTa image-level inference is acceptable only if one image per patient is verified; otherwise use patient groups. If adequate grouping cannot be established, flag inference as exploratory rather than assuming independence.
4. Use the mean of these unit-level differences, giving each unit equal weight, for the inferential effect estimate.
5. Apply a paired t-test to the corresponding unit means, equivalently a one-sample t-test of unit differences against zero. Report its unadjusted 95% t confidence interval, raw p, and Holm p.

Label this `unit_weighted_delta`. It may differ from the image-weighted difference in the main table. Do not attach a unit-weighted CI or p-value to an image-weighted estimate.

Report the number of units and their image counts, plus per-unit effects. When only a few volumes are available, describe the estimates as exploratory regardless of an apparently small p-value; grouping does not create additional independent evidence. For insufficient or degenerate data, report the effect and NA for undefined inference. Preserve the 11-comparison family, using p=1 only internally for unavailable tests in Holm adjustment; do not display a fabricated raw p.

For supplementary mIoU, HD95, and ASSD, paired unit-weighted differences and 95% paired unit-bootstrap CIs suffice: 10,000 resamples, `default_rng(0)`, with identical sampled units on both sides. No additional significance tests are required. Uncertainty is conditional on the three fixed trained seeds and frozen donor assignments.

MosMed grouping addresses correlated slices in the analysis. It does **not** repair the known train/validation/test volume overlap of the original benchmark split. Keep that limitation explicit; no new split or training is required in Phase 2.

Create `results/phase2_pairwise_results.csv` with split, method, comparison, eligible image count, unit count/type, matched condition means, image-weighted delta, unit-weighted delta, CI, raw p, Holm p, and inference caveats.

---

# Task 2.6: Mismatch Extent

Create `results/phase2_severity_results.csv` for compatibility with the original plan, but describe its variable as **attribute-distance**, not validated clinical severity.

For each dataset, define the intersection of target cases eligible for C4, C5, and C6. Compare all three conditions on **that same set**, using identical frozen donor rules across methods. Report C0 on the same set and the paired harms C0-C4, C0-C5, and C0-C6.

Report intersection coverage. Do not infer an increasing-harm trend by comparing different subsets. If the intersection is empty or small, state the limitation and present condition-specific paired effects without a strong trend claim.

This task is descriptive with paired intervals; no extra trend hypothesis test is required. Do not force monotonicity. Break down changed-attribute combinations descriptively where coverage allows, because changing location alone need not be weaker than changing two other attributes.

---

# Task 2.7: Compare Model Susceptibility

On the same C6-eligible cases and identical donors, compare the per-image harm:

```text
H_model = D_model(C0) - D_model(C6)
```

For each of the three model pairs, calculate paired differences in harm on common cases and report unit-weighted means and 95% intervals using the Task 2.5 grouping. Treat these as secondary descriptive comparisons; no additional significance-testing family is required.

Always present susceptibility alongside clean C0 performance, C0-C1, and C1-C6 on the same cases. A model with little text benefit and little text response is not automatically a useful robust model. Different model behavior also does not isolate the causal effect of a particular fusion module.

If C6 is unavailable, report that limitation and the existing C4/C5 results, without relabelling them C6 or selecting whichever condition produces the strongest ranking.

Include the interpretation in `reports/PHASE2_RESULTS.md`.

---

# Required Figures

1. C0/C1/C2/C3a–C3c comparisons on each dataset, with eligible counts.
2. C4–C6 attribute-distance curves on their common eligible set, with C0 reference.
3. Wrong-vs-null paired penalties, **C1 minus C4/C5/C6**, so positive values consistently mean harm.
4. Per-unit or per-case effect distributions, showing heterogeneity rather than mean scores alone.

Do not connect all nine conditions as if they formed one ordered severity scale. Representative original/null/mismatched predictions are optional; use a documented selection rule, not only the most dramatic failures.

---

# Milestones

## Milestone 2.1: Baseline and Text Conditions Ready

Required: `PHASE2_PROTOCOL.md`, attribute/condition CSVs, checkpoint records, validation C0 reconciliation, effective-text-path check, and eligibility/grouping summary.

## Milestone 2.2: Validation Reliability Experiments Complete

All three methods and seeds evaluated on available validation conditions. Complete the validation tables and decide whether Phase 3 is justified. Do not run counterfactual test inference yet.

## Milestone 2.3: Protocol Frozen and Final Analysis Complete

Freeze Phase 2 and either the Phase 3 protocol or its cancellation. Then complete the locked test evaluation and produce:

```text
results/phase2_main_results.csv
results/phase2_pairwise_results.csv
results/phase2_severity_results.csv
reports/PHASE2_RESULTS.md
```

# Decision Before Phase 3: Validation Only

Retain the original planning signals, with explicit units:

```text
C0 - C5/C6 >= 0.02 on [0,1] Dice = 2.0 percentage points
or
C1 - C5/C6 >= 0.01 on [0,1] Dice = 1.0 percentage point
```

These are project-planning thresholds, not clinically validated importance thresholds. Use paired eligible cases, inspect all seeds and unit-level effects, and require a repeated signal rather than one extreme donor, seed, or volume. At least two of the three seeds should show harm in the same direction before describing a signal as repeatable; this alone is not a significance test.

C0-Cwrong can reflect loss of useful guidance. A positive C1-Cwrong more directly shows harm beyond missing information. A mitigation proposal must also preserve useful clean-text performance; do not optimize only corrupted-text scores.

If text has little usable variation or eligible donor coverage is inadequate, conclude that this benchmark cannot resolve that reliability question. If verified text changes produce little harm, retain the negative result. Neither case justifies manufacturing stronger corruptions or forcing Phase 3.

# Phase 2 Completion Checklist

- [ ] Correct checkpoints fixed, including MosMed LViT batch-24 models.
- [ ] Validation-first and final-test release decisions recorded.
- [ ] C0 and actual text path verified on final checkpoints.
- [ ] C1 defined; C2 semantic preservation and C3 deletions reviewed.
- [ ] Same frozen donor assignments used across methods and seeds.
- [ ] Nine conditions represented separately; unavailable cases documented.
- [ ] Donors/seeds aggregated within images; paired subsets aligned.
- [ ] Group-aware inference and image-versus-unit weighting labelled.
- [ ] C4–C6 extent comparison uses the common eligible set.
- [ ] Scored masks, per-case metrics, empty rates, and needed float32 probabilities retained.
- [ ] Phase 3 go/no-go decided from validation, before counterfactual test results.
- [ ] Final report distinguishes observed findings, limitations, and unsupported claims.

## Basis for this revision

Project sources: `PHASE_2_TEXT_RELIABILITY_BENCHMARK_FINAL(1).md`; `PHASE1_SENSITIVITY_LVIT.md`, sections 1–3; `PHASE1_RESULTS.md`, methods and limitations; `PHASE1_SUMMARY.md`, section 7; and the documented Phase 0 MosMed split audit. These source reports motivate the changes but are not independent verification of the underlying code or data.

Implementation references for the proposed statistical operations: SciPy documentation for `scipy.stats.ttest_rel` (paired mean difference and confidence interval) and `scipy.stats.bootstrap` (`paired=True` preserves paired resampling indices). The grouping, estimand, and multiplicity choices above are Phase 2 design decisions, not requirements imposed by SciPy.
