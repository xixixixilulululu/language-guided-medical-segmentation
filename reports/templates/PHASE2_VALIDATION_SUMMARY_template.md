# Phase 2 — Validation summary for the supervisor

> **All results in this summary come from the validation split.**
> - The test split has not been evaluated in Phase 2. The Phase 2 lock is closed, and no test text or test
>   result was read.
> - Following [PHASE2_PLAN.md](PHASE2_PLAN.md) (validation-first order, Milestones 2.2 and 2.3), the test
>   split is evaluated only after the Phase 3 plan (or its cancellation) and the Phase 2 rules and analysis are
>   frozen.
> - The Phase 2 protocol ([PHASE2_PROTOCOL.md](PHASE2_PROTOCOL.md)) is still a **draft and not frozen**.
>
> Tables and quoted numbers are generated from the results files by `src/phase2/make_phase2_tables.py`.
> Dice is given in points (×100); a Δ is the difference A − B in points. MosMedData+ has 11 validation units
> (CT volumes + pooled `bjorke`), so **every MosMedData+ interval and p-value is exploratory**. Items decided
> after results had been seen are marked **post-hoc**.

## 1. Summary

1. **The benchmark ran as specified.** All three methods (LViT-T, RecLMIS, DD-CMD), both datasets, three
   seeds and nine text conditions ran on the validation split: 18 frozen checkpoints and 252 condition
   passes.
   - C0 reproduced the archived Phase 1 validation evaluation (LViT-T and RecLMIS identical; DD-CMD within 1–4
     pixels; section 10).
   - In the effective-input audit, {{N:audit_unchanged}} of {{N:audit_rows}} condition rows had a changed
     text but an unchanged model input.
2. **QaTa-COV19: the empty text (C1) and a three-attribute mismatch (C6) give lower Dice than the original
   text, for all three methods.**
   - Unit-weighted C0−C1 is {{N:q_C0-C1}} points across the methods, and C0−C6 is {{N:q_C0-C6}}.
   - {{N:q_sig}} of 33 comparisons have Holm p < 0.05.
3. **QaTa-COV19: a three-attribute mismatch gives lower Dice than an empty text.** C1−C6 is {{N:q_C1-C6}}
   points. For one- and two-attribute mismatches the sign differs: C1−C4 is {{N:q_C1-C4}} and C1−C5 is
   {{N:q_C1-C5}} points.
4. **MosMedData+ (all exploratory, 11 units).**
   - The mismatched text has a small effect: C0−C4 and C0−C5 are {{N:m_C0-C4C5}} points, C0−C6 is
     {{N:m_C0-C6}} points (unit-weighted).
   - The empty text has a large effect for RecLMIS: C0−C1 {{N:m_c0c1_RecLMIS}}.
   - For the other two methods: DD-CMD {{N:m_c0c1_DD-CMD}}, and LViT-T {{N:m_c0c1_LViT-T}} (CI
     {{N:m_c0c1_LViT-T_ci0}}).
   - C1−C6 is negative for {{N:m_c1c6_neg}} of 3 methods: the empty text gives lower Dice than the mismatched
     report.
5. **Equivalent wording (C2) changes almost every mask.**
   - The scored masks change in {{N:c2_q_changed}}% of the QaTa images.
   - The mean Dice change C0−C2 (unit-weighted [95% t-CI]) is {{N:c2_big_ci}} for {{N:c2_big}}
     ({{N:c2_big_sig}}).
   - For the other five method × dataset cells it is {{N:c2_rest}} points unit-weighted
     ({{N:c2_rest_img}} image-weighted).
   - The mean |D(C0) − D(C2)| is {{N:c2_abs_range}} points.
6. **Incomplete text (C3), QaTa-COV19.**
   - Deleting the location (C0−C3c) lowers Dice for all three methods by {{N:q_C0-C3c}} points.
   - Deleting the scope or the count lowers LViT-T by {{N:q_lvit_c3a}} / {{N:q_lvit_c3b}} points, and RecLMIS
     and DD-CMD by {{N:q_rd_c3ab}}.
7. **The Phase 3 rule of the protocol (section 5.1, validation only) gives: {{N:outcome}}.**
   {{N:n_repeated}} of 24 method × dataset × comparison cells have a repeated signal: {{N:repeated_list}}.
   Whether Phase 3 is started is for the supervisor and the project lead to decide.
8. **None of these comparisons shows that the models "use" the text in a clinical sense.** They measure how
   the frozen models' Dice responds to the benchmark texts given to them.

## 2. Setup

**Checkpoints.** The Phase 1 checkpoints (seeds 1001 / 1002 / 1003, selected on validation Dice), except one:
- **MosMedData+ LViT-T uses the batch-24 runs** of the post-hoc Phase 1 sensitivity study
  ([PHASE1_SENSITIVITY_LVIT.md](PHASE1_SENSITIVITY_LVIT.md)): training batch 24 instead of 2, everything else
  unchanged.
- sha256 values: `results/phase2_checkpoint_sha256.tsv`.
- Inference settings are unchanged from Phase 1 (eval mode, batch 32, no TTA, probability > 0.5, common
  evaluator).
- Validation C0 Dice: QaTa {{N:c0_q}}; MosMed {{N:c0_m}}.

**Units.**
- QaTa: patient = BIDS subject `sub-S…`. `covid_*` images, which have no patient id, are their own unit.
- MosMed: CT volume = file name without the slice number. `bjorke` slices form one pooled unit.
- The unit-weighted Δ is the mean over units of the per-unit mean image difference, with a 95% t-CI over
  units.

**Aggregation.** Metrics are averaged over the donors (up to 3) within image and seed, then over the 3 seeds.

**The nine conditions.** The example is the QaTa validation target `{{N:ex_id}}`, chosen by a fixed rule: the
smallest hash among the targets with all nine conditions.

| Condition | Definition | Example (validation) |
|---|---|---|
| C0 | original paired report | {{N:ex_C0}} |
| C1 | empty string, through each method's normal text pathway | {{N:ex_C1}} |
| C2 | the original with `pulmonary` → `lung` and `infected area(s)` → `infected region(s)` only (P2-1) | {{N:ex_C2}} |
| C3a | scope word deleted | {{N:ex_C3a}} |
| C3b | count clause deleted | {{N:ex_C3b}} |
| C3c | location clause deleted | {{N:ex_C3c}} |
| C4 | real report of another case, exactly 1 of (scope, count, location) different | {{N:ex_C4}} |
| C5 | real report, exactly 2 different | {{N:ex_C5}} |
| C6 | real report, all 3 different | {{N:ex_C6}} |

**Donors.**
- C4–C6 donors come from the same split, never from the same patient or volume.
- Up to 3 distinct donor texts per target, chosen by a fixed hash before any model output existed.
- The same donors are used for all methods and seeds (`results/phase2_donor_manifest_val.csv`).

**Coverage on validation** (images and units per condition; identical for the three methods):

{{T:coverage}}

**Effective-input audit.** Text changed but model input unchanged, counted over all validation condition
rows:

{{T:audit}}

## 3. Main table (Task 2.4)

Dice, mean ± sample SD over the 3 per-seed image means (donor-averaged); image-weighted. Columns are on
different image sets where coverage differs (section 2), so column differences are not paired effects.

{{T:main}}

Per-seed values, mIoU, HD95, ASSD and empty-prediction rates are in Appendix A.

## 4. Paired comparisons (Task 2.5)

**Method.** 11 comparisons per method × dataset, each on the common eligible images of its two conditions.
- Unit-weighted Δ with 95% t-CI and the one-sample t-test over units.
- Holm correction across the 11 comparisons of each method × dataset, α = 0.05.
- The image-weighted Δ is shown for reference only, without inference.
- Positive = lower Dice under the second condition.

**QaTa-COV19** ({{N:q_sig}} of 33 comparisons with Holm p < 0.05):

{{T:pairwise_QaTa-COV19}}

**MosMedData+ — exploratory, 11 units** ({{N:m_sig}} of 33 with Holm p < 0.05; the p-values rest on 11 units):

{{T:pairwise_MosMedData+}}

**Null, equivalent and incomplete text** (C0 − C1, C2, C3a, C3b, C3c):

![Figure 1](figures/phase2_val_fig1_c0_c1_c3.png)

**Wrong text versus no text** (C1 − C4, C5, C6; positive = the mismatched report gives lower Dice than the
empty text):

![Figure 3](figures/phase2_val_fig3_wrong_vs_null.png)

**Per-unit effects.** The distribution over units of C0−C1 and C0−C6. For MosMed all 11 units are shown.

![Figure 4](figures/phase2_val_fig4_units.png)

## 5. Equivalent wording (C2)

C2 differs from C0 only in two words (section 2). Values are per seed, then mean ± SD over the 3 seeds:

{{T:c2}}

- **Masks change, Dice hardly.** The scored mask changes in {{N:c2_q_changed}}% of the QaTa images and
  {{N:c2_m_changed}}% of the MosMed images. The mean absolute Dice change is {{N:c2_abs_range}} points.
- **Empty ↔ non-empty flips.** {{N:c2_flips}} in total over all methods, datasets and seeds.
- **DD-CMD.** Its C0 comes from a different process (job 369) than its C2 (job 370). Cross-process noise
  alone changes 1–4 pixels (protocol 4.8). The share of images changed by more than 4 pixels is
  {{N:c2_dd_gt4_q}}% (QaTa) and {{N:c2_dd_gt4_m}}% (MosMed). The 4-pixel cut-off was chosen **post-hoc**, from
  the reconciliation result.
- **Review.** The manual review of the C2 rewrites is recorded in protocol section 4.3.1.

## 6. Mismatch extent (Task 2.6)

**S** = the target images with C4, C5 and C6 donors:
- QaTa: {{N:S_q}};
- MosMed: {{N:S_m}}.

S is identical for the three methods ({{N:S_same}}). Mean Dice on S (donor and seed averaged), and the paired
harms C0 − Cx (image-weighted / unit-weighted [95% t-CI]). Descriptive, no test:

{{T:severity}}

![Figure 2](figures/phase2_val_fig2_extent.png)

- **The attribute distance is not a validated severity scale.** On QaTa the mean Dice under C5 is higher than
  under C4 for {{N:q_c5_gt_c4}} of 3 methods.
- **Which attributes differ, QaTa.**
  - C4 is almost entirely location-only: {{N:q_c4_location_pairs}} donor pairs, harm {{N:q_c4_location_harm}}
    points.
  - C5 is mostly count + location: {{N:q_c5_cl_pairs}} pairs, harm {{N:q_c5_cl_harm}} points.
  - So the higher mean Dice under C5 than under C4 goes with *which* attributes differ, not only with *how
    many*.
- **What `changed_attributes` records.** Only whether an attribute differs, not how far apart two locations
  are (for example, whether they share a lung zone). This is a description, not an inference.

**By changed attributes.** On S, per (target, donor) pair, seed-averaged. The mean harm averages the donors of
a target within its group first. Descriptive:

{{T:attribute}}

## 7. Model susceptibility (Task 2.7)

On the C6-eligible images. All values in points, donor and seed averaged; descriptive.

{{T:sus_methods}}

Paired differences of H between methods (same images, same donors):

{{T:sus_pairs}}

On QaTa:
- LViT-T − RecLMIS: {{N:sus_q_lr}};
- LViT-T − DD-CMD: {{N:sus_q_ld}};
- RecLMIS − DD-CMD: {{N:sus_q_rd}}.

These are different architectures and training recipes; a difference in H does not isolate the effect of any
one module.

## 8. Real cases

**Selection rule** (written before the figures were drawn; **post-hoc** with respect to the aggregate
validation results, which were known by then):
- Take the C6-eligible validation images.
- Sort them by the C0−C6 harm, averaged over the 3 methods, 3 seeds and the donors.
- Take the image at position round(q × (n − 1)) for q = 10/30/50/70/90%; ties are broken by sample id.
- Each row shows the image with the ground truth (yellow dashed contour) and, for each method, C0 | C1 | C6
  with donor d1, seed 1001. The masks are the scored masks of the benchmark; nothing was re-inferred.
- Below each row: the original text and the d1 donor's text.

{{T:cases}}

![QaTa-COV19 cases](figures/phase2_val_cases_QaTa-COV19.png)

QaTa-COV19 (seed 1001, donor d1; what the figure shows):
- **q30 (`{{N:q30_id}}`).**
  - The original text names "{{N:q30_text}}", but the ground truth lies in the lower lung fields.
  - In this row RecLMIS and DD-CMD reach a higher Dice under C6 than under C0: RecLMIS {{N:q30_recl_c6}} vs.
    {{N:q30_recl_c0}}; DD-CMD {{N:q30_dd_c6}} vs. {{N:q30_dd_c0}}.
  - The d1 donor text is "{{N:q30_donor_text}}".
- **q90 (`{{N:q90_id}}`).**
  - LViT-T predicts an empty mask under C1 (Dice {{N:q90_lvit_c1}}; C0 {{N:q90_lvit_c0}}).
  - Under C6 (donor: "{{N:q90_donor_text}}") it predicts an additional region on the other side of the image
    (Dice {{N:q90_lvit_c6}}).

![MosMedData+ cases](figures/phase2_val_cases_MosMedDataplus.png)

MosMedData+:
- {{N:m_case_vol_n}} of the selected cases come from the same CT volume (`{{N:m_case_vol}}`).
- The harm of the five rows ranges from {{N:m_case_harm}} points, so this figure reflects only a few volumes
  (exploratory).

## 9. Phase 3 rule (protocol section 5.1)

**When the rule was fixed.** On 2026-10-27, after the 64-case subset C1 / C6 numbers had been seen and before
any full-validation C1–C6 result.

**The rule.** For each method × dataset × comparison:
- per seed, the image-weighted Δ ("img") and the unit-weighted Δ ("unit");
- **repeated signal** = at least 2 of 3 seeds with img ≥ threshold **and** unit > 0 in the same seed;
- **unstable** = the image-weighted threshold is met in ≥ 2 seeds, but not both conditions;
- otherwise **no signal**.

**Results.** Repeated: {{N:n_repeated}}; unstable: {{N:n_unstable}}; no signal: {{N:n_none}}.

{{T:phase3}}

**Outcome by the rule: {{N:outcome}}.** Repeated-signal cells: {{N:repeated_list}}.

**Per-donor check** (descriptive). Is a signal carried by one donor? For the repeated-signal cells the smallest
per-donor image-weighted Δ is {{N:donor_rep_min}} points, and {{N:donor_rep_cells}} seed × donor cells meet the
threshold.

{{T:donor}}

**D6 sensitivity check** (pre-specified).
- {{N:d6_excluded}} MosMed validation images are excluded: their text says "Unilateral" but names both sides.
- The unit-weighted Δs of the 11 comparisons then change by at most {{N:d6_maxdiff}} points (all three
  methods; exploratory).

{{T:d6}}

## 10. Issues found and how they were handled

- **The first C2 template was truncated.** RecLMIS (CLIP, 16 tokens) and DD-CMD (22 word pieces) cut the end
  of the longer rewrite in most cases. Decision P2-1, taken before any C2 segmentation was run: C2 now changes
  only two words of the original. After P2-1, C2 cuts no more report content than C0 for any method.
- **DD-CMD is not bit-reproducible across processes** (official `deterministic: false`). Compared with
  Phase 1, its validation C0 differs by 1–4 pixels in up to 111 of 1429 cases; within one process it is
  identical. Tolerance rule P2-2 was fixed for the later test reconciliation.
- **C0 is reused.** The benchmark reuses the C0 pass of the text-path jobs. For DD-CMD this mixes processes;
  the measured bound is ≤ 4 pixels per case and ≤ 0.00002 mean Dice. Every pass is logged with its job id.
- **RecLMIS truncation of C0.** RecLMIS uses a CLIP context of 18 tokens including start and end (16 text
  tokens). In 101 of 1429 QaTa validation texts it cuts report words (81 the final "lung", 20 parts of the
  right-lung location), under C0 and the conditions that keep the original location (C2, C3a, C3b).
- **LViT-T uses only the first 10 word vectors.** The location clause is never fully inside them. In
  {{N:lvit_c4_window_same_q}} QaTa C4 rows ({{N:lvit_c4_window_same_m}} MosMed), the ten window words are
  unchanged; the vectors still differ through BERT's context.
- **C1 looks like a "no infection" report for LViT-T and RecLMIS.** The official padding turns `""` into
  `EOF XXX` × 9. In MosMed training this padding occurs only with "No pulmonary infection, zero infected
  areas.".
- **"No pulmonary infection" MosMed images have lesions.** 46 of the 47 train/val images with this text have
  lesion pixels in their mask (median about 10 pixels). Counted only, never used to change a text.
- **D6.** 420 MosMed train/val texts say "Unilateral" but name both sides. They are kept as written, and a
  pre-specified sensitivity check is run (section 9).
- **Grouping.**
  - QaTa patient identity rests on the BIDS `sub-S` naming.
  - `covid_*` validation images cannot be grouped and are their own units.
  - MosMed `bjorke` slices are pooled into one unit.
- **A software error** (DD-CMD's own `evaluate.py` shadowed the common evaluator) stopped DD-CMD in the first
  text-path job. It was fixed and rerun (protocol 4.5).

## 11. Limitations

- **Validation only.** The test split is evaluated only after the freeze, so none of these numbers is a test
  result.
- **MosMedData+ is exploratory.** It has 11 units, and the benchmark split has a known volume overlap between
  train, validation and test (Phase 0).
- **What the texts are.** They are benchmark annotations, not independently verified reports. The three
  attributes are dependent (scope follows the sides named), and the attribute distance is not a clinical
  severity scale.
- **What the comparisons can show.** They describe the frozen models' response to text changes. They do not
  show the causal value of adding text during training, and different architectures do not isolate a module.
- **Seeds and donors are fixed.** Inference is conditional on the three trained seeds and the fixed donor
  assignments. The unit-level t-intervals treat units as independent.
- **Effective inputs differ by method.** LViT-T's 10-word window and RecLMIS's 18-token context limit what each
  method receives, so the same text change is not the same input change for every method.

## 12. Observations relevant to Phase 3 (facts only) and the order of the next steps

**Observations:**
- **The rule's repeated signals:** {{N:repeated_list}}. MosMedData+ has none (exploratory, 11 units).
- **QaTa, incomplete text.**
  - Deleting the location (C0−C3c) lowers Dice by {{N:q_C0-C3c}} points for all three methods.
  - Deleting the scope or the count lowers LViT-T by {{N:q_lvit_c3a}} / {{N:q_lvit_c3b}} points, and RecLMIS
    and DD-CMD by {{N:q_rd_c3ab}}.
- **QaTa, three-attribute mismatch.** C0−C6 is {{N:q_C0-C6}} points (unit-weighted) and C1−C6 is
  {{N:q_C1-C6}} points.
- **QaTa, one- and two-attribute mismatches versus the empty text.** C1−C4 is {{N:q_C1-C4}} and C1−C5 is
  {{N:q_C1-C5}} points: mostly negative, i.e. there the mismatched report gives higher Dice than the empty
  text.
- **QaTa, susceptibility.** LViT-T has a larger C0−C6 harm than RecLMIS ({{N:sus_q_lr}}) and DD-CMD
  ({{N:sus_q_ld}}). RecLMIS − DD-CMD is {{N:sus_q_rd}}.
- **C2 changes masks in most images**, but the mean |ΔDice| is {{N:c2_abs_range}} points.

**Order of the next steps.**
1. The supervisor writes the Phase 3 plan, or decides to cancel Phase 3.
2. The Phase 3 protocol (or the cancellation), the Phase 2 rules and the analysis are frozen and pushed; the
   Phase 2 lock opens.
3. Only then is the test split evaluated, with the frozen programs.

## Appendix A. Supplementary main table (Task 2.4)

mIoU in points, HD95 and ASSD in pixels: mean ± SD over the 3 per-seed means. Empty = the share of
predictions with no foreground (mean over seeds). Dice per seed, in points.

{{T:app_main}}

## Appendix B. Supplementary paired differences (Task 2.5)

Unit-weighted Δ with a paired unit bootstrap 95% CI (10,000 resamples, `default_rng(0)`). Descriptive, no
test. HD95 and ASSD: positive = larger distance under the first condition.

{{T:app_pw}}

## Files

**Results** (`results/`):
- `phase2_val_per_case.csv.gz` (per-case metrics, no text);
- `phase2_donor_manifest_val.csv`;
- `phase2_main_results_val.csv`;
- `phase2_pairwise_results_val.csv`;
- `phase2_severity_results_val.csv`;
- `phase2_severity_by_attribute_val.csv`;
- `phase2_susceptibility_val.csv`;
- `phase2_c2_change_val.csv` and `phase2_c2_mask_change_val.csv.gz`;
- `phase2_effective_input_val_audit.csv`;
- `phase2_phase3_rule_val.csv` and `phase2_phase3_per_donor_val.csv`;
- `phase2_d6_sensitivity_val.csv`;
- `phase2_val_case_selection.json`.

**Code** (`src/phase2/`):
- `analyze_phase2.py`, `analyze_phase2_extra.py` and `make_phase2_tables.py`;
- tests: `test_analyze_phase2.py` and `test_analyze_phase2_extra.py`.
