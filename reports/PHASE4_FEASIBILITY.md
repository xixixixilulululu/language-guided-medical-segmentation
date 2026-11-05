# Phase 4 — Spatial evidence feasibility

Status: 2026-11-05. **Stage 0: GO (with a documented exception and a known limitation); Stage 1: STOP. Phase 4
stopped at Stage 1.** Plan: [PHASE4_PLAN.md](PHASE4_PLAN.md) (sha256 f04901bc…040bd). Validation split only; the
test split is refused by `src/phase4/guard.py`. Before the Stage 1 run below, no validation spatial score, Dice or
AUROC had been computed; GATE_CAL was never read.

## Stage 1: spatial score on GATE_FIT — STOP

`src/phase4/stage1.py`, run once on 2026-11-05 (18:18 UTC) after commit 19b8666; orientation
`patient_left_on_image_left` (text-to-image alignment); the plan's grid unchanged; DD-CMD seed 1001 report
assignments, nnU-Net seed 1001 masks; GATE_FIT only. The eligible images and groups equal the Stage 0 coverage
(831 / 693). Outputs: `results/phase4_spatial_scores_val.csv` (frozen; ids and numbers only) and
`results/phase4_stage1_summary.csv`. Both AUROCs were recomputed independently with
`sklearn.metrics.roc_auc_score` from the scores file and agree (differences ≤ 3 × 10⁻⁷, from the 6-digit storage).

| Quantity | Value |
|---|---|
| GATE_FIT targets in the denominator | 851 |
| Eligible images / groups | 831 / 693 |
| Excluded images (no C0 score: lateral view or empty IMG) | 20 |
| Unavailable donor rows, C4 location-only / C6 (all 851 targets) | 61 / 62 |
| AUROC, C0 vs donor-averaged C4 location-only | **0.620** |
| AUROC, C0 vs donor-averaged C6 (descriptive) | 0.714 |
| mean(C4 − C0) spatial mismatch, image-weighted | +0.118 |
| mean(C6 − C0) spatial mismatch, image-weighted | +0.317 |
| Group-weighted mean(C4 − C0) / mean(C6 − C0) | +0.102 / +0.298 |
| Groups with a positive mean C4 − C0 / C6 − C0 | 414 / 474 of 693 |
| Criterion (spec): AUROC(C0 vs C4) ≥ 0.65 and both mean differences > 0 | AUROC criterion not met |
| **Decision** | **STOP** |

**Phase 4 stopped at Stage 1.** The score moves in the expected direction (higher mismatch for the changed
reports, more for C6 than for location-only C4), but it does not separate C0 from location-only C4 well enough by
the fixed screening criterion. As the plan requires, no other grid, alignment, threshold or feature was tried, and
Stage 2 was not started. The known limitation of the vertical bands (section "Stage 0 decision": for lower-only
training reports a median of only 13% of the lesion lies in the grid's lower band) may be one of the reasons: the
equal-thirds grid may place many "lower" lesions in the middle band. This is a possible explanation, not a tested
one.

## Stage 0 decision: GO, with a documented exception and a known limitation

**The plan's rule.** Stage 0 step 3 asks for the correspondence between image coordinates and the patient's left
and right to be established from documented metadata and the existing code; if it cannot be, Phase 4 stops. The
code establishes only that all masks and images share one array layout; nothing in the data or the documentation
says which image half the texts call "left" (section "Stage 0 records" below).

**The exception (supervisor, 2026-11-05).** For this pilot the supervisor approved option 1: the left/right
correspondence learned on the training split is used as a *text-to-image coordinate alignment* — a report's "left"
means image columns 0–111 and "right" columns 112–223, with the plan's rows 0–74 / 75–148 / 149–223 for upper /
middle / lower. It is not a statement about the patient's anatomy; in the usual chest X-ray display the patient's
left is on the image right, the opposite of what the texts do here. In the spec this is `orientation.value =
patient_left_on_image_left` (the code name is kept for compatibility), with the decision, its basis and the
exception recorded under `orientation_decision`. Option 2 was not run (supervisor's decision); there is no
sensitivity orientation.

**Option 1 on the training split** (rule `configs/phase4/option1_train_rule.json`, commit ee14b30, fixed before
any data were read; result d83862a). One-sided training reports were compared with the image half that holds most
of the ground-truth lesion. Seven criteria were fixed in advance: one convention better in all reports, `sub-*`
and `covid_*`; agreement ≥ 0.90 and Wilson lower bound ≥ 0.85 in each; ≥ 0.85 for left-only and right-only
reports; ≥ 30 reports per group; the same outcome under a stricter lesion-side definition; ≥ 0.90 at patient
level. All seven held for the convention "text left = image left": 1,085 of 1,129 reports (96.1%; Wilson lower
bound 0.948), 603 of 624 patients.

**SGSeg's revised texts** (rule v2 4fdd05d, result 3d44316). With SGSeg's corrected training texts the agreement
is the same (96.1%); none of the 44 disagreeing reports was changed to the other side (pre-set reading: the
revisions explain little of the disagreement).

**Known limitation: the vertical bands** (an extra check added by the project lead; rule 23d79a8, result 6f9b473;
not a criterion of the plan). Up and down are not flipped: the median lesion centroid is row 66 for upper-only and
row 130 for lower-only training reports. But for lower-only reports a median of only 13% of the lesion lies in the
grid's lower band (rows 149–223); most of it is in the middle band. The pre-set criterion (≥ 90%) failed:
lower-only 82.1%, combined 82.7%. The plan's equal-thirds grid is kept unchanged, as the plan requires (regions
are not derived from ground-truth masks). A Stage 1 failure may therefore reflect the grid rather than the idea.

**Decision.** Coverage 1384 / 1421 images (97.4%) and 1156 / 1186 groups (97.5%), both ≥ 80%; orientation by the
approved exception: **GO** (`results/phase4_stage0_decision.json`).

## Stage 0 records so far

**Rules fixed before any score:** `configs/phase4/stage0_spec.json` (commit cb5687d): 2 × 3 grid (columns
0–111 / 112–223; rows 0–74 / 75–148 / 149–223), lookup table for all 56 validation location values (Phase 2
parse with the frozen D5 typo rules), unavailable rules (lateral view `_vp-ll_`, empty IMG, location not parsed,
"unspecified", unknown token), the denominator and the Stage 0 / Stage 1 criteria. Orientation: "pending
supervisor decision", to be set by a separate commit before any score.

**Data linkage (DD-CMD seed 1001, QaTa validation).** Every row of C0 (1429), C2 (1429), C4 (4268), C5 (4269)
and C6 (4269) links to its scored LANG mask (C0: job 369; C4–C6: job 370; C2: the Phase 3 rerun, job 371), the
nnU-Net seed 1001 IMG mask (job 371), F1–F5 and the per-case Dice; no row is missing.

**Orientation (Stage 0 step 3).**
- Established from the code: LANG, IMG, ground truth and the common 224 × 224 images share one array layout
  (row 0 top, column 0 image left); no flip, transpose or rotation at evaluation (`prepare_common_224.py`
  resizes only, QaTa is already 224 × 224; `prepare_lvit_datasets.py` and `prepare_nnunet.py` copy files; the
  DD-CMD evaluation transforms are resize / normalise only; nnU-Net's mirroring TTA is undone internally).
- Not established: whether the image-left half is the patient's left or right. The QaTa package has PNG files
  without metadata, the text files have only image name and text, the archive readme says nothing about
  orientation, and the LViT paper does not say whether "left" in its texts means the patient's or the image's
  left. The QaTa-COV19 dataset paper (Degerli et al.) and other papers that use these texts (e.g. SSA, SGSeg) do
  not state it either. File names give the view (ap / pa / ll) for most `sub-*` images only; 590 `covid_*` images
  of the denominator have no metadata. Options put to the supervisor by the project lead: (a) STOP at Stage 0 as
  the plan specifies; (b) a global rule fixed on the training split (one-sided reports versus the side of the
  ground-truth lesion, criterion fixed in advance, validation masks not used); (c) no left/right commitment: both
  mappings computed, the higher agreement kept. Option (b) ("option 1") was later run on the training split and
  chosen by the supervisor (decision section above); (c) ("option 2") was not run.

**Orientation sensitivity, text only (`src/phase4/stage0_mirror.py` → `results/phase4_stage0_mirror_counts.csv`,
commit 328d514; no score computed).** On the 1388 text-eligible targets (the denominator minus 32 lateral views
and the 1 "unspecified" C0; IMG emptiness not read), each pair of a target and a mappable donor was classed by
patient-side cells: a, the donor cells cover the target cells; b, they cover them only after a left/right swap; c,
otherwise. Location-only C4: 4092 pairs, a 1249 (30.5%), b 327 (8.0%), c 2516 (61.5%). C6: 4162 pairs, a 700
(16.8%), b 56 (1.3%), c 3406 (81.8%). Class b can be flagged by option (b) but not by option (c); class a cannot
be flagged under either. This compares report text with report text; the actual score uses the nnU-Net mask.

**Training augmentation (recorded, no conclusion drawn).** The language-guided models were trained with
geometric augmentation applied to image and mask while the text was unchanged:
- DD-CMD (official `utils/dataset.py` lines 79–84; `src/phase1/train_ddcmd.py` lines 51–56): `RandZoomd`
  (p 0.3), `RandRotated` (±0.3 rad, p 0.3), `RandFlipd` on spatial axis 0 (p 0.5) and on spatial axis 1
  (p 0.5), `RandRotate90d` (p 0.2, k ≤ 3). Both the up–down and the left–right direction are randomised.
- LViT-T (`external/LViT/Load_Dataset.py` `RandomGenerator`, used by `src/phase1/train_lvit.py`): with
  probability 0.5 a random `rot90` (k = 0–3) followed by a flip on a random axis; otherwise with probability
  0.5 a rotation by an integer angle in [−20, 20) degrees.

**Coverage (orientation-independent; `src/phase4/stage0_coverage.py` → `results/phase4_stage0_coverage.csv`).**
Denominator: validation targets with C0, ≥ 1 location-only C4 donor and ≥ 1 C6 donor (frozen manifest).

| Part | Eligible images | Eligible groups (≥ 1 eligible image) |
|---|---|---|
| All | 1384 / 1421 (97.4%) | 1156 / 1186 (97.5%) |
| GATE_FIT | 831 / 851 (97.6%) | 693 / 708 (97.9%) |
| GATE_CAL | 553 / 570 (97.0%) | 463 / 478 (96.9%) |

Unavailable targets (37; the reasons do not overlap): lateral view 32, empty IMG mask 4, C0 location
"unspecified" 1; no target lacks a mappable C4 location-only or C6 donor. Mappable location-only C4 donors per
target: 3 for 1352, 2 for 64, 1 for 5; mappable C6 donors: 3 for 1419, 2 for 2. The coverage passes the 80%
thresholds; the decision is in the section above.
