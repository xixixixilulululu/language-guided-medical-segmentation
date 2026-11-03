# Phase 4 — Spatial evidence feasibility (DRAFT, Stage 0 in progress)

Status: 2026-11-03. **Stage 0 not decided** (no GO / STOP): the patient left/right orientation is pending the
supervisor's decision. Plan: [PHASE4_PLAN.md](PHASE4_PLAN.md) (sha256 f04901bc…040bd). Validation split only;
the test split is refused by `src/phase4/guard.py`. No spatial score, Dice or AUROC has been computed.

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
  mappings computed, the higher agreement kept. None of them has been run.

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
thresholds, but the Stage 0 decision also requires the orientation, which is open.
