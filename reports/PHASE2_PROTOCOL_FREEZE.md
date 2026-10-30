# Phase 2 — Protocol freeze (FROZEN)

Status: 2026-10-30, **frozen** (`python src/phase2/lock.py freeze`; record `configs/phase2_frozen/phase2_lock.json`).
- At the freeze, no test file had been read by Phase 2 or Phase 3 code and no test job had been submitted.
- **Phase 3 decision** (`configs/phase2_frozen/phase3_decision.json`): Phase 3 continues; the gates are frozen as
  fixed at b58cdf6, with no amendment. On 2026-10-28 the project lead informed the supervisor that the C0 / C2
  losses on GATE_FIT lie beyond the −0.005 margin, with the default of keeping the gates as planned; by
  2026-10-30 the supervisor had not asked for a change.
- **C2 manual reviews.** Both 40-row reviews (P2-1, section 4.3.1, and the superseded D9 template, section 4.3.2)
  were done by the project lead personally; their records are verified.
- This file and every file of the record are not changed after the freeze. Test results and their explanations
  go to files outside the record (e.g. `reports/PHASE2_RESULTS.md`).
- **Scope.** PHASE3_PLAN Task 3.6, items 1 and 4 for Phase 2: the condition and donor-generation rules,
  eligibility, statistical rules, the validation / test exposure history and the reproducibility information.
  The full rules are in [PHASE2_PROTOCOL.md](PHASE2_PROTOCOL.md); this file states what is frozen and where.
- **Companion.** [PHASE3_PROTOCOL_FREEZE.md](PHASE3_PROTOCOL_FREEZE.md). Both are frozen together, in one
  commit (section 8).

## 1. Condition and donor-generation rules (frozen as in PHASE2_PROTOCOL.md sections 3–4)

- **Text attributes** (section 3): scope, count, location parsed by `src/phase2/texts.py`; unknown values stay
  empty; normalisation D4, D5, D7.
- **Conditions** (section 4), the same code for test as for validation (`src/phase2/conditions.py`):

| Condition | Rule |
|---|---|
| C0 | original text |
| C1 | `""` through each method's normal text pathway (D8) |
| C2 | P2-1: two word substitutions on the original (`pulmonary` → `lung`, `infected area(s)` → `infected region(s)`) |
| C3a / C3b / C3c | delete the scope word / the count clause / the location clause |
| C4 / C5 / C6 | real report of another case, same dataset and **same split**, differing in exactly 1 / 2 / 3 attributes |

- **Donors** (C4–C6). On the test split, donors come from the test split only. Not from the target's patient /
  CT volume (D1, D2, D11); identical texts counted once; up to 3 distinct donor texts, chosen by the smallest
  sha256 of `phase2-donor-v1|dataset|split|target|condition|donor` (salt D12). One manifest for all methods and
  seeds. Generated automatically by the program after the freeze; nobody chooses donors by hand.
- **Eligibility.** A condition is evaluated on the targets for which it is available (section 4). Every method
  and seed uses the same eligible images and donors.

## 2. Inference

- **Checkpoints.** The 18 Phase 1 checkpoints (3 methods × 2 datasets × 3 seeds), identified by sha256 in
  `results/phase2_checkpoint_sha256.tsv`; MosMed LViT-T uses the batch-24 configuration
  (`configs/phase1_frozen/lvit_b24.sha256`). The Phase 1 driver records (`configs/phase1_frozen/`) are
  checked at the start of every job.
- **Settings** (section 1): eval mode, float32, no TTA, threshold `probability > 0.5`, the common evaluator.
- **Storage** (D13 with amendment **A1**, section 9 of the protocol): on the test split the float32
  probabilities of **every** condition and donor are kept, because Phase 3 reads them. Masks (packed bits) and
  per-case metrics for every pass.
- **Per-pass input check** (section 4.8): the number of cases whose model input changes must equal the audit
  count; otherwise the job stops.

## 3. C0 reconciliation (P2-2, section 4.7)

`src/phase2/reconcile_test_c0.py` compares the test C0 of Phase 2 with the archived Phase 1 test evaluation of
the same checkpoint (`runs/<method>/<dataset>/seed<k>/eval_test`; MosMed LViT-T: `runs/lvit_b24/...`).
- LViT-T, RecLMIS: identical masks. DD-CMD: ≤ 10 differing pixels per case, no empty / non-empty flip,
  |Δ mean Dice| ≤ 0.0005 per checkpoint.
- Outside the limits: stop and find the cause; a pipeline error → fix, rerun that checkpoint once (both runs
  logged); GPU non-determinism → keep and report. Never rerun until it passes.
- Output `results/phase2_test_c0_reconciliation.csv`. All Phase 2 paired comparisons use the C0 computed by
  Phase 2 itself.

## 4. Statistical rules (PHASE2_PROTOCOL.md section 5)

- **Units.** QaTa: patients from the BIDS `sub-S` naming (`covid_*` images, validation only, are their own
  units). MosMed: CT volumes, `bjorke` slices pooled into one unit (D1, D2). 9 MosMed test units.
- **Main analysis** (`src/phase2/analyze_phase2.py --split test`): Task 2.4 table (mean ± SD over the 3
  per-seed image means, donors averaged within image); Task 2.5 the 11 comparisons per method × dataset
  (C0−C1, C0−C2, C0−C3a, C0−C3b, C0−C3c, C0−C4, C0−C5, C0−C6, C1−C4, C1−C5, C1−C6) on common eligible image
  sets; unit-weighted delta with a one-sample t-test over units; Holm across the 11; α = 0.05.
- **Exploratory labels.** MosMed unit-level inference is always exploratory; QaTa inference is exploratory with
  fewer than 20 units (D3).
- **D6 sensitivity check** (MosMed without the scope-conflict images): descriptive, test as validation.
- **Additional analyses** (`src/phase2/analyze_phase2_extra.py --split test`): Task 2.6 severity on
  S = targets with C4, C5 and C6 donors, Task 2.7 susceptibility, C0 vs C2 change and figures. Descriptive.
  Case figures use the validation rule: the C6-eligible images sorted by harm C0 − C6 (Dice; mean over the
  3 methods × 3 seeds, donors averaged), ties by sample id; the images at the 10 / 30 / 50 / 70 / 90%
  positions (round(q (n − 1))); shown for seed 1001 and donor d1.
- **The Phase 3 go / no-go rule** (section 5.1) used validation only. It is not re-applied on test.

## 5. Amendments

- **A1** (2026-10-28): test float32 probabilities for every condition and donor (changes D13 for the test
  split only). Reason: one test run serves Phase 2 and Phase 3; otherwise a second test inference of the same
  checkpoints would be needed. Nothing else changes. Details: PHASE2_PROTOCOL.md section 9.

## 6. Validation / test exposure history

**Test split, seen before this freeze** (clean, original-text results only):
- Phase 0: image-only baselines (U-Net, nnU-Net, TransUNet) on both test sets.
- Phase 1: LViT-T, RecLMIS, DD-CMD with the original texts (C0) on both test sets, and the post-hoc MosMed
  LViT-T batch-24 sensitivity check and six-method table (reports/PHASE1_SENSITIVITY_LVIT.md). Phase 1 read
  the test texts through its own pipeline; those are the C0 texts.

**Test split, never seen** (the protected outcome):
- no counterfactual test condition (C1–C6) has been generated or run; the Phase 2 text view
  `~/datasets/phase2/text_view` holds Train and Val only;
- no Phase 2 test inference (C0 included), no test reconciliation, no Phase 2 test analysis;
- no Phase 3 test feature, gate decision or IMG rerun on test.
- `results/phase2_test_eval_log.tsv` does not exist yet (no test run logged).

**Validation, used for design** (in order; all decisions dated in PHASE2_PROTOCOL.md):
1. text parser, conditions, C2 review on train / validation texts (2026-10-27);
2. text-path check on a 64-case validation subset (jobs 368, 369), then P2-1 and P2-2;
3. the Phase 3 go / no-go rule, fixed after the 64-case C1 / C6 subset numbers and before any full-validation
   C1–C6 result;
4. the full validation benchmark (job 370) and analysis; the Phase 3 rule applied as fixed;
5. Phase 3 development on validation (see PHASE3_PROTOCOL_FREEZE.md section 7).

## 7. Test pipeline (order; after the freeze only)

1. Freeze (login node): write `configs/phase2_frozen/phase3_decision.json`, commit; `python src/phase2/lock.py
   freeze`; commit the record `configs/phase2_frozen/phase2_lock.json`; push; `python src/phase2/lock.py check`
   must report OPEN.
2. `python src/phase2/lock.py link-test` (login node; logged in `runs/phase2_lock_log.jsonl`).
3. Job `slurm/phase2_test_benchmark.slurm` (by hand): lock check first (exit 5 if closed), test texts linked,
   Phase 1 drivers unchanged; all 18 checkpoints × all conditions and donors; then the C0 reconciliation.
4. `python src/phase2/analyze_phase2.py --split test` and `python src/phase2/analyze_phase2_extra.py --split
   test` (login node, CPU).
5. Phase 3 test job (PHASE3_PROTOCOL_FREEZE.md section 8), using the **same** Phase 2 test outputs.
6. `reports/PHASE2_RESULTS.md` with the final locked test results.

## 8. Frozen files and reproducibility

- **Record.** `python src/phase2/lock.py freeze` writes `configs/phase2_frozen/phase2_lock.json`: the commit and
  the sha256 of every file of `LOCKED_FILES` in `src/phase2/lock.py` (Phase 2 and Phase 3 files, both freeze
  documents and the Phase 3 decision record). That record is authoritative; the table below lists the sha256
  of the Phase 2 files in the frozen commit (`python src/phase2/lock.py manifest`, generated just before the
  freeze).
- **Code commit.** The frozen commit is the one recorded in `phase2_lock.json` (it cannot be written into this
  file, which is part of it). Gates fixed as of b58cdf6.
- **Environment.** `~/envs/lgms_p1`: torch 2.14.0+cu130, scikit-learn 1.9.0, nnunetv2; Phase 1 method repositories
  at the commits of `configs/phase1_frozen/*.commit`.

<!-- BEGIN MANIFEST-PHASE2 -->
| file | sha256 |
|---|---|
| `src/phase2/lock.py` | `70fca4bd7806f533a0061dd1054e4f57ca65fbbabeef3e9a19f4399fdea65a64` |
| `src/phase2/texts.py` | `5740a18cf5efe946e2bc656c0412d246794657421156cd7099a2c08fa4687de9` |
| `src/phase2/conditions.py` | `109ccb9d0f59205990ce0cd96759fb5c7d3b23da7f8dc4c335ed98efce29d5b0` |
| `src/phase2/infer.py` | `e7f17195caa9f714a1f23fdf4543116e011274cda26445accb1150ecd041f3ad` |
| `src/phase2/analyze_phase2.py` | `ac901fcbb05ec22e059a0932991d2ea131bd6415d8d8cacb267dd0573d111208` |
| `src/phase2/analyze_phase2_extra.py` | `9c84625c6a51017b3d5b58a5411d825656a52c705d2271901f64c4575852278d` |
| `src/phase2/reconcile_test_c0.py` | `48d0e8b900a2e496bd76d724c82b45def40171c42b9efb57f2233bca1f7d46bb` |
| `results/phase2_checkpoint_sha256.tsv` | `fa3eeaed86d427ca94f5f0f12011b0faa194a5dc7bc0512236fb8c301cca42dd` |
| `slurm/phase2_test_benchmark.slurm` | `514cfa87ea50533c2ed69df432692cdd6ef8238dfaf4b9a25227daac5f4da7a9` |
| `reports/PHASE2_PROTOCOL.md` | `76c9b22940a48f4a888e7509a4b782e12c82b7bfa0261574a79ce80140d19aff` |
| `configs/phase2_frozen/phase3_decision.json` | `fd19c8edc820a89ac020f58fef91978db31ecb1971403c425f584b61517d2d37` |
<!-- END MANIFEST-PHASE2 -->

## 9. After the freeze

1. Commit and push the record; `python src/phase2/lock.py check` must report OPEN.
2. Then the test pipeline of section 7, from step 2 (`link-test`).
