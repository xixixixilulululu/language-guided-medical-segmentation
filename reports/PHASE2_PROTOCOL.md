# Phase 2 — Protocol (DRAFT, not frozen)

Status: 2026-10-27, **draft**.
- **Covered so far:** Milestone 2.1 text work, i.e. checkpoints, text attributes, conditions and the lock.
- **Still open:**
  - the inference settings check, the C0 reconciliation and the tolerance rule (after the text-path job);
  - the analysis program;
  - the Phase 3 decision.
- **How it is frozen:** together with the code by `src/phase2/lock.py freeze`, after the validation benchmark
  and the Phase 3 decision. Changes after the freeze are recorded as amendments.
- **Sources:** the supervisor's plan [PHASE2_PLAN.md](PHASE2_PLAN.md), and the decisions of 2026-10-27
  ([PHASE2_SETUP_PLAN.md](PHASE2_SETUP_PLAN.md), section 9, D1–D15, plus the five additions recorded in
  section 7 below).

## 1. Baseline: checkpoints, settings, splits

| Dataset | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| QaTa-COV19 | Phase 1 main runs `runs/lvit/QaTa-COV19/seed<k>/best_model.pt` (config `configs/lvit.yaml`, batch 2) | `runs/reclmis/QaTa-COV19/seed<k>/best_model.pt` (`configs/reclmis.yaml`) | `runs/ddcmd/QaTa-COV19/seed<k>/checkpoints/best.ckpt` (`configs/ddcmd.yaml`) |
| MosMedData+ | **post-hoc batch-24 sensitivity runs** `runs/lvit_b24/MosMedData+/seed<k>/best_model.pt` (`configs/lvit_sensitivity_b24.yaml`) | `runs/reclmis/MosMedData+/seed<k>/best_model.pt` | `runs/ddcmd/MosMedData+/seed<k>/checkpoints/best.ckpt` |

**Checkpoints.**
- Seeds 1001, 1002, 1003; 18 checkpoints.
- sha256 values: `results/phase2_checkpoint_sha256.tsv`.
- All were selected on validation Dice only. The epoch stored in each file equals the best epoch of its
  `training_summary.json` (checked 2026-10-27).

**Provenance of MosMed LViT-T.**
- The batch-24 runs come from a post-hoc sensitivity study decided after the Phase 1 results had been seen
  ([PHASE1_SENSITIVITY_LVIT.md](PHASE1_SENSITIVITY_LVIT.md)): test Dice 75.03 / 72.82 / 74.19, mean 74.01.
- The Phase 1 main results (batch 2, 61.26) are unchanged and are not the Phase 2 C0 baseline.
- 24 is the training batch size; the inference settings are those of the Phase 1 drivers.

**Inference settings** (from the frozen Phase 1 drivers, unchanged; to be confirmed by the C0 reconciliation):
- `eval()` mode, float32, no TTA;
- inference batch 32 for all three methods;
- the common 224 × 224 images and targets;
- threshold `probability > 0.5`;
- evaluator `src/evaluate.py`: Dice, mIoU, HD95, ASSD; an empty prediction is scored HD95 = ASSD = 316.78 px.

**Text encoders.**
- LViT-T: BERT-base uncased, PyTorch re-implementation of `bert_embedding`, 25 tokens, first 10 word vectors.
- RecLMIS: CLIP ViT-B/32 text tower, `clip.tokenize`, context 18, `truncate=True`.
- DD-CMD: BiomedBERT, 24 tokens, padding to max length.
- All three are frozen.

**Split manifests** (sha256 of `~/datasets/common224/<dataset>/<split>.txt`):

| Dataset | train | val | test |
|---|---|---|---|
| QaTa-COV19 | `10d5a105…19a8cb` | `ed9056dc…36b4cc` | `0662d5c3…ade475b` |
| MosMedData+ | `49b2d993…6333c1` | `cd2b5c46…37ea4` | `f38fc890…fec49e` |

**Code.**
- `src/phase2/texts.py`, `src/phase2/conditions.py`, `src/phase2/lock.py` and `src/phase2/infer.py`
  (CPU checks: `src/phase2/test_phase2_infer.py`, 12 passed).
- The freeze will record the final commit.

## 2. Validation-first order and the Phase 2 lock

1. Parser and text transformations: train/validation texts only (done).
2. Text-path check and C0 reconciliation on validation, then the validation benchmark (next jobs).
3. Phase 3 decision from validation results only.
4. Freeze (`src/phase2/lock.py freeze`): code, this protocol and the Phase 3 decision record
   (`configs/phase2_frozen/phase3_decision.json`). Push to GitHub.
5. Only then: `lock.py link-test`, test conditions, test inference.

**Lock** (`src/phase2/lock.py`, tests `src/phase2/test_phase2_lock.py`, 14 checks passed on 2026-10-27).
- Every Phase 2 test-split entry point stops with exit code 5, before any test file is read, unless all of
  the following hold:
  - the frozen record exists;
  - it covers the protocol and the Phase 3 decision record;
  - every file is unchanged;
  - its commit is on `origin/main`.
- This covers **all** conditions, C0 included (D14).
- Independently, the text view `~/datasets/phase2/text_view/` holds only the Train and Val text files until
  `link-test`.
- Phase 1 clean test results have been seen before; the lock protects against using new counterfactual test
  outcomes to design the benchmark.
- After the freeze the program generates test donors automatically by the frozen rules. Nobody chooses them
  by hand.

## 3. Text attributes (Task 2.1)

`src/phase2/texts.py` → `results/phase2_text_attributes.csv` (train + val; 9601 rows).

**Template.** `<Scope> pulmonary infection, <count> infected area(s), <location>.`

**Definitions.**
- `scope`: the stated laterality / presence word: Bilateral | Unilateral | No. Infection type and severity
  do not occur in these texts and are not mixed in.
- `count`: the number of infected areas, 0–9.
- `location`: zones per side, canonical form `left:<zones>;right:<zones>`.
  - Zones are upper / middle / lower.
  - The full set equals `all` (D4).
  - A side named without a zone is `unspecified`.
  - `none` for "No pulmonary infection" texts (D7).
- Unknown values stay empty (never 0).

**Normalisation** (D5):
- spacing and commas;
- trailing full stop(s);
- the elision `<z> left and <z> right lung`;
- 26 fixed typo corrections.

Anything ambiguous leaves that attribute unknown: `higher`, `two one infected area`, `all lung`, a zone list
mixing `all` with zones.

| | QaTa-COV19 train | QaTa-COV19 val | MosMedData+ train | MosMedData+ val |
|---|---:|---:|---:|---:|
| Images | 5716 | 1429 | 2183 | 273 |
| Distinct texts | 273 | 128 | 101 | 45 |
| All three attributes parsed | 5692 (99.58%) | 1423 (99.58%) | 2183 (100%) | 273 (100%) |

Unknown attributes over train + val: QaTa count 5 images, location 25 images; MosMed none.

**Attribute frequencies** (train + val images):
- QaTa scope: Bilateral 5663, Unilateral 1482.
- QaTa count (parsed): one 1436, two 5254, three 364, four 58, five 6, six 1.
- QaTa location: 65 canonical values; most frequent `left:all;right:all` 2164, `left:lower;right:lower` 1560,
  `right:lower` 551.
- MosMed scope: Unilateral 1234, Bilateral 1175, No 47.
- MosMed count: zero 47, one 908, two 460, three 368, four 258, five 148, six 100, seven 62, eight 38, nine 67.
- MosMed location: 23 canonical values; most frequent `left:middle;right:middle` 578, `left:middle` 528,
  `left:unspecified;right:middle` 336.

**Flags** (`parse_note`; values are kept as written):
- MosMed: 420 images "Unilateral …, left lung and <zone> right lung" (D6; 14 of them in val).
- QaTa: 4 "Unilateral" naming two sides, and 9 "Bilateral" naming one side.

**Groups** (D1, D2):

| | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Unit | patient = BIDS subject `sub-S…` of the file name | CT volume = file name without the final `_<slice>` (Phase 0 source mapping) |
| Without id | `covid_*` (2354 train, 597 val, 0 test): own unit each, flagged "patient unverified"; donor exclusion only of the case itself | `bjorke_*` (80 train, 10 val, 10 test): one pooled unit `bjorke_pooled` |
| Val units | 1194 = 597 patients + 597 unverified images | **11** = 10 volumes + `bjorke_pooled` |
| Test units (from sample ids) | 474 patients (all images `sub-S…`) | **9** = 8 volumes + `bjorke_pooled` |

## 4. Text conditions (Task 2.2)

`src/phase2/conditions.py` → `results/phase2_text_conditions.csv` (val) and
`results/phase2_text_conditions_donor_counts.csv`. Tests: `src/phase2/test_phase2_texts.py`, 49 checks
passed (after P2-1).

| Condition | Rule |
|---|---|
| C0 | original text |
| C1 | `""` (D8), processed by each method's normal pathway (section 4.1) |
| C2 | **(P2-1, 2026-10-27)** the original text with exactly two word substitutions, `pulmonary` → `lung` and `infected area(s)` → `infected region(s)` (case and number kept); everything else unchanged, including zones (`upper middle lower` is not turned into `all`), typos and grammar. Unavailable if neither substitution applies; rows with only one are available and counted (validation: 0). *Superseded: the D9 template from parsed values, section 4.3.* |
| C3a | delete the scope word: `Bilateral pulmonary infection` → `Pulmonary infection`; unavailable for "No" texts (D10) |
| C3b | delete the whole count clause `, <n> infected area(s)` |
| C3c | delete the whole location clause; unavailable for "No" texts (no location) |
| C4 / C5 / C6 | real report of another case, same dataset and split, differing in exactly 1 / 2 / 3 of (scope, count, location), all known for both; not from the target's patient / volume (D1, D2, D11); identical texts counted once; up to 3 distinct donor texts, the smallest sha256 of `phase2-donor-v1|dataset|split|target|condition|donor` (salt D12); one manifest for all methods and seeds |

**C3 edits.** The C3 deletions edit the original text and nothing else. The residual cues are recorded in
`residual_cue`:
- C3a: two sides still named imply bilateral; one side implies unilateral.
- C3b: "Bilateral" implies at least two areas.
- C3c: the scope word implies the sides.

**Eligible target images on validation:**

| | C0 | C1 | C2 | C3a | C3b | C3c | C4 | C5 | C6 | C4∩C5∩C6 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| QaTa-COV19 (1429) | 1429 | 1429 | 1429 | 1429 | 1427 | 1425 | 1423 | 1423 | 1423 | 1423 |
| MosMedData+ (273) | 273 | 273 | 273 | 269 | 273 | 269 | 269 | 269 | 273 | 269 |

**Donors** (validation):
- QaTa: 12,806 donor rows from 1373 distinct donor cases; the most-used donor case serves 148 targets. All
  eligible targets have 3 donors for C5 and C6, and 1422 of 1423 for C4. The median number of available
  distinct donor texts is 54 (C4), 41 (C5) and 16 (C6).
- MosMed: 2417 donor rows from 259 donor cases; maximum reuse 85. 254 / 269 / 273 targets have 3 donors for
  C4 / C5 / C6; the medians of available distinct texts are 5 / 11 / 23.
- The 4 MosMed "No" texts have no C4 or C5 donor: they differ from every other report in all three
  attributes.

**Changed attributes** (donor rows):

| | C4 | C5 |
|---|---|---|
| QaTa-COV19 | location 4194, count 71, scope 3 | count + location 3748, scope + location 488, scope + count 33 |
| MosMedData+ | location 513, count 278 | count + location 656, scope + location 151 |

**C4 in QaTa is in practice a location-only mismatch.** Scope and location are dependent (the sides named
determine the laterality word). This is reported as a property of the benchmark, not changed.

### 4.1 C1 as input (tokenizer level; encoder outputs are checked in the text-path job)

| Method | `""` becomes | Tokens / mask |
|---|---|---|
| LViT-T | official padding: `" EOF XXX"` × 9 | 18 BERT words `eof xxx …`; the first 10 word vectors are used |
| RecLMIS | official padding: `" EOF XXX"` × 9 | CLIP `[49406, 68, 539, 8352, …, 49407]`, 18 tokens, mask all 1 (truncated at 18) |
| DD-CMD | no padding rule | `[CLS] [SEP]` + 22 × [PAD] (`input_ids` 2, 3, 0 …), attention mask `1 1 0 …` |

No model-specific fallback was needed: the empty string is a valid input for all three.

### 4.2 Manual review examples (validation; one per rule, from the generated manifest)

| Rule | Original | Condition text |
|---|---|---|
| C2 (P2-1) | Bilateral pulmonary infection, two infected areas, lower left lung and all right lung. | Bilateral lung infection, two infected regions, lower left lung and all right lung. |
| C2 (P2-1), typo kept | Bilateral pulmonary infection, two infected areas, lower left lung and midddle lower right lung. | Bilateral lung infection, two infected regions, lower left lung and midddle lower right lung. |
| *superseded D9 template:* C2 | Bilateral pulmonary infection, two infected areas, lower left lung and all right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and all of the right lung. |
| *superseded D9 template:* C2, D6 text | Unilateral pulmonary infection, one infected areas, left lung and lower right lung. | Unilateral lung infection, one area of infection, in the left lung and the lower part of the right lung. |
| *superseded D9 template:* C2, typo | Bilateral pulmonary infection, two infected areas, lower left lung and midddle lower right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and the middle and lower part of the right lung. |
| C3a | Bilateral pulmonary infection, two infected areas,all left and all right lung. | Pulmonary infection, two infected areas,all left and all right lung. (cue: both sides named) |
| C3b | Unilateral pulmonary infection, two infected areas, lower left lung. | Unilateral pulmonary infection, lower left lung. |
| C3c | Bilateral pulmonary infection, two infected areas, middle left lung and lower right lung. | Bilateral pulmonary infection, two infected areas. (cue: Bilateral implies both lungs) |
| C3b, odd spacing | Bilateral pulmonary infection, three infected areas,upper lower left lung and lower right lung. | Bilateral pulmonary infection,upper lower left lung and lower right lung. |
| C4 (location) | Bilateral pulmonary infection, three infected areas, all left lung and upper middle lower right lung. | Bilateral pulmonary infection, three infected areas, lower left lung and upper lower right lung. |
| C4 (count, MosMed) | Bilateral pulmonary infection, two infected areas, middle left lung and middle right lung. | Bilateral pulmonary infection, nine infected areas, middle left lung and middle right lung. |
| C5 (scope + location, MosMed) | Bilateral pulmonary infection, two infected areas, middle left lung and middle right lung. | Unilateral pulmonary infection, two infected areas, upper left lung. |
| C6 | Bilateral pulmonary infection, three infected areas, all left lung and upper middle lower right lung. | Unilateral pulmonary infection, one infected area, middle left lung. |

### 4.3 C2 review: 20 train + 20 validation texts

#### 4.3.1 Current C2 (P2-1, two word substitutions)

- **Why the C2 rule was changed** (decision P2-1 of the project lead, 2026-10-27):
  1. C0 and C2 now differ only in these two words, so every piece of information in the original, including
     content the parser does not cover, is kept as written. This meets the plan's requirement that C2
     "preserves all substantive information".
  2. It avoids the extra differences of the D9 template: the rewrite of `upper middle lower` as `all`, the
     correction of typos and grammar, and the longer wording.
  3. The decision rests only on the effective-input audit (truncation and the LViT-T window, section 4.5).
     No segmentation result of C2 had been computed when it was made. The only segmentation numbers seen so far
     in Phase 2 are C0 and the validation subset of 64 cases under C1 and C6 (section 4.5).
- **Sample.** Same rule as before: `python src/phase2/conditions.py --c2-review`, distinct fully parsed texts
  per split, both datasets pooled, sorted by dataset and text, 20 per split with `numpy.random.default_rng(0)`.
  The 40 originals are the same as in the superseded review.
- **Who checks what.**
  - The AI assistant (Claude) checked automatically that each C2 equals its C0 except for the two
    substitutions: 40 of 40 pass (column "Automatic word check").
  - The project lead checked all 40 rows for identical meaning: **all 40 have the same meaning**
    (last column, 2026-10-27).

| # | Split | Dataset | Original (C0) | C2 (two word substitutions) | Automatic word check (AI assistant) | Same meaning? (project lead) |
|---:|---|---|---|---|---|---|
| 1 | train | MosMedData+ | Bilateral pulmonary infection, eight infected areas, middle left lung and middle right lung. | Bilateral lung infection, eight infected regions, middle left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 2 | train | MosMedData+ | Bilateral pulmonary infection, five infected areas, middle left lung and middle right lung. | Bilateral lung infection, five infected regions, middle left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 3 | train | MosMedData+ | Bilateral pulmonary infection, four infected areas, middle left lung and middle right lung. | Bilateral lung infection, four infected regions, middle left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 4 | train | MosMedData+ | Bilateral pulmonary infection, three infected areas, lower left lung and upper right lung. | Bilateral lung infection, three infected regions, lower left lung and upper right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 5 | train | MosMedData+ | Unilateral pulmonary infection, two infected areas, left lung and upper right lung. | Unilateral lung infection, two infected regions, left lung and upper right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 6 | train | QaTa-COV19 | Bilateral pulmonary infection, four infected areas, upper left lung and upper middle lower right lung. | Bilateral lung infection, four infected regions, upper left lung and upper middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 7 | train | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, upper middle lower left lung and upper middle lower right lung. | Bilateral lung infection, three infected regions, upper middle lower left lung and upper middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 8 | train | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, upper middlr lower left lung and lower right lung. | Bilateral lung infection, three infected regions, upper middlr lower left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 9 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and middle right lung. | Bilateral lung infection, two infected regions, all left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 10 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and upper right lung. | Bilateral lung infection, two infected regions, all left lung and upper right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 11 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and middle right lung. | Bilateral lung infection, two infected regions, lower left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 12 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and upper middle lower right lung. | Bilateral lung infection, two infected regions, lower left lung and upper middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 13 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and middle right lung. | Bilateral lung infection, two infected regions, middle left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 14 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and upper middle lower right lung. | Bilateral lung infection, two infected regions, middle left lung and upper middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 15 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper left lung andupper right lung. | Bilateral lung infection, two infected regions, upper left lung andupper right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 16 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle lower left lung and upper middle lower right lung. | Bilateral lung infection, two infected regions, upper middle lower left lung and upper middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 17 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas,all left lung and lower right lung. | Bilateral lung infection, two infected regions,all left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 18 | train | QaTa-COV19 | Unilateral pulmonary infection, one infected area, middle  left lung. | Unilateral lung infection, one infected region, middle  left lung. | pass (pulmonary → lung, area → region) | Yes |
| 19 | train | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle lower right lung. | Unilateral lung infection, one infected region, upper middle lower right lung. | pass (pulmonary → lung, area → region) | Yes |
| 20 | train | QaTa-COV19 | Unilateral pulmonary infection, two infected area, lower left lung. | Unilateral lung infection, two infected region, lower left lung. | pass (pulmonary → lung, area → region) | Yes |
| 21 | val | MosMedData+ | Bilateral pulmonary infection, five infected areas, upper left lung and middle right lung. | Bilateral lung infection, five infected regions, upper left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 22 | val | MosMedData+ | Bilateral pulmonary infection, four infected areas, middle left lung and lower right lung. | Bilateral lung infection, four infected regions, middle left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 23 | val | MosMedData+ | Bilateral pulmonary infection, nine infected areas, middle left lung and middle right lung. | Bilateral lung infection, nine infected regions, middle left lung and middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 24 | val | MosMedData+ | Unilateral pulmonary infection, one infected areas, left lung and lower right lung. | Unilateral lung infection, one infected regions, left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 25 | val | QaTa-COV19 | Bilateral pulmonary infection, four infected areas, middle lower left lung and lower right lung. | Bilateral lung infection, four infected regions, middle lower left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 26 | val | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, lower left lung and lower right lung. | Bilateral lung infection, three infected regions, lower left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 27 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas,  upper lower right lung. | Bilateral lung infection, two infected regions,  upper lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 28 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and lower right lung. | Bilateral lung infection, two infected regions, all left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 29 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and upper right lung. | Bilateral lung infection, two infected regions, all left lung and upper right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 30 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and all right lung. | Bilateral lung infection, two infected regions, lower left lung and all right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 31 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and midddle lower right lung. | Bilateral lung infection, two infected regions, lower left lung and midddle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 32 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and upper lower right lung. | Bilateral lung infection, two infected regions, lower left lung and upper lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 33 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and lower right lung. | Bilateral lung infection, two infected regions, middle left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 34 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle lower left and middle lower right lung. | Bilateral lung infection, two infected regions, middle lower left and middle lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 35 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper left lung and upper middle right lung. | Bilateral lung infection, two infected regions, upper left lung and upper middle right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 36 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle left lung and lower right lung. | Bilateral lung infection, two infected regions, upper middle left lung and lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 37 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle left lung and upper lower right lung. | Bilateral lung infection, two infected regions, upper middle left lung and upper lower right lung. | pass (pulmonary → lung, areas → regions) | Yes |
| 38 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, all left and all right lung. | Unilateral lung infection, one infected region, all left and all right lung. | pass (pulmonary → lung, area → region) | Yes |
| 39 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle left lung. | Unilateral lung infection, one infected region, upper middle left lung. | pass (pulmonary → lung, area → region) | Yes |
| 40 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle lower left lung. | Unilateral lung infection, one infected region, upper middle lower left lung. | pass (pulmonary → lung, area → region) | Yes |

#### 4.3.2 SUPERSEDED: D9 template review (kept for the record)

The D9 template below was replaced by P2-1 on 2026-10-27, before any C2 segmentation was run; this table is not
used.


- **Sample.** `python src/phase2/conditions.py --c2-review`: distinct fully parsed texts of each split (both
  datasets pooled, sorted by dataset and text), 20 per split drawn with `numpy.random.default_rng(0)`.
- **Review result (2026-10-27): all 40 rewrites have the same meaning as the original.**
  - The project lead checked the 40 pairs row by row.
  - In addition, the AI assistant (Claude) parsed each rewrite back and compared its (scope, count, location)
    with the parsed original: 40 of 40 identical.
- **Points that were checked:**
  - by D4, `upper middle lower` is rewritten as `all of the … lung`;
  - grammatical slips of the original (e.g. "one infected areas") are not carried over.

| # | Split | Dataset | Original (C0) | Rewrite (C2) | Same meaning? |
|---:|---|---|---|---|---|
| 1 | train | MosMedData+ | Bilateral pulmonary infection, eight infected areas, middle left lung and middle right lung. | Bilateral lung infection, eight areas of infection, in the middle part of the left lung and the middle part of the right lung. | Yes |
| 2 | train | MosMedData+ | Bilateral pulmonary infection, five infected areas, middle left lung and middle right lung. | Bilateral lung infection, five areas of infection, in the middle part of the left lung and the middle part of the right lung. | Yes |
| 3 | train | MosMedData+ | Bilateral pulmonary infection, four infected areas, middle left lung and middle right lung. | Bilateral lung infection, four areas of infection, in the middle part of the left lung and the middle part of the right lung. | Yes |
| 4 | train | MosMedData+ | Bilateral pulmonary infection, three infected areas, lower left lung and upper right lung. | Bilateral lung infection, three areas of infection, in the lower part of the left lung and the upper part of the right lung. | Yes |
| 5 | train | MosMedData+ | Unilateral pulmonary infection, two infected areas, left lung and upper right lung. | Unilateral lung infection, two areas of infection, in the left lung and the upper part of the right lung. | Yes |
| 6 | train | QaTa-COV19 | Bilateral pulmonary infection, four infected areas, upper left lung and upper middle lower right lung. | Bilateral lung infection, four areas of infection, in the upper part of the left lung and all of the right lung. | Yes |
| 7 | train | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, upper middle lower left lung and upper middle lower right lung. | Bilateral lung infection, three areas of infection, in all of the left lung and all of the right lung. | Yes |
| 8 | train | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, upper middlr lower left lung and lower right lung. | Bilateral lung infection, three areas of infection, in all of the left lung and the lower part of the right lung. | Yes |
| 9 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and middle right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and the middle part of the right lung. | Yes |
| 10 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and upper right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and the upper part of the right lung. | Yes |
| 11 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and middle right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and the middle part of the right lung. | Yes |
| 12 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and upper middle lower right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and all of the right lung. | Yes |
| 13 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and middle right lung. | Bilateral lung infection, two areas of infection, in the middle part of the left lung and the middle part of the right lung. | Yes |
| 14 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and upper middle lower right lung. | Bilateral lung infection, two areas of infection, in the middle part of the left lung and all of the right lung. | Yes |
| 15 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper left lung andupper right lung. | Bilateral lung infection, two areas of infection, in the upper part of the left lung and the upper part of the right lung. | Yes |
| 16 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle lower left lung and upper middle lower right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and all of the right lung. | Yes |
| 17 | train | QaTa-COV19 | Bilateral pulmonary infection, two infected areas,all left lung and lower right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and the lower part of the right lung. | Yes |
| 18 | train | QaTa-COV19 | Unilateral pulmonary infection, one infected area, middle  left lung. | Unilateral lung infection, one area of infection, in the middle part of the left lung. | Yes |
| 19 | train | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle lower right lung. | Unilateral lung infection, one area of infection, in all of the right lung. | Yes |
| 20 | train | QaTa-COV19 | Unilateral pulmonary infection, two infected area, lower left lung. | Unilateral lung infection, two areas of infection, in the lower part of the left lung. | Yes |
| 21 | val | MosMedData+ | Bilateral pulmonary infection, five infected areas, upper left lung and middle right lung. | Bilateral lung infection, five areas of infection, in the upper part of the left lung and the middle part of the right lung. | Yes |
| 22 | val | MosMedData+ | Bilateral pulmonary infection, four infected areas, middle left lung and lower right lung. | Bilateral lung infection, four areas of infection, in the middle part of the left lung and the lower part of the right lung. | Yes |
| 23 | val | MosMedData+ | Bilateral pulmonary infection, nine infected areas, middle left lung and middle right lung. | Bilateral lung infection, nine areas of infection, in the middle part of the left lung and the middle part of the right lung. | Yes |
| 24 | val | MosMedData+ | Unilateral pulmonary infection, one infected areas, left lung and lower right lung. | Unilateral lung infection, one area of infection, in the left lung and the lower part of the right lung. | Yes |
| 25 | val | QaTa-COV19 | Bilateral pulmonary infection, four infected areas, middle lower left lung and lower right lung. | Bilateral lung infection, four areas of infection, in the middle and lower part of the left lung and the lower part of the right lung. | Yes |
| 26 | val | QaTa-COV19 | Bilateral pulmonary infection, three infected areas, lower left lung and lower right lung. | Bilateral lung infection, three areas of infection, in the lower part of the left lung and the lower part of the right lung. | Yes |
| 27 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas,  upper lower right lung. | Bilateral lung infection, two areas of infection, in the upper and lower part of the right lung. | Yes |
| 28 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and lower right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and the lower part of the right lung. | Yes |
| 29 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, all left lung and upper right lung. | Bilateral lung infection, two areas of infection, in all of the left lung and the upper part of the right lung. | Yes |
| 30 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and all right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and all of the right lung. | Yes |
| 31 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and midddle lower right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and the middle and lower part of the right lung. | Yes |
| 32 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, lower left lung and upper lower right lung. | Bilateral lung infection, two areas of infection, in the lower part of the left lung and the upper and lower part of the right lung. | Yes |
| 33 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle left lung and lower right lung. | Bilateral lung infection, two areas of infection, in the middle part of the left lung and the lower part of the right lung. | Yes |
| 34 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, middle lower left and middle lower right lung. | Bilateral lung infection, two areas of infection, in the middle and lower part of the left lung and the middle and lower part of the right lung. | Yes |
| 35 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper left lung and upper middle right lung. | Bilateral lung infection, two areas of infection, in the upper part of the left lung and the upper and middle part of the right lung. | Yes |
| 36 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle left lung and lower right lung. | Bilateral lung infection, two areas of infection, in the upper and middle part of the left lung and the lower part of the right lung. | Yes |
| 37 | val | QaTa-COV19 | Bilateral pulmonary infection, two infected areas, upper middle left lung and upper lower right lung. | Bilateral lung infection, two areas of infection, in the upper and middle part of the left lung and the upper and lower part of the right lung. | Yes |
| 38 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, all left and all right lung. | Unilateral lung infection, one area of infection, in all of the left lung and all of the right lung. | Yes |
| 39 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle left lung. | Unilateral lung infection, one area of infection, in the upper and middle part of the left lung. | Yes |
| 40 | val | QaTa-COV19 | Unilateral pulmonary infection, one infected area, upper middle lower left lung. | Unilateral lung infection, one area of infection, in all of the left lung. | Yes |

## 4.4 Text-path check (first job; validation only)

Defined before the job is run.
- **Driver and job.** `src/phase2/infer.py --stage textpath`, one process per method × dataset. Slurm file
  `slurm/phase2_textpath.slurm`, all 18 checkpoints, validation only.
- **Checks per checkpoint:**
  1. C0 on the full validation split, through the Phase 2 path (the Phase 1 drivers' dataset classes, model
     and `predict`, unchanged; eval mode; inference batch 32; seed set before every pass).
  2. Reconciliation with the archived Phase 1 validation evaluation of the same checkpoint: per-case
     |ΔDice| and |ΔHD95|, and the scored-mask pixels that differ.
  3. A second C0 pass as the numerical-noise reference: maximum probability difference, mask pixels, |ΔDice|.
  4. Fixed subset: 64 C6-eligible validation cases per dataset, the smallest
     sha256(`phase2-textpath-subset-v1|dataset|id`). C1 and C6 (first donor) are run on it, recording whether
     the raw text and the model input changed, the mean and maximum probability change, the changed mask
     pixels, and Dice under C0 vs. the condition.
  5. Image/mask pairing: the targets returned by the loader must equal the common masks, or the job stops.
  6. LViT-T: the C0 word vectors of the Phase 2 path must equal the Phase 1 cache (looked up by key).
- **Effective-input audit** for every validation condition row, the same for all seeds:
  - raw-text change;
  - change of the actual model input: LViT-T the 10 × 768 word vectors, RecLMIS the CLIP tokens, DD-CMD
    the BiomedBERT ids and mask;
  - truncation by the method's tokenizer;
  - for LViT-T, the words actually inside its 10-word window, taken from the tokenizer as the text path forms
    them, and whether they differ from C0. The C2 window changes are reported separately.

  The audit is written to `runs/phase2/val/effective_input_<method>_<dataset>.csv`.
- **Afterwards, before any test run.** The numerical-tolerance rule for the C0 reconciliation is fixed from
  the repeat and reconciliation results and recorded here. A text change does not have to reduce Dice to
  pass; an unchanged prediction with a changed input is reported as insensitivity, not as a failure.

### 4.5 Text-path check: results (jobs 368 and 369, 2026-10-27; validation only)

- **Jobs.** Job 368 completed LViT-T and RecLMIS. DD-CMD failed in job 368 on a software error:
  `external/DD-CMD/evaluate.py` shadowed `src/evaluate.py`. The common evaluator is now loaded by file path
  (commit fa758f5), and DD-CMD was rerun in job 369.
- **Files** (numbers only): `results/phase2_textpath_summary.csv` and
  `results/phase2_effective_input_val_summary.csv`.

**C0 reconciliation with the archived Phase 1 validation evaluation:**
- **LViT-T and RecLMIS** (12 checkpoints): identical. Mean Dice, every per-case Dice and every scored mask
  pixel are equal.
- **DD-CMD** (6 checkpoints): almost identical.

  | DD-CMD | Masks that differ from Phase 1 (cases) | Pixels per differing mask |
  |---|---|---|
  | QaTa-COV19 | 5, 90 and 111 of 1429 per seed | 1–4 |
  | MosMedData+ | 0–4 of 273 | 1 |

  - Per case: max |ΔDice| 0.0048 (one MosMed case), max |ΔHD95| 0.20 px.
  - Mean Dice per checkpoint: |Δ| ≤ 0.00002.
  - DD-CMD runs with `deterministic: false` (official). Within Phase 2, its repeated pass is bit-identical, so
    the differences come from GPU algorithm selection between processes. Phase 0 recorded the same kind of
    difference for nnU-Net.
- **Repeated C0 pass** (all 18 checkpoints): probability maps bit-identical (max difference 0). The 5 × 10⁻⁷
  Dice differences in the logs come from the 6-decimal CSV rounding of the first pass.
- **LViT-T text path:** the C0 word vectors equal the Phase 1 cache (max difference 0) for all six
  checkpoints.

**Fixed subset of 64 cases, C1 and C6 (first donor):**
- The model input changed in 64 of 64 cases for every checkpoint and both conditions.
- The scored mask changed in 59–64 cases.
- So the text reaches the predictions in all three methods. The mean Dice values on the subset are
  development numbers and are not interpreted here.

**Effective-input audit** (all validation condition rows):
- Every raw-text change changes the actual model input, for all three methods and all conditions.
- **LViT-T window** (words from the tokenizer):
  - C2 changes the window content in 100% of the cases (QaTa 1423 / 1423, MosMed 273 / 273), as recorded in
    section 6.
  - C4 leaves the 10 window words unchanged in 544 of 4268 QaTa rows (12.7%) and 69 of 791 MosMed rows
    (8.7%). All of these are location-only mismatches. Their word vectors still differ, through BERT's
    context.
- **C2 is truncated where C0 is not** (superseded D9 template; resolved by P2-1, see below):
  - RecLMIS: 1155 of 1423 QaTa cases and 269 of 273 MosMed cases;
  - DD-CMD: 670 of 1423 QaTa cases and 112 of 273 MosMed cases.

  The cut removed the end of the location: with the D9 template, C2 was **not information-preserving at the
  model input** for most RecLMIS cases and about half of the DD-CMD cases. This led to decision P2-1.
- The rows for C2 in `results/phase2_effective_input_val_summary.csv` belong to the superseded D9 template. The
  current C2 is audited in `results/phase2_c2_input_audit_val.csv` (section 4.6).
- The "truncated" counts for C1, C3b and C3c of LViT-T and RecLMIS mostly cut only the `EOF XXX` padding,
  not report content.

**Decisions taken (project lead, 2026-10-27):** P2-1 (C2, section 4.3.1) and P2-2 (tolerance, section 4.7).
The proposals that preceded them are superseded and not repeated here.

### 4.6 C2 effective-input audit after P2-1 (tokenizer only, validation)

`src/phase2/audit_c2_inputs.py` → `results/phase2_c2_input_audit_val.csv`. No model was run.

| | QaTa-COV19 (1429 C2 rows) | MosMedData+ (273) |
|---|---|---|
| Word check: C2 = C0 except the two substitutions | 1429 pass, 0 fail | 273 pass, 0 fail |
| Rows with only one substitution | 0 | 0 |
| LViT-T: window words differ from C0 only at the substituted words | **1429 / 1429 (100%)** | **273 / 273 (100%)** |
| C2 cuts more report words than C0: LViT-T / RecLMIS / DD-CMD | 0 / 0 / 0 | 0 / 0 / 0 |
| C2 cuts fewer report words than C0: LViT-T / RecLMIS / DD-CMD | 0 / 0 / 0 | 0 / 0 / 0 |

"Report words kept" = the report's letter-run words whose tokens all lie inside the method's window (LViT-T 23
BERT word pieces, RecLMIS 16 CLIP tokens, DD-CMD 22 BiomedBERT word pieces). C0 and C2 have the same words in
the same order. With P2-1, C2 keeps exactly the same report content as C0 in every method, and for LViT-T only
the two substituted words differ inside its window.

**RecLMIS C0 truncation** (answer to the project lead's question; validation):
- QaTa: all 272 truncated C0 texts lose report content, not only `EOF XXX` padding:
  - 171 lose only the final full stop;
  - **101 lose words**: 81 the final "lung", 12 "right lung", 7 "lower right lung", 1 "l right lung".
  - In training, 397 of 5716 QaTa texts lose words in the same way.
- MosMed: the 4 truncated C0 texts lose only padding.

This is an input limitation of RecLMIS (section 6).

### 4.7 Tolerance rule for the test C0 reconciliation (P2-2, fixed 2026-10-27, before any test run)

- **Rules.**
  - LViT-T and RecLMIS: the scored masks must be identical.
  - DD-CMD:
    - at most 10 differing mask pixels per case;
    - no case flips between an empty and a non-empty prediction;
    - per checkpoint, |Δ mean Dice| ≤ 0.0005.
  - No per-case |ΔDice| limit: on a small lesion, one pixel can change Dice by 0.0048 (validation, MosMed seed
    1002). That measures lesion size, not numerical noise. The maximum per-case |ΔDice| and |ΔHD95| are
    reported.
- **Validation check of the rule.** It holds on all 18 validation reconciliations (section 4.5). DD-CMD: no
  empty / non-empty flip, at most 4 pixels, |Δ mean Dice| ≤ 0.000021.
- **Outside the limits:**
  1. Stop, find the cause and record it here.
  2. A pipeline error (checkpoint, preprocessing, image–text pairing, cache): fix it, then rerun all test
     inference of that checkpoint once. Both runs are recorded in the test log.
  3. The same kind of GPU non-determinism, only larger: keep the results and report the exceedance. No rerun.
  4. Never rerun repeatedly until it "passes".
- **Paired differences.** All Phase 2 paired comparisons (C0 − Cx) use the C0 computed by Phase 2 itself. The
  archived Phase 1 evaluation is used only for this reconciliation.

### 4.8 Pass bookkeeping and process noise

- **C0 reuse.** The benchmark stage reuses the C0 pass of the text-path job.
  - LViT-T and RecLMIS: C0 comes from job 368. Their passes are bit-identical across processes (section 4.5).
  - DD-CMD: C0 comes from job 369, and its other conditions from later jobs, i.e. different processes. Its
    paired differences can therefore contain cross-process noise. The measured bound, from the Phase 1
    vs. Phase 2 reconciliation, is ≤ 4 pixels per case and ≤ 0.00002 mean Dice: far below the Phase 3 planning
    thresholds (1–2 Dice points). C0 is not rerun.
- **Pass log.** Every condition pass is recorded with its Slurm job id in
  `runs/phase2/<split>/<method>/<dataset>/seed<k>/passes.tsv`. The 18 validation C0 passes were backfilled
  from the job logs (368, 369).
- **Test stage.** All conditions of one checkpoint, C0 included, run in the same job where possible. A split
  into several jobs because of the 12 h limit is recorded.
- **Input check per pass** (benchmark stage).
  - At the start of each method × dataset, the effective-input audit is redone for the current manifest:
    `runs/phase2/<split>/effective_input_<method>_<dataset>.csv`. For C2 this replaces the D9 rows.
  - For every condition pass, the number of cases whose model input differs from C0 must equal the audit
    count for that pass and the number of cases whose raw text differs. Otherwise the job stops with an
    error.
  - The count is written to `passes.tsv` (`model_input_changed=n/N`).
  - CPU check on validation: for RecLMIS and DD-CMD every pass meets this (`test_phase2_infer.py`). For
    LViT-T it needs the BERT features and is checked in the job.

## 5. Analysis rules fixed so far

- **Main analysis.** As in PHASE2_PLAN Task 2.5:
  - 11 comparisons per method × dataset, Holm across the 11, α = 0.05;
  - the unit-weighted delta with a one-sample t-test over units;
  - common eligible image sets for each comparison.
- **MosMedData+ unit-level inference is exploratory** (addition 5). MosMed has 11 independent units on
  validation and 9 on test, one of them the pooled `bjorke` unit. Its CIs and p-values are reported, but
  every MosMed inferential result is labelled exploratory in this protocol, in the results tables and in the
  final report, whatever the p-value. For QaTa, inference is labelled exploratory when fewer than 20 units
  are available (D3).
- **Pre-specified sensitivity check for D6** (addition 1, fixed 2026-10-27, before any Phase 2 inference):
  - Exclude the MosMed images whose stated scope contradicts the sides named
    (`parse_note` contains "inconsistent: Unilateral but both sides named"; 14 val images).
  - Recompute the MosMed main paired differences: the 11 comparisons, image-weighted and unit-weighted
    deltas, per method.
  - On the remaining cases, donors are those already assigned; no new donors are drawn.
  - Descriptive only, no test. Reported next to the main analysis, for validation and test.

### 5.1 Phase 3 go / no-go rule (validation only)

**When it was fixed.** On 2026-10-27, by the project lead:
- **after** the C1 / C6 numbers of the 64-case validation subset had been seen (section 4.5);
- **before** any full-validation result of C1–C6.

It uses validation results only and never test results. Thresholds are on the [0, 1] Dice scale.

**Algorithm.** For each method × dataset:

1. **Per seed and comparison.** For each seed k ∈ {1001, 1002, 1003} and each comparison
   X ∈ {C0−C5, C0−C6, C1−C5, C1−C6}:
   1. Take the paired eligible cases: the targets eligible for the mismatch condition (C5 or C6). C0 and C1
      exist for every case.
   2. For each image, average the Dice of the mismatch condition over its donors (up to 3). C0 and C1 have
      one value per image.
   3. Compute the image difference d_i, e.g. D(C0) − mean over donors of D(C5).
   4. **Image-weighted difference** Δ_img(k, X) = the mean of d_i over images.
   5. **Unit-weighted difference** Δ_unit(k, X): average d_i within each unit (patient / CT volume, section 3,
      including the D1 / D2 units), then take the mean over units.
2. **Thresholds.** T(X) = 0.02 for C0−C5 and C0−C6; T(X) = 0.01 for C1−C5 and C1−C6.
3. **Classification of comparison X:**
   - **Repeated signal:** at least 2 of the 3 seeds have Δ_img(k, X) ≥ T(X) **and** Δ_unit(k, X) > 0
     (both in the same seed).
   - **Unstable signal** (descriptive only): at least 2 seeds have Δ_img(k, X) ≥ T(X), but fewer than 2
     seeds satisfy both conditions.
   - **No signal:** otherwise.
4. **The method × dataset** has a repeated signal if at least one of its four comparisons has one.
5. **Descriptive checks, reported but not part of the rule:**
   - Δ_img and Δ_unit for **each donor separately** (variants d1, d2, d3), to see whether one donor drives
     the signal;
   - the per-unit differences;
   - the per-seed values.
6. **Outcome.**
   - If at least one method × dataset has a repeated signal: there is a basis for Phase 3. Record which
     method × dataset and comparison.
   - If none has one: record **no-go**, and keep the negative result.
   - Whether Phase 3 actually starts is decided by the project lead and the supervisor. The decision is
     recorded in `configs/phase2_frozen/phase3_decision.json` before the freeze (section 2).
7. **MosMed.** Its unit-weighted values rest on 11 validation units and are exploratory (section 5).

## 6. Limitations recorded so far

- **C1 for LViT-T and RecLMIS** (D8, addition 3). The official padding turns `""` into `" EOF XXX"` × 9.
  - In MosMed training, the `EOF XXX` padding occurs only together with "No pulmonary infection, zero
    infected areas." (43 of 2183 training images; the only MosMed text with fewer than 9 words).
  - In QaTa training it occurs in 11 texts (short typo variants).
  - C1 for these two methods may therefore resemble a "no infection" report. It is reported, not changed.
- **"No pulmonary infection" texts vs. masks** (D7, addition 2). Counted only for annotation quality; no text
  attribute was changed.
  - Of the 47 MosMed train/val images with this text, 46 have lesion pixels in their common 224 × 224 mask:
    42 of 43 train and 4 of 4 val; median 11 and 9.5 lesion pixels; at most 52.
  - For comparison, only 1 of all 2456 MosMed train/val masks is empty.
- **D6 texts.** 420 MosMed train/val images say "Unilateral" while naming both sides. They are kept as
  written; see the sensitivity check in section 5.
- **LViT-T 10-word window.** The location clause is never fully inside the 10 kept word vectors. It is
  partly inside (27–31% of location words); the rest acts only through BERT's context.
- **C2 and the LViT-T window.** This was recorded 2026-10-27 for the superseded D9 template. **Resolved by
  P2-1:** the window now differs from C0 only at the two substituted words (section 4.6). The original note
  is kept below.
- *(Superseded D9 template)* **C2 for LViT-T changes the content of the 10-word window** (recorded
  2026-10-27, before any inference).
  The C2 template is longer than the original, so the first 10 BERT words differ in content, not only in
  wording. For example, the original window `bilateral pulmonary infection , two infected areas , middle left`
  becomes `bilateral lung infection , two areas of infection , in` (the location starts after word 10).
  For LViT-T, C0−C2 therefore also measures a change of what is inside the window. It is interpreted with
  this in mind. The effective-input audit records, for every case and condition, the actual words in the
  LViT-T window as produced by the tokenizer (not inferred) and reports separately how often C2 changes the
  window content.
- **RecLMIS truncation.** CLIP's context of 18 tokens cuts the end of some QaTa reports already under C0.
  - Validation: 272 of 1429 texts are cut. 171 lose only the final full stop; 101 lose words (81 "lung",
    12 "right lung", 7 "lower right lung", 1 "l right lung").
  - Training: 397 of 5716 lose words.
  - For these cases RecLMIS never sees the end of the location, whatever the condition. MosMed: no report
    content is cut.
- **Effective input.** Whether each change reaches the model is measured per case in the text-path audit,
  not assumed.
- **Grouping.**
  - QaTa patient identity rests on the BIDS `sub-S` naming (no separate patient list).
  - `covid_*` images (validation only) cannot be grouped.
  - MosMed `bjorke` slices are pooled into one unit.
- **MosMed split.** The known train/val/test volume overlap of the benchmark split is not repaired.
- **Exploratory MosMed inference.** MosMed unit-level inference is exploratory (section 5).
- **What the texts are.** They are benchmark annotations, not independently verified clinical reports.

## 7. Decision record

- **D1–D15.** As recommended in PHASE2_SETUP_PLAN.md section 9, confirmed by the project lead on 2026-10-27.
- **Additions of 2026-10-27:**
  1. D6 + the pre-specified sensitivity check (section 5);
  2. D7 mask count (section 6);
  3. D8 limitation (section 6);
  4. C2 review of 20 + 20 texts (section 4.3);
  5. MosMed unit-level inference stated as exploratory (sections 3 and 5, and the final report).
- **P2-1** (C2 = two word substitutions on the original; section 4.3.1) and **P2-2** (tolerance rule; section
  4.7), decided by the project lead on 2026-10-27 after the text-path check and before any C2 or test
  inference. Plus the bookkeeping records of section 4.8.
- **Phase 3 go / no-go rule** (section 5.1), decided by the project lead on 2026-10-27, after the 64-case
  subset C1 / C6 numbers and before any full-validation C1–C6 result. Plus the per-pass input check
  (section 4.8).

## 8. Files

- **Texts.**
  - `results/phase2_text_attributes.csv` (sha256 `5db47e6f…0e1866`).
  - `results/phase2_text_conditions.csv` (`a63500c4…c14b06`, after P2-1; before: `88979de9…c490`).
  - `results/phase2_text_conditions_donor_counts.csv` (`7040ee7e…690251`).
  - These contain the dataset's annotation texts. Like the raw data, they are kept out of git
    (`.gitignore`) and are reproduced from the code above.
- **Checkpoints.** `results/phase2_checkpoint_sha256.tsv`.
