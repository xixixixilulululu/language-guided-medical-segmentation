# Phase 2 — Setup checks and plan (before any Phase 2 run)

Status: 2026-10-27. Read-only checks for [PHASE2_PLAN.md](PHASE2_PLAN.md) (the supervisor's plan). Nothing has
been run on a GPU, no job was submitted, and no Phase 0/1 file was changed. **Only training and validation
texts were read.** The test texts (`Test_text.xlsx`) were not opened, and no new test result was read. For
the test split, only the sample ids of the split files were counted, for the grouping (section 3.3).

The numbers below come from read-only scripts in `src/phase2/setup_probes/`, which read train/val texts only:
- `parse_probe2.py`: parser;
- `elig_probe.py`: eligibility;
- `combo_probe.py`: changed-attribute combinations;
- `trunc_probe.py`: tokenizer-only truncation and empty input.

Checkpoint hashes are in `results/phase2_checkpoint_sha256.tsv`. **The decisions needed from the project lead
are collected in section 9.**

## 1. The 18 Phase 2 checkpoints

| Dataset | Method | Checkpoint (per seed 1001 / 1002 / 1003) | Size |
|---|---|---|---|
| QaTa-COV19 | LViT-T | `runs/lvit/QaTa-COV19/seed<k>/best_model.pt` | 153 MB |
| QaTa-COV19 | RecLMIS | `runs/reclmis/QaTa-COV19/seed<k>/best_model.pt` | 843 MB |
| QaTa-COV19 | DD-CMD | `runs/ddcmd/QaTa-COV19/seed<k>/checkpoints/best.ckpt` | 991 MB |
| MosMedData+ | LViT-T (**batch-24 training**, post-hoc sensitivity runs) | `runs/lvit_b24/MosMedData+/seed<k>/best_model.pt` | 153 MB |
| MosMedData+ | RecLMIS | `runs/reclmis/MosMedData+/seed<k>/best_model.pt` | 795 MB |
| MosMedData+ | DD-CMD | `runs/ddcmd/MosMedData+/seed<k>/checkpoints/best.ckpt` | 991 MB |

- **Files and hashes.** All 18 files exist. Their sha256 are in `results/phase2_checkpoint_sha256.tsv`.
- **Epoch check.** The epoch stored in each checkpoint equals the best epoch in `training_summary.json`.
  DD-CMD stores Lightning's 0-based epoch, i.e. one less; its stored `best_model_score` equals the best
  validation Dice. The weights were only loaded on the CPU to read this metadata; the models were not run.
- **Archived validation outputs** exist for all 18 runs and are used for the C0 reconciliation:
  - `eval_val/per_case.csv`, with 1429 QaTa and 273 MosMed rows;
  - `eval_val/masks/`, the scored binary masks;
  - `val_probs.npz`, float16.
- **Settings to reproduce.** Inference settings from the frozen configs:
  - LViT-T: batch 32, `deterministic: true`;
  - RecLMIS: batch 32;
  - DD-CMD: batch 32;
  - all three: no TTA, threshold `probability > 0.5`, common 224×224 targets, empty predictions scored as
    HD95 = ASSD = 316.78 px.

  The training batch size of MosMed LViT-T (24) does not change its inference batch size (32).
- **Provenance.** The provenance of the MosMed LViT-T checkpoints (post-hoc sensitivity study) will be
  recorded in `PHASE2_PROTOCOL.md`.

## 2. Actual text processing of the three methods

| | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| Padding rule before tokenizing | official `read_text`: a text with < 9 whitespace words gets `" EOF XXX"` × (9 − n) | same official rule | none |
| Tokenizer | BERT-base uncased word pieces (HF `google-bert/bert-base-uncased`; PyTorch re-implementation of `bert_embedding`) | OpenAI CLIP BPE (`clip.tokenize`) | BiomedBERT word pieces (HF) |
| Maximum length | 25 incl. [CLS]/[SEP] (23 word pieces) | `context_length` 18 incl. SOT/EOT (16 BPE tokens) | 24 incl. [CLS]/[SEP] (22 word pieces) |
| Truncation | cut after 23 word pieces | `truncate=True`: cut at 18, last token replaced by EOT | cut after 22 word pieces |
| Padding | [PAD] to 25 | zeros to 18; mask = token ≠ 0 | [PAD] to 24, attention mask |
| What the segmentation model receives | the **first 10 word vectors** (word pieces averaged into words), last BERT layer, 10 × 768, precomputed cache | all 18 CLIP token features with the mask | last-layer hidden states of all 24 positions (the decoders receive no mask) |
| Encoder | frozen (features precomputed) | frozen CLIP text tower | frozen BiomedBERT |

**Truncation estimate** (train + val texts, tokenizer only):

| Dataset | LViT-T: location words inside the 10 kept vectors | RecLMIS: texts truncated at 16 BPE tokens | DD-CMD: texts truncated |
|---|---|---|---|
| QaTa-COV19 (7145 images) | 26.9% of location words; **the location clause is never fully inside** (always partly) | 18.6% (location only partly kept); 81.4% fully kept | 0% |
| MosMedData+ (2456 images) | 30.8%; never fully inside (always partly) | 1.9% | 0% |

- **LViT-T** keeps the first 10 BERT "words". Punctuation counts as a word, so these 10 words are typically
  `bilateral pulmonary infection , two infected areas , all left`: scope, count and the first one or two
  words of the location.
  - **Scope and count changes are positionally inside the window. Location changes mostly lie after
    word 10.** They are therefore not visible as words of their own.
  - They are not necessarily erased: BERT is bidirectional, and each of the 10 kept vectors is computed
    from the whole sentence (up to 25 tokens).
  - Whether such a change survives is measured in the effective-input audit (feature difference per case,
    Task 2.2), not assumed.
- **RecLMIS on QaTa** cuts the end of the location (usually the right-lung part) in 18.6% of the texts. A
  change only in that part does not reach the model.
- **DD-CMD** sees the full text.
- **Cache keys.** The current LViT cache is keyed by the padded text string. Phase 2 needs new features for
  every counterfactual text; the Phase 2 cache key will include the text and the encoder settings (model id,
  max length 25, 10 words).
  - The existing Phase 1 cache also holds test texts as keys. Phase 2 code will only look texts up in it and
    never list its keys before the freeze (section 8).

## 3. Text attributes (train + val only)

### 3.1 What the texts look like

The texts are templates: `<Scope> pulmonary infection, <count> infected area(s), <location>.`

| | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Images (train / val) | 5716 / 1429 | 2183 / 273 |
| Distinct texts (train / val; train+val) | 273 / 128; 301 | 101 / 45; 110 |
| Words per text (min / median / max) | 8–17, median 13 | 6–13, median 12 (train) / 9 (val) |

### 3.2 Attribute meaning, values and parsing coverage

Frequencies are images in train + val, from the parser with the normalisation described below.


| Attribute | QaTa-COV19 | MosMedData+ |
|---|---|---|
| `scope` | the stated laterality word: Unilateral (1478) / Bilateral (5641). No other type or severity word occurs. | Unilateral (1234) / Bilateral (1175) / **No** (47: "No pulmonary infection, zero infected areas.") |
| `count` | number of infected areas: one 1436, two 5254, three 364, four 58, five 6, six 1 | zero 47, one 908, two 460, three 368, four 258, five 148, six 100, seven 62, eight 38, nine 67 |
| `location` | zones per side: subsets of {upper, middle, lower} or "all", for the left and/or right lung | one zone word per side (upper / middle / lower / all); the left side may have no zone ("left lung and middle right lung"); no location for "No" texts |

- **Proposed scope definition.** `scope` = the stated laterality/presence word only. Infection type is
  always "pulmonary infection", and no severity word occurs, so nothing else is mixed in.
- **Dependence between attributes.** In both datasets, scope is largely determined by which sides the
  location names. In QaTa, count also follows scope (bilateral ↔ two areas in most texts).
- **Coverage.** A fixed normalisation was used, all of it in `parse_probe2.py`:
  - spaces and commas, a trailing double full stop, and the elision "lower left and lower right lung";
  - a fixed table of 26 unambiguous typos (e.g. `lowerl`, `midldle`, `alll`, `thrr`).

  Everything ambiguous stays unparsed, i.e. `unknown`: `higher`, `two one infected area`, `all lung`,
  `middle right left lung`, `lower ll left`, …

  | | QaTa-COV19 | MosMedData+ |
  |---|---|---|
  | Images parsed (train+val) | 7119 / 7145 = **99.64%** (26 unknown) | 2456 / 2456 = **100%** |
  | Images parsed (val) | 1423 / 1429 | 273 / 273 |
  | Stated scope inconsistent with the sides named | 13 (4 Unilateral with two sides, 9 Bilateral with one side) | **420**: all "Unilateral …, left lung and <zone> right lung" |

  Without the normalisation, QaTa coverage would be 96.7% (236 images fail).

### 3.3 Grouping (patient / CT volume)

| | QaTa-COV19 | MosMedData+ |
|---|---|---|
| Id formats | `sub-S<subject>_ses-E<session>_…` (BIDS naming of BIMCV-COVID19+) and `covid_<n>` | `Morozov_study_<n>_<slice>`, `Jun_coronacases_case<n>_<slice>`, `Jun_radiopaedia_…_case<n>_<slice>`, `bjorke_<n>` |
| Group id that can be read | subject `sub-S…` | volume = name without the final `_<slice>`; `bjorke_<n>` has none (pooled source, Phase 0 audit) |
| Train | 3362 images in 1493 subjects; **2354 `covid_*` images without id** | 2103 slices in 58 volumes; 80 `bjorke` |
| Val | 832 images in 597 subjects (max 7 images per subject; 145 subjects with > 1 session); **597 `covid_*` without id** | 263 slices in **10 volumes** (4–111 slices each); 10 `bjorke` |
| Test (sample ids only) | **2113 images, all `sub-S…`, 474 subjects** | 263 slices in **8 volumes** (1–135 slices) + 10 `bjorke` |

What can and cannot be verified:
- **QaTa.** The `sub-S` subject id follows the documented BIDS naming of BIMCV-COVID19+, and the same subject
  appears with several sessions. That `sub-S` = one patient therefore rests on that naming convention; there
  is no separate patient list. The `covid_*` images (42% of val, none in test) carry no patient id. Whether
  several of them belong to one patient cannot be verified.
- **MosMed.** The volume rule is the documented Phase 0 source mapping. `bjorke` slices cannot be assigned to
  volumes.
- **Small number of units.** MosMed has only about 10–11 (val) and 8–9 (test) independent units. Unit-level
  inference on MosMed will be exploratory in any case (PHASE2_PLAN, Task 2.5).
- **Known split overlap.** The train/val/test volume overlap of the MosMed split (Phase 0 audit) remains a
  stated limitation.

## 4. Eligible cases on the validation split

Donors: same split, a different case and a different text, all three attributes known for target and donor.
Donors from the target's own subject or volume are excluded where the id is known.

| Condition | QaTa-COV19 val (1429) | MosMedData+ val (273) |
|---|---|---|
| C0, C1 | 1429 | 273 |
| C2 (rewrite from parsed values) | 1423 | 273 |
| C3a / C3b / C3c (explicit deletion) | 1423 / 1423 / 1423 | 273 / 273 / 269 (no location in "No" texts); C3a not possible for "No" texts (section 9, D10) |
| C4 (1 attribute differs), ≥ 1 / ≥ 3 distinct donor texts | 1422 / 1421 | 269 / 254 |
| C5 (2 attributes), ≥ 1 / ≥ 3 | 1423 / 1423 | 269 / 269 |
| **C6 (all 3 attributes)**, ≥ 1 / ≥ 3 | **1423 / 1423** (median 16 distinct donor texts) | **273 / 273** (median 23) |
| C4 ∩ C5 ∩ C6 (Task 2.6) | 1422 | 269 |

- These counts treat cases without a group id (QaTa `covid_*`, MosMed `bjorke`) as eligible, excluding only
  the case itself.
- If those cases are excluded instead (decision D1/D2), the QaTa counts become 831 for C4–C6 and the MosMed
  counts 259–263.
- Treating `all` as equal to `upper middle lower` changes the counts by at most 1.

**Which attributes differ, in practice** (validation, target × distinct donor text):

| | QaTa-COV19 | MosMedData+ |
|---|---|---|
| C4 (1 attribute) | location only: 98.7% of pairs; count only: 1.3%; **scope only: ≈ 0%** (9 targets) | location only 58%, count only 42%, scope only 0% |
| C5 (2 attributes) | count + location 85.7%, scope + location 13.4%, scope + count 0.9% | count + location 83.6%, scope + location 16.4% |

- **C4 in QaTa is in practice a location-only change.** Because scope and location are dependent, a
  scope-only mismatch almost never exists.
- With the LViT-T 10-word window (section 2), a location-only change is mostly outside the kept word
  positions. This is a property of the benchmark text and pipeline, to be reported, not changed.

## 5. C1 (empty text) as an input

The empty string `""` is a structurally valid input for all three methods through their normal pathway
(tokenizer level; the encoder output will be checked in the text-path check):

| Method | `""` after the normal processing |
|---|---|
| LViT-T | official padding → `" EOF XXX"` × 9 → 18 BERT words `eof xxx …` → 10 word vectors. Valid. Without the padding there would be 0 words, which is invalid (the model needs exactly 10). |
| RecLMIS | official padding → `EOF XXX` × 9 → 18 CLIP tokens `[SOT eof xxx … EOT]`. Valid. |
| DD-CMD | `[CLS] [SEP]` + 22 × [PAD], attention mask `1 1 0 …`. Valid. |

- **Caveat** (decision D8): for LViT-T and RecLMIS, the official padding turns the empty text into a string
  that occurs in training. In MosMed training, the `EOF XXX` padding occurs **only** in the 43 texts
  "No pulmonary infection, zero infected areas." (6 words). In QaTa it occurs in 11 texts.
- The suggested neutral fallback ("No case-specific clinical information is available.", 7 words) would also
  be padded with `EOF XXX` × 2, so it does not avoid this.
- The plan prefers the empty string when it is valid, which it is. The overlap will be reported as a
  limitation of C1 for these two methods.

## 6. Compute, time and jobs

**Inference cost.** The Phase 1 final evaluation (inference + all four metrics) took 28–39 s for the 2113
QaTa test images and about 5 s for 273 MosMed images, per checkpoint, i.e. about 15–20 ms per image. Per
checkpoint, Phase 2 needs 16 passes over the split:

| Passes | Count |
|---|---:|
| C0 | 1 |
| C0 repeat (noise reference) | 1 |
| C1, C2, C3a, C3b, C3c | 5 |
| C4, C5, C6 × 3 donors | 9 |
| **Total** | **16** |

| Stage | Images × passes × checkpoints | Estimate |
|---|---|---|
| Text-path check (fixed val subset, e.g. 64 cases, 18 checkpoints) | small | < 30 min |
| Validation, QaTa | 1429 × 16 × 9 | about 1–1.5 h (+ model loading, text encoding) |
| Validation, MosMed | 273 × 16 × 9 | about 15–20 min |
| Test (after the freeze), QaTa | 2113 × 16 × 9 | about 1.5–2.5 h |
| Test (after the freeze), MosMed | 273 × 16 × 9 | about 15–20 min |

- **Text features.** LViT-T's new texts need BERT features (a few thousand distinct texts, minutes on the
  GPU). RecLMIS and DD-CMD encode the text in their own forward pass.
- **Storage.** float32 probabilities for every validation pass would take about 41 GB (QaTa) + 8 GB (MosMed)
  uncompressed. 16 TB are free; what to keep is decision D13. Scored masks are stored as packed bits, plus
  per-case metrics.
- **Jobs** (cluster rules: submitted by hand, 1 running + 1 waiting, 12 h each):
  1. text-path check;
  2. validation QaTa;
  3. validation MosMed;
  4. after the freeze: test QaTa;
  5. after the freeze: test MosMed.

  Each job is far below 12 h. The driver will still be resumable per (checkpoint, condition), and every job
  sends the ntfy end notification.
- **Code.** The jobs use the frozen Phase 1 drivers' model-building and `predict` functions, imported and not
  modified.

## 7. Statistical analysis program (design)

`src/analyze_phase2.py`. It does not change any Phase 1 statistics.

**Inputs.** One per-case table: dataset, split, method, checkpoint id, seed, sample, condition, variant/donor,
Dice, mIoU, HD95, ASSD, empty flag.

**Aggregation.** Metrics are averaged over donors within image and seed, then over the 3 seeds within the
image. Probability maps are never averaged.

**Main table** (`phase2_main_results.csv`, val and test separate):
- image-weighted mean Dice ± SD over the 3 per-seed image means;
- eligible image counts and unit counts;
- NA for unavailable conditions.

**11 comparisons per method × dataset**: C0−C1, C0−C2, C0−C3a/b/c, C0−C4/5/6, C1−C4/5/6. For each:
1. the common eligible image set (e.g. C0 vs. C6 on C6-eligible cases);
2. the image-level paired difference;
3. the mean within unit (patient / volume, section 3.3), then `unit_weighted_delta` = the mean over units;
4. a one-sample t-test of the unit differences, with the unadjusted 95% t-CI, raw p, and Holm p across the 11
   comparisons (α = 0.05; an unavailable test enters Holm as p = 1 but is shown as NA);
5. the image-weighted delta reported next to it, without p or CI;
6. an exploratory flag when the units are few (decision D3).

**Supplementary.**
- mIoU / HD95 / ASSD: unit-weighted differences with a paired unit bootstrap (10,000 resamples,
  `default_rng(0)`).
- Mask-change frequency and absolute per-image C0−C2 change.
- Empty-prediction rate and per-seed results.

**Task 2.6.** C4/C5/C6 on their common eligible set, with C0 on the same set and the paired harms. The
changed-attribute combinations are broken down descriptively.

**Task 2.7.** Harm H = D(C0) − D(C6) per model on the common C6 set, three model pairs, unit-weighted mean
differences with 95% t-CI over units. Shown next to C0, C0−C1 and C1−C6.

**When it is fixed.**
1. The program is written and tested on synthetic data and run on the validation results. Validation is the
   development set: rules may still change there, and every change is recorded.
2. After the validation benchmark and the Phase 3 decision (validation only), the Phase 2 rules are frozen
   together with this program (and the Phase 3 protocol or its cancellation): text parser, condition
   generator, donor salt, inference driver, analysis and `PHASE2_PROTOCOL.md`. The record is pushed to GitHub.
3. Only then can test inference with changed texts run (section 8).

## 8. Protection of the test split (Phase 2 lock)

The same pattern as the Phase 1 program lock (`src/phase1/analysis_lock.py`):

- **Record.** `configs/phase2_frozen/phase2_lock.json` holds the commit and the sha256 of every file that
  defines Phase 2:
  - the parser and the condition generator, with the donor salt;
  - the inference driver and `analyze_phase2.py`;
  - `PHASE2_PROTOCOL.md`;
  - the Phase 3 decision record (`configs/phase2_frozen/phase3_decision.json`: protocol frozen, or
    cancelled).
- **Open only if all hold:**
  1. the record exists;
  2. every recorded file has its recorded sha256 now;
  3. the recorded commit is an ancestor of `origin/main` after `git fetch`. On a compute node without
     network, the local `origin/main` ref is used, as in Phase 1, and the fetch result is logged.
  4. in that commit, every file has the recorded sha256;
  5. the Phase 3 decision record exists and is included.
- **Gates.**
  - (a) The condition generator refuses `--split test` while the lock is closed. It never opens
    `Test_text.xlsx` before the check, so test donors cannot be generated, or even read, early.
  - (b) The inference driver refuses `--split test` while the lock is closed (decision D14: for all
    conditions, or only C1–C6). The exit code is 5, as in Phase 1, and the check runs before any image, text
    or checkpoint is loaded.
  - (c) Every test run writes start and end lines to `results/phase2_test_eval_log.tsv`: job id, time,
    condition, lock commit and fetch status.
- **Second, independent lock.** Phase 2 reads texts through a text view `~/datasets/phase2/text_view/` that
  holds only `Train_text.xlsx` and `Val_text.xlsx` until a `link-test` command (allowed only when the lock is
  open) adds `Test_text.xlsx`. This is like the Phase 1 data view.
- **Tests** (`src/phase2/test_phase2_lock.py`, run on the login node, no GPU, no test data). Each case runs in
  a temporary git repository with a bare "remote":
  1. no record → closed;
  2. a recorded file changed → closed;
  3. commit not pushed → closed;
  4. Phase 3 decision record missing → closed;
  5. all conditions met → open.

  Driver-level tests:
  - the generator and the inference entry with `--split test` exit with code 5 while closed, and a
    monkey-patched `read_excel` / image loader proves that no test file was opened;
  - `--split val` works without the lock;
  - `link-test` refuses while closed.

## 9. Decisions for the project lead

| # | Question | Recommendation |
|---|---|---|
| D1 | QaTa `covid_*` images (597 val, 0 test) have no patient id. | Keep them as targets and donors (self excluded only). In the unit analysis, count each as its own unit, flagged "patient unverified". Test is unaffected (all `sub-S`). |
| D2 | MosMed `bjorke` slices (10 val, 10 test) have no volume id. | One pooled unit `bjorke` (conservative); donors from `bjorke` are excluded for `bjorke` targets. |
| D3 | When is inference "exploratory"? | Always for MosMed (≤ 11 units). For QaTa, when < 20 units. |
| D4 | Is `all <side> lung` equal to `upper middle lower <side> lung`? | Treat as equal canonical location (same meaning); the effect on eligibility is ≤ 1 case. |
| D5 | Accept the fixed typo table (26 unambiguous corrections); ambiguous words (`higher`, `two one`, `all lung`, …) → unknown? | Yes. |
| D6 | MosMed "Unilateral … left lung and <zone> right lung" (420 train/val images): stated scope contradicts the two sides named. | Keep the stated values (scope Unilateral; location left: unspecified, right: <zone>) and flag `parse_note`. They count as known. Alternative: scope unknown. |
| D7 | MosMed "No pulmonary infection, zero infected areas." (47): location absent. | location = known value `none`; usable as target and donor (a real report). C3c is unavailable for these. |
| D8 | C1 = empty string for all three methods? For LViT-T/RecLMIS it becomes `EOF XXX` × 9, which in MosMed training co-occurs only with "No infection" texts. | Yes, empty string (valid input, as the plan prefers), with this overlap reported as a limitation. |
| D9 | C2 template | Keep the order scope → count → location and only reword, e.g. `<Scope> lung infection, <count> area(s) of infection, in the <zones> part of the left lung and the <zones> part of the right lung.` Rewritten from parsed values; typos are not carried over. Unparsed → unavailable. |
| D10 | C3 deletions | C3a: delete the scope word only (`Pulmonary infection, two infected areas, …`); **unavailable for "No" texts**, because deleting "No" would invert the meaning. C3b: delete the whole count clause. C3c: delete the whole location clause. Residual cues are recorded (e.g. two sides still imply bilateral). |
| D11 | Donors for targets without a group id | Exclude only the case itself (D1); report that same-patient donors cannot be ruled out there. |
| D12 | Donor salt | A fixed string, e.g. `phase2-donor-v1`, recorded in the protocol before any run. |
| D13 | What to store | Validation: float32 probabilities for C0, C1, C4–C6 (needed by a possible Phase 3), compressed. All conditions: packed binary masks and per-case metrics. Test: masks and metrics only. |
| D14 | The lock covers which test runs? | **All** Phase 2 test inference, including C0 (stricter than required, simpler to verify), plus the test texts (text view). |
| D15 | Numerical tolerance for the C0 reconciliation | Fixed after the repeat-inference check on validation, before any test run. Phase 0 showed that GPU inference is not bit-identical (≤ 10 pixels per case); the rule will be stated in pixels and in Dice. |

## 10. Next steps after confirmation

1. Write the parser, condition generator and Phase 2 lock with their tests. These run on the login node, on
   train/val texts only.
2. Write `results/phase2_text_attributes.csv` and `results/phase2_text_conditions.csv` for train/val, and
   `PHASE2_PROTOCOL.md` (draft).
3. Write the inference driver.
4. Plan the first job (text-path check). It is submitted by you.
