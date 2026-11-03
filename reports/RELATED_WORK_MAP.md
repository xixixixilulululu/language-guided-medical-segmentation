# Related work and possible paper directions (for discussion)

Status: 2026-11-02 (second version, more papers read). A literature note to support the choice of the next
step. It is **not** a claim that a gap exists.

**How it was made:**
- web searches on 2026-11-02;
- the papers in section 7, each read to the depth given in the "Read" column of section 2;
- our numbers, all from [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

**Caveats:**
- "Not found" means "not found in these searches", not "does not exist".
- Two of the closest papers appeared in October 2026, so the search must be repeated before any submission.
- Numbers quoted from other papers come from their own settings and are not comparable with ours.

## 1. Summary for a quick read

1. **Wrong reports are rarely tested.**
   - Among the text-guided segmentation papers read in full, wrong text at test time appears in only two:
     - RADAR: its own single model, with no statistics;
     - Text Branch Sensitivity: other datasets and random swaps.
   - The others read in full test the correct report, missing or incomplete text, or mismatched text during
     training only (RELiF).
   - The abstracts of C2Seg, BiCLIP and MRSeg do not mention wrong text.
2. **The clinical motivation is documented.** Left/right errors occur in real radiology reports:
   - 1,607 corrected laterality errors in 2,923,094 reports (0.055%) [Luetmer 2013];
   - 1.20% of 10,000 real-world reports, found with a traditional detection approach enhanced by GPT-4o
     [Weng 2025];
   - 70.9% of addendum-corrected side errors graded clinically important [Sangwaiya 2009].
   - In our Phase 2, the location is exactly the part of the report whose error causes the harm.
3. **Text overrides the image in clinical vision-language models.**
   - In chest X-ray classification, swapping the report for one with the opposite label dropped accuracy to
     0.17–0.34 (e.g. GPT-5 0.83 → 0.18), while swapping the image barely changed it (0.83 → 0.82) [Restrepo
     2026].
   - With the report available, a model's answer changed in only 4.26% of image swaps, against 20.94% without
     it [ModaLens 2026].
   - Our Phase 2 shows the same tendency in segmentation: on QaTa-COV19, a fully mismatched report is worse
     than no report.
4. **Some new methods tie the mask to the location words directly.**
   - SSA turns direction words ("left", "lower") into a spatial prior mask. Its experiments use only correct
     text.
   - C2Seg aims to make masks follow the text more closely. Its abstract does not mention wrong text.
5. **Existing mitigations do not cover wrong reports.**
   - Missing text: ProLearn. Incomplete text: ARSeg. An internal text weight that never reaches zero: RELiF.
   - Image-level abstention (selective prediction) for segmentation: Soft Dice Confidence.
   - A post-hoc fallback to an image-only model, evaluated under wrong reports, was not found.
6. **Suggestion.** A benchmark-and-analysis paper (Option A, section 5) can be written from results we already
   have. A cheap addition, a left/right-swap condition (section 4, D7), would answer the most likely reviewer
   question about realistic errors.

## 2. Evidence matrix

✓ = done; ✗ = not done; ~ = partly; ? = not reported in what was read. "Wrong text" means a wrong report given at
test time.

| Work | Read | Wrong text at test | Which attribute is wrong is controlled | ≥ 3 models | QaTa / MosMed | CIs or tests | Mitigation for wrong text | Fallback to an image-only model |
|---|---|---|---|---|---|---|---|---|
| RADAR (2026) | appendix D.1 | ✓ (one undefined "Wrong" condition) | ~ (incomplete prompts only) | ✗ (one) | ✓ | ✗ | ✗ | ✗ |
| Text Branch Sensitivity (2026) | full text | ✓ (random swap, contradictory) | ~ (missing location only) | ✓ (5 encoders) | ✗ | ? | ✗ | ✗ |
| RELiF (2026) | full text | ✗ (mismatch in training only) | ✗ | ✗ | ✓ | ? | ~ (text weight 0.5–1.5, never zero) | ✗ |
| ProLearn (ICCV 2025) | full text | ✗ (missing text only) | ✗ | n/a | ✓ | ? | ✗ | n/a (text-free model) |
| ARSeg (MICCAI 2025) | page and reviews | ✗ (missing attributes only) | ~ (missing, not wrong) | n/a | ✓ | ? | ✗ | ✗ |
| SSA (BIBM 2025) | full text | ✗ | ✗ | n/a | ✓ | ? | ✗ | ✗ |
| C2Seg (ICLR 2026) | abstract | ? | ? | ? | ? | ? | ? | ? |
| BiCLIP (2026) | abstract | ? (its robustness claims concern image degradation) | ? | n/a | ✓ | ? | ? | ? |
| **Ours (Phases 2–3)** | — | ✓ (real reports of other cases) | ✓ (exactly 1 / 2 / 3 of scope, count, location; deletions) | ✓ (3 models × 3 seeds) | ✓ | ✓ (patient-level, Holm, frozen protocol) | ~ (post-hoc gate: 9/9 significant, 11–33% of headroom) | ✓ (nnU-Net) |

## 3. The landscape by theme

**A. Text-guided segmentation methods on QaTa-COV19 / MosMedData+.**
- Methods: LViT, RecLMIS and DD-CMD (reproduced in our Phase 1), and newer ones (SSA, RELiF, RADAR, BiCLIP,
  MRSeg, C2Seg).
- In the papers read in full, the main results use the paired report.
- Some report high Dice, e.g. SSA 91.65 on QaTa-COV19 and 80.37 on MosMedData+.
- Our Phase 1 found that under one common evaluator the gain over nnU-Net is 2–5 points on QaTa-COV19 and is
  not consistent on MosMedData+.

**B. Analyses of text reliance.**
- Not all fine-tuned vision-language segmentation models use the prompt [Poudel 2024].
- Text sensitivity depends on the dataset, and text acts mainly through global semantics. For example,
  MedCLIP on BTMRI drops from 91.61 to 40.76 without text, and missing location matters on BUSI but hardly on
  BTMRI [Text Branch Sensitivity 2026].
- On QaTa-COV19 our results show a different pattern: the location is what matters.

**C. Mitigations.**
- Text-free inference [ProLearn].
- Incomplete attributes [ARSeg].
- A text-reliability weight inside the network [RELiF].
- Image-level selective prediction for segmentation, which abstains on low-confidence images [Soft Dice
  Confidence 2026].
- None is evaluated under wrong reports.

**D. Report–image consistency.**
- Report error detection for chest X-rays [CorBenchX 2025; ReXErr-v1 2025; phrase-grounded fact-checking,
  MICCAI 2025].
- Location verification with anatomy masks, used only to build a dataset [MIMIC-ILS, CVPR 2026].
- Not connected to deciding how far a segmentation model should trust its report.

**E. Text dominance in clinical vision-language models.**
- Predictions follow a swapped report rather than the image [Restrepo 2026].
- Image sensitivity falls when the report is available [ModaLens 2026].
- These are classification and VQA studies. In segmentation, the closest evidence is RADAR and Text Branch
  Sensitivity (section 2).

**F. Clinical motivation: real location errors.**
- Laterality errors corrected by addendum: 0.008% of 1,065,322 reports, 70.9% of them clinically important;
  in a one-month sample, 0.26% of reports had an uncorrected side error [Sangwaiya 2009].
- 0.055% corrected and 0.0019% major in 2,923,094 reports [Luetmer 2013].
- 1.20% of 10,000 reports [Weng 2025].
- The rates depend strongly on how errors are found.

## 4. Directions, with the evidence we already have

| | Direction | What we have | What is missing |
|---|---|---|---|
| D1 | A controlled, multi-model benchmark of report errors for report-guided segmentation | Complete: 3 models × 3 seeds × 2 datasets × 9 conditions; patient-level statistics; frozen protocol; validation and test agree (31/33 same sign) | Repeat the literature search |
| D2 | Which part of the report drives the harm, and what the models actually read | Location-only change 12.6–13.8 points vs count or scope 0.1–1.1; deleting the location 11.43–21.76; LViT-T reads 10 word vectors; RecLMIS cuts 101 of 1429 validation reports | — |
| D3 | A wrong report is worse than no report | C1 − C6 +3.12 / +7.60 / +11.39 (all significant). RADAR agrees on QaTa (wrong 70.33 < empty 76.90), and on MosMed both show the reverse | The hypothesis that stronger text–mask coupling (SSA, C2Seg) increases the harm is **untested** |
| D4 | A model-agnostic fallback to an image-only model | Post-hoc gate: 9/9 significant gains, cost 0.17–0.30 under the correct report; limits measured (11–33% of headroom; nnU-Net alone better when the report is wrong; AUROC 0.69–0.77 on GATE_CAL) | A better detector (D5) |
| D5 | Check the report against the image before trusting it (new method) | Nothing yet | A location signal from the image (e.g. lung-zone anatomy masks, as in MIMIC-ILS); a validation test; a fresh test protocol |
| D6 | Strong image-only baselines under one evaluator (supporting) | Phase 0–1: LViT Table II's U-Net on MosMedData+ is 9.4 points below our reproduction; gains over nnU-Net 2–5 points on QaTa | Not a paper on its own |
| **D7** | **A realistic error type: swap left and right in the original report** | The parser already reads scope, count and location; Phase 2 shows the location drives the harm | A new condition: free on validation; on test it needs a pre-registered addendum (supervisor's decision) |

## 5. A concrete paper sketch (Option A, for discussion)

**Possible titles:**
- *When the Report Is Wrong: A Counterfactual Audit of Language-Guided Lesion Segmentation*
- *Wrong Reports Hurt More Than No Reports in Language-Guided Segmentation*

**Research questions** (all answered by existing results, except the left/right part of RQ2):
- RQ1. How much do frozen language-guided models depend on the paired report? (C0 vs C1)
- RQ2. What does a wrong report do, and which part of it matters? (C4–C6, attribute breakdown, C3; D7 adds
  left/right swaps)
- RQ3. Is a wrong report worse than no report? (C1 vs C4–C6)
- RQ4. Can a simple post-hoc fallback to an image-only model limit the harm, and what limits it? (Phase 3)

**Contributions:**
1. A controlled counterfactual protocol: real reports of other cases with exactly 1 / 2 / 3 attributes changed,
   and deletions and rewording. It is frozen and evaluated once on a locked test set.
2. Findings on three models and two datasets:
   - on QaTa, wrong reports are worse than no report;
   - the location drives the harm;
   - rewording alone hurts one model;
   - the effective-input audit.
3. A model-agnostic fallback baseline with an honest account of its limits. The bottleneck is detecting the
   wrong report.

**Existing material:**
- Main table: Phase 2 test Dice under nine conditions.
- Figures: mismatch extent; the attribute breakdown; Phase 3 Dice by condition; real cases chosen by a fixed rule.
- Gate table: LANG / gates / IMG / ORACLE.

**Likely reviewer questions and our answers:**

| Question | Answer now | What would strengthen it |
|---|---|---|
| Are swapped reports realistic? | Laterality errors are documented clinically (section 3F); our location-only swaps approximate them | D7: a left/right swap condition |
| Only three models? | They are three different architectures, reproduced under one evaluator | Optionally one recent model with public code |
| Why not just use nnU-Net? | Under the correct report it is 2–5 points worse on QaTa; the gate is for the case where report quality is unknown | D5 as future work |
| MosMedData+ is small | Labelled exploratory throughout (9 units); its slice-level split is stated | — |
| The texts are templated | Stated, with the effective-input audit | — |

**Option B** adds D5 as a method. It needs validation experiments and then a fresh test protocol, and its effect
is unknown.

**Timeline** (MICCAI 2027: Auckland, 26 Sep–1 Oct 2027; the deadline is not announced, and the 2026 deadline was
26 Feb):

| When | Step |
|---|---|
| November | Choose the direction |
| December | Optional additions: D7 on validation, then a pre-registered test addendum if approved |
| January | Writing |
| February | Internal review and submission |

ISBI 2027 has passed. MIDL 2027 dates are not announced.

## 6. Practical points

- **Public repository.** The repository is public and carries the author's name. For a double-blind venue,
  decide before submission whether it stays public.
- **The QaTa test set has been used once.** Any new method or condition evaluated on it should be frozen first,
  and this should be stated in the paper.

## 7. References (read on 2026-11-02)

- RADAR: Gui et al., [arXiv 2610.14944](https://arxiv.org/html/2610.14944) (Oct 2026)
- Text Branch Sensitivity: Liu, Zhu, Shi, [arXiv 2609.02663](https://arxiv.org/abs/2609.02663) (Sep 2026)
- RELiF: Talaei et al., [arXiv 2610.28118](https://arxiv.org/html/2610.28118) (Oct 2026)
- ProLearn: Ye et al., ICCV 2025, [arXiv 2507.11055](https://arxiv.org/abs/2507.11055)
- ARSeg: Wang, Lin, Yan, [MICCAI 2025](https://papers.miccai.org/miccai-2025/0944-Paper3688.html)
- SSA: Liao, Geng, Liu, BIBM 2025, [arXiv 2512.22981](https://arxiv.org/html/2512.22981v1)
- C2Seg: Gui et al., [ICLR 2026](https://mlanthology.org/iclr/2026/gui2026iclr-textmask/)
- BiCLIP: Talaei et al., [arXiv 2603.00156](https://arxiv.org/abs/2603.00156)
- MRSeg: Rahman et al., MICCAI 2026, [arXiv 2609.28860](https://arxiv.org/abs/2609.28860)
- Poudel et al., [MIDL 2024](https://proceedings.mlr.press/v250/poudel24a.html)
- MIMIC-ILS / ROSALIA: Choi et al., CVPR 2026, [arXiv 2511.15186](https://arxiv.org/html/2511.15186)
- CorBenchX, [arXiv 2505.12057](https://arxiv.org/abs/2505.12057); ReXErr-v1, [PhysioNet](https://physionet.org/content/rexerr-v1/1.0.0/); phrase-grounded fact-checking, [MICCAI 2025](https://papers.miccai.org/miccai-2025/0693-Paper3526.html)
- Restrepo et al., "Medical Context Distorts Decisions in Clinical Vision Language Models", [arXiv 2605.17436](https://arxiv.org/html/2605.17436v1) (May 2026)
- ModaLens: Cajas Ordóñez et al., [arXiv 2609.15635](https://arxiv.org/abs/2609.15635) (Sep 2026)
- Soft Dice Confidence: Borges, Pacheco, Silva, Machine Learning 115(7), 2026, [arXiv 2402.10665](https://arxiv.org/abs/2402.10665)
- Sangwaiya et al., AJR 2009;192(5):W239-44, PMID 19380530
- Luetmer et al., J Am Coll Radiol 2013;10(7):538-43, PMID 23827005
- Weng et al., JMIR Form Res 2025;9:e76384, PMID 41161340
