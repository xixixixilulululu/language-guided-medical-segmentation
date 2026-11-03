# Related work and possible paper directions (for discussion)

Status: 2026-11-02. A literature note to support the choice of the next step. It is **not** a claim that a gap
exists.

**How it was made:**
- web searches on 2026-11-02;
- the papers listed below, read to the depth given in the "Read" column;
- our own numbers, all from [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md).

**Caveats:**
- "Not found" means "not found in these searches", not "does not exist".
- The field is moving fast: two of the closest papers appeared in October 2026.
- The search should be repeated before any submission.
- Numbers quoted from other papers come from their own settings and are not comparable with ours.

## 1. Closest work

| Paper | Read | What it tests | What it does not test (relative to our work) |
|---|---|---|---|
| **RADAR** ([arXiv 2610.14944](https://arxiv.org/html/2610.14944), Oct 2026) | appendix D.1 in full | One trained checkpoint (RADAR) on QaTa-COV19 and MosMedData+, test-time text: Original / Empty / Incomplete / Mild Noise / Synonym / Wrong. QaTa: 84.12 → 76.90 (empty) → 70.33 (wrong). MosMed: 75.47 → 67.80 (empty) → 70.63 (wrong). | One model only; how the "Wrong" text is built is not described; no CIs, seeds or tests; no control of which attribute is wrong; no mitigation |
| **Text Branch Sensitivity** ([arXiv 2609.02663](https://arxiv.org/abs/2609.02663), Sep 2026) | full text | Five pretrained medical VLM encoders (CLIP, MedCLIP, BioMedCLIP, PubMedCLIP, UniMed-CLIP) on BUSI, BTMRI, Kvasir-SEG, ISIC. Test-time: no text, random text from another sample, missing location, contradictory text. Text sensitivity is dataset-dependent (e.g. MedCLIP on BTMRI 91.61 → 40.76 without text); text acts mainly through global semantics, not spatial localisation. | Not QaTa-COV19 / MosMedData+ and not the dedicated report-guided models (LViT family); swaps are random, not attribute-controlled; no mitigation |
| **RELiF** ([arXiv 2610.28118](https://arxiv.org/html/2610.28118), Oct 2026) | full text | A reliability-guided fusion gate inside the network scales text updates within [0.5, 1.5]. Training-time ablation on MosMed / Kvasir: no text 77.78 / 88.90, mismatched text 78.73 / 91.01, full 79.27 / 93.28. | No test-time wrong reports; the gate never switches the text off; no fallback to an image-only model |
| **ProLearn** ([ICCV 2025](https://arxiv.org/abs/2507.11055)) | full text | Reduced paired text in training (50% to 1%) and text-free inference on QaTa-COV19, MosMedData+, Kvasir-SEG. | Missing text only; no wrong or mismatched reports |
| **ARSeg** ([MICCAI 2025](https://papers.miccai.org/miccai-2025/0944-Paper3688.html)) | paper page and reviews | Incomplete prompts (attributes missing at set rates) on QaTa-COV19 and MosMedData+. | Missing attributes only; wrong text not tested (no reviewer asked about it) |
| **C2Seg** ([ICLR 2026](https://mlanthology.org/iclr/2026/gui2026iclr-textmask/)) | abstract | Makes masks follow multi-site / multi-lesion text more closely (text–mask consistency). | Robustness to wrong or missing text is not mentioned in the abstract |
| **Poudel et al.** ([MIDL 2024](https://proceedings.mlr.press/v250/poudel24a.html)) | abstract | After fine-tuning, not all vision-language segmentation models use the language prompt; image features dominate. | No counterfactual text conditions |
| **MIMIC-ILS / ROSALIA** ([CVPR 2026, arXiv 2511.15186](https://arxiv.org/html/2511.15186)) | full text | Instruction-guided chest X-ray lesion segmentation.<br>- Dataset construction keeps only report locations that overlap a generated lesion mask, using anatomy masks ("location verification").<br>- The model is trained with negative instructions ("no [target] in [location]") and reaches 91.8% empty-target accuracy. | The check is used to build the data, not at inference. It is not about a segmentation model guided by a possibly wrong report, and there is no fallback. |
| **Report error detection**: CorBenchX ([arXiv 2505.12057](https://arxiv.org/abs/2505.12057)), ReXErr-v1 ([PhysioNet](https://physionet.org/content/rexerr-v1/1.0.0/)), phrase-grounded fact-checking ([MICCAI 2025](https://papers.miccai.org/miccai-2025/0693-Paper3526.html)) | abstracts | Detect errors in chest X-ray reports, mostly generated or injected, by checking them against the image. | Not connected to segmentation guidance |

## 2. Possible directions, with the evidence we already have

Each direction lists what the closest work did, what we have, and what is still missing.

### D1. A controlled counterfactual benchmark of report errors for report-guided segmentation

- **Closest work.**
  - RADAR: one model, an undefined "Wrong" condition, no statistics.
  - Text Branch Sensitivity: other datasets and models, random swaps.
  - RELiF: mismatch only in training.
- **What we have (data complete).**
  - 3 models × 3 seeds × 2 datasets under 9 conditions.
  - Real reports of other cases, with exactly 1 / 2 / 3 of scope, count and location changed.
  - Patient-level paired tests with Holm correction, and a protocol frozen before the test.
  - Validation and test agree: same sign in 31 of 33 comparisons, same Holm decision in 30 of 33.
- **Not found in the search:** a multi-model benchmark with attribute-controlled report swaps on these datasets.

### D2. Which part of the report drives the model, and what the model actually reads

- **Closest work.** Text Branch Sensitivity finds that text acts through global semantics, and that removing
  the location matters on BUSI but hardly on BTMRI.
- **What we have.** On QaTa-COV19:
  - a location-only change costs 12.6–13.8 points, a count-only or scope-only change 0.1–1.1;
  - deleting the location costs 11.43–21.76;
  - an audit of the effective input shows that LViT-T uses only the first 10 word vectors, and RecLMIS cuts
    the end of 101 of 1429 validation reports.
- **What this adds.** It is a dataset where the location is what matters, which complements the "global
  semantics" finding.

### D3. A wrong report is worse than no report

- **What we have.** C1 − C6 is +3.12, +7.60 and +11.39 points (all three models significant).
- **Independent agreement.** RADAR's single model shows the same pattern on QaTa (wrong 70.33 < empty 76.90).
  It also shows the reverse on MosMed (wrong 70.63 > empty 67.80), as our MosMed point estimates do.
- **Why it matters.** Recent methods aim to make masks follow the text more closely (e.g. C2Seg).
- **Open hypothesis, not tested.** Stronger text–mask consistency may increase the harm of a wrong report.
  Testing it would need those models.

### D4. A model-agnostic fallback when the report may be wrong

- **Closest work.**
  - RELiF's gate is internal and never removes the text.
  - ProLearn removes the need for text but does not detect wrong text.
- **What we have (baseline complete).**
  - A post-hoc gate between any language-guided model and nnU-Net: 9 of 9 significant gains under mismatched
    reports, at a cost of 0.17–0.30 points under the correct report.
  - Its limits are measured. It recovers 11–33% of the oracle headroom, and nnU-Net alone is better when the
    report is wrong.
  - The current signals separate C6 from C0 with AUROC 0.69–0.77 on GATE_CAL (validation diagnostic).
- **Not found in the search:** a post-hoc fallback evaluated against wrong reports.

### D5. Check the report against the image before trusting it (new method; no data yet)

- **Related work.**
  - Report-error detection (CorBenchX, ReXErr, phrase-grounded fact-checking) checks reports against images,
    for report generation.
  - MIMIC-ILS maps lesions to anatomical locations with anatomy masks, for building data.
- **Not found in the search:** using such a check to decide how far a segmentation model should trust its
  report.
- **What is needed.**
  - A location signal from the image, for example anatomy masks for lung zones, as in MIMIC-ILS.
  - A validation-only first test: does it separate C0 from C4–C6 better than 0.69–0.77?
  - A fresh test protocol, since the QaTa test set has been used.
- **Risk.** The effect is unknown.

### D6 (supporting). Strong image-only baselines under one evaluator

- **Observation.** Our Phase 0 found LViT Table II's U-Net on MosMedData+ 9.4 points below our reproduction,
  scored with LViT's own evaluation code. Whether later papers reuse these baseline values or retrain them was
  not checked.
- **What we have.** Under one evaluator, the gain of the language-guided models over nnU-Net is 2–5 points on
  QaTa-COV19 and is not consistent on MosMedData+.
- **Role.** Part of the setup of a paper, not a paper on its own.

## 3. How these could combine into a paper (for discussion)

| Option | Content | What is still needed | Main risk |
|---|---|---|---|
| A | D1 + D2 + D3 as the core, D4 as a first mitigation, D6 as setup | writing; repeat the literature search | A benchmark paper must convince through its findings. Reviewers may ask about real report errors and why the gate is not simply replaced by nnU-Net. |
| B | A + D5 as a method | validation experiments, then a fresh test protocol | time; the effect of D5 is unknown |

**Venues** (checked 2026-11-02):
- ISBI 2027: the deadline has passed.
- MICCAI 2027: Auckland, 26 Sep–1 Oct 2027. The deadline is not announced yet; the 2026 deadline was 26 Feb.
  Main-conference acceptance was 29% in 2025.
- MIDL 2027: dates not announced yet.

**Practical point.** This repository is public and carries the author's name. For a double-blind venue, it
should be decided before submission whether it stays public.
