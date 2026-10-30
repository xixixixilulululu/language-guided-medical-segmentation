# Research learning log

A running record of what I learned during the research training and the research project: questions I
asked, mistakes that were found (including mistakes in AI-assisted advice), how they were checked, and the
rule I took away from each. It covers the training stage (BUSI, Tasks 1–8) and Phases 0–3 of this project.

Last updated: 2026-10-30 (Phases 2 and 3 completed: Part D; reflections E4–E5 and principles 12–15 added).

---

## Part A — Training stage (BUSI breast ultrasound, Tasks 1–8)

Repository: `research_training`.

### A1. Task 1: methodology problems found and fixed

- **Test-set leakage.** The first setup used the test set during model development. The split was changed
  to train / validation / test, with the test set used only for the final evaluation.
- **Metrics computed for the wrong class.** Precision and recall were first reported with "benign" as the
  positive class. For the clinical question the positive class is "malignant". After the fix, the work
  focused on malignant recall.
- **Further steps:** partial unfreezing (layer3) of ResNet50, k-fold validation, and checkpoint selection
  aimed at malignant recall.

**Lesson:** first check *what* is being measured (which split, which class) before trusting any number.

### A2. Tasks 2–5: comparing ResNet50, VGG16 and DenseNet121, and CAM

- **Fair comparison.** All three models were trained with the same recipe; only the architecture changed.
- **AUC/ROC.** AUC measures ranking quality over all thresholds; accuracy uses one threshold.
- **"DenseNet121 is more conservative" was wrong.** An AI explanation said DenseNet121 misses malignant
  cases because it hesitates near the 0.5 threshold. Checking the per-case probabilities showed that 9 of
  its 10 missed malignant cases were predicted benign with high confidence (several below 0.05); only one
  was near 0.5. The finding in `comparison.md` was corrected.
- **"DenseNet121 has the least trustworthy CAMs" was wrong.** VGG16 had more complete misses (35 vs. 9) and
  more border-focused CAMs. The corrected finding: the three models show *different failure modes*
  (VGG16 often looks at the wrong place; DenseNet121 rarely misses but is diffuse; ResNet50 is the most
  precise), not a single ranking.
- **A wrong case reference** (#129 was cited as `malignant (45)`; it is `malignant (131)`) was found by
  opening the actual image and fixed.
- **Mean vs. single case.** A lesion-energy value of 0 for one image and a mean of 0.195 over 130 images are
  not contradictory.
- **Accuracy vs. evidence.** A model can be right for the wrong reason (case #129: correct prediction, CAM
  entirely outside the lesion). Classification metrics cannot show this; CAM analysis can.

**Lessons:** verify an explanation against the per-case data before writing it down; an AI-written
conclusion is a draft, not a fact.

### A3. Tasks 6–8: segmentation (U-Net, nnU-Net, DeepLabV3+) and statistics

- **Results:** DeepLabV3+ 0.797 > nnU-Net 0.780 > U-Net 0.730 Dice. U-Net failed completely on more images
  (15 images with Dice < 0.1), which pulled its mean down although its median was similar.
- **Fairness question:** DeepLabV3+ used ImageNet pre-training and the others did not. This is acceptable if
  stated, but then the claim is "DeepLabV3+ with pre-training", not "the DeepLabV3+ architecture".
- **What a test does:** a statistical test does not check that a number was computed correctly; it asks
  whether a difference could be due to chance in this test set.
- **Holm correction** is not a second test; it adjusts p-values because several comparisons were made.
- **t-test vs. Wilcoxon** can disagree: the t-test compares means (sensitive to a few complete failures),
  Wilcoxon compares the typical paired difference.
- **"Not significant" means "not shown to differ"**, not "equal".
- **Plots:** mean ± SD bars can exceed 1.0 for Dice; box plots and bar plots answer different questions.
- **What serious work adds:** several seeds, confidence intervals, and cross-validation. These became
  standard in Phases 0–2.

---

## Part B — Phase 0: image-only benchmark (U-Net, TransUNet, nnU-Net)

### B1. Planning

- **Unknown method.** The first plan listed "TriPathLesionNet". It could not be found anywhere. Asking the
  supervisor showed it was an unpublished model of a former student; it was replaced by public methods.
  **Lesson:** when a required item has no source, ask instead of inventing one.
- **Code vs. paper.** LViT's code (`Config.py`) and paper disagreed on batch size and learning rate. For
  the U-Net baseline the paper's settings were used and the choice was documented.

### B2. A result that disagreed with the literature

- Our U-Net on MosMedData+ was about 9 points above the LViT paper (74.0 vs. 64.6), while QaTa matched.
- **What was checked:** the evaluation reproduced LViT's official test code exactly; the data came from the
  same source and split; the released LViT code has no MosMedData+ path.
- **Two over-statements were corrected:** a GitHub issue was first read as proof that the paper used
  different U-Net hyper-parameters (it only concerned constructor arguments), and "only U-Net is low in the
  paper" was false (Swin-Unet, UCTransNet and AttUNet were also low).
- **A data-based argument:** our three seeds were 73.2–74.9 (SD 0.86). 64.6 is far outside normal seed
  variation, so "the authors picked a bad seed" cannot explain it alone; a different training setting or
  data processing is more likely. The written report stays neutral; possible author intent is not claimed.
- **Consequence:** later phases compare against our own baselines, not copied literature values.

### B3. Keeping the analysis honest

- **Analysis fixed before results.** The statistics code was committed before the results were summarised.
- **Honest record of what was seen early.** Some test scores had been seen before the analysis was fixed;
  they were listed in the report. The significance level (5%) was confirmed after the run; the one
  borderline result (Holm p = 0.043) was flagged because of that.
- **Wording matters.** "Fixed before the results were seen" was an over-statement and was corrected to
  "fixed before the results of the 18 runs were summarised". "The project lead recomputed the results" was
  also corrected: the independent recomputation was done with separate code by the AI assistant.
- **GPU inference is not bit-identical.** Re-predicting with the same checkpoint changed a few boundary
  pixels. Rule: scores come from the first stored evaluation; a repair is attempted once, with a fallback
  decided in advance, never repeated until it "looks right".

### B4. Working on a shared cluster

- The partition time limit changed from unlimited to 24 h to 12 h, queued jobs were removed, and a
  fair-use policy followed (1 running + 1 waiting job per user, manual submission).
- nnU-Net was trained in segments with its official resume function; continuity (epoch, learning rate,
  loss) was checked after each resume.
- **Lessons:** ask about the rules early; lower priority voluntarily; plan long jobs for the night.

---

## Part C — Phase 1: language-guided methods (LViT-T, RecLMIS, DD-CMD)

### C1. Reproducing official code is not straightforward

Problems found before any training:

- **RecLMIS** silently switched off its reconstruction losses (its main contribution) when a dependency
  (`fairseq`) was missing, and loaded the QaTa configuration when asked for MosMedData+.
- **LViT**'s released loader paired files by another dataset's naming, so QaTa images and masks did not
  match.
- **LViT-T**'s text tool (`bert_embedding`, MXNet) cannot be installed on the cluster; it was
  re-implemented with the same BERT weights (not verifiable against the original).
- **DD-CMD**: an old framework version used `np.Inf`, removed in NumPy 2; a one-line alias fixed it.

**Lesson:** "official code" can contain silent failures; check that every component is really active.

### C2. Unnecessary changes cause problems

- To speed up DD-CMD, the number of data-loader workers was changed from the official 0 to 8. On this
  cluster the training hung before the first batch; a diagnostic job confirmed it, and the official value
  was restored. **Lesson:** keep official settings unless a change is necessary.

### C3. Protecting the test set (stricter than Phase 0)

- The analysis program was fixed and pushed (GitHub server time 2026-10-19 04:02:14 UTC) before the first
  Phase 1 test evaluation.
- Two independent locks: Phase 1 jobs could not read the test split (data lock), and every training job
  checked that the analysis was fixed on GitHub before evaluating the test set (program lock).
- Every test evaluation was logged; all 36 entries were later than the fixing time.
- **Lesson:** "I did not look" is weaker than evidence that could not have been otherwise.

### C4. A NaN during training

- RecLMIS (QaTa, seed 1001) produced a NaN in one loss term at epoch 152. The run was stopped before the
  NaN reached the weights; the best validation checkpoint (epoch 129) was kept.
- **Rule fixed before seeing more results:** any non-finite loss ends training and the best validation
  checkpoint is evaluated. On the same trajectory this gives the checkpoint the official code would also
  select.
- On resume, the NaN reappeared at exactly the same step, which also showed that resuming was exact.
  Seed 1002 had no NaN.
- **Caveat kept in the report:** the NaN appeared in our environment; whether it occurs in the authors'
  environment cannot be determined.

### C5. The biggest mistake: mixing settings from two sources

- For LViT-T, batch size 2 was taken from the code and the MosMedData+ learning rate 1e-3 from the paper,
  following the rule "take each parameter from its most specific source". Neither source used this
  combination.
- **Result:** LViT-T on MosMedData+ scored 61.3 (paper 74.6), with 38–41 empty predictions per seed.
- **Check:** a supplementary run, planned and committed in advance, changed only the batch size to 24.
  LViT-T reached 74.0 with almost no empty predictions.
- **Why it happened:** batch size and learning rate are coupled; a learning rate tuned for a large batch
  can make training unstable with a small batch. The coherent code setting (3e-4 with batch 2) had been
  proposed and was overridden.
- **Lessons:** never mix coupled hyper-parameters from different sources; take one coherent set, or compare
  coherent sets on the validation set before freezing; state which parameters interact.
- **Why the corrected run is reported separately:** it was decided after seeing the test results. Putting
  it into the main table would mean adjusting a method after seeing its test score, and only this method
  got a second chance. The main results stay as fixed; the supplementary results are reported next to them.
- **Effect on conclusions:** only LViT-T on MosMedData+ was affected. The main conclusion (language-guided
  methods are not consistently better than the best image-only method) still holds.

### C6. Reading published numbers carefully

- LViT's 83.66 is a mean of three runs (Table IV), not a single run; this was first stated wrongly.
- DD-CMD's paper uses a pixel-pooled Dice, so its numbers are not directly comparable with per-image Dice.

### C7. Practical lessons

- Time estimates must include the 12-hour job limit.
- With only one waiting slot, the job queued before sleeping should be one that surely runs long.
- When the VPN failed, the supervisor provided a temporary SSH tunnel. Before entering a password, the
  server's host-key fingerprint was compared with the known one.

---

## Part D — Phases 2 and 3: text reliability and selective fallback (completed 2026-10-30)

### D1. Phase 2: does the model depend on its text?

- **Goal:** test whether the trained models depend on their text, by changing the text (null, equivalent
  wording, deleted attributes, real reports from other cases) without retraining.
- **Validation first:** rules are developed on train/validation text; the decision about Phase 3 uses
  validation only; test inference is locked until everything is fixed.
- **What the setup checks found:**
  - LViT-T keeps only the first 10 word vectors; most location words are outside this window. A longer
    "equivalent" rewrite also changes what is inside the window, so C0−C2 for LViT-T is not only a wording
    effect.
  - The texts are templated, and some contradict themselves ("unilateral" with both lungs named) or the
    masks ("no pulmonary infection" while the mask has a small lesion). Such cases are kept as written and
    flagged.
  - MosMedData+ has only about 10 independent CT-volume units, so its unit-level inference is exploratory.
- **Honesty about who checked what:** the 40 rewrites were reviewed for equal meaning, and the record states
  who reviewed them.
- **Freeze and test.**
  - Before any counterfactual test run, the Phase 2 and Phase 3 rules were frozen on GitHub (commit b593d7f,
    27 files with their sha256). A lock program refused every test run until then.
  - The test benchmark ran once (job 392): 270 passes, 0 failures.
  - The clean-text (C0) results matched the Phase 1 test archive within the pre-set rule.
- **Results on the QaTa-COV19 test set** (472–474 patient units, Dice points; `reports/PHASE2_RESULTS.md`):
  - Removing the report (C0 − C1) costs 12.35–20.13 points. A fully mismatched report (C0 − C6) costs
    23.31–24.40, more than removing it (C1 − C6 +3.12 to +11.39).
  - The harm follows the location: a location-only change costs 12.6–13.8 points, a count-only or scope-only
    change 0.1–1.1.
  - RecLMIS is sensitive to equivalent rewording (−1.25 points); LViT-T and DD-CMD are not, on average.
  - Validation and test agree:
    - same sign in 31 of 33 comparisons;
    - same Holm decision in 30 of 33;
    - the three that did not replicate are "wrong versus no text" comparisons for partial mismatches.
  - MosMedData+ (9 test units) is exploratory; its effects are small.
- **Lessons:**
  - **The replication can be trusted because the rules were fixed before the test.** There was nothing left to
    choose after seeing the results.
  - **A frozen program can still contain a small error.** The test case figures carry the title "validation",
    because the title is fixed text in the frozen code. It was documented, not fixed.
  - **A frozen program can write files that must not be used.** The frozen analysis also wrote the Phase 3
    go/no-go files for the test split. That rule is defined on validation only, so the files were labelled
    descriptive and not used.

### D2. Phase 3: falling back to an image-only model when the text may be wrong

- **Goal.** A gate chooses, per image, between the language-guided prediction (LANG) and nnU-Net (IMG).
  - D-Gate is a threshold on one disagreement feature.
  - LR-Gate is a logistic regression on five features of the two probability maps.
  - Both were developed on QaTa validation only: GATE_FIT for fitting, GATE_CAL for thresholds.
  - MosMedData+ is exploratory.
- **Before the freeze:**
  - On GATE_FIT the clean-text losses were already beyond the planned margin (−0.66 to −0.94 points against
    −0.5). I wrote this to the supervisor before the freeze, with "keep the gates as specified" as the default.
    With no reply by the freeze, the gates were kept as specified and the decision was recorded.
  - The first cost measurement (job 379) reported 66.6 GB of GPU memory for LViT-T's D-Gate (395 MB for its LR-Gate).
    This was an artifact of the measurement order. It was withdrawn and remeasured with one process per mode (job 391).
- **Results on the QaTa-COV19 test set** (`reports/PHASE3_RESULTS.md`):
  - **Primary family** (LR-Gate − LANG under C4–C6): 9 of 9 significant after Holm, +1.36 to +6.53 points.
  - **Secondary family** (LR-Gate − D-Gate): 9 of 9 significant, +0.76 to +2.68.
  - **Clean and reworded text (C0/C2):** the gates cost −0.17 to −0.30 points.
    - D-Gate keeps the −0.005 margin in 6 of 6 cells.
    - LR-Gate keeps it in 4 of 6; for LViT-T, the CI lower bounds for C0 and C2 reach −0.56 and −0.52.
  - **Limits:**
    - LR-Gate recovers only 11–33% of the oracle headroom.
    - When the report is wrong or missing, nnU-Net alone is 4.85–17.12 points better than either gate.
    - On MosMedData+ the QaTa gates almost never switch.
- **Checks:**
  - **Independent recomputation with separate code: 14,373,709 checks, 0 differences.** It recomputed the gate
    decisions from the frozen parameters, the tables, the routing counts, and the statistics with Holm.
  - **Every table in the summary was right, but four of its sentences did not match the files.** For example,
    "77–97% beneficial" where the file says 77.4–94.8%. All four were corrected before the report was final.
  - **The log shows one failed `git fetch` on the compute node.** The lock allows this by design: it falls back
    to the local reference. It was written into the run record instead of being left out.
- **Lessons:**
  - Check the sentences, not only the tables.
  - A measurement setup can create its own numbers (job 379).
  - A small irregularity in a log is reported, not left out.
  - What the results mean for improving the gate is in E4.

---

## Part E — Reflections

### E1. A convincing AI answer can still be wrong

- **What happened.** Several times an AI assistant (including the one that helped write this log) gave an
  answer that sounded right, came with reasons, and was confirmed again when I asked "are you sure?", but was
  wrong:
  - Training stage: "DenseNet121 misses malignant cases because it hesitates near 0.5". The per-case
    probabilities showed confident misses.
  - Phase 0: a GitHub issue was presented as evidence that the paper's U-Net used different settings. It was
    about something else.
  - Phase 1: taking the batch size from the code and the learning rate from the paper was justified as "each
    parameter from its most specific source". It sounded principled, and it cost LViT-T about 13 Dice points
    on MosMedData+.
  - Phase 1: LViT's 83.66 was called a single run. The paper's table shows a mean of three runs.
  - Phase 2: the first C2 rewrite passed a 40-row meaning check, but in many cases two of the methods cut off
    its last words (the location) at their token limit. The check looked at the text, not at what the model
    actually reads.
  - Phase 2: a quick check written by the AI printed "MISMATCH". The data were fine; the check command was
    wrong.
- **Why.** An AI writes fluent reasons for whatever it currently believes. Asking "are you sure?" usually
  produces more reasons, not new evidence.
- **What works:**
  - ask for evidence that can be checked: the line of code, the table in the paper, the per-case numbers;
  - check by independent means: separate code, the raw data, a validation run;
  - ask what result would prove the claim wrong, and look for it;
  - two AI sessions checking each other caught several errors, but they can share a blind spot. The final
    word belongs to the data and the code.

### E2. Papers show their methods in the best light

A paper is written to support its claim, and its space is limited. Authors show the settings, baselines and
metrics that support the claim; details and limits that do not fit are often left out. This is not
necessarily intentional, but it caused real problems in this project:

- **Code and paper disagree.** LViT's code and paper give different batch sizes and learning rates, and it was
  not clear which setting produced the reported numbers.
- **Released code does not do what the paper describes.** LViT's loader paired QaTa images with the wrong
  masks and had no MosMedData+ path. RecLMIS silently switched off its main loss when a package was missing,
  and its MosMedData+ option loaded the QaTa settings.
- **Weak baselines.** In the LViT paper, U-Net scores 64.6 on MosMedData+. Our U-Net, scored with LViT's own
  test code, reached 74.0. A weak baseline makes a new method look better. We do not know the reason, so our
  report stays neutral.
- **Metric choice.** DD-CMD reports a Dice pooled over all pixels, which gives different numbers from the
  per-image Dice used by the others.
- **What the model really reads.** LViT-T keeps only the first 10 word vectors, so most location words never
  reach it. RecLMIS reads 16 tokens; in 101 of 1429 QaTa validation cases even the original text loses its
  last location words. This is hard to see from the papers, which present the methods as guided by the report.
- **The overall result.** Under one common evaluation, the language-guided methods were not consistently
  better than the best image-only method (Phase 1).

**Lesson:** read a paper as the authors' best case, not as a complete description. Read the code too, rerun
the baselines ourselves, score every method with one evaluator, and check what each component actually does.

### E3. Small datasets make conclusions fragile

- **Count independent units, not images.** MosMedData+ has 273 validation images but only 11 independent
  CT-volume units (9 on test). Slices of one volume are similar, so 273 images are worth far fewer than 273
  independent cases.
- **Examples from this project:**
  - Phase 0: the disagreement with the LViT paper appeared on MosMedData+, the small dataset; on QaTa our
    numbers matched.
  - Phase 2: on MosMedData+, the harm of a fully wrong report for RecLMIS was 1.7 Dice points when every image
    counts equally, and 4.0 points when every CT volume counts equally. Both are reasonable averages; one is
    below the planning threshold of 2 points and the other above it. The decision rule had been written down
    before the full results, so the answer could not be picked afterwards.
  - Training stage: 15 complete failures pulled U-Net's mean Dice down while its median was similar.
  - The text is small too: the reports are templated. QaTa's 5716 training images have only 273 distinct
    texts, and MosMedData+'s 2183 have only 101.
- **Lesson:** report the number of independent units, the seeds and the confidence intervals; call
  small-sample results exploratory; do not read meaning into small differences; fix the analysis before
  seeing the results.

### E4. Look for the root cause and a breakthrough, not a list of fixes

- **What happened.** After the Phase 3 test, I asked how the gate's limitations could be improved.
  - **First answer: a list of reasonable fixes.** More features, a stronger classifier, a stricter
    calibration, per-dataset thresholds, fewer collinear features. Each would help a little; none would change
    the picture.
  - **Second answer, after I asked for a breakthrough: a "root cause" that was wrong.**
    - It said the limit was the selection design itself. Under C6 (a fully mismatched report), even a perfect
      selector that knows the answer (the ORACLE) reaches only about 81 Dice, 1.3–1.5 points above nnU-Net
      alone (79.7).
    - It proposed correcting the report instead of discarding it. It sounded convincing, came with real
      numbers, and was written into this log and into an e-mail draft.
  - **Third answer, after I asked for a rethink from scratch: the second answer did not hold.**
- **Why the second answer was wrong.**
  - **The ceiling is not the bottleneck.** Under C6 the gates reach 63–68 Dice, 12–17 points below simply
    using nnU-Net. They are far below the ceiling, so the ceiling is not what limits them now. What limits them
    is that they do not recognise a wrong report and do not switch when they should.
  - **"Correcting" a wrong report cannot bring back the information.** A correct report is worth 2–5 points
    over nnU-Net (81.7–84.8 Dice), largely because it states where the lesions are (an interpretation,
    consistent with the large loss when the text is removed). Once the report is wrong,
    that information is not in the input any more. A location derived from nnU-Net's mask carries only
    nnU-Net's own opinion. Whether feeding it back to the language-guided model helps is an open question, not
    a ceiling of 82–85.
- **The root cause, as far as I can tell now.** The gates decide only from the two segmentations and never
  look at what the report says. So when the two segmentations disagree, they cannot reliably tell whether the
  report or nnU-Net is wrong.
  - **C1 shows it most clearly.** With no text at all, the trivial rule "no text → nnU-Net" gives 79.7; the
    gates give 65–74. C1 is only an illustration, since a real system would apply that rule anyway. The real
    target is the wrong reports (C4–C6).
  - **Validation data (development data) support this.**
    - The existing gate signals separate clean (C0) from fully wrong (C6) reports only moderately: AUROC
      0.68–0.76.
    - When the two masks strongly disagree under the clean report (mask Dice < 0.3, 2.5–4% of the rows), the
      language-guided model is still the better one in 49–79% of these rows. The share depends on the cutoff
      (35–85% for mask Dice below 0.5 to 0.1); the point is that a strong disagreement alone does not say which
      side is wrong.
    - Under C6, in the rows where switching is clearly right (LANG < 0.3, nnU-Net > 0.6), LR-Gate switches only
      33–53%.
- **Direction now (an idea to test on validation, not a result).** Give the gate a signal of whether the
  report fits the image. There are two ways to get it:
  - **A simple rule.** Compare the location stated in the report with the location found by nnU-Net. It needs
    no training. Its risk is that it shares nnU-Net's errors: on validation, 49–68% of LR-Gate's harmful
    switches under the clean report come with a poor nnU-Net mask (Dice < 0.5), which such a rule would also
    flag.
  - **A small classifier trained to recognise mismatch.** Train it on the training split with real and swapped
    reports, so that it learns directly whether a report fits an image. Its errors are not directly tied to
    nnU-Net's segmentation. Its risk is that it learns how the benchmark swaps reports, which may differ from real
    report errors.
  - **The first measurement** is how well each signal separates C0 from C4–C6 on validation, compared with
    0.68–0.76 now.
- **Lessons:**
  - Before improving a method, compare three numbers: the current result, the simplest baseline (here
    nnU-Net alone) and the ceiling (here the ORACLE). The real gap shows where the bottleneck is. Looking only
    at the ceiling misled the second answer.
  - If even the ideal version of an approach cannot solve the problem, change the approach. If the current
    result is far below a simple baseline, fix the real bottleneck first.
  - A first "root cause" can be wrong too. It also needs checking against the numbers, and a rethink from
    scratch, before it goes into a document or an e-mail (see E1).
  - Aim for a breakthrough: ask "what is the root cause?" every time, not only "what else can we try?".

### E5. Let the data speak: real numbers, and no more speculation than the data allow

- **What happened** (2026-10-30, while finishing Phases 2 and 3):
  - **Reasoning without checking.** The second "root cause" in E4 used correct numbers but was not checked
    against the data. Once it was checked on validation, it did not hold.
  - **Statements that went beyond what was measured.** Each was found and corrected:
    - "the features cost < 1 ms" was never measured;
    - "IMG is 5–15 points better than the gates" was a rough estimate; the file says 4.85–17.12;
    - "C5 is the same size on validation and test" was wrong: for LViT-T it was about three times larger on
      test (+2.99 versus +0.97);
    - "the gates recover 11–33%" is true only of LR-Gate; D-Gate recovers 5–21%;
    - "RecLMIS misses both lungs" went beyond the figure, which only shows an almost empty prediction.
  - **A number that depends on an arbitrary choice.** "The language-guided model is still the better one in
    49–79% of strong disagreements" became 35–85% when the cutoff for "strong" changed. The e-mail used a
    measure without a cutoff (AUROC) instead.
  - **Documentation versus code.** The setup document describes F2 as "Dice of the two masks", but the code
    stores 1 − Dice. The AI assistant's first diagnostic read F2 the wrong way round and found that 78–80% of
    clean-text rows strongly disagree. The number was implausible, and the code showed the real definition.
  - **Summaries versus files.** VS Code's summary had four sentences that did not match its own files. The AI
    assistant that checks VS Code's work swapped two numbers in its own message (DD-CMD +3.4 / +3.7); VS Code
    caught it by reading the file.
  - **"Are you sure?"** When I asked whether everything was now based on data, the honest answer was in two
    parts. The numbers are. The conclusions are interpretations of the numbers, and the proposed next step is
    an untested idea, so both have to be labelled as such.
- **Lessons:**
  - Every number must come from a file or a script that can be rerun, and the text around it must not say
    more than the number does.
  - Check the sentences of a summary against the files, not only its tables. This includes every AI summary,
    the checker's own messages, and my own e-mails.
  - Mark what is a measured fact, what is an interpretation and what is an untested idea.
  - If a number depends on a cutoff or another choice, say so, or use a measure that does not depend on it.
  - An implausible result is a reason to check the definition in the code, not to explain the result.
  - Answer "are you sure?" with evidence: what was checked, against which file, and what is still a judgment.

---

## Part F — Principles I now follow

1. **Check what is measured** (split, class, metric definition) before trusting a number.
2. **Verify explanations against per-case data** before writing them down; treat AI-written text as a draft.
3. **Fix analysis rules before seeing results**, and record honestly anything seen earlier.
4. **Keep official settings** unless a change is necessary; never mix coupled settings from different
   sources.
5. **Report problems instead of hiding them**: keep the pre-specified result, and add clearly labelled
   supplementary checks.
6. **Say what is certain and what is a judgment**, and state who did each check.
7. **Ask the supervisor early** when a requirement has no source or a rule is unclear.
8. **Compare with your own baselines**, not only with numbers copied from papers.
9. **A convincing explanation is not evidence.** Ask for checkable evidence, not "are you sure?".
10. **Read a paper as the authors' best case.** Check the code, the baselines and the metrics yourself.
11. **Count independent units, not images.** With few units, conclusions are exploratory.
12. **Find the root cause before improving.** Compare the current result with the simplest baseline and with
    the ceiling to see where the real gap is; change the approach only if even its ideal version cannot solve
    the problem.
13. **Rethink a root-cause analysis from scratch** before it goes into a document or an e-mail.
14. **Let the data speak.** Every number comes from a file or a script that can be rerun. Statements stay
    within what was measured, and facts, interpretations and untested ideas are labelled as such.
15. **Check the sentences, not only the tables**, in AI summaries and in my own e-mails.
