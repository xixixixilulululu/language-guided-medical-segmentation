# Research learning log

A running record of what I learned during the research training and the research project: questions I
asked, mistakes that were found (including mistakes in AI-assisted advice), how they were checked, and the
rule I took away from each. It covers the training stage (BUSI, Tasks 1–8) and Phases 0, 1 and 2 of this
project.

Last updated: 2026-10-27 (Phase 2 in progress).

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

## Part D — Phase 2: text reliability (in progress)

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

---

## Part E — Principles I now follow

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
