# Phase 1 setup: read-only checks and execution plan

Date: 2026-10-18. Scope of this step: read-only checks and the plan only. Nothing was trained, no Slurm
job was submitted, no package was installed (only `pip install --dry-run` and `pip download` of one
wheel into a scratch folder, to read its source), and no Phase 0 file was changed.
The full Phase 1 requirements are in `reports/PHASE1_PLAN.md`.

Section 9 lists the decisions that were needed before any work started. They were taken on 2026-10-18
and are recorded in section 10, with what has been done since.

---

## 1. Repositories

| Method | Repository | Commit (cloned 2026-10-18) | Commit date |
|---|---|---|---|
| LViT-T | https://github.com/HUANGLIZI/LViT | `ba90775ead37d6f445f65ce83d1a0be64157f76f` (same as Phase 0) | 2026-07-21 |
| RecLMIS | https://github.com/ShawnHuang497/RecLMIS | `d3265c8b4d4582a94595fe1c8b1892c618bc9715` | 2025-06-19 |
| DD-CMD | https://github.com/maklachur/DD-CMD | `27387c47163971448f8fcdf5e37f310509dbd948` | 2026-08-12 |
| OpenAI CLIP package (tokenizer for RecLMIS) | https://github.com/openai/CLIP | `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` (HEAD, not installed yet) | |

All three are cloned to `external/` (ignored by git). The external clones stay unchanged at these commits.
Every compatibility change is made in our own code (section 3), not in `external/`.
The two new rows are added to `external/VERSIONS.md` when the work starts. That file was not
edited in this read-only step.

## 2. Text annotations: pairing with the Phase 0 split

Checked against the Phase 0 manifest `~/datasets/common224/manifest.csv`. The per-split xlsx files in
`~/datasets/<dataset>/<Split>_Folder/<Split>_text.xlsx` were checked against the original LViT xlsx
lists in `~/datasets/raw/lvit_split_lists/`.

| Dataset | Split | Manifest | xlsx rows | Duplicate keys | Images without text | Text without image | Empty text | Differs from original LViT xlsx | Distinct texts |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| QaTa-COV19 | train | 5716 | 5716 | 0 | 0 | 0 | 0 | 0 | 273 |
| QaTa-COV19 | val | 1429 | 1429 | 0 | 0 | 0 | 0 | 0 | 128 |
| QaTa-COV19 | test | 2113 | 2113 | 0 | 0 | 0 | 0 | 0 | 198 |
| MosMedData+ | train | 2183 | 2183 | 0 | 0 | 0 | 0 | 0 | 101 |
| MosMedData+ | val | 273 | 273 | 0 | 0 | 0 | 0 | 0 | 45 |
| MosMedData+ | test | 273 | 273 | 0 | 0 | 0 | 0 | 0 | 61 |

**Pairing is one-to-one: every image of the Phase 0 split has exactly one non-empty text.**

- **Keys:** the xlsx key is the mask filename. For QaTa this is `mask_<image>.png`; for MosMed the
  mask filename equals the image filename.
- **Texts are templated** (e.g. "Bilateral pulmonary infection, two infected areas, all left lung
  and middle lower right lung."). 495 distinct texts cover the 11,987 images (806 when distinct texts are counted per split, as in the table).
  (Corrected 2026-10-18: an earlier version of this report said "806 distinct texts cover 12,087 images";
  12,087 was an addition error.)

**How each method reads the text:**

| | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| File format | xlsx, columns `Image`, `Description` | xlsx, columns `Image`, `Description` | **CSV**, key column `Image`, text column `text` (or `Description`) |
| Key | mask filename | mask filename | mask filename with `mask_` removed |
| Train / val text file (QaTa) | one joint `Train_Val_text.xlsx` | one joint `Train_Val_text.xlsx` | separate per split |
| Padding of short texts | `' EOF XXX'` appended to texts with < 9 words | same `' EOF XXX'` rule | none (tokenizer padding to 24 tokens) |
| Text model | BERT-base uncased (bert_embedding / MXNet), first 10 word vectors, 768-d | CLIP ViT-B/32 text tower (frozen), 18 tokens | BiomedBERT (frozen), 24 tokens, last hidden state |
| Text at test time | yes | yes | yes |

The texts are identical for the three methods. Only the file format, the key spelling and the text
encoder differ. All three use the same text strings from our xlsx files. DD-CMD's CSV files are
generated from them by an exact conversion (no edits), and the conversion is checked row by row.

Two code bugs would change the texts silently in our environment:

1. **Padding is lost under pandas 3 (LViT and RecLMIS).** Both pad with the chained assignment
   `df.Description[i] = ...` (LViT `utils.py:534`, RecLMIS `utils.py:543`). Under pandas 3
   copy-on-write this does nothing, so short texts would stay unpadded. For LViT-T this can even crash
   the model, because it needs exactly 10 word vectors. Our reader performs the same padding
   explicitly, and a test checks the padded strings.
2. **The released LViT loader does not pair QaTa correctly.** The active lines use MoNuSeg pairing
   (`Load_Dataset.py:169-170`); the Covid19 pairing is commented out (171-172). With the released
   lines, QaTa would look for a mask named like the image and fail. MosMed is not affected, because
   its image and mask names are equal. Our loader uses the Covid19 pairing for QaTa, as the
   commented-out lines intend.

## 3. Environment

The machine is aarch64 (GPU nodes `promaxgb10-[4415,4986]`). The `lgms` environment has Python 3.11.16,
torch 2.14.0+cu130 and numpy 2.4.6. None of the installs below changes torch: every dry-run left torch
out of its "Would install" list.

### 3.1 LViT-T: `bert_embedding` / MXNet

- **The official stack cannot be installed.** `bert-embedding 1.0.1` pins `mxnet==1.4.0`,
  `numpy==1.14.6`, `gluonnlp==0.6.0` and `typing==3.6.6`. mxnet 1.4.0 has no aarch64 wheel and
  numpy 1.14.6 has no Python 3.11 wheel, so the dry-run fails at the metadata step.
- **A partial route exists but is untested.** `mxnet 1.9.1` has an aarch64 wheel
  (`py3-none-manylinux2014_aarch64`) but would downgrade numpy to 1.26.4. `gluonnlp 0.10.0` exists
  only as an sdist (needs Cython and a compiler). `bert-embedding` itself would have to be installed
  with `--no-deps` against these newer versions. MXNet is retired (Apache Attic); whether it imports
  and runs on Python 3.11 is unknown.
- **What LViT actually computes.** This was read from the `bert_embedding-1.0.1` wheel source, not run.
  `BertEmbedding()` with defaults is `bert_12_768_12`, `dataset_name='book_corpus_wiki_en_uncased'`
  and `max_seq_length=25`. For each sentence it:
  1. tokenises with the uncased BERT word-piece tokenizer (with [CLS] and [SEP], truncated to 25
     tokens);
  2. runs BERT-base in inference mode, so dropout is off;
  3. takes the last-layer token outputs;
  4. drops [CLS], [SEP] and [PAD];
  5. merges the word pieces of each word by averaging (`oov_way='avg'`).

  LViT then keeps the first 10 word vectors (10 × 768) (`Load_Dataset.py:186-189`). It recomputes them
  on the CPU in every `__getitem__`, but the result is deterministic per text.
- **Closest replacement (recommended):**
  1. Load the same BERT-base uncased weights in PyTorch (`google-bert/bert-base-uncased` on Hugging
     Face). GluonNLP's `book_corpus_wiki_en_uncased` and HF's `bert-base-uncased` are both
     conversions of Google's original uncased BERT-base checkpoint.
  2. Re-implement steps 1-5 exactly (same tokenizer rules, max length 25, same word-piece averaging,
     first 10 vectors).
  3. Compute the vectors **once** for the distinct texts (495) and store them as a cache keyed by text.
     LViT's data loader reads from the cache instead of calling MXNet.

  The model input is then the same 10 × 768 matrix as in the official code. The only expected
  differences are floating-point rounding between frameworks and the vocabulary id numbering (GluonNLP
  uses [PAD]=1, [CLS]=2, [SEP]=3; only the special tokens are filtered, so this does not change the
  output). Precomputing also removes a large CPU cost per batch, but does not change training.
- **Optional verification (needs your approval).** In a throw-away virtual environment separate from
  `lgms` (no torch: mxnet 1.9.1, numpy 1.26, gluonnlp 0.10.0, bert-embedding `--no-deps`), run the
  official `BertEmbedding()` on the distinct texts and compare with our cache (max absolute difference,
  cosine similarity). If MXNet does not run there, the replacement is reported as "verified by
  construction only". Downloads: the GluonNLP BERT zip, 494 MB, is reachable (HTTP 200).

### 3.2 RecLMIS (official: torch 1.8, Python 3.9)

- **`fairseq` cannot be built here,** and its silent failure path is critical.
  `nets/transformer/mutihead_attention.py:184` does `from fairseq import utils`. That call sits
  inside a bare `try/except` in `RecLMIS.py:310-320`, which on any error sets `self.aux = False` for
  good. Without fairseq, training would **silently drop the conditioned-reconstruction and CCL
  losses**, i.e. the method's main contribution, and `mode="test"` would raise NameError. This is
  confirmed in the source.
  - fairseq is only used for `utils.softmax(x, dim)`, which is `F.softmax(x.float(), dim)`.
  - Fix: our driver registers a one-function stand-in module `fairseq.utils` with exactly that
    function, so the math is identical.
  - Guard: the driver also checks every step that the reconstruction losses are present and finite,
    so a silent fallback would stop the run.
- **Config bug in the training script.** `train_model.py:17-22` loads `Config_covid19` for any
  argument other than `Config_monuseg` or `Config_Kvasir_Clinic`. `-c Config_MosMedPlus` therefore
  trains MosMed with the **QaTa** settings, while the log still prints the MosMed file name. Our
  driver imports the correct config module directly.
- **Missing packages:**
  - `clip` (OpenAI, git install; adds ftfy and regex);
  - `thop` (imported, unused).

  Both dry-runs are clean (`clip-1.0 ftfy-6.3.1 regex-2026.9.29`; `thop-0.1.1`).
- **Other compatibility points:**
  - `torch.load` now defaults to `weights_only=True`; the checkpoints are plain dicts, to be verified.
  - The deprecated `scipy.ndimage.interpolation`/`morphology` and `timm.models.layers` still import.
  - `Image.ANTIALIAS` is only in an unused function.
  - If `nets/ViT-B-32.pt` is missing, `torch.jit.load` raises an error that skips the intended
    fallback, so the file must be in place.
- **Pretrained weight:** CLIP ViT-B/32 from
  `openaipublic.azureedge.net/clip/models/40d36571.../ViT-B-32.pt` is reachable (HTTP 200, 353,976,522
  bytes). It is stored outside git; only the text tower is used.

### 3.3 DD-CMD (official: Python 3.8, torch 1.12, pytorch_lightning 1.9, transformers 4.24)

- **requirements.txt is in conda syntax,** so `pip install -r` fails. Packages it does not list:
  pyyaml, Pillow and tensorboard are already present; thop is optional.
- **Dry-runs (none changes torch):**

| Package | Dry-run result | Note |
|---|---|---|
| `pytorch_lightning==1.9.0` + `torchmetrics==0.10.3` (official pins) | wheels, installs | first choice: keeps the official Lightning hooks unchanged; must be tested under torch 2.14 |
| `pytorch_lightning 2.6.6` (fallback) | installs | needs a port: `*_step_end` and `*_epoch_end(outputs)` were removed in 2.x (`creratemodel.py:84-126`) |
| `transformers==4.24.0` | wheels (tokenizers 0.13.3 aarch64 wheel) | downgrades `huggingface_hub` 2.1.1 → 0.36.2 |
| `transformers 4.57.6` (latest 4.x) | wheels | also downgrades `huggingface_hub` to 0.36.2 |
| `transformers 5.20.1` | wheels, keeps hub 2.1.1 | major version: changed defaults; not recommended |
| `monai==1.0.1` (official) | installs | will not work with numpy 2 / torch 2.14 |
| `monai 1.6.1` | installs | the APIs used (`UnetrUpBlock`, `UnetOutBlock`, `DiceCELoss`, dictionary transforms) are stable |

- **`huggingface_hub` downgrade.** In `lgms`, `huggingface_hub` is required by `timm` and
  `segmentation_models_pytorch`. Installing any transformers 4.x into `lgms` would therefore change a
  package that Phase 0 code depends on. See decision D2: a separate overlay environment avoids this.
- **Pretrained weights.** `facebook/convnext-tiny-224` and
  `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` (pytorch_model.bin, 440 MB) are
  reachable on huggingface.co (HTTP 200). They go to the HF cache, outside git.
- **No released checkpoints:** `pretrained_models/note.txt` says they will be shared later.

### 3.4 Reachability summary (HTTP HEAD from the login node)

| Resource | Status |
|---|---|
| PyPI, GitHub, huggingface.co | 200 |
| CLIP ViT-B-32.pt (openaipublic) | 200 |
| ConvNeXt-Tiny, BiomedBERT, bert-base-uncased (HF) | 200 |
| GluonNLP BERT zip (MXNet S3) | 200 |

## 4. Official training settings and differences from our rules

Our rules:
- select the checkpoint on validation Dice;
- score with the common 224 × 224 evaluator at threshold 0.5;
- use seeds 1001, 1002 and 1003;
- use the Phase 0 common 224 × 224 image/mask preparation;
- never tune on the test set.

| | LViT-T | RecLMIS | DD-CMD |
|---|---|---|---|
| Input | 224×224, BGR 3-ch via cv2, bilinear resize, /255 | 224×224, BGR 3-ch via cv2, bilinear resize, /255 | 224×224, MONAI loader, per-image z-score, grey repeated to 3 ch |
| Mask | bilinear resize, > 0 → 1 | bilinear resize, > 0 → 1 | nearest resize, > 0.5 |
| Augmentation | p 0.5: rot90 + flip; else p 0.5: rotate ±20° | Gaussian blur, photometric distortion, rot90/flip or ±20° rotation | zoom, rotate, Gaussian noise, flips, rot90 |
| Loss | 0.5 Dice + 0.5 weighted BCE | 5·(0.5 Dice + 0.5 BCE) + 0.2 CCL + 1 text-rec + 1 image-rec | DiceCE (1:1) |
| Optimizer / LR | Adam; lr 3e-4 for Covid19 per the config comment (default value 1e-3 is MoNuSeg's); no MosMed setting | Adam 3e-4 | AdamW 5e-5 |
| Schedule | cosine warm restarts, T_0 = 10, min 1e-4 | QaTa: cosine warm restarts T_0 = 10, min 1e-4; MosMed: 0.99^epoch (≥ 0.1) | cosine, T_max 200, min 1e-6 (never fully anneals at 160 epochs) |
| Batch size | 2 | QaTa 32, MosMed 20 | 8 |
| Max epochs | 2000 | QaTa 2000, MosMed 400 | 160 |
| Early stopping | patience 50 on val Dice | patience QaTa 100, MosMed 30, on val Dice | patience 50 on **val_MIoU** |
| Checkpoint selection | best val Dice (batch-mean, thr 0.5; from epoch 6 on) | best val Dice (batch-mean, thr 0.5) | best **val_MIoU** (global pixel IoU over the epoch) |
| Seed | 666, cudnn deterministic | 3407, cudnn deterministic | 42 hard-coded (config seed ignored), not deterministic |
| Resume | none | `resume_path` never defined (not usable) | none (Lightning could, but no `ckpt_path` passed) |
| Test-time text | yes | yes | yes |
| Test script | does not run as shipped (`test_session` undefined); JPG output | `test_model.py` (README says `test.py`); no prediction files | `evaluate.py`; PNGs named by index, probably transposed (MONAI reader) |
| Data loader workers | 8 | 8 | 0 (hard-coded) |

Planned handling:

1. **Input data.** All three methods read the Phase 0 `common224` images and masks, through symlinked
   folder views in the layout each official loader expects. QaTa source images are already
   224 × 224. For MosMed, `common224` used the same `cv2` bilinear resize and `> 0` mask rule as
   LViT and RecLMIS, so for these two the official resize becomes the identity and nothing changes.
   DD-CMD's own resize (MONAI) also becomes the identity. This is a small, documented deviation for
   DD-CMD and makes all six methods see the same pixels.
2. **Seeds.** 1001, 1002 and 1003 replace 666, 3407 and 42. Each method keeps its own determinism
   flags.
3. **Checkpoint selection.** In all three drivers, after every epoch, the mean per-case
   validation Dice at threshold 0.5 is computed with `src/lgms/metrics.py`, the same code as in Phase 0.
   - For LViT-T and RecLMIS this is the official criterion, minus a batch-weighting detail (the official
     mean of batch means over a shuffled val loader).
   - For DD-CMD it replaces val_MIoU (decision D3).
   - Early stopping uses the same quantity, with each method's official patience.
4. **Test set.** It is touched only once, after training, with the selected checkpoint.
5. **Everything else** keeps the official values: loss, optimizer, LR, schedule, batch size,
   augmentation, maximum epochs and patience. The exceptions are decisions D4 and D6.

Reference numbers (each paper's own metric definition, for orientation only):
- LViT README: LViT-T QaTa Dice 83.66 / IoU 75.11, MosMed 74.57 / 61.33.
- DD-CMD paper (arXiv 2608.11335): QaTa 91.46 / 84.26, MosMed 81.95 / 69.42.
  Its Lightning metric accumulates pixels over the whole epoch (a pooled Dice, not the mean per-image
  Dice we report), which likely explains part of the gap.
- RecLMIS: the numbers are not in the README and are to be read from the paper's table (placeholder).

## 5. The 12-hour limit

No method has a usable resume function. Our drivers add one without changing the training process:

- **Every epoch** they save `last.pth`, containing:
  - model, optimizer and scheduler state, and the epoch;
  - best validation Dice and best epoch, and the early-stopping counter;
  - the Python, NumPy, torch CPU and CUDA RNG states and the DataLoader generator state.

  The best checkpoint is saved separately.
- **For DD-CMD,** Lightning's own `trainer.fit(ckpt_path=last.ckpt)` restores optimizer, scheduler,
  loop counters and callback state (EarlyStopping counter, ModelCheckpoint best). Only a wall-clock
  stop callback is added.
- **Wall-clock guard.** Before each epoch, if the elapsed time plus twice the longest epoch so far
  would pass 11 h 30 min, the job saves and ends cleanly as "segment finished, training not
  finished". The ntfy message then says "👉 RUN" the next segment.
- **Next segment.** The user submits the next segment by hand with the submit helper. It resumes from
  `last.pth`. This is the same pattern as the Phase 0 nnU-Net segments: one job per segment, nothing
  resubmits itself.
- **Equivalence, not identity.** A resumed run continues the same optimisation with the same states.
  It is equivalent in distribution but not bit-identical, because cuDNN kernels and DataLoader worker
  timing are not deterministic. This is stated in the report.
- **Resume test.** It is part of each quick check: train k epochs, stop, resume, and compare the
  restored state.

Rough time per run on one GB10 GPU. **These are estimates, not measurements;** they will be replaced
with the measured seconds per epoch from the quick checks. The Phase 0 reference is U-Net on QaTa at
about 116 s per epoch (train + val).

| Method | QaTa-COV19 | MosMedData+ | Main uncertainty |
|---|---|---|---|
| LViT-T | ~3-5 min/epoch (batch 2, 2858 iterations); 100-250 epochs → **likely 6-20 h, 1-2 segments** | ~1-2 min/epoch → 2-6 h | stopping epoch (patience 50) |
| RecLMIS | ~1.5-2.5 min/epoch; patience 100 with LR restarts → 200-400 epochs → **5-15 h, 1-2 segments** | ~0.5-1 min/epoch, patience 30 → 1-4 h | stopping epoch |
| DD-CMD | ~2-4 min/epoch with 0 workers; ≤ 160 epochs → **≤ 6-11 h** | ~1 min/epoch → ≤ 3 h | data loading with 0 workers (D6) |

Estimated total: roughly 100-150 GPU hours for the 18 runs, i.e. about 25-35 jobs including extra
segments. Under the 1-running + 1-waiting rule this takes about 1.5-2.5 weeks of wall time.

## 6. Exporting predictions to the Phase 0 format

The format is the same as Phase 0 (`src/lgms/runner.py`):
- `val_probs.npz` / `test_probs.npz` hold one float16 224 × 224 foreground probability (sigmoid of
  the logits) per `common224` sample id.
- They are scored by the unchanged `src/evaluate.py`, which writes `eval_<split>/per_case.csv`,
  `summary.json` and `masks/<id>.png`.

Layout: `runs/{lvit,reclmis,ddcmd}/<dataset>/seed<k>/`, holding the training log, `epochs.csv`, the
best checkpoint, `last.pth`, the probability files and the `eval_val/` and `eval_test/` folders.

**Alignment guard.** During export, the target mask produced by each method's own data pipeline must
equal the `common224` mask of the same id, pixel for pixel. Otherwise the export stops. This catches:
- index-based or reordered file names (DD-CMD names files by index);
- transposed arrays (MONAI's PIL reader);
- flips;
- QaTa `mask_` key errors.

The quick checks also confirm that a perfect prediction scores Dice 1 in `src/evaluate.py`.

## 7. Statistical analysis (fixed and committed before any Phase 1 test result exists)

A new script `src/analyze_phase1.py` and tests `src/test_analyze_phase1.py` will be added.
`src/analyze_phase0.py` is not modified. Its functions (bootstrap CI, paired t, Wilcoxon, Holm) are
imported, so both phases use the same code.

- **Methods**, in this fixed order: U-Net, TransUNet, nnU-Net, LViT-T, RecLMIS, DD-CMD.
  - Phase 0 inputs: the existing `runs/{unet,transunet,nnunet}/<dataset>/seed<k>/eval_test/per_case.csv`,
    used unchanged.
  - Every run must contain exactly the test ids, and all methods must have the same ids.
- **Main tables** (`phase1_main_results.csv` for the 3 language-guided methods,
  `final_six_method_results.csv` for all 6):
  - Dice mean ± SD over the 3 per-seed test means;
  - Dice 95% CI from a per-case bootstrap of the seed-averaged Dice (10,000 resamples, seed 0, as in
    Phase 0);
  - mIoU, HD95 and ASSD means;
  - the six per-seed Dice values;
  - "Text at train / test": No / No for the image-only methods, Yes / Yes for the language-guided ones.
- **Pairwise comparisons** (`final_six_method_pairwise.csv`), 15 pairs per dataset, difference
  = A − B with A earlier in the order above:
  1. Average each test case over the 3 seeds within each method.
  2. Align the cases by id.
  3. Compute the per-case Dice difference (percentage points).
  4. 95% CI = mean ± t(0.975, n−1)·SD/√n.
  5. Primary test: two-sided paired t-test, with **Holm correction across the 15 pairs of the dataset**.
  6. Supplementary: Wilcoxon signed-rank test, with its own Holm correction across the same 15 pairs.
  7. **Significance level: two-sided α = 0.05 after Holm, fixed now.** In Phase 0 the 5% level was only
     confirmed after the analysis.
  8. mIoU, HD95 and ASSD: paired mean difference with the same t-based 95% CI, and no p-values (as in
     the plan). Because HD95 and ASSD are heavy-tailed (an empty prediction scores 316.78 px), a
     per-case bootstrap percentile CI (10,000 resamples, seed 0) is reported next to the t-CI.
- **Figures:**
  - six-method Dice with 95% CI, one per dataset;
  - seed stability: the 3 per-seed Dice values of each language-guided method (and the 3 baselines for
    reference);
  - optional case figure: U-Net, nnU-Net, LViT-T, RecLMIS and DD-CMD. Cases are chosen by the Phase 0
    rule (quantiles of the Dice averaged over seeds and methods). Each row shows the image, the
    ground truth with a yellow dashed contour, and each method's seed-1001 prediction, with Dice and
    HD95 on every panel.
- **Tests:**
  - synthetic data with known answers (Holm over 15, pair direction, id alignment, mismatch errors);
  - a regression check: run with only the 3 Phase 0 methods on the real Phase 0 per-case files and
    compare with `results/phase0_main_results.csv`. Main-table numbers must match exactly; pairwise
    raw p-values must match, while the Holm values differ because the family is 15 instead of 3. This
    uses only Phase 0 results that are already published.
- The script, tests and this specification are committed and pushed **before the first Phase 1
  training job is submitted**. Test-set inference runs inside the training jobs, so this ordering
  guarantees that the analysis is fixed before any Phase 1 test number exists.

## 8. Execution plan

Each step ends with a commit and push. Steps marked **(you)** need you.

1. **Environment** (after D2 and D7). Create the environment and install the approved packages, with a
   dry-run before each install. Check that torch is still 2.14.0+cu130. Download the weights outside
   git. Add the new commits to `external/VERSIONS.md`.
2. **Data views.**
   - Build the symlinked `common224` views for each loader under `~/datasets/phase1/` (no new pixels,
     no new split).
   - Generate the DD-CMD CSV files from our xlsx and check them row by row.
   - Precompute the LViT-T text cache and run its tests (shape 10 × 768 for all distinct texts, padding
     rule).
   - Write `reports/PHASE1_DATA_SUMMARY.md`.
3. **Drivers.** `src/phase1/train_{lvit,reclmis,ddcmd}.py`. Each imports the official model, dataset
   transforms and loss from `external/` and reproduces the official loop. It adds the seed, the common
   validation Dice, early stopping, the resume/wall-clock guard, `epochs.csv`, export and the alignment
   guard. The `.slurm` files source `env.sh`, set `JOB_LABEL` and `NOTIFY_RESULT`, and have
   `--time=12:00:00`. A Phase 1 plan file and submit helper are added as new files; the Phase 0 ones
   are not changed.
4. **Analysis code** (section 7): written, tested on synthetic data and on the Phase 0 regression
   check, committed.
5. **Quick checks (you: 1 job, or 3 if run separately).** For each method, on a small subset plus
   2-3 full epochs:
   - pairing check;
   - forward pass;
   - text reaches the language branch: gradients on the text-branch parameters are non-zero, and
     replacing the text changes the output;
   - for RecLMIS, the reconstruction losses are present;
   - loss decreases and validation Dice rises;
   - masks are valid;
   - export and alignment guard;
   - resume test;
   - measured seconds per epoch.

   Results go in `results/phase1_quick_check.json` and `reports/PHASE1_MODEL_SETUP.md`. Then **the
   configurations are frozen** (commit) and the time estimates are updated.
6. **18 runs (you: one job at a time).** The submission order puts long QaTa runs at night, as in
   Phase 0. Extra segments are submitted by hand when a notification asks for one.
7. **Analysis and report.** Run `analyze_phase1.py` and write `reports/PHASE1_RESULTS.md`, answering
   the 8 questions of the plan without causal claims about text.

## 9. Decisions needed from you

| # | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | LViT-T text features | (a) PyTorch re-implementation of `bert_embedding` with the same BERT-base uncased weights, precomputed; (b) the same plus an attempt to verify against official MXNet output in a throw-away env (installs mxnet 1.9.1 / numpy 1.26 / gluonnlp 0.10.0 there, never in `lgms`) | (b): try the verification; if MXNet does not run, keep (a) and say so in the report |
| D2 | Where to install | (a) into `lgms` (downgrades `huggingface_hub` 2.1.1 → 0.36.2, used by timm and segmentation_models_pytorch in Phase 0); (b) overlay venv `~/envs/lgms_p1` created with `--system-site-packages` on top of `lgms`, which uses `lgms`'s torch 2.14 unchanged and leaves `lgms` itself untouched | (b) |
| D3 | DD-CMD selection metric | (a) per-case val Dice for both checkpoint selection and early stopping (patience 50); (b) per-case val Dice for selection, official val_MIoU for early stopping | (a): one rule for all six methods |
| D4 | LViT-T learning rate on MosMedData+ (the code only names MoNuSeg 1e-3 and Covid19 3e-4) | 3e-4 or 1e-3 | 3e-4, the same as QaTa (both are COVID lung datasets with the same text format); chosen now, not tuned |
| D5 | Runs longer than 12 h | segmented training with resume (section 5), as for nnU-Net in Phase 0, with each segment submitted by hand | approve; it replaces the U-Net-specific evaluation-only rule of CLAUDE.md for the Phase 1 methods |
| D6 | DD-CMD data-loader workers | official 0, or 8 with seeded workers (same augmentation distribution, different random stream, much faster) | 8 |
| D7 | Packages to install (all dry-run clean, torch unchanged) | `transformers 4.57.6` (+ tokenizers 0.22.2, hub 0.36.2), `pytorch_lightning 1.9.0` + `torchmetrics 0.10.3` (fallback 2.6.6 + port), `monai 1.6.1`, `clip` (git `d05afc4`) + `ftfy`, `regex`, `thop` | approve the list |
| D8 | Weights (outside git: HF cache, `~/pretrained/`) | CLIP ViT-B-32.pt (354 MB), ConvNeXt-Tiny, BiomedBERT (440 MB), bert-base-uncased (440 MB) | approve the downloads |

## 10. Decisions taken (2026-10-18) and status

| # | Decision | Status |
|---|---|---|
| D1 | (b): PyTorch re-implementation with the same BERT-base uncased weights, precomputed; try to verify against official MXNet in a throw-away environment, without spending long on it | **Verification not possible.** In a separate venv without torch, `mxnet 1.9.1` (aarch64 wheel) installs but does not import: it needs the ARM Performance Libraries (`libarmpl_lp64_mp.so`), which the cluster does not have. `gluonnlp 0.10.0` cannot be built because there is no `gcc`. The throw-away venv was deleted. (a) is kept and **the report states that the text features could not be checked against the official MXNet output.** The cache (`src/phase1/lvit_text.py`) holds 495 distinct padded texts, each 10 × 768, and covers all 11,987 samples. None of the texts has fewer than 9 words, so the `EOF XXX` padding never applies. |
| D2 | (b): overlay venv `~/envs/lgms_p1` | Done. torch is still `2.14.0+cu130`; `lgms` is unchanged (`huggingface_hub` 2.1.1). Details in `external/VERSIONS.md`. |
| D3 | (a): DD-CMD checkpoint selection **and** early stopping on the mean per-case validation Dice (patience 50) | Implemented (`configs/ddcmd.yaml`, monitor `val_case_dice`). The official `val_MIoU` is still logged. Reported as a deviation from the official setting. |
| D4 | LViT-T learning rate 3e-4 (QaTa) / **1e-3 (MosMed)**, batch size 2 | LViT paper Sec. IV-C, checked on 2026-10-18: "The initial learning rate is set to 3e-4 for the QaTa-COV19 dataset and 1e-3 for the MosMedData+ dataset" and "The default batch size is 24 for the QaTa-COV19 dataset and the MosMedData+ dataset". Batch size 2 follows `Config.py` ("For LViT-T, 2 is better than 4"). This is our judgement; both sources are recorded in `configs/lvit.yaml` and will be in `PHASE1_MODEL_SETUP.md`. |
| D5 | Segmented training with resume; each segment submitted by hand. After each method's first real resume, check that epoch numbers, learning rate and loss are continuous (as for Phase 0 nnU-Net) | Implemented. Also tested in the quick check (6 epochs uninterrupted vs 3 + 3). |
| D6 | DD-CMD: 8 seeded data-loader workers | Implemented (MONAI `worker_init_fn`; base seed from a seeded generator). Reported as a change of the augmentation's random order, not of the method. |
| D7 | Install the listed packages; the quick check must show that pytorch_lightning 1.9 and transformers 4.57 train normally under torch 2.14. If not: stop and report, no other solution | Installed (dry-run first). The training check is part of the quick check. |
| D8 | Download the weights, outside git | Done (`external/VERSIONS.md`). |

Further findings during the implementation:
- **RecLMIS reconstruction needs a GPU.** `_mask_feat` hard-codes `.cuda()` (`nets/RecLMIS.py:176`), so
  on a CPU the reconstruction raises an error and the official bare `except` switches it off. The CPU
  smoke test on the login node showed exactly this (`aux_active: false`), which confirms that the guard
  works. On the GPU the quick check verifies that `aux` stays on and that all three auxiliary losses are
  present and finite in every step. The training driver stops otherwise.
- **The official RecLMIS reconstructor replaces a NaN reconstruction loss with 0.** The driver counts such
  steps (`rec_loss_zero_steps` in `training_summary.json`).
- **Code layout.** The data loaders follow the Phase 0 pattern: the `common224` arrays are read in
  memory, and model, loss, scheduler and augmentation are imported from the official repositories. The
  symlinked folder views of section 8 step 2 are therefore not needed, and neither is the DD-CMD CSV
  conversion. For DD-CMD, arrays are fed in (H, W) orientation; see `configs/ddcmd.yaml`.
- **Quick checks never touch the test set.** They are scored on validation only (`--eval-splits val`).
- **Submission.** Phase 1 jobs are submitted with `slurm/submit_next_p1.sh` (new; plan
  `slurm/phase1_plan.tsv`). The Phase 0 tool is unchanged. The 18 main runs are added to the plan only
  after the quick-check results and the measured timetable have been confirmed.
