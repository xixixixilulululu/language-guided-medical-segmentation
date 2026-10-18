# Phase 0 — Common Data and Model Setup (Tasks 0.2 and 0.3)

Status: data prepared, all three models set up, quick checks passed on both datasets. The 18 main runs
(Task 0.4) were submitted on 2026-10-11 in the order of section 6 (Slurm jobs 212–229) and are running.

| Method | Source | Configuration | Input | Pretraining | Notes |
|---|---|---|---|---|---|
| U-Net | LViT (`nets/UNet.py`, commit `ba90775`) | binary U-Net, sigmoid output, 14.75 M params | 224×224, gray → 3 channels | none | batch 24; Adam, lr 3e-4 (QaTa) / 1e-3 (MosMed), LViT paper Sec. IV-C; cosine warm restarts; early stopping 50 epochs |
| TransUNet | Official (commit `02ef001`) | R50-ViT-B_16, 2 classes, n_skip 3, 105.3 M params | 224×224, 1 channel (net repeats to 3) | ImageNet-21k `R50+ViT-B_16.npz` | official recipe: batch 24, SGD 0.01, poly, 150 epochs; best epoch by val Dice |
| nnU-Net | Official `nnunetv2==2.8.1` | 2d, PlainConvUNet 6 stages, 20.6 M params | 224×224 prepared data (patch 224×224, batch 65) | none | 1000 × 250 iterations; mirroring on (train + TTA); best of 20 snapshots by val Dice |

Configuration files: [`configs/unet.yaml`](../configs/unet.yaml), [`configs/transunet.yaml`](../configs/transunet.yaml),
[`configs/nnunet.yaml`](../configs/nnunet.yaml). The common rules for all methods:

- training uses the LViT train split only;
- the checkpoint is selected by the mean per-case **validation Dice** of the common evaluator, and nothing else;
- the test set is predicted once, with the selected checkpoint, and scored by the same evaluator.

---

## 1. Task 0.2 — one common 224×224 dataset

`src/prepare_common_224.py` writes `~/datasets/common224/<dataset>/{train,val,test}/{images,masks}/<id>.png`
and one id list per split. Every method reads these files. The preparation repeats the LViT loader
(`Load_Dataset.py`, `ImageToImage2D.__getitem__`) exactly:

| Step | Image | Mask |
|---|---|---|
| read | `cv2.imread(path)` (3-channel uint8) | `cv2.imread(path, 0)` (grayscale) |
| resize | `cv2.resize(·, (224, 224))`, default `INTER_LINEAR` | same |
| binarise | — | `mask <= 0 → 0`, `mask > 0 → 1` (after resizing) |
| store | one channel (the three channels are identical in both datasets; asserted for every image) | uint8 {0, 1} |

Methods that need three channels (the LViT U-Net) duplicate the gray channel in their loader. QaTa files are
already 224×224, so the resize is the identity for QaTa. For MosMed the anti-aliased border values 12/17/21
(Task 0.1) become lesion, as in LViT.

Check against Task 0.1 (`results/phase0_common224_summary.json`). Counts and lesion areas are identical to
`PHASE0_DATA_SUMMARY.md`:

| Dataset | Train / Val / Test | Mean lesion area (Train / Val / Test) | Empty masks |
|---|---|---|---|
| QaTa-COV19 | 5716 / 1429 / 2113 | 12.2% / 11.9% / 11.9% | 0 |
| MosMedData+ | 2183 / 273 / 273 | 1.7% / 1.3% / 1.8% | 1 (Train) |

nnU-Net receives the same PNG files (`src/prepare_nnunet.py`). The datasets are Dataset501_QaTaCOV19 and
Dataset502_MosMedDataPlus:

- `imagesTr` holds train + val and `imagesTs` holds test.
- A custom `splits_final.json` with a single fold 0 = LViT train / LViT val. After preprocessing it was
  verified to hold 5716 / 1429 and 2183 / 273 cases.
- The channel name is `"gray"`, so nnU-Net applies z-score normalisation. The PNGs are not in Hounsfield
  units, so its CT scheme would be wrong.

---

## 2. Task 0.3 — training settings

### U-Net (LViT)

| Item | Setting | Source |
|---|---|---|
| Network | `external/LViT/nets/UNet.py`, `UNet(n_channels=3, n_classes=1)` | LViT repo |
| Pretraining | none | LViT `Config.pretrain = False` |
| Loss | `WeightedDiceBCE(dice_weight=0.5, BCE_weight=0.5)` on sigmoid output | LViT `utils.py`, `train_model.py` |
| Optimizer | Adam, no weight decay | LViT `train_model.py` |
| Learning rate | **QaTa 3e-4, MosMed 1e-3** | **LViT paper, Sec. IV-C** |
| Batch size | **24 (both datasets)** | **LViT paper, Sec. IV-C** |
| Scheduler | LViT `CosineAnnealingWarmRestarts`, T0 = 10, T_mult = 1, eta_min = 1e-4, stepped once per epoch | LViT `utils.py`, `Config.cosineLR` |
| Augmentation | LViT `RandomGenerator`: 50% rot90 + flip, 25% rotation ∈ [−20°, 20°) (nearest), 25% none | LViT `Load_Dataset.py` |
| Epochs | max 2000; stop when val Dice has not improved for 50 epochs; "best" counted from epoch 6 on | LViT `Config.py`, `train_model.py` |

**Batch size and learning rate.** They follow the LViT paper (Li et al., IEEE TMI 2023), Sec. IV-C
"Implementation Details": *"The initial learning rate is set to 3e-4 for the QaTa-COV19 dataset and 1e-3 for
the MosMedData+ dataset … The default batch size is 24 for the QaTa-COV19 dataset and the MosMedData+
dataset."*
- `Config.py`'s `batch_size = 2` is **not** used. Its comment says it applies to LViT-T ("For LViT-T, 2 is
  better than 4").
- The paper gives no separate settings for its U-Net comparison model. We use these general settings.
- After training, our U-Net is checked against the U-Net scores the paper reports (section 3 below).

The model, loss and scheduler are imported unchanged from `external/LViT`. The image-only part of the LViT
data loader is re-implemented line by line in `src/train_unet.py`, because the original module imports
`bert_embedding` for the text branch.

### TransUNet (official)

| Item | Setting | Source |
|---|---|---|
| Network | `VisionTransformer`, `R50-ViT-B_16`, 2 classes, n_skip 3, patch grid 14×14 | official `train.py` |
| Pretraining | ImageNet-21k `R50+ViT-B_16.npz` (`net.load_from`), official Google URL, sha256 `ff009bf3…eaafbe` | official README |
| Loss | 0.5 × CrossEntropy + 0.5 × Dice (softmax, 2 classes) | official `trainer.py` |
| Optimizer | SGD, lr 0.01, momentum 0.9, weight decay 1e-4 | official `trainer.py` |
| Scheduler | poly, `lr = 0.01 × (1 − iter / max_iter)^0.9`, updated every iteration | official `trainer.py` |
| Batch size | 24 (official default; peak GPU memory 9.1 GB, well within the GB10's memory) | official `train.py` |
| Augmentation | official `RandomGenerator` (same rot/flip/rotate scheme as LViT) | `dataset_synapse.py` |
| Intensity | uint8 / 255 → [0, 1], one channel (the network repeats it to 3) | — |
| Epochs | 150; validation Dice after every epoch, the best epoch is kept | official `train.py` + agreed plan |

The official trainer has no validation and keeps the last epoch. Following the agreed plan, we only add a
per-epoch validation Dice and keep the best epoch. The training schedule itself is unchanged.

### nnU-Net v2 (official, 2d)

| Item | Setting |
|---|---|
| Plans | default planner, `2d` only. Patch 224×224, batch 65, PlainConvUNet with 6 stages (32–512 features), z-score normalisation, no resampling (spacing 1×1) |
| Trainer | `nnUNetTrainerLGMS` = default `nnUNetTrainer` + seed + 50-epoch snapshots (`src/nnunet_ext/`) |
| Recipe (unchanged nnU-Net defaults) | Dice + CE with deep supervision; SGD lr 1e-2, momentum 0.99, Nesterov, weight decay 3e-5; poly LR; 1000 epochs × 250 iterations; default 2D augmentation |
| Mirroring | kept (decision of 2026-10-11): training augmentation, and TTA both for the validation predictions that select the checkpoint and for the test predictions |
| Checkpoint selection | 20 snapshots (epochs 50, 100, …, 1000); each predicts the **full** validation set; the highest mean per-case Dice of the common evaluator wins (ties → earlier epoch). nnU-Net's own `checkpoint_best.pth` (EMA pseudo-Dice on random patches) is not used |
| Seed | `LGMS_SEED` seeds python / numpy / torch and every augmentation worker. nnU-Net's multi-process loader still delivers batches in non-deterministic order, so runs are seeded but not bit-reproducible |

### Notes on faithful but unusual official behaviour

- **Worker seeding (LViT and TransUNet).** Both official loaders use
  `worker_init_fn = random.seed(seed + worker_id)`, and workers are recreated every epoch. As a result,
  Python's `random` stream, which chooses between rot/flip, rotation and no augmentation, restarts
  identically in every epoch.
  - Shuffling still changes which sample receives which branch.
  - The numpy draws (rotation angle, k, flip axis) differ between epochs.
  - This is kept as in the official code.
- **LViT "best" rule.** A best model is only recorded from the 6th epoch on (`epoch + 1 > 5`), and the
  50-epoch patience is counted from that best epoch. Both are kept.
- **LViT validation Dice.** LViT selects on its own batch-averaged Dice of thresholded masks. We use the mean
  per-case validation Dice of the common evaluator, the selection rule of the plan. Apart from the averaging,
  the two are the same quantity: thresholded at 0.5, per image.

---

## 3. Reference values from the LViT paper (for checking only)

LViT paper, Table II (100% labels, test set), values in %. These numbers are used **only** to check
afterwards that our pipeline has no gross error. They are never used to tune anything.

| Dataset | Model in LViT Table II | Dice (%) | mIoU (%) |
|---|---|---:|---:|
| QaTa-COV19 | U-Net | 79.02 | 69.46 |
| MosMedData+ | U-Net | 64.60 | 50.73 |
| QaTa-COV19 | TransUNet | 78.63 | 69.13 |
| MosMedData+ | TransUNet | 71.24 | 58.44 |
| QaTa-COV19 | nnUNet | 80.42 | 70.81 |
| MosMedData+ | nnUNet | 72.59 | 60.36 |

The paper lists the same parameter counts as our models: U-Net 14.8 M (ours 14.75 M) and TransUNet 105 M
(ours 105.3 M). The LViT README repeats the U-Net values. The comparison of our U-Net seed-1001 results
with these values is in section 7.

---

## 4. Common evaluator (Task 0.5 code, checked now)

`src/lgms/metrics.py` and `src/evaluate.py` define the evaluator used for all three methods:

- **Input and threshold:** the lesion probability map is thresholded at `prob > 0.5`. For the two-class
  softmax models this equals argmax.
- **Dice:** `2|P∩G| / (|P| + |G|)`.
- **mIoU:** the per-case lesion IoU averaged over cases, as in LViT `test_model.py`.
- **HD95 / ASSD:** medpy 0.5.2, spacing 1, i.e. in pixels of the 224×224 image.
- **Empty prediction:** Dice = mIoU = 0 and HD95 = ASSD = 224·√2 = 316.78 px.

In medpy 0.5.2, **ASSD is the mean of the pooled surface distances P→G and G→P**. medpy 0.4.x averaged the
two directed means instead, which gives slightly different values for shapes with unequal surface sizes. We
use medpy 0.5.2 unchanged.

**Hand-made mask test** (`src/test_metrics.py`, output in `results/phase0_metric_check.txt`). Expected values
are analytic where possible, otherwise from an independent brute-force surface-distance implementation. All
checks pass:

| Case | Dice | mIoU | HD95 | ASSD |
|---|---:|---:|---:|---:|
| identical masks (40×40 square) | 1 | 1 | 0 | 0 |
| disjoint single pixels, 10 px apart | 0 | 0 | 10 | 10 |
| disjoint 10×10 squares, 30 px gap | 0 | 0 | 40 | 35.5 |
| half overlap (10×10 shifted 5 px) | 0.5 | 0.3333 | 5 | 2.5 |
| empty prediction | 0 | 0 | 316.78 | 316.78 |
| empty ground truth | 0 | 0 | 316.78 | 316.78 |
| both empty | 1 | 1 | 0 | 0 |
| 5 random multi-blob pairs | = reference | = reference | = reference | = reference |
| threshold on [0, .4999, .5, .5001, 1] | → [0, 0, 0, 1, 1] | | | |

End-to-end checks on the real MosMed validation split:
- **Ground truth as prediction:** Dice 1, HD95 0.
- **All predictions empty:** Dice 0, HD95 316.78 for all 273 cases.

**Independent agreement with nnU-Net.** After its quick-check training, nnU-Net's own final validation
reported a mean Dice of 0.69766 (QaTa) and 0.55578 (MosMed). Our evaluator, scoring the same checkpoint's
validation predictions, gives 0.6977 and 0.5557.

---

## 5. Quick check (Task 0.3)

All checks ran on the GPU nodes through Slurm (`slurm/quick_check_*.slurm`, jobs 206, 210, 211; records in
`results/phase0_quick_check.json`). U-Net was re-checked with the final configuration (batch 24,
paper learning rates) in job 211.

### Steps 1–4: one batch, one forward pass, shapes, loss (+ one backward pass)

| Method | Dataset | Batch image | Mask | Output | Loss | Gradients finite | Passed |
|---|---|---|---|---|---:|---|---|
| U-Net | QaTa | 24×3×224×224, [0, 1] | 24×224×224, {0,1} | 24×1×224×224 | 0.671 | yes | ✅ |
| U-Net | MosMed | 24×3×224×224, [0, 1] | 24×224×224, {0,1} | 24×1×224×224 | 0.861 | yes | ✅ |
| TransUNet | QaTa | 24×1×224×224, [0, 1] | 24×224×224, {0,1} | 24×2×224×224 | 0.567 | yes | ✅ |
| TransUNet | MosMed | 24×1×224×224, [0, 1] | 24×224×224, {0,1} | 24×2×224×224 | 0.620 | yes | ✅ |
| nnU-Net | QaTa | 65×1×224×224, z-scored | 65×224×224, {0,1} | 65×2×224×224 (+ 4 deep-supervision outputs down to 14×14) | 0.963 | yes | ✅ |
| nnU-Net | MosMed | 65×1×224×224, z-scored | 65×224×224, {0,1} | 65×2×224×224 (+ 4 deep-supervision outputs down to 14×14) | 1.171 | yes | ✅ |

The nnU-Net check builds `nnUNetTrainerLGMS` exactly as `nnUNetv2_train` does, so it also confirms that the
trainer is found through `nnUNet_extTrainer` and that `do_split()` returns the LViT split (5716 / 1429 and
2183 / 273).

### Step 5: short small-subset training, followed by the complete evaluation pipeline

| Method | Dataset | Subset | Training | Train loss first → last | Best val Dice (subset) |
|---|---|---|---|---|---:|
| U-Net | QaTa | first 48 train / 32 val | 60 epochs (2 it/epoch) | 0.643 → 0.267 | 0.616 |
| U-Net | MosMed | first 48 train / 32 val | 60 epochs | 0.764 → 0.133 | 0.777 |
| TransUNet | QaTa | first 48 train / 32 val | 60 epochs | 0.551 → 0.087 | 0.740 |
| TransUNet | MosMed | first 48 train / 32 val | 60 epochs | 0.602 → 0.080 | 0.480 |
| nnU-Net | QaTa | full data | 3 epochs × 250 it | — | 0.597 → 0.673 → 0.698 (full val, per snapshot) |
| nnU-Net | MosMed | full data | 3 epochs × 250 it | — | 0.507 → 0.519 → 0.556 (full val, per snapshot) |

Every run then went through the complete final pipeline:
- the selected checkpoint (nnU-Net: the best snapshot) predicted the full validation and test sets;
- the common evaluator wrote `per_case.csv` (2113 test rows for QaTa), `summary.json` and the binary PNG
  masks.

These subset models are pipeline tests only. Their test scores are not reported or used.

---

## 6. Measured run times and the 18-run schedule

Hardware: 2 Slurm nodes (`gpu` partition), one NVIDIA GB10 each, one job per node (exclusive). Running
3 U-Net jobs together on one GPU gave no speed-up: 3 packed epochs took 441 s, against about 405 s
for 3 epochs one after another. Every run therefore gets a whole node.

Measured epoch times (full data, final configurations; second epoch, which excludes start-up):

| Method | QaTa-COV19 | MosMedData+ | Notes |
|---|---:|---:|---|
| U-Net (batch 24) | 114.7 s (105 train + 10 val) | 42.7 s | peak GPU memory 6.0 GB |
| TransUNet (batch 24) | 161.3 s (148 + 14) | 59.9 s | peak GPU memory 9.1 GB |
| nnU-Net (250 iterations, batch 65) | 80.3 s | 80.5 s | GPU-bound (96% utilisation); epoch time is independent of dataset size |
| nnU-Net snapshot selection | 50 s / snapshot (1429 cases) | 11 s / snapshot (273 cases) | × 20 snapshots |

Time per run:

| Method | QaTa-COV19 | MosMedData+ |
|---|---:|---:|
| U-Net (early stopping; 100 / 200 / 400 epochs) | 3.2 / 6.4 / 12.8 h | 1.2 / 2.4 / 4.8 h |
| U-Net, upper bound (2000 epochs) | 63.7 h | 23.7 h |
| TransUNet (150 epochs) | 6.8 h | 2.5 h |
| nnU-Net (1000 epochs + selection + final prediction) | 22.9 h | 22.5 h |

U-Net's length depends on early stopping. 200 epochs is the central assumption; 100 and 400 give the range.

Total for the 18 runs: **≈ 190 GPU-hours** (177–217 h for U-Net at 100–400 epochs). On 2 nodes this is
**≈ 96 h ≈ 4.0 days** of wall-clock time (3.7–4.6 days).

Proposed submission order (Slurm starts jobs in order as nodes become free):

1. U-Net QaTa seed 1001 and U-Net MosMed seed 1001 first. Their results, compared with the LViT
   reference above, are available after about 2–13 h, early enough to catch a pipeline problem before
   most of the compute is spent.
2. The 6 nnU-Net runs (the longest).
3. The 6 TransUNet runs.
4. The remaining 4 U-Net runs.

A simulation of this order on two nodes gives a makespan of 95.8 h. The optimal longest-first order gives
95.3 h, so starting with U-Net costs nothing measurable.

Every job sends a phone notification (ntfy) when it finishes, fails or is cancelled. The message carries
the job ID and the best validation Dice.

---

## 7. U-Net on MosMedData+ vs. the LViT paper

*This section is the record made when only seed 1001 was available. The final results over the three
seeds are in [PHASE0_RESULTS.md](PHASE0_RESULTS.md), section 3.*

The first two main runs (U-Net, seed 1001) were compared with the LViT paper as the agreed pipeline check.
On QaTa-COV19 they match the paper. On MosMedData+ our U-Net is 9.3 Dice points **above** the value the
paper reports. This section records what was checked. No setting of any run was changed, and all runs
continue.

**Test-set comparison (seed 1001 only; seeds 1002 / 1003 are pending).** Values in %:

| Dataset | Ours, Dice | LViT paper, Dice | Δ | Ours, mIoU | LViT paper, mIoU | Δ |
|---|---:|---:|---:|---:|---:|---:|
| QaTa-COV19 | 79.12 | 79.02 | +0.10 | 69.68 | 69.46 | +0.22 |
| MosMedData+ | 73.92 | 64.60 | +9.32 | 60.86 | 50.73 | +10.13 |

**The same predictions under different scoring rules** (`src/analysis/lvit_metric_check.py`,
`results/phase0_lvit_metric_check.json`). Dice in %:

| Scoring of the seed-1001 test predictions | QaTa-COV19 | MosMedData+ |
|---|---:|---:|
| Common evaluator (per image, prob > 0.5, 224×224) | 79.12 | 73.92 |
| LViT `test_model.py` algorithm, re-implemented exactly | 79.12 | 73.92 |
| LViT algorithm, threshold ≥ 0.5 | 79.12 | 73.92 |
| Pooled over the whole test set (global Dice) | 87.41 | 78.21 |
| Per image at the native 512×512 resolution (MosMed masks > 0) | — | 72.62 |

### Confirmed facts

1. **The LViT evaluation algorithm gives exactly our values.** `external/LViT/test_model.py` is unchanged
   since the first upload (2022-06-30). It scores each test image separately:
   - batch size 1, at `config.img_size` = 224, prediction = sigmoid output `> 0.5`;
   - Dice = `2·Σ(G·P) / (ΣG + ΣP + 1e-5)`;
   - IoU = sklearn `jaccard_score` of the binary lesion mask;
   - both are summed over images and divided by the number of test images.

   This is the same per-image mean as our common evaluator. Re-implemented on our predictions it
   reproduces our numbers exactly (table above). None of the other scoring variants gives 64.60.
2. **The released LViT code has no MosMedData+ path.** No `.py` file in the LViT repository mentions
   MosMed. `Config.py`, `train_model.py` and `test_model.py` only have the `MoNuSeg` and `Covid19`
   branches. The MosMed numbers in the paper were therefore produced with code that is not public.
3. **We use the same MosMed data as LViT.**
   - The LViT README links the Kaggle mirror `maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset`
     for MosMedData+, and that is the mirror we use.
   - The 2729 image ids in the LViT split lists are exactly the 2729 frames of this mirror (identical sets).
4. **The LViT paper and repository do not describe how the comparison models were trained.**
   - Table II gives one value per comparison model, without standard deviation. Only the LViT variants are
     reported as mean ± std over three runs.
   - In GitHub issue #4 (2022-09) the authors state that some comparison results and models were taken
     from UCTransNet and that the U-Net setting and results follow UCTransNet. UCTransNet does not cover
     QaTa-COV19 or MosMedData+. The U-Net setting used for these two datasets is not published.
   - Issue #18 concerns U-Net *construction arguments*: the reported error comes from passing the ViT
     config to `UNet(...)`. It says nothing about the training settings.
5. **U-Net is not the only comparison model that is low on MosMedData+ in LViT Table II.** MosMed Dice
   values of the comparison models without text:

   | Group | Model | MosMed Dice |
   |---|---|---:|
   | between 63 and 66 | Swin-Unet | 63.29 |
   | | U-Net | 64.60 |
   | | UCTransNet | 65.90 |
   | | AttUNet | 66.34 |
   | between 71 and 73 | TransUNet | 71.24 |
   | | UNet++ | 71.75 |
   | | nnUNet | 72.59 |

   On QaTa-COV19 the same models lie within 78.07–80.42.
6. **No public report of reproducing or failing to reproduce the MosMed U-Net value was found.**
   - No LViT GitHub issue reports re-training U-Net on MosMedData+ (all 61 issues and pull requests checked on 2026-10-12).
   - Later papers that use the LViT benchmark list identical U-Net values (79.02 / 69.46 / 64.60 /
     50.73):
     - FMISeg (arXiv 2509.19719) states that its uni-modal results "are taken from the reproduction
       provided by LViT";
     - RecLMIS (arXiv 2404.02845) lists the same row without stating its source.
7. **The Kaggle mirror is a curated merge, not the original medicalsegmentation.com release**
   (`src/analysis/inspect_mosmed_sources.py`, `results/phase0_mosmed_sources.json`).
   - **Sources.** Its description states that it merges the lesion masks and the corresponding frames of
     three public datasets: MosMedData (Morozov et al.), COVID-19-CT-Seg (Jun Ma et al., Zenodo) and the
     medicalsegmentation.com (MedSeg) data. It also states that "all different types of lesions are mapped
     to white color".
   - **Composition** (from the file names):

     | Source | Slices | Volumes |
     |---|---:|---:|
     | `Morozov_study_*` (MosMedData) | 785 | 50 |
     | `Jun_coronacases_*` | 1351 | 10 |
     | `Jun_radiopaedia_*` | 493 | 10 |
     | `bjorke_*` (MedSeg 100 axial images) | 100 | — |

   - **Slice selection.** Only slices that have a lesion mask are included: 1 of 2729 masks is empty.
   - **Encoding.** Every image is an 8-bit, 3-channel PNG (identical channels), 512×512. The original
     NIfTI / JPG data were therefore converted.
   - **Intensity.** The scaling differs by source and is not documented. Mean grey value and share of
     pixels at 0, in a fixed sample of 60 slices per source:

     | Source | Mean grey value | Pixels at 0 |
     |---|---:|---:|
     | `Jun_coronacases_*` | 40 | 2% |
     | `Jun_radiopaedia_*` | 121 | 9% |
     | `Morozov_study_*` | 91 | 21% |
     | `bjorke_*` | 136 | 5% |

   - **Relation to the "original" link.** The medicalsegmentation.com page that the LViT README calls
     "Link (Original)" now redirects to medseg.ai, which no longer describes the data. Secondary sources
     describe the original MedSeg release as 100 axial images (converted from SIRM JPGs) plus 9 Radiopaedia
     volumes. The Morozov and Jun Ma parts of the mirror therefore do not come from that link.

   This affects all three methods in the same way, because every method reads the same prepared files.
   By fact 3 it is the same data that LViT used.

### Possible causes of the MosMed gap (not confirmed)

- **Different, unpublished training of the LViT comparison models on MosMed** (number of epochs, early
  stopping, learning rate, batch size, input pipeline). This cannot be checked, because the MosMed code
  and the comparison-model settings are not public (facts 2 and 4).
  - Several comparison models in the paper are in the same low range on MosMed (fact 5). This would fit a
    common cause in how those models were trained or evaluated on MosMed, but it does not identify one.
- **Variance of a single run.** The paper's comparison values are single numbers, and ours is one seed so
  far. The spread over our seeds 1001–1003 will show how stable 73.9 is (to be reported in
  PHASE0_RESULTS.md).
- **Not a cause:** the evaluation algorithm (fact 1) and the data (fact 3). The MosMed slice-level split
  (Task 0.1) makes absolute scores optimistic. But LViT used the same split, so the split cannot explain a
  difference between our run and theirs.

**Decision (2026-10-12):** the common evaluator is not changed, and all runs continue unchanged. The
TransUNet and nnU-Net results will be compared with LViT Table II in the same way in PHASE0_RESULTS.md:
- TransUNet: QaTa 78.63 / MosMed 71.24;
- nnUNet: QaTa 80.42 / MosMed 72.59.

---

## 8. Infrastructure notes

- **nnU-Net I/O.** The compute nodes reach `/mnt/cv_data` over NFS. Small-file writes run at about
  2–3 files/s there: nnU-Net preprocessing of QaTa was projected at about 2 h.
  - nnU-Net therefore preprocesses on the node-local NVMe, which took 50 s for both datasets.
  - The result is stored as one tar per dataset in `~/datasets/nnunet/preprocessed_tar/`.
  - Every nnU-Net job unpacks the tar to local disk before training (`slurm/nnunet_local.sh`).
  - Checkpoints and predictions are few, large files and are written to `runs/` as usual.
  - This changes only where files are stored, not any nnU-Net setting.
- U-Net and TransUNet keep a whole split in RAM, so data loading does not touch NFS during training.
- `torch.compile` is enabled by nnU-Net by default and works on this machine. The first epoch includes the
  compilation (about 138 s instead of 80 s).

## Reproduce

```bash
python src/prepare_common_224.py                  # Task 0.2 (login node: file preparation only)
python src/test_metrics.py                        # evaluator check on hand-made masks
python src/prepare_nnunet.py                      # nnU-Net raw datasets + splits_final.json
sbatch slurm/quick_check_unet_transunet.slurm     # TransUNet steps 1-5 + epoch timing (U-Net part superseded)
sbatch slurm/quick_check_unet_final.slurm         # U-Net steps 1-5 with the final configuration
sbatch slurm/quick_check_nnunet.slurm             # nnU-Net preprocessing (node-local) + steps 1-5
python src/summarize_quick_check.py               # -> results/phase0_quick_check.json
python src/analysis/lvit_metric_check.py --runs runs/unet/QaTa-COV19/seed1001 runs/unet/MosMedData+/seed1001
python src/analysis/inspect_mosmed_sources.py     # MosMed Kaggle mirror: sources and image encoding
# Task 0.4 (after confirmation), e.g.:
sbatch --export=ALL,DATASET=QaTa-COV19,SEED=1001 slurm/train_unet.slurm
```
