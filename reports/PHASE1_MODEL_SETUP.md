# Phase 1 model setup (Task 1.5)

Status: 2026-10-19. The LViT-T and RecLMIS quick checks are complete (job 316). The DD-CMD quick check is
being repeated after a NumPy 2 compatibility fix (section 3.3), so its numbers below are placeholders.
The configurations are **not frozen yet**.

Code: `src/phase1/train_{lvit,reclmis,ddcmd}.py`, `configs/{lvit,reclmis,ddcmd}.yaml`. Pinned repository
commits, environment and weights: `external/VERSIONS.md`. Checks and decisions:
`reports/PHASE1_SETUP_PLAN.md`.

## 1. Summary table

| Method | Repository | Input | Text Encoder | Main Cross-Modal Mechanism | Notes |
|---|---|---|---|---|---|
| LViT-T | HUANGLIZI/LViT (`ba90775`) | 224×224, grayscale repeated to 3 channels, /255 | BERT-base uncased, frozen, precomputed: first 10 word vectors (10 × 768), PyTorch re-implementation of `bert_embedding` (section 3.1) | LViT fusion: text features enter the ViT branch at every level of the hybrid CNN–Transformer encoder and decoder | lr 3e-4 QaTa / 1e-3 MosMed, batch 2 (section 2), patience 50 |
| RecLMIS | ShawnHuang497/RecLMIS (`d3265c8`) | 224×224, grayscale repeated to 3 channels, /255 | CLIP ViT-B/32 text tower, frozen, 18 tokens | conditioned reconstruction: masked text/image tokens reconstructed across modalities (2 reconstruction losses), conditional contrastive loss (CCL), bidirectional cross-attention Interactor | official per-dataset configs; fairseq stand-in (section 3.2) |
| DD-CMD | maklachur/DD-CMD (`27387c4`) | 224×224, per-image z-score, grayscale repeated to 3 channels | BiomedBERT (`microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`), frozen, 24 tokens | dual-domain gated decoding: text-guided cross-attention with a gated fusion, and DCT frequency-domain modulation conditioned on the text; ConvNeXt-Tiny image encoder | Lightning 1.9 with a `np.Inf` alias (section 3.3); selection on per-case val Dice (D3) |

All three methods use the text at training **and** test time. All three read the same Phase 0
`common224` images and masks, use seeds 1001, 1002 and 1003, and choose the checkpoint on the mean
per-case validation Dice (threshold 0.5). The final scoring uses the Phase 0 common evaluator
(`src/evaluate.py`).

## 2. LViT-T settings and their sources

Rule (decision D4, 2026-10-18): each parameter is taken from its most specific official source.

| Parameter | Value | Source |
|---|---|---|
| Learning rate | QaTa-COV19 3e-4, MosMedData+ 1e-3 | LViT paper (Li et al., IEEE TMI 2023), Sec. IV-C: "The initial learning rate is set to 3e-4 for the QaTa-COV19 dataset and 1e-3 for the MosMedData+ dataset." This section describes LViT's own training. `Config.py` names only Covid19 (3e-4) and MoNuSeg (1e-3); it has no MosMed value. |
| Batch size | 2 | `Config.py`: `batch_size = 2  # For LViT-T, 2 is better than 4`. This comment refers specifically to LViT-T. |
| Optimizer, schedule | Adam; cosine warm restarts (T_0 10, min 1e-4), stepped once per epoch | `train_model.py`, `utils.py` |
| Loss | 0.5 Dice + 0.5 weighted BCE | `train_model.py` |
| Epochs, early stopping | max 2000, patience 50; best model saved from the 6th epoch on | `Config.py`, `train_model.py`; paper: 50 epochs |

**The batch size of 2 is our judgement, not a confirmed setting.** The paper states "The default batch
size is 24 for the QaTa-COV19 dataset and the MosMedData+ dataset". The code's LViT-T comment says 2.
Whether the paper's LViT-T scores were produced with 2 or with 24 cannot be confirmed from the paper or
the repository. We follow the source that refers to LViT-T specifically.

**Why this differs from the Phase 0 U-Net (batch 24).** The Phase 0 U-Net uses the same LViT training
procedure. The LViT-T comment does not apply to U-Net, so the paper's general default of 24 was the most
specific source for U-Net (decision of 2026-10-11, `configs/unet.yaml`). For LViT-T, the code comment is
the more specific source.

## 3. Compatibility changes (no change of the methods)

The official code is not modified (`external/` stays at the pinned commits). Every change is in our
drivers.

### 3.1 LViT-T: text features without MXNet (decision D1)

- `bert_embedding` pins `mxnet 1.4.0`, `numpy 1.14.6` and `gluonnlp 0.6.0`; none of these exists for
  aarch64 / Python 3.11.
- `src/phase1/lvit_text.py` re-implements the steps of `BertEmbedding()` with its defaults:
  1. BERT-base uncased tokenizer, at most 25 tokens including [CLS] and [SEP];
  2. last-layer token outputs;
  3. [CLS], [SEP] and [PAD] removed;
  4. word pieces averaged into words;
  5. the first 10 word vectors kept.

  It uses the same weights (HF `google-bert/bert-base-uncased`). The vectors are computed once: 495
  distinct texts, all 10 × 768, covering all 11,987 samples.
- **They could not be checked against the official MXNet output.** `mxnet 1.9.1` (aarch64 wheel) does
  not import, because the ARM Performance Libraries are missing on the cluster. `gluonnlp` cannot be
  built, because there is no `gcc`. Expected differences are floating-point rounding only, but this is
  not verified.
- Text padding (`' EOF XXX'` for texts with fewer than 9 words) is applied explicitly, because the
  official chained pandas assignment does nothing under pandas 3. In these datasets no text has fewer
  than 9 words, so the padding never applies.
- QaTa image/mask pairing follows LViT's commented-out Covid19 lines; the active lines are MoNuSeg's.

### 3.2 RecLMIS

- **fairseq.** fairseq cannot be built here. The code only uses `fairseq.utils.softmax`, which the
  driver provides with fairseq's own definition, `F.softmax(x, dim, dtype=float32)`.
  - Without it, the official bare `except` around the reconstruction silently switches off the CCL
    and both reconstruction losses.
  - The driver stops if any of these losses is missing or not finite, or if `aux` is off. The official
    reconstruction can only run on a GPU, because it hard-codes `.cuda()`.
- **Config bug.** The official `train_model.py` loads `Config_covid19` for `-c Config_MosMedPlus`.
  The driver imports the correct module for each dataset and checks every value against
  `configs/reclmis.yaml`.
- **NaN reconstruction losses.** The official reconstructor replaces a NaN reconstruction loss by 0.
  Such steps are counted in `training_summary.json` (`rec_loss_zero_steps`).
- **CLIP weights.** The official ViT-B/32 file is linked as `external/RecLMIS/nets/ViT-B-32.pt`. Its
  SHA-256 equals the hash in the official URL.

### 3.3 DD-CMD

- **NumPy 2 alias (decision of 2026-10-19).** pytorch_lightning 1.9.0, the official pin, uses
  `torch.tensor(np.Inf)` in `EarlyStopping` and `ModelCheckpoint`. NumPy 2.0 removed the alias `np.Inf`,
  so training failed while the callbacks were being created, before any step (quick-check job 316). The
  driver restores `np.Inf = np.inf` before importing Lightning. **The value is identical** (IEEE
  positive infinity), so training is not changed. A scan of Lightning 1.9.0 and lightning_utilities
  found no other alias removed in NumPy 2. torchmetrics 0.10.3 uses `np.float_` only in its FID metric,
  which is not used.
- **Logging.** Only `CSVLogger` is used; tensorboard is not installed, and logging does not affect
  training.
- **Data.** Arrays are fed in (H, W) orientation. The official PIL reader returns (W, H), i.e. a
  transposed image. Every augmentation is symmetric under transposition, so the training distribution is
  unchanged (`configs/ddcmd.yaml`).

## 4. Deviations from the official settings

| Method | Deviation | Reason |
|---|---|---|
| all | Seeds 1001 / 1002 / 1003 (official: 666 / 3407 / 42) | Phase 1 plan |
| all | Checkpoint selection on the mean per-case validation Dice of the common evaluator | Phase 1 plan; the same rule as Phase 0 |
| LViT-T, RecLMIS | Same criterion as the official one, except for the batch weighting (official: mean of batch means over a shuffled loader) | – |
| DD-CMD | **D3:** checkpoint selection **and** early stopping (patience 50) on the per-case validation Dice instead of the official `val_MIoU` (a pixel-pooled IoU over the epoch). The official `val_MIoU` is still logged. | one rule for all six methods |
| DD-CMD | **D6:** 8 seeded data-loader workers instead of 0. This changes the random order of the augmentation, not the method. | run time |
| all | Training can be split into segments of at most 12 h (resume from the last epoch) | cluster 12 h limit (D5) |

## 5. Quick-check results

Only the validation set is used; the test set is never touched. Small-sample training uses the first 64
training and 32 validation cases, 40 epochs, seed 1001. Text-swap probe: the trained subset model is
evaluated with the true texts and with every text replaced by the most frequent training text. Resume
test: MosMedData+, 32/16 cases, 6 epochs uninterrupted vs 3 + 3 with a segment end in between. Timing:
full training and validation sets, second epoch. All values come from `results/phase1_quick_check.json`.

| | LViT-T QaTa | LViT-T MosMed | RecLMIS QaTa | RecLMIS MosMed | DD-CMD QaTa | DD-CMD MosMed |
|---|---|---|---|---|---|---|
| Probes on the untrained model (pairing, forward, text gradient ≠ 0, text swap changes the output) | passed | passed | passed | passed | passed | passed |
| RecLMIS: CCL and both reconstruction losses present and finite, `aux` on | – | – | yes, every step | yes, every step | – | – |
| Subset training: training loss, epoch 1 → 40 | 0.535 → 0.251 | 0.463 → 0.070 | 6.10 → 2.79 | 6.63 → 2.02 | _pending_ | _pending_ |
| Subset training: validation Dice, epoch 1 → best | 0.005 → 0.649 | 0.449 → 0.783 | 0.226 → 0.607 | 0.026 → 0.763 | _pending_ | _pending_ |
| Text swap (trained subset model): val Dice true text / fixed text | 0.6488 / 0.6092 | 0.7826 / 0.7826 | 0.6073 / 0.5929 | 0.7635 / 0.7635 | _pending_ | _pending_ |
| Seconds per epoch (train + val, full data) | 245 | 92 | 171 | 63 | _pending_ | _pending_ |

- **RecLMIS auxiliary losses over the 40 subset epochs:**
  - QaTa: CCL 4.37 → 2.99, text reconstruction 0.95 → 0.53, image reconstruction 0.98 → 0.02.
  - MosMed: CCL 5.08 → 2.48, text reconstruction 0.85 → 0.45, image reconstruction 0.91 → 0.02.
  - No step had a reconstruction loss replaced by 0 (QaTa 0 of 80 steps, MosMed 0 of 160).
- **Resume test.** For LViT-T and RecLMIS, the uninterrupted and the segmented runs have identical epoch
  numbers (1-6), learning rates, training losses and validation Dice (maximum absolute difference 0).
  DD-CMD: _pending_.
- **Text swap on MosMedData+.** For both LViT-T and RecLMIS, the validation Dice of the trained subset
  model is the same to four decimals with the true texts and with every text replaced by one fixed text.
  In these 32 validation cases, 26 have a text different from the fixed text. This is recorded as
  observed, without interpretation. The text dependence of the frozen models is tested in Phase 2.
- **D3 check for DD-CMD:** both callbacks must monitor `val_case_dice`, and the selected best epoch must
  equal the epoch with the highest per-case validation Dice in `epochs.csv`. _pending_.
