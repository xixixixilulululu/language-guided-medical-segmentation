# Phase 0: Image-Only Segmentation Benchmark

## Goal
Establish a strong and reproducible image-only benchmark on the same two datasets used by the later language-guided experiments.

Required methods:
1. U-Net
2. TransUNet
3. nnU-Net

Datasets:
1. QaTa-COV19 (LViT split)
2. MosMedData+ (LViT split)

Seeds: 1001, 1002, 1003
Main experiment count: 3 methods × 2 datasets × 3 seeds = 18 runs
Primary metric: Dice
Secondary metrics: mIoU, HD95, ASSD

These three baselines provide a compact but useful range of image-only approaches:
- U-Net: classic CNN segmentation baseline;
- TransUNet: CNN-Transformer hybrid;
- nnU-Net: strong self-configuring medical segmentation framework.

---

# Public Implementations

## U-Net
Use the U-Net implementation distributed with the official LViT project so that this baseline is directly connected to the established QaTa/MosMed benchmark.
LViT repository: https://github.com/HUANGLIZI/LViT
The LViT paper/repository reports U-Net as its image-only reference model.
Use binary segmentation: background, lesion.

## TransUNet
Paper: TransUNet: Transformers Make Strong Encoders for Medical Image Segmentation
Official repository: https://github.com/Beckschen/TransUNet
Use the 2D configuration: R50-ViT-B_16, with two output classes: background, lesion.
Use the pretrained weights referenced by the official repository when available.

## nnU-Net
Paper: nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation, Nature Methods, 2021
Official repository: https://github.com/MIC-DKFZ/nnUNet
Use: nnU-Net v2, 2D configuration.
Treat each 2D QaTa/MosMed image as one 2D segmentation sample.
Do not use the 3D configuration.

---

# Task 0.1: Create the Project and Download the Data
Create a new GitHub repository and local project folder.
Recommended repository name: language-guided-medical-segmentation
Recommended structure:
language-guided-medical-segmentation/
├── external/
├── src/
├── configs/
├── data/
├── runs/
├── results/
└── reports/

Do not store raw datasets in GitHub.

## Dataset package
Use the dataset package linked by the LViT project:
https://drive.google.com/drive/folders/1BjkeVIYwiV4d94AMDZfBtIk_7MNIb2AH?usp=sharing

### QaTa-COV19
Use the LViT benchmark split: Train 5716, Validation 1429, Test 2113
Original/public reference: https://www.kaggle.com/datasets/aysendegerli/qatacov19-dataset

### MosMedData+
Use the LViT benchmark split: Train 2183, Validation 273, Test 273
Original/public reference: https://medicalsegmentation.com/covid19/
Alternative mirror: https://www.kaggle.com/datasets/maedemaftouni/covid19-ct-scan-lesion-segmentation-dataset

Do not create a different split.

## Output
Create reports/PHASE0_DATA_SUMMARY.md

| Dataset | Train | Validation | Test |
|---|---:|---:|---:|
| QaTa-COV19 | 5716 | 1429 | 2113 |
| MosMedData+ | 2183 | 273 | 273 |

---

# Task 0.2: Prepare One Common Segmentation Dataset
Use one common image/mask preparation pipeline before model-specific loading.
Target benchmark resolution: 224 × 224

## MosMed masks
Follow the target-mask construction used by the LViT/RecLMIS benchmark:
1. read the original mask;
2. resize to 224×224 using the same OpenCV resizing behavior used by the official loader;
3. binarize after resizing: mask <= 0 -> 0, mask > 0 -> 1
Use this target definition for training, validation, and test.

## QaTa masks
Use the QaTa masks prepared under the LViT benchmark pipeline at 224×224.

## Images
Use the same underlying images for all methods.
If a model requires three channels, duplicate the grayscale channel.

---

# Task 0.3: Set Up the Three Models
Create:
configs/
├── unet.yaml
├── transunet.yaml
└── nnunet.yaml

Record for each method: input size; pretrained initialization; loss; optimizer; learning rate; scheduler; augmentation; maximum epochs.
Preserve the intended training strategy of each method.
Do not tune settings using test performance.
Checkpoint selection must use validation Dice only.

## Quick check
Before full training:
1. load one batch;
2. run one forward pass;
3. verify the output and mask shapes;
4. verify the loss can be computed;
5. run a short small-subset training check.

## Output
Create reports/PHASE0_MODEL_SETUP.md

| Method | Source | Configuration | Input | Pretraining | Notes |
|---|---|---|---|---|---|
| U-Net | LViT | binary U-Net | 224×224 | | |
| TransUNet | Official | R50-ViT-B_16 | 224×224 | | |
| nnU-Net | Official | 2D | 224×224 prepared data | | |

---

# Task 0.4: Run the 18 Main Experiments

| Method | Dataset | Seed 1001 | Seed 1002 | Seed 1003 |
|---|---|---|---|---|
| U-Net | QaTa-COV19 | | | |
| TransUNet | QaTa-COV19 | | | |
| nnU-Net | QaTa-COV19 | | | |
| U-Net | MosMedData+ | | | |
| TransUNet | MosMedData+ | | | |
| nnU-Net | MosMedData+ | | | |

For every run save: training log, best checkpoint, test summary, per-case metrics, predicted masks.
Per-case metrics: sample_id, dice, miou, hd95, assd
Use validation Dice to select the best checkpoint.

---

# Task 0.5: Common Final Evaluation
All three methods must be evaluated using the same final evaluator.
Use: evaluation size = 224 × 224, binary threshold = 0.5
Metrics: Dice, mIoU, HD95, ASSD
Report HD95 and ASSD in resized-image pixel units.
If a native model produces predictions at another resolution, map the prediction to the common 224×224 evaluation space before scoring.

---

# Task 0.6: Summarize the Results
Create results/phase0_main_results.csv and reports/PHASE0_RESULTS.md
For every method and dataset report: mean ± SD across 3 seeds, Dice 95% CI, mIoU, HD95, ASSD

| Dataset | Method | Dice Mean ± SD | Dice 95% CI | mIoU | HD95 | ASSD |
|---|---|---|---|---:|---:|---:|
| QaTa-COV19 | U-Net | | | | | |
| QaTa-COV19 | TransUNet | | | | | |
| QaTa-COV19 | nnU-Net | | | | | |
| MosMedData+ | U-Net | | | | | |
| MosMedData+ | TransUNet | | | | | |
| MosMedData+ | nnU-Net | | | | | |

---

# Task 0.7: Pairwise Comparison
Within each dataset compare: U-Net vs TransUNet, U-Net vs nnU-Net, TransUNet vs nnU-Net
For Dice:
1. average each test case across the three seeds within each method;
2. align the same test cases;
3. compute paired Dice differences;
4. report a 95% CI;
5. report a paired p-value;
6. apply Holm correction across the three comparisons within each dataset.
Do not test significance using only the three seed means.
Create results/phase0_pairwise_comparisons.csv

| Dataset | Method A | Method B | ΔDice | 95% CI | Raw p | Holm p |
|---|---|---|---:|---|---:|---:|

---

# Required Figures
1. QaTa-COV19 Dice comparison with 95% CI.
2. MosMedData+ Dice comparison with 95% CI.
3. Three-seed stability for all methods.
Optional: Representative segmentation examples.

---

# Milestones
0.1 Data Ready: QaTa-COV19 = 5716 / 1429 / 2113, MosMedData+ = 2183 / 273 / 273
0.2 Models Ready: U-Net, TransUNet, nnU-Net 2D run correctly; training settings are fixed.
0.3 18 Runs Complete: 18 / 18 runs complete
0.4 Final Analysis Complete: results/phase0_main_results.csv, results/phase0_pairwise_comparisons.csv, reports/PHASE0_RESULTS.md

# Phase 0 Completion Checklist
- [ ] QaTa-COV19 uses the LViT split.
- [ ] MosMedData+ uses the LViT split.
- [ ] U-Net is complete.
- [ ] TransUNet is complete.
- [ ] nnU-Net is complete.
- [ ] all 18 runs are complete.
- [ ] all three seeds are included.
- [ ] per-case predictions and metrics are saved.
- [ ] common 224×224 evaluation is used.
- [ ] main result table is complete.
- [ ] paired comparisons are complete.
- [ ] PHASE0_RESULTS.md is complete.

After this, Phase 0 is frozen.
