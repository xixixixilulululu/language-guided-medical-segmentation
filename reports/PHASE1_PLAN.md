# Phase 1: Language-Guided Medical Image Segmentation

## Goal

Reproduce three representative language-guided medical image segmentation methods and compare them with the Phase 0 image-only baselines.

Language-guided methods:

1. **LViT-T**
2. **RecLMIS**
3. **DD-CMD**

Datasets:

1. **QaTa-COV19 (LViT split)**
2. **MosMedData+ (LViT split)**

Seeds:

```text
1001
1002
1003
```

Phase 1 training experiments:

```text
3 methods × 2 datasets × 3 seeds = 18 runs
```

Final benchmark methods:

```text
Image-only:
- U-Net
- TransUNet
- nnU-Net

Language-guided:
- LViT-T
- RecLMIS
- DD-CMD
```

Total:

```text
6 methods
```

This set is intentionally compact:

- LViT-T provides the foundational language-guided benchmark;
- RecLMIS provides a later reconstruction-based cross-modal method;
- DD-CMD provides a recent strong cross-modal decoder with gated text fusion.

---

# Public Implementations

## LViT-T

Paper:

```text
LViT: Language Meets Vision Transformer in Medical Image Segmentation
```

Official repository:

```text
https://github.com/HUANGLIZI/LViT
```

Use:

```text
LViT-T
```

---

## RecLMIS

Paper:

```text
Cross-Modal Conditioned Reconstruction for Language-Guided Medical Image Segmentation
IEEE Transactions on Medical Imaging
```

Official repository:

```text
https://github.com/ShawnHuang497/RecLMIS
```

Use the official QaTa/MosMed implementation and pretrained CLIP component specified by the repository.

---

## DD-CMD

Paper:

```text
Dual-Domain Cross-Modal Decoding for Clinical Text-Guided Medical Image Segmentation
MICCAI 2026
```

Official repository:

```text
https://github.com/maklachur/DD-CMD
```

DD-CMD is included because it is a recent strong method on the same QaTa-COV19 and MosMedData+ benchmarks and uses gated cross-modal fusion, making it particularly relevant to the later text-reliability experiments.

---

# Task 1.1: Prepare the Data and Repositories

Use the same datasets and splits as Phase 0.

### QaTa-COV19

```text
Train: 5716
Validation: 1429
Test: 2113
```

### MosMedData+

```text
Train: 2183
Validation: 273
Test: 273
```

Dataset package:

```text
https://drive.google.com/drive/folders/1BjkeVIYwiV4d94AMDZfBtIk_7MNIb2AH?usp=sharing
```

The package contains the paired text annotations used by LViT-style experiments.

Use the same common 224×224 image/mask preparation used in Phase 0.

Do not create a different split.

## Output

Create:

```text
reports/PHASE1_DATA_SUMMARY.md
```

| Dataset | Train | Validation | Test | Text Available |
|---|---:|---:|---:|---|
| QaTa-COV19 | 5716 | 1429 | 2113 | Yes |
| MosMedData+ | 2183 | 273 | 273 | Yes |

---

# Task 1.2: Reproduce LViT-T

Confirm:

1. image, mask, and text pairing;
2. forward pass;
3. text input reaches the language branch;
4. training loss decreases;
5. validation Dice improves;
6. predicted masks are valid.

Use the official LViT-T training design.

Freeze the configuration before the three-seed experiment.

---

# Task 1.3: Reproduce RecLMIS

Confirm:

1. image, mask, and text pairing;
2. official CLIP/pretrained component loads correctly;
3. forward pass works;
4. conditioned reconstruction/fusion path is active;
5. training behaves normally;
6. validation predictions are valid.

Freeze the configuration before the three-seed experiment.

---

# Task 1.4: Reproduce DD-CMD

Confirm:

1. image, mask, and text pairing;
2. official model and text encoder load correctly;
3. forward pass works;
4. cross-modal decoder is active;
5. training behaves normally;
6. validation predictions are valid.

Use the official repository configuration as the starting point.

Freeze the configuration before the three-seed experiment.

---

# Task 1.5: Model Setup Summary

Create:

```text
reports/PHASE1_MODEL_SETUP.md
```

| Method | Repository | Input | Text Encoder | Main Cross-Modal Mechanism | Notes |
|---|---|---|---|---|---|
| LViT-T | HUANGLIZI/LViT | 224×224 | | LViT fusion | |
| RecLMIS | ShawnHuang497/RecLMIS | 224×224 | | conditioned reconstruction | |
| DD-CMD | maklachur/DD-CMD | 224×224 | | dual-domain gated decoding | |

---

# Task 1.6: Run the 18 Main Experiments

| Method | Dataset | Seed 1001 | Seed 1002 | Seed 1003 |
|---|---|---|---|---|
| LViT-T | QaTa-COV19 | | | |
| RecLMIS | QaTa-COV19 | | | |
| DD-CMD | QaTa-COV19 | | | |
| LViT-T | MosMedData+ | | | |
| RecLMIS | MosMedData+ | | | |
| DD-CMD | MosMedData+ | | | |

For each run save:

```text
training log
best checkpoint
test summary
per-case metrics
predicted masks
```

Use validation Dice for checkpoint selection.

Do not tune using the test set.

---

# Task 1.7: Common Final Evaluation

Evaluate all six Phase 0/1 methods using the same final evaluation protocol:

```text
evaluation size = 224 × 224
binary threshold = 0.5
```

Metrics:

```text
Dice
mIoU
HD95
ASSD
```

All methods must use the same test samples and the same target masks.

---

# Task 1.8: Phase 1 Result Table

Create:

```text
results/phase1_main_results.csv
```

| Dataset | Method | Dice Mean ± SD | Dice 95% CI | mIoU | HD95 | ASSD |
|---|---|---|---|---:|---:|---:|
| QaTa-COV19 | LViT-T | | | | | |
| QaTa-COV19 | RecLMIS | | | | | |
| QaTa-COV19 | DD-CMD | | | | | |
| MosMedData+ | LViT-T | | | | | |
| MosMedData+ | RecLMIS | | | | | |
| MosMedData+ | DD-CMD | | | | | |

---

# Task 1.9: Final Six-Method Comparison

Combine:

```text
U-Net
TransUNet
nnU-Net
LViT-T
RecLMIS
DD-CMD
```

Create:

```text
results/final_six_method_results.csv
```

| Dataset | Method | Text at Train | Text at Test | Dice Mean ± SD | Dice 95% CI | mIoU | HD95 | ASSD |
|---|---|---|---|---|---|---|---:|---:|---:|

This is the main benchmark table.

---

# Task 1.10: Statistical Comparison

There are:

```text
6 choose 2 = 15
```

method pairs per dataset.

For each dataset:

1. average each test case across the three seeds within each method;
2. align the same cases;
3. compute paired Dice differences;
4. report 95% CI;
5. report paired p-value;
6. apply Holm correction across the 15 comparisons.

Do not test significance using only three seed means.

Create:

```text
results/final_six_method_pairwise.csv
```

| Dataset | Method A | Method B | ΔDice | 95% CI | Raw p | Holm p |
|---|---|---|---:|---|---:|---:|

For mIoU, HD95, and ASSD, paired differences and 95% CIs are sufficient.

---

# Task 1.11: Interpret the Results

Create:

```text
reports/PHASE1_RESULTS.md
```

Answer:

1. Which method performs best on QaTa-COV19?
2. Which method performs best on MosMedData+?
3. How strong are the image-only baselines?
4. Do language-guided methods consistently outperform image-only methods?
5. Does the recent DD-CMD model change the conclusion compared with LViT-T and RecLMIS?
6. Which methods are most stable across seeds?
7. Do boundary metrics agree with Dice?
8. Which differences remain significant after Holm correction?

Do not claim that text itself causes an improvement solely from comparing different architectures. Phase 2 will test text dependence within the same frozen model.

---

# Required Figures

1. Six-method Dice comparison on QaTa-COV19.
2. Six-method Dice comparison on MosMedData+.
3. Seed stability for the three language-guided methods.

Optional:

Representative predictions from:

```text
U-Net
nnU-Net
LViT-T
RecLMIS
DD-CMD
```

---

# Milestones

## Milestone 1.1: Data and Repositories Ready

Expected:

- official splits loaded;
- text annotations paired correctly;
- all three official repositories available.

## Milestone 1.2: Three Language-Guided Methods Reproduced

Expected:

- LViT-T runs correctly.
- RecLMIS runs correctly.
- DD-CMD runs correctly.
- final configs are fixed.

## Milestone 1.3: 18 Runs Complete

Expected:

```text
18 / 18 runs complete
```

## Milestone 1.4: Final Six-Method Analysis Complete

Required:

```text
results/phase1_main_results.csv
results/final_six_method_results.csv
results/final_six_method_pairwise.csv
reports/PHASE1_RESULTS.md
```

---

# Phase 1 Completion Checklist

- [ ] LViT-T reproduction is complete.
- [ ] RecLMIS reproduction is complete.
- [ ] DD-CMD reproduction is complete.
- [ ] all 18 Phase 1 runs are complete.
- [ ] all three seeds are included.
- [ ] common 224×224 evaluation is used.
- [ ] test predictions and per-case metrics are saved.
- [ ] final six-method table is complete.
- [ ] all 15 pairwise comparisons are complete for QaTa-COV19.
- [ ] all 15 pairwise comparisons are complete for MosMedData+.
- [ ] `PHASE1_RESULTS.md` is complete.

After this, Phase 1 is frozen.
