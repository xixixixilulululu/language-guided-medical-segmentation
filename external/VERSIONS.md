# External code versions

The repositories below are cloned into `external/` locally and are not committed (see `.gitignore`).
To reproduce:

```bash
git clone https://github.com/HUANGLIZI/LViT.git external/LViT
git -C external/LViT checkout ba90775ead37d6f445f65ce83d1a0be64157f76f
git clone https://github.com/Beckschen/TransUNet.git external/TransUNet
git -C external/TransUNet checkout 02ef0010b36eb8328b5e689eadaf613602edf9b8
```

| Component | Source | Version |
|---|---|---|
| LViT (U-Net implementation, official data loader, split lists) | https://github.com/HUANGLIZI/LViT | commit `ba90775ead37d6f445f65ce83d1a0be64157f76f` (2026-07-21) |
| TransUNet (R50-ViT-B_16) | https://github.com/Beckschen/TransUNet | commit `02ef0010b36eb8328b5e689eadaf613602edf9b8` (2026-02-24) |
| nnU-Net v2 | `pip install nnunetv2` (https://github.com/MIC-DKFZ/nnUNet) | `nnunetv2==2.8.1` |

Environment: conda env `~/envs/lgms` (cloned from the research_training env), Python 3.11,
`torch 2.14.0+cu130`, extra packages `ml_collections 1.1.0`, `tensorboardX 2.6.5`, `h5py 3.16.0`.
LViT's own training script needs `bert_embedding`, which is not installed: Phase 0 only uses
`external/LViT/nets/UNet.py` and re-implements the image-only part of its data loader.
