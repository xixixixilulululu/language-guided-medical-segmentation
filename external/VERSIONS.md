# External code versions

The repositories below are cloned into `external/` locally and are not committed (see `.gitignore`).
To reproduce:

```bash
git clone https://github.com/HUANGLIZI/LViT.git external/LViT
git -C external/LViT checkout ba90775ead37d6f445f65ce83d1a0be64157f76f
git clone https://github.com/Beckschen/TransUNet.git external/TransUNet
git -C external/TransUNet checkout 02ef0010b36eb8328b5e689eadaf613602edf9b8
# Phase 1
git clone https://github.com/ShawnHuang497/RecLMIS.git external/RecLMIS
git -C external/RecLMIS checkout d3265c8b4d4582a94595fe1c8b1892c618bc9715
ln -s ~/pretrained/clip/ViT-B-32.pt external/RecLMIS/nets/ViT-B-32.pt   # official CLIP weights (see below)
git clone https://github.com/maklachur/DD-CMD.git external/DD-CMD
git -C external/DD-CMD checkout 27387c47163971448f8fcdf5e37f310509dbd948
```

| Component | Source | Version |
|---|---|---|
| LViT (U-Net implementation, official data loader, split lists) | https://github.com/HUANGLIZI/LViT | commit `ba90775ead37d6f445f65ce83d1a0be64157f76f` (2026-07-21) |
| TransUNet (R50-ViT-B_16) | https://github.com/Beckschen/TransUNet | commit `02ef0010b36eb8328b5e689eadaf613602edf9b8` (2026-02-24) |
| RecLMIS (Phase 1) | https://github.com/ShawnHuang497/RecLMIS | commit `d3265c8b4d4582a94595fe1c8b1892c618bc9715` (2025-06-19) |
| DD-CMD (Phase 1) | https://github.com/maklachur/DD-CMD | commit `27387c47163971448f8fcdf5e37f310509dbd948` (2026-08-12) |
| OpenAI CLIP package (RecLMIS tokenizer) | https://github.com/openai/CLIP | commit `d05afc436d78f1c48dc0dbf8e5980a9d471f35f6` |
| nnU-Net v2 | `pip install nnunetv2` (https://github.com/MIC-DKFZ/nnUNet) | `nnunetv2==2.8.1` |

Environment: conda env `~/envs/lgms` (cloned from the research_training env), Python 3.11,
`torch 2.14.0+cu130`, extra packages `ml_collections 1.1.0`, `tensorboardX 2.6.5`, `h5py 3.16.0`.
LViT's own training script needs `bert_embedding`, which is not installed: Phase 0 only uses
`external/LViT/nets/UNet.py` and re-implements the image-only part of its data loader.

Phase 1 environment (decision D2, 2026-10-18): overlay venv `~/envs/lgms_p1`, created with
`~/envs/lgms/bin/python -m venv --system-site-packages ~/envs/lgms_p1`, so it uses `lgms`'s
`torch 2.14.0+cu130` unchanged while `lgms` itself is not modified. Added packages: `transformers 4.57.6`
(+ `tokenizers 0.22.2`, `huggingface_hub 0.36.2`, shadowing `lgms`'s 2.1.1 only inside `lgms_p1`),
`pytorch-lightning 1.9.0`, `torchmetrics 0.10.3`, `monai 1.6.1`, `clip` (git above), `ftfy 6.3.1`, `regex`,
`thop`.

Phase 1 pretrained weights (outside git):

| Weight | Source | Location / revision |
|---|---|---|
| CLIP ViT-B/32 (RecLMIS) | openaipublic.azureedge.net/clip/models/40d36571.../ViT-B-32.pt | `~/pretrained/clip/ViT-B-32.pt`, SHA-256 `40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af` (= URL hash) |
| BERT-base uncased (LViT-T text features) | HF `google-bert/bert-base-uncased` | HF cache, revision `86b5e0934494` |
| ConvNeXt-Tiny (DD-CMD) | HF `facebook/convnext-tiny-224` | HF cache, revision `6166b7613034` |
| BiomedBERT (DD-CMD) | HF `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext` | HF cache, revision `e1354b7a3a09` |
