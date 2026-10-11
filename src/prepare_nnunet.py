"""Export the common 224x224 data (Task 0.2) as nnU-Net v2 raw datasets, with the LViT split as fold 0.

    Dataset501_QaTaCOV19, Dataset502_MosMedDataPlus   in $nnUNet_raw
      imagesTr/<id>_0000.png   LViT train + val images (uint8 grayscale, 224x224)
      labelsTr/<id>.png        their masks {0, 1}
      imagesTs/<id>_0000.png   LViT test images (labelsTs/ holds their masks, never used by nnU-Net)
      dataset.json
    $nnUNet_preprocessed/<DatasetName>/splits_final.json
      a single fold: fold 0 = {"train": LViT train ids, "val": LViT val ids}

nnU-Net is therefore trained on exactly the LViT training set, and its fold-0 "validation" set is the
LViT validation set (used for checkpoint selection only). Only fold 0 is trained.

Channel name "gray" (not "CT"): the PNG intensities are not Hounsfield units, so nnU-Net's CT
normalisation would be wrong; any other name selects nnU-Net's default z-score normalisation.

Usage:
    export nnUNet_raw=~/datasets/nnunet/nnUNet_raw nnUNet_preprocessed=~/datasets/nnunet/nnUNet_preprocessed
    python src/prepare_nnunet.py
"""
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgms import data  # noqa: E402

NNUNET_IDS = {"QaTa-COV19": (501, "Dataset501_QaTaCOV19"), "MosMedData+": (502, "Dataset502_MosMedDataPlus")}


def main():
    raw = os.environ["nnUNet_raw"]
    pre = os.environ["nnUNet_preprocessed"]
    for dataset, (_, name) in NNUNET_IDS.items():
        root = os.path.join(raw, name)
        for sub in ["imagesTr", "labelsTr", "imagesTs", "labelsTs"]:
            os.makedirs(os.path.join(root, sub), exist_ok=True)
        ids = {s: data.split_ids(dataset, s) for s in data.SPLITS}
        # train + val go to imagesTr (fold 0 separates them again), test to imagesTs
        for split, (img_dir, lab_dir) in {"train": ("imagesTr", "labelsTr"), "val": ("imagesTr", "labelsTr"),
                                          "test": ("imagesTs", "labelsTs")}.items():
            for i in ids[split]:
                shutil.copyfile(data.image_path(dataset, split, i), os.path.join(root, img_dir, f"{i}_0000.png"))
                shutil.copyfile(data.mask_path(dataset, split, i), os.path.join(root, lab_dir, f"{i}.png"))
        assert not set(ids["train"]) & set(ids["val"]), "train/val ids overlap"
        dataset_json = {
            "channel_names": {"0": "gray"},
            "labels": {"background": 0, "lesion": 1},
            "numTraining": len(ids["train"]) + len(ids["val"]),
            "file_ending": ".png",
            "name": name,
            "description": f"{dataset}, LViT benchmark split, common 224x224 preparation (Phase 0 Task 0.2)",
        }
        with open(os.path.join(root, "dataset.json"), "w") as f:
            json.dump(dataset_json, f, indent=2)
        # Custom split: one fold = the LViT train / val partition
        os.makedirs(os.path.join(pre, name), exist_ok=True)
        with open(os.path.join(pre, name, "splits_final.json"), "w") as f:
            json.dump([{"train": ids["train"], "val": ids["val"]}], f, indent=1)
        print(f"{name}: imagesTr {len(ids['train']) + len(ids['val'])} (train {len(ids['train'])} + "
              f"val {len(ids['val'])}), imagesTs {len(ids['test'])}")


if __name__ == "__main__":
    main()
