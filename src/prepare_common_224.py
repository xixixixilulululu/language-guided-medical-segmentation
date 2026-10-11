"""Task 0.2: build the one common 224x224 image/mask set used by all three methods.

Input: the LViT folder layout written by src/prepare_lvit_datasets.py
       (~/datasets/<dataset>/<Split>_Folder/{img,labelcol}/).
Output: ~/datasets/common224/<dataset>/{train,val,test}.txt and
        ~/datasets/common224/<dataset>/<split>/{images,masks}/<sample_id>.png  (layout: src/lgms/data.py)

The preparation follows the official LViT loader, external/LViT/Load_Dataset.py,
ImageToImage2D.__getitem__ (LViT commit ba90775):

    image = cv2.imread(img_path)                      # 3-channel uint8
    image = cv2.resize(image, (224, 224))             # default interpolation: cv2.INTER_LINEAR
    mask  = cv2.imread(mask_path, 0)                  # grayscale uint8
    mask  = cv2.resize(mask, (224, 224))              # INTER_LINEAR, then binarise:
    mask[mask <= 0] = 0; mask[mask > 0] = 1

Images: both datasets are grayscale (QaTa stored 1-channel, MosMed stored as 3 identical channels),
so after the resize the three channels are identical; this is asserted and one channel is stored.
A method that needs three channels duplicates it (Task 0.2: "duplicate the grayscale channel").
QaTa files are already 224x224, so the resize is the identity for QaTa.

Sample ids: the image filename without ".png" (QaTa mask "mask_<id>.png", as in LViT).

Usage:
    python src/prepare_common_224.py --lvit-root ~/datasets --out ~/datasets/common224
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgms.data import DATASETS, IMG_SIZE  # noqa: E402

SPLIT_FOLDERS = {"train": "Train_Folder", "val": "Val_Folder", "test": "Test_Folder"}


def mask_name(dataset, image_file):
    """LViT naming: QaTa masks are 'mask_<image name>', MosMed masks share the image filename."""
    return f"mask_{image_file}" if dataset == "QaTa-COV19" else image_file


def lvit_image(path):
    """Image exactly as LViT loads it, reduced to its (identical) single channel."""
    img = cv2.imread(path)                                   # BGR, 3 channels
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))              # INTER_LINEAR (OpenCV default)
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 0] == img[..., 2]).all(), f"not grayscale: {path}"
    return img[..., 0]


def lvit_mask(path):
    """Target mask exactly as LViT builds it: grayscale read, INTER_LINEAR resize, then > 0 -> 1."""
    m = cv2.imread(path, 0)
    m = cv2.resize(m, (IMG_SIZE, IMG_SIZE))
    m[m <= 0] = 0
    m[m > 0] = 1
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lvit-root", default=os.path.expanduser("~/datasets"))
    ap.add_argument("--out", default=os.path.expanduser("~/datasets/common224"))
    args = ap.parse_args()

    rows = []
    for dataset in DATASETS:
        for split, folder in SPLIT_FOLDERS.items():
            src = os.path.join(args.lvit_root, dataset, folder)
            img_dir, mask_dir = os.path.join(src, "img"), os.path.join(src, "labelcol")
            out_img = os.path.join(args.out, dataset, split, "images")
            out_mask = os.path.join(args.out, dataset, split, "masks")
            os.makedirs(out_img, exist_ok=True)
            os.makedirs(out_mask, exist_ok=True)

            files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
            # Image <-> mask pairing must be one-to-one (checked in Task 0.1, re-checked here)
            assert sorted(mask_name(dataset, f) for f in files) == sorted(os.listdir(mask_dir)), (dataset, split)
            ids = []
            for f in files:
                sid = f[:-4]
                img = lvit_image(os.path.join(img_dir, f))
                m = lvit_mask(os.path.join(mask_dir, mask_name(dataset, f)))
                cv2.imwrite(os.path.join(out_img, f"{sid}.png"), img)
                cv2.imwrite(os.path.join(out_mask, f"{sid}.png"), m)
                ids.append(sid)
                rows.append({"dataset": dataset, "split": split, "sample_id": sid,
                             "source_image": os.path.join(dataset, folder, "img", f),
                             "source_mask": os.path.join(dataset, folder, "labelcol", mask_name(dataset, f)),
                             "lesion_pixels": int(m.sum())})
            with open(os.path.join(args.out, dataset, f"{split}.txt"), "w") as fh:
                fh.write("\n".join(ids) + "\n")
            print(f"{dataset:12s} {split:5s} {len(ids):5d} cases")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(args.out, "manifest.csv"), index=False)
    # Small summary for the report (counts and lesion area at 224x224)
    summary = {}
    for (d, s), g in df.groupby(["dataset", "split"], sort=False):
        frac = g.lesion_pixels / IMG_SIZE ** 2
        summary.setdefault(d, {})[s] = {"n": int(len(g)), "empty_masks": int((g.lesion_pixels == 0).sum()),
                                        "mean_lesion_fraction": float(frac.mean()),
                                        "median_lesion_fraction": float(frac.median())}
    print(json.dumps(summary, indent=2))
    with open(os.path.join(args.out, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
