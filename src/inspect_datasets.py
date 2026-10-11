"""Task 0.1 data check: image/mask formats, sizes, empty masks, pairing, and an example figure.

Reads the LViT-layout folders produced by src/prepare_lvit_datasets.py and writes
  * results/phase0_data_check.json   - all numbers quoted in reports/PHASE0_DATA_SUMMARY.md
  * reports/figures/data_examples.png - 4 cases per dataset, original image and mask overlay

Usage:
    python src/inspect_datasets.py --data ~/datasets
"""
import argparse
import collections
import json
import os
import textwrap

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SPLITS = ["Train", "Val", "Test"]
DATASETS = {
    # name: function mapping a mask filename to its image filename (LViT convention)
    "QaTa-COV19": lambda m: m.replace("mask_", "", 1),
    "MosMedData+": lambda m: m,
}
LESION_COLOR = "#eb6834"   # mask overlay colour
IMG_SIZE = 224             # LViT benchmark resolution


def lvit_mask_224(mask):
    """The LViT target-mask construction (external/LViT/Load_Dataset.py, ImageToImage2D.__getitem__,
    lines 177-180): cv2.resize to 224x224 with OpenCV's default interpolation, then mask > 0 -> 1."""
    m = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    return (m > 0).astype(np.uint8)


def inspect_split(folder, image_name):
    """Collect per-split statistics; every mask must have exactly one image and vice versa."""
    masks = sorted(os.listdir(os.path.join(folder, "labelcol")))
    images = set(os.listdir(os.path.join(folder, "img")))
    paired = {image_name(m) for m in masks}
    s = {"n": len(masks), "unpaired_masks": len(paired - images), "unpaired_images": len(images - paired)}
    img_channels, img_gray_rgb, img_dtypes = collections.Counter(), 0, collections.Counter()
    img_sizes, mask_values, mask_dims = collections.Counter(), set(), collections.Counter()
    size_mismatch, empty_raw, empty_224, fg_frac = 0, 0, 0, []
    for m in masks:
        img = cv2.imread(os.path.join(folder, "img", image_name(m)), cv2.IMREAD_UNCHANGED)
        msk = cv2.imread(os.path.join(folder, "labelcol", m), cv2.IMREAD_UNCHANGED)
        ch = 1 if img.ndim == 2 else img.shape[2]
        img_channels[ch] += 1
        img_dtypes[str(img.dtype)] += 1
        if ch >= 3 and np.array_equal(img[..., 0], img[..., 1]) and np.array_equal(img[..., 1], img[..., 2]):
            img_gray_rgb += 1          # stored with 3 channels but all channels identical
        img_sizes[f"{img.shape[1]}x{img.shape[0]}"] += 1   # W x H
        mask_dims[msk.ndim] += 1
        msk2 = msk if msk.ndim == 2 else msk[..., 0]
        mask_values |= set(np.unique(msk2).tolist())
        size_mismatch += msk2.shape[:2] != img.shape[:2]
        empty_raw += int(msk2.max() == 0)
        m224 = lvit_mask_224(msk2)
        empty_224 += int(m224.max() == 0)
        fg_frac.append(float(m224.mean()))
    fg = np.array(fg_frac)
    s.update({
        "image_channels": dict(img_channels), "image_dtype": dict(img_dtypes),
        "images_3ch_with_identical_channels": img_gray_rgb,
        "n_distinct_image_sizes": len(img_sizes), "top_image_sizes_WxH": dict(img_sizes.most_common(5)),
        "mask_ndim": dict(mask_dims), "mask_values": sorted(mask_values),
        "mask_image_size_mismatch": size_mismatch,
        "empty_masks_original": empty_raw, "empty_masks_after_lvit_224": empty_224,
        "lesion_fraction_224": {"mean": round(float(fg.mean()), 4), "median": round(float(np.median(fg)), 4),
                                 "min": round(float(fg.min()), 4), "max": round(float(fg.max()), 4)},
    })
    return s, masks, fg


def pick_examples(masks, fg, k=4):
    """Deterministic, non-cherry-picked choice: the non-empty masks at the 10/40/70/95th
    percentile of lesion area, so small and large lesions are both shown."""
    nz = np.where(fg > 0)[0]
    order = nz[np.argsort(fg[nz], kind="stable")]
    return [masks[order[int(q * (len(order) - 1))]] for q in [0.10, 0.40, 0.70, 0.95][:k]]


def draw_examples(examples, data_root, out_png):
    """Rows: per dataset, the original image and the mask overlay; columns: the selected cases."""
    fig, axes = plt.subplots(2 * len(examples), 4, figsize=(13, 3.4 * 2 * len(examples)))
    for d, (name, cases) in enumerate(examples.items()):
        for c, (split, mask_file, frac) in enumerate(cases):
            folder = os.path.join(data_root, name, f"{split}_Folder")
            img = cv2.imread(os.path.join(folder, "img", DATASETS[name](mask_file)), cv2.IMREAD_GRAYSCALE)
            msk = cv2.imread(os.path.join(folder, "labelcol", mask_file), cv2.IMREAD_GRAYSCALE)
            msk = (msk > 0).astype(np.uint8)
            top, bottom = axes[2 * d, c], axes[2 * d + 1, c]
            top.imshow(img, cmap="gray")
            top.set_title(f"{name} | {split}\n" + "\n".join(textwrap.wrap(mask_file, 34)), fontsize=8)
            bottom.imshow(img, cmap="gray")
            bottom.imshow(np.ma.masked_where(msk == 0, msk), cmap=matplotlib.colors.ListedColormap([LESION_COLOR]),
                          alpha=0.4)
            bottom.contour(msk, levels=[0.5], colors=LESION_COLOR, linewidths=1.2)
            bottom.set_title(f"mask overlay | lesion area {100 * frac:.1f}% (at 224x224)", fontsize=8)
            for ax in (top, bottom):
                ax.axis("off")
    fig.suptitle("Example cases: original image (upper row) and ground-truth lesion mask overlay (lower row)",
                 fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_png, dpi=110, bbox_inches="tight")
    print("wrote", out_png)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.expanduser("~/datasets"))
    ap.add_argument("--project", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    args = ap.parse_args()
    report, examples = {}, {}
    for name, image_name in DATASETS.items():
        report[name] = {}
        for split in SPLITS:
            stats, masks, fg = inspect_split(os.path.join(args.data, name, f"{split}_Folder"), image_name)
            report[name][split] = stats
            if split == "Train":     # examples are drawn from the training split only
                examples[name] = [("Train", m, float(fg[masks.index(m)])) for m in pick_examples(masks, fg)]
            print(name, split, json.dumps(stats))
    out_json = os.path.join(args.project, "results", "phase0_data_check.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out_json)
    draw_examples(examples, args.data, os.path.join(args.project, "reports", "figures", "data_examples.png"))


if __name__ == "__main__":
    main()
