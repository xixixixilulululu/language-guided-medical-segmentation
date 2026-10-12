"""Re-score saved test predictions with the LViT test_model.py algorithm and with common variants.

Read-only analysis (no training, no change to any run): loads a run's `test_probs.npz` and the common
224x224 masks, and prints Dice / mIoU computed

  * by our common evaluator (the run's eval_test/per_case.csv),
  * exactly as LViT external/LViT/test_model.py does it (show_image_with_dice): per image,
    prediction = output > 0.5, Dice = 2*sum(G*P) / (sum(G) + sum(P) + 1e-5),
    IoU = sklearn jaccard_score of the binary lesion mask, mean = sum over images / number of images,
  * with the threshold >= 0.5 (used by LViT's training-time helpers save_on_batch / iou_on_batch),
  * pooled over the whole test set (global Dice / IoU instead of the per-image mean),
  * MosMedData+ only: per image at the native 512x512 resolution (probability map resized with
    cv2.INTER_LINEAR, original mask > 0).

Usage:
    python src/analysis/lvit_metric_check.py --runs runs/unet/QaTa-COV19/seed1001 runs/unet/MosMedData+/seed1001
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from lgms import data  # noqa: E402


def lvit_case(pred, gt):
    """One image, exactly as LViT test_model.py show_image_with_dice."""
    p, g = pred.astype(np.float32), gt.astype(np.float32)
    dice = 2 * np.sum(g * p) / (np.sum(g) + np.sum(p) + 1e-5)
    iou = jaccard_score(g.reshape(-1), p.reshape(-1), zero_division=0)
    return dice, iou


def rescore(run_dir, dataset):
    """All variants for one run directory; values in % (rounded to 2 decimals)."""
    z = np.load(os.path.join(run_dir, "test_probs.npz"))
    ids = data.split_ids(dataset, "test")
    probs = np.stack([z[i].astype(np.float32) for i in ids])
    gts = np.stack([data.read_mask(dataset, "test", i) for i in ids]).astype(bool)
    ours = pd.read_csv(os.path.join(run_dir, "eval_test", "per_case.csv"))

    r = {"run": os.path.relpath(run_dir, REPO), "dataset": dataset, "n_test": len(ids),
         "ours_dice": ours.dice.mean(), "ours_miou": ours.miou.mean()}
    lv = np.array([lvit_case(p > 0.5, g) for p, g in zip(probs, gts)])
    r["lvit_dice"], r["lvit_miou"] = lv[:, 0].sum() / len(ids), lv[:, 1].sum() / len(ids)
    r["lvit_ge05_dice"] = np.mean([lvit_case(p >= 0.5, g)[0] for p, g in zip(probs, gts)])
    P = probs > 0.5
    tp = np.logical_and(P, gts).sum()
    r["global_dice"] = 2 * tp / (P.sum() + gts.sum())
    r["global_iou"] = tp / np.logical_or(P, gts).sum()
    if dataset == "MosMedData+":
        d512 = []
        for i, p in zip(ids, probs):
            g = cv2.imread(os.path.expanduser(f"~/datasets/MosMedData+/Test_Folder/labelcol/{i}.png"), 0) > 0
            pu = cv2.resize(p, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR) > 0.5
            d512.append(2 * np.sum(pu & g) / (pu.sum() + g.sum() + 1e-5))
        r["native512_dice"] = float(np.mean(d512))
    return {k: (round(100 * float(v), 2) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True, help="run directories (…/<dataset>/seed<k>)")
    ap.add_argument("--out", default=os.path.join(REPO, "results", "phase0_lvit_metric_check.json"))
    args = ap.parse_args()
    rows = []
    for run in args.runs:
        run = os.path.abspath(run)
        dataset = os.path.basename(os.path.dirname(run))  # runs/<method>/<dataset>/seed<k>
        rows.append(rescore(run, dataset))
    print(pd.DataFrame(rows).set_index("run").T.to_string())
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"written: {os.path.relpath(args.out, REPO)}")


if __name__ == "__main__":
    main()
