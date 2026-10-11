"""Common final evaluator (Task 0.5): the same code scores U-Net, TransUNet and nnU-Net.

Input: the foreground probability maps of one split, in the common 224x224 space, as either
  * `--probs probs.npz`   one float16 array (224, 224) per sample id (written by every method), or
  * `--pred-dir DIR`      one PNG per sample id (`<id>.png`, 0/1 or 0/255 values).
Every sample id of the split must be present; extra or missing ids stop the evaluation.

Output (in `--out`):
  * per_case.csv  sample_id, dice, miou, hd95, assd
  * summary.json  mean of each metric over cases, number of cases, empty predictions
  * masks/<id>.png  (with --save-masks) the binarised prediction, 0/255

Usage:
    python src/evaluate.py --dataset QaTa-COV19 --split test --probs runs/.../test_probs.npz --out runs/.../eval_test
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lgms import data, metrics  # noqa: E402


def load_predictions(ids, probs=None, pred_dir=None):
    """Return {id: float/uint8 array (224,224)} for exactly the ids of the split."""
    if probs is not None:
        z = np.load(probs)
        keys = set(z.files)
        get = lambda i: z[i].astype(np.float32)
    else:
        keys = {f[:-4] for f in os.listdir(pred_dir) if f.endswith(".png")}

        def get(i):
            # PNG masks may be stored as 0/1 or 0/255: map to [0, 1] before the common threshold
            m = cv2.imread(os.path.join(pred_dir, f"{i}.png"), cv2.IMREAD_GRAYSCALE).astype(np.float32)
            return m / 255.0 if m.max() > 1 else m
    missing, extra = set(ids) - keys, keys - set(ids)
    if missing or extra:
        raise ValueError(f"prediction ids do not match the split: {len(missing)} missing, {len(extra)} extra")
    out = {}
    for i in ids:
        p = get(i)
        if p.shape != (metrics.IMG_SIZE, metrics.IMG_SIZE):
            raise ValueError(f"{i}: prediction shape {p.shape} is not 224x224")
        out[i] = p
    return out


def _score(args):
    """Worker: metrics of one case."""
    sample_id, prob, gt = args
    return {"sample_id": sample_id, **metrics.case_metrics(metrics.binarize(prob), gt > 0)}


def evaluate(dataset, split, preds, workers=8):
    """Per-case metrics DataFrame for {id: prob} against the common ground truth."""
    ids = data.split_ids(dataset, split)
    jobs = [(i, preds[i], data.read_mask(dataset, split, i)) for i in ids]
    with Pool(workers) as pool:
        rows = pool.map(_score, jobs, chunksize=32)
    return pd.DataFrame(rows, columns=["sample_id", "dice", "miou", "hd95", "assd"])


def summarize(df, preds):
    """Mean of every metric over cases plus the number of empty predictions."""
    n_empty = int(sum(not metrics.binarize(p).any() for p in preds.values()))
    return {"n_cases": int(len(df)), "n_empty_predictions": n_empty,
            **{m: float(df[m].mean()) for m in ["dice", "miou", "hd95", "assd"]}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=data.DATASETS)
    ap.add_argument("--split", required=True, choices=data.SPLITS)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--probs", help="npz with one probability map per sample id")
    src.add_argument("--pred-dir", help="folder of <id>.png predicted masks")
    ap.add_argument("--out", required=True)
    ap.add_argument("--save-masks", action="store_true", help="also write the binarised masks as PNG")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    ids = data.split_ids(args.dataset, args.split)
    preds = load_predictions(ids, args.probs, args.pred_dir)
    df = evaluate(args.dataset, args.split, preds, args.workers)
    summary = {"dataset": args.dataset, "split": args.split, **summarize(df, preds)}

    os.makedirs(args.out, exist_ok=True)
    df.to_csv(os.path.join(args.out, "per_case.csv"), index=False, float_format="%.6f")
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    if args.save_masks:
        mdir = os.path.join(args.out, "masks")
        os.makedirs(mdir, exist_ok=True)
        for i, p in preds.items():
            cv2.imwrite(os.path.join(mdir, f"{i}.png"), metrics.binarize(p).astype(np.uint8) * 255)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
