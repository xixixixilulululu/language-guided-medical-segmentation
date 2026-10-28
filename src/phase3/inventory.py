"""Phase 3, Task 3.1A: inventory of the existing VALIDATION artifacts (read only; no test file is read).

LANG (Phase 2 benchmark, runs/phase2/val/<method>/<dataset>/seed<k>/): for every condition pass
  per_case_<cond>_<variant>.csv, masks_<cond>_<variant>.npz (packed bits), probs_<cond>_<variant>.npz (float32).
IMG (Phase 0, runs/{unet,nnunet}/<dataset>/seed<k>/): val_probs.npz, eval_val/per_case.csv, eval_val/masks/*.png.
For each item: present?, number of cases, probability dtype, keys matching the per-case ids, and (for the
first CHECK_N cases of every pass that has probabilities) whether the stored mask equals probability > 0.5.
Also the sha256 of every Phase 3 checkpoint (LANG: from results/phase2_checkpoint_sha256.tsv, re-hashed here;
IMG: U-Net best_model.pt, nnU-Net selected snapshot).

Usage:  python src/phase3/inventory.py   -> results/phase3_inventory_val.csv, results/phase3_checkpoints.tsv
"""
import glob
import hashlib
import json
import os
import re
import sys

import cv2
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS = ["QaTa-COV19", "MosMedData+"]
SEEDS = [1001, 1002, 1003]
CHECK_N = 50
NNUNET_DS = {"QaTa-COV19": "Dataset501_QaTaCOV19", "MosMedData+": "Dataset502_MosMedDataPlus"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def unpack(npz, k):
    return np.unpackbits(npz[k])[:224 * 224].reshape(224, 224).astype(bool)


def lang_rows():
    rows = []
    for d in sorted(glob.glob(os.path.join(REPO, "runs", "phase2", "val", "*", "*", "seed*"))):
        method, ds, seed = d.split(os.sep)[-3], d.split(os.sep)[-2], int(d.split(os.sep)[-1][4:])
        for f in sorted(glob.glob(os.path.join(d, "per_case_*.csv"))):
            cond, var = re.match(r"per_case_(C\w+?)_(\w+)\.csv", os.path.basename(f)).groups()
            pc = pd.read_csv(f, dtype={"sample_id": str})
            ids = set(pc.sample_id)
            m = np.load(os.path.join(d, f"masks_{cond}_{var}.npz"))
            pp = os.path.join(d, f"probs_{cond}_{var}.npz")
            r = {"model_type": "LANG", "method": method, "dataset": ds, "seed": seed, "condition": cond,
                 "variant": var, "n_cases": len(pc), "per_case": True, "masks": set(m.files) == ids,
                 "probs": os.path.exists(pp), "prob_dtype": "", "keys_match": set(m.files) == ids,
                 "donor_ids_present": bool((pc.donor_sample_id.notna()).all()) if var.startswith("d") else "",
                 "mask_equals_prob_gt_0.5": ""}
            if r["probs"]:
                p = np.load(pp)
                r["prob_dtype"] = str(p[p.files[0]].dtype)
                r["keys_match"] = r["keys_match"] and set(p.files) == ids
                r["mask_equals_prob_gt_0.5"] = all((unpack(m, k) == (p[k] > 0.5)).all() for k in sorted(ids)[:CHECK_N])
            rows.append(r)
    return rows


def img_rows():
    rows = []
    for arch in ("unet", "nnunet"):
        for ds in DATASETS:
            for s in SEEDS:
                d = os.path.join(REPO, "runs", arch, ds, f"seed{s}")
                pc = pd.read_csv(os.path.join(d, "eval_val", "per_case.csv"), dtype={"sample_id": str})
                p = np.load(os.path.join(d, "val_probs.npz"))
                ids = list(pc.sample_id)
                eq = all(((cv2.imread(os.path.join(d, "eval_val", "masks", f"{k}.png"), cv2.IMREAD_GRAYSCALE) > 0)
                          == (p[k].astype(np.float32) > 0.5)).all() for k in ids[:CHECK_N])
                rows.append({"model_type": "IMG", "method": arch, "dataset": ds, "seed": s, "condition": "image only",
                             "variant": "", "n_cases": len(pc), "per_case": True,
                             "masks": len(os.listdir(os.path.join(d, "eval_val", "masks"))) == len(pc),
                             "probs": True, "prob_dtype": str(p[p.files[0]].dtype), "keys_match": set(p.files) == set(ids),
                             "donor_ids_present": "", "mask_equals_prob_gt_0.5": eq})
    return rows


def checkpoints():
    rows = []
    t = pd.read_csv(os.path.join(REPO, "results", "phase2_checkpoint_sha256.tsv"), sep="\t", dtype=str)
    for r in t.itertuples():
        run = os.path.join("runs", r.method, r.dataset, f"seed{r.seed}")
        path = os.path.join(run, "checkpoints", "best.ckpt") if r.method == "ddcmd" else os.path.join(run, "best_model.pt")
        h = sha(os.path.join(REPO, path))
        rows.append({"role": "LANG", "model": r.method, "dataset": r.dataset, "seed": r.seed, "path": path,
                     "sha256": h, "matches_phase2_record": h == r.sha256})
    for ds in DATASETS:
        for s in SEEDS:
            p = os.path.join("runs", "unet", ds, f"seed{s}", "best_model.pt")
            rows.append({"role": "IMG", "model": "unet", "dataset": ds, "seed": s, "path": p, "sha256": sha(os.path.join(REPO, p)),
                         "matches_phase2_record": ""})
            sel = json.load(open(os.path.join(REPO, "runs", "nnunet", ds, f"seed{s}", "selection.json")))
            p = os.path.join("runs", "nnunet", f"nnUNet_results_seed{s}", NNUNET_DS[ds], "nnUNetTrainerLGMS__nnUNetPlans__2d",
                             "fold_0", sel["best_checkpoint"])
            rows.append({"role": "IMG", "model": "nnunet", "dataset": ds, "seed": s, "path": p, "sha256": sha(os.path.join(REPO, p)),
                         "matches_phase2_record": ""})
    return pd.DataFrame(rows)


def main():
    inv = pd.DataFrame(lang_rows() + img_rows())
    inv.to_csv(os.path.join(REPO, "results", "phase3_inventory_val.csv"), index=False)
    ck = checkpoints()
    ck.to_csv(os.path.join(REPO, "results", "phase3_checkpoints.tsv"), sep="\t", index=False)
    print(inv.groupby(["model_type", "condition"]).agg(passes=("n_cases", "size"), probs=("probs", "sum"),
                                                       dtypes=("prob_dtype", lambda x: ",".join(sorted(set(x) - {""}))),
                                                       keys_match=("keys_match", "all")).to_string())
    print("mask == prob > 0.5 (first", CHECK_N, "cases):",
          inv[inv.probs].groupby("model_type")["mask_equals_prob_gt_0.5"].apply(lambda x: f"{int(x.sum())}/{len(x)}").to_dict())
    print("LANG checkpoints match Phase 2 record:", bool(ck[ck.role == "LANG"].matches_phase2_record.all()))


if __name__ == "__main__":
    sys.exit(main())
