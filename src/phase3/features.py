"""Phase 3, Task 3.2: the five reliability features F1-F5 for every validation row (all conditions, both datasets).

A row = dataset x language model x seed x sample x condition x donor variant. Sources (PHASE3_SETUP.md item 1):
  P_lang  C0, C1, C4-C6: Phase 2 (runs/phase2/val/<method>/<ds>/seed<k>/probs_<cond>_<variant>.npz, float32)
          C2, C3a-C3c:  Phase 3 rerun (runs/phase3/val/lang/<method>/<ds>/seed<k>/probs_<cond>_0.npz)
  P_img   the chosen IMG architecture (results/phase3_fallback_selection.csv: nnU-Net), same seed, Phase 3 rerun
          (runs/phase3/val/img/<arch>/<ds>/seed<k>/probs.npz)
  B = P > 0.5 (equal to the scored masks: results/phase3_rerun_checks.csv, results/phase3_inventory_val.csv)
Features (PHASE3_PLAN Task 3.2; float64 from the float32 maps; per row, before any averaging):
  F1 mean |P_lang - P_img|                    F2 1 - Dice(B_lang, B_img); both empty -> Dice 1, one empty -> 0
  F3 |mean(B_lang) - mean(B_img)|             F4 mean H(P_lang)    F5 mean H(P_img)
  H(p) = -p ln p - (1 - p) ln(1 - p), with p clipped to [1e-6, 1 - 1e-6] for the entropy only.
Outputs (item 12):
  results/phase3_reliability_features.csv.gz   keys + F1-F5 only (the only input of the gates)
  results/phase3_feature_labels.csv.gz         keys + dice_lang, dice_img, gain = dice_img - dice_lang,
                                               y = gain > 0.005, group (Phase 2 unit), partition
                                               (GATE_FIT / GATE_CAL for QaTa, "MosMed" for MosMedData+)
Usage:  python src/phase3/features.py
"""
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
P2 = os.path.join(REPO, "runs", "phase2", "val")
P3 = os.path.join(REPO, "runs", "phase3", "val")
DATASETS = ["QaTa-COV19", "MosMedData+"]
METHODS = ["lvit", "reclmis", "ddcmd"]
SEEDS = [1001, 1002, 1003]
RERUN = {"C2", "C3a", "C3b", "C3c"}
KEYS = ["dataset", "method", "seed", "sample_id", "condition", "variant_id", "donor_sample_id"]
EPS = 1e-6
LABEL_MARGIN = 0.005


def entropy(p):
    q = np.clip(p.astype(np.float64), EPS, 1 - EPS)
    return -(q * np.log(q) + (1 - q) * np.log(1 - q))


def features(pl, pi, f5=None):
    """F1-F5 for one pair of probability maps (2-D arrays); f5 = precomputed mean H(P_img) (same value)."""
    bl, bi = pl > 0.5, pi > 0.5
    nl, ni = int(bl.sum()), int(bi.sum())
    if nl == 0 and ni == 0:
        dice = 1.0
    elif nl == 0 or ni == 0:
        dice = 0.0
    else:
        dice = 2.0 * int((bl & bi).sum()) / (nl + ni)
    return (float(np.abs(pl.astype(np.float64) - pi.astype(np.float64)).mean()), 1.0 - dice,
            abs(float(bl.mean()) - float(bi.mean())), float(entropy(pl).mean()),
            float(entropy(pi).mean()) if f5 is None else f5)


def passes(method, dataset, seed):
    """[(condition, variant, per-case csv, probs npz)] of one language model x dataset x seed."""
    out = []
    for f in sorted(glob.glob(os.path.join(P2, method, dataset, f"seed{seed}", "per_case_*.csv"))):
        cond, var = re.match(r"per_case_(C\w+?)_(\w+)\.csv", os.path.basename(f)).groups()
        d = os.path.join(P3, "lang", method, dataset, f"seed{seed}") if cond in RERUN else os.path.dirname(f)
        out.append((cond, var, os.path.join(d, f"per_case_{cond}_{var}.csv"), os.path.join(d, f"probs_{cond}_{var}.npz")))
    return out


def one_run(arch, ds, s, m):
    """Feature and label rows of one dataset x seed x language model."""
    di = os.path.join(P3, "img", arch, ds, f"seed{s}")
    pimg = dict(np.load(os.path.join(di, "probs.npz")))
    f5 = {k: float(entropy(v).mean()) for k, v in pimg.items()}                   # F5 depends on the image only
    dimg = pd.read_csv(os.path.join(di, "per_case.csv"), dtype={"sample_id": str}).set_index("sample_id").dice
    feats, labels = [], []
    for cond, var, fcsv, fnpz in passes(m, ds, s):
        pc = pd.read_csv(fcsv, dtype={"sample_id": str, "variant_id": str}, keep_default_na=False)
        pl = np.load(fnpz)
        for r in pc.itertuples():
            key = (ds, m, s, r.sample_id, cond, var, r.donor_sample_id)
            feats.append(key + features(pl[r.sample_id], pimg[r.sample_id], f5[r.sample_id]))
            labels.append(key + (r.dice, float(dimg[r.sample_id])))
    print(f"{ds} seed {s} {m}: {len(feats)} rows", flush=True)
    return feats, labels


def main():
    from phase2 import texts as T
    arch = {"U-Net": "unet", "nnU-Net": "nnunet"}[
        pd.read_csv(os.path.join(REPO, "results", "phase3_fallback_selection.csv")).query("chosen").arch.iloc[0]]
    part = pd.read_csv(os.path.join(REPO, "results", "phase3_gate_partitions.csv"), dtype=str).set_index("sample_id").partition
    from multiprocessing import Pool
    with Pool(9) as pool:                                       # one task per dataset x seed x language model
        parts = pool.starmap(one_run, [(arch, ds, s, m) for ds in DATASETS for s in SEEDS for m in METHODS])
    feats = [x for p in parts for x in p[0]]
    labels = [x for p in parts for x in p[1]]
    F = pd.DataFrame(feats, columns=KEYS + ["F1", "F2", "F3", "F4", "F5"])
    L = pd.DataFrame(labels, columns=KEYS + ["dice_lang", "dice_img"])
    L["gain"] = L.dice_img - L.dice_lang
    L["y"] = (L.gain > LABEL_MARGIN).astype(int)
    grp = {(d, i): T.groups(d, i)[2] for d, i in L[["dataset", "sample_id"]].drop_duplicates().itertuples(index=False)}
    L["group"] = [grp[(d, i)] for d, i in zip(L.dataset, L.sample_id)]
    L["partition"] = [part.get(i, "?") if d == "QaTa-COV19" else "MosMed" for d, i in zip(L.dataset, L.sample_id)]
    if (L.partition == "?").any():
        raise ValueError("QaTa sample without partition")
    F.to_csv(os.path.join(REPO, "results", "phase3_reliability_features.csv.gz"), index=False, float_format="%.9g")
    L.to_csv(os.path.join(REPO, "results", "phase3_feature_labels.csv.gz"), index=False, float_format="%.6f")
    print(F.groupby(["dataset", "condition"]).size().unstack(0).to_string())
    print("IMG architecture:", arch)


if __name__ == "__main__":
    main()
