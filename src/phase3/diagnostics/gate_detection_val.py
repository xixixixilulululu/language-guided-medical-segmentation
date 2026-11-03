"""Validation only (QaTa, both parts): can the existing gate signals tell a clean report (C0) from a wrong one
(C4 / C6)? And where do the LR-Gate's C0 harmful switches and C6 missed switches come from?"""
import csv, gzip, os, sys
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
R = os.path.join(sys.argv[1], "results")
def rd(f): return csv.DictReader(gzip.open(os.path.join(R, f), "rt"))
K = lambda r: (r["method"], r["seed"], r["sample_id"], r["condition"], r["variant_id"])
feat = {K(r): [float(r[f"F{i}"]) for i in range(1, 6)] for r in rd("phase3_reliability_features.csv.gz") if r["dataset"] == "QaTa-COV19"}
lab = {K(r): (float(r["dice_lang"]), float(r["dice_img"])) for r in rd("phase3_feature_labels.csv.gz") if r["dataset"] == "QaTa-COV19"}
dec = {K(r): (float(r["lr_score"]), r["lr_switch"] == "True") for r in rd("phase3_val_gate_decisions.csv.gz") if r["dataset"] == "QaTa-COV19"}
def auroc(pos, neg):
    x = np.r_[pos, neg]; r = rankdata(x)
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
for m in ("lvit", "reclmis", "ddcmd"):
    rows = defaultdict(list)
    for k in dec:
        if k[0] == m: rows[k[3]].append(k)
    out = [m]
    for c in ("C4", "C6"):
        for name, fn in (("F1", lambda k: feat[k][0]), ("F2", lambda k: feat[k][1]), ("F3", lambda k: feat[k][2]), ("LR", lambda k: dec[k][0])):
            out.append(f"{c} {name} {auroc([fn(k) for k in rows[c]], [fn(k) for k in rows['C0']]):.2f}")
    print("AUROC wrong-vs-clean:", " | ".join(out))
    # C6 rows where switching is clearly right (LANG < 0.3, IMG > 0.6): share switched by LR-Gate
    clear = [k for k in rows["C6"] if lab[k][0] < 0.3 and lab[k][1] > 0.6]
    print(f"  C6 clear-switch rows: {len(clear)}/{len(rows['C6'])} ({100*len(clear)/len(rows['C6']):.0f}%), LR switched {100*np.mean([dec[k][1] for k in clear]):.0f}%")
    # C0 rows with the same kind of large disagreement (1-F2 > 0.7): how often is LANG the better one?
    big0 = [k for k in rows["C0"] if feat[k][1] > 0.7]
    big6 = [k for k in rows["C6"] if feat[k][1] > 0.7]
    print(f"  rows with F2 (1 - mask Dice) > 0.7: C0 {len(big0)} ({100*len(big0)/len(rows['C0']):.1f}%), LANG better in {100*np.mean([lab[k][0] > lab[k][1] for k in big0]):.0f}% | "
          f"C6 {len(big6)} ({100*len(big6)/len(rows['C6']):.1f}%), LANG better in {100*np.mean([lab[k][0] > lab[k][1] for k in big6]):.0f}%")
    harm0 = [k for k in rows["C0"] if dec[k][1] and lab[k][1] < lab[k][0] - 1e-6]
    if harm0:
        print(f"  C0 harmful LR switches: {len(harm0)}, IMG Dice median {np.median([lab[k][1] for k in harm0]):.2f}, share with IMG < 0.5: {100*np.mean([lab[k][1] < 0.5 for k in harm0]):.0f}%")
