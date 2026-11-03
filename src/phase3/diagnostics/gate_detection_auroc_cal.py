"""Descriptive, validation only: how well do the existing gate signals separate the original report (C0) from a
mismatched one (C4, C5, C6)? QaTa-COV19, GATE_CAL part only (not used to fit the LR coefficients), all three seeds
pooled per method, unweighted rows. AUROC = P(signal on a mismatched row > signal on a C0 row).
Usage: python3 -I diag_detect_cal.py <repo> [out.csv]"""
import csv, gzip, os, sys
import numpy as np
from scipy.stats import rankdata
R = os.path.join(sys.argv[1], "results")
def rd(f): return csv.DictReader(gzip.open(os.path.join(R, f), "rt"))
K = lambda r: (r["method"], r["seed"], r["sample_id"], r["condition"], r["variant_id"])
NAME = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD"}
part = {}
dec = {}
for r in rd("phase3_val_gate_decisions.csv.gz"):
    if r["dataset"] == "QaTa-COV19":
        dec[K(r)] = float(r["lr_score"]); part[K(r)] = r["part"]
feat = {K(r): [float(r[f"F{i}"]) for i in (1, 2, 3)] for r in rd("phase3_reliability_features.csv.gz") if r["dataset"] == "QaTa-COV19"}
def auroc(pos, neg):
    x = np.r_[pos, neg]; rk = rankdata(x)
    return (rk[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
rows = []
for m in NAME:
    by = {c: [k for k in dec if k[0] == m and k[3] == c and part[k] == "GATE_CAL"] for c in ("C0", "C4", "C5", "C6")}
    for c in ("C4", "C5", "C6"):
        for sig, fn in (("F1", lambda k: feat[k][0]), ("F2", lambda k: feat[k][1]), ("F3", lambda k: feat[k][2]), ("LR score", lambda k: dec[k])):
            a = auroc([fn(k) for k in by[c]], [fn(k) for k in by["C0"]])
            rows.append({"method": NAME[m], "mismatch": c, "signal": sig, "n_mismatch_rows": len(by[c]), "n_C0_rows": len(by["C0"]), "auroc": round(a, 4)})
for r in rows:
    if r["mismatch"] == "C6": print(r)
c6 = [r["auroc"] for r in rows if r["mismatch"] == "C6"]
print("C6 vs C0 AUROC range over methods and signals:", min(c6), max(c6))
print("C4-C6 vs C0 AUROC range:", min(r["auroc"] for r in rows), max(r["auroc"] for r in rows))
if len(sys.argv) > 2:
    with open(sys.argv[2], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
