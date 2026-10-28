"""Phase 2: reconciliation of the TEST C0 pass with the archived Phase 1 test evaluation (tolerance rule P2-2).

Runs only with the Phase 2 lock open (it reads test outputs). For each language model x dataset x seed it compares
runs/phase2/test/<method>/<dataset>/seed<k>/{per_case_C0_0.csv, masks_C0_0.npz} with the Phase 1 archive of the
same checkpoint (runs/<method or lvit_b24>/<dataset>/seed<k>/eval_test/{per_case.csv, masks/<id>.png}).
Rule P2-2 (PHASE2_PROTOCOL.md section 4.7):
  LViT-T, RecLMIS   scored masks identical (0 differing pixels)
  DD-CMD            <= 10 differing pixels per case; no empty <-> non-empty flip; |mean Dice difference| <= 0.0005
The maximum per-case |dDice| and |dHD95| are reported for all runs. Outside the limits: the script exits with
status 1 and the protocol's procedure applies (stop, find the cause, record it). Phase 2 paired differences always
use the Phase 2 C0; the archive is used only here.

Usage:  python src/phase2/reconcile_test_c0.py   -> results/phase2_test_c0_reconciliation.csv
"""
import os
import sys

import cv2
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
DATASETS, SEEDS = ["QaTa-COV19", "MosMedData+"], [1001, 1002, 1003]


def archive_dir(method, dataset, seed):
    m = "lvit_b24" if (method == "lvit" and dataset == "MosMedData+") else method
    return os.path.join(REPO, "runs", m, dataset, f"seed{seed}", "eval_test")


def check(method, dataset, seed):
    new = os.path.join(REPO, "runs", "phase2", "test", method, dataset, f"seed{seed}")
    pc = pd.read_csv(os.path.join(new, "per_case_C0_0.csv"), dtype={"sample_id": str}).set_index("sample_id")
    m = np.load(os.path.join(new, "masks_C0_0.npz"))
    old_dir = archive_dir(method, dataset, seed)
    old = pd.read_csv(os.path.join(old_dir, "per_case.csv"), dtype={"sample_id": str}).set_index("sample_id")
    px, flips = [], 0
    for i in pc.index:
        a = np.unpackbits(m[i])[:224 * 224].reshape(224, 224).astype(bool)
        b = cv2.imread(os.path.join(old_dir, "masks", f"{i}.png"), cv2.IMREAD_GRAYSCALE) > 0
        px.append(int((a != b).sum()))
        flips += int(a.any() != b.any())
    px = np.array(px)
    dd = (pc.dice - old.dice.reindex(pc.index)).abs()
    r = {"method": method, "dataset": dataset, "seed": seed, "n": len(pc), "cases_differing": int((px > 0).sum()),
         "max_pixels_differing": int(px.max()), "empty_flips": flips,
         "abs_mean_dice_difference": float(abs(pc.dice.mean() - old.dice.mean())),
         "max_abs_case_dice_difference": float(dd.max()),
         "max_abs_case_hd95_difference": float((pc.hd95 - old.hd95.reindex(pc.index)).abs().max())}
    if method in ("lvit", "reclmis"):
        r["within_rule"] = r["cases_differing"] == 0
    else:
        r["within_rule"] = r["max_pixels_differing"] <= 10 and flips == 0 and r["abs_mean_dice_difference"] <= 0.0005
    return r


def main():
    from phase2 import lock
    lock.require_open(what="Phase 2 test C0 reconciliation")
    df = pd.DataFrame([check(m, d, s) for m in ("lvit", "reclmis", "ddcmd") for d in DATASETS for s in SEEDS])
    df.to_csv(os.path.join(REPO, "results", "phase2_test_c0_reconciliation.csv"), index=False, float_format="%.6g")
    print(df.to_string(index=False))
    sys.exit(0 if df.within_rule.all() else 1)


if __name__ == "__main__":
    main()
