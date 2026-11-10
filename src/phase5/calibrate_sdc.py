"""Phase 5, Task 5.1A: SDC-Fallback thresholds on QaTa GATE_CAL (validation only).

Rules: configs/phase5/spec.json ("sdc", "sdc_threshold"). Rows as src/phase3/gates.py main(): features and labels
merged one-to-one on the keys, QaTa, conditions C0, C2, C4, C5, C6, partition GATE_CAL per model x seed. SDC per
row from the LANG probability map of src/phase3/features.py passes() (validation C2 from the Phase 3 rerun).
Threshold: gates.calibrate(cal, "neg_sdc") with neg_sdc = -SDC (unchanged rule); reported as tau = -t.
Self-checks (exit 10): spec committed once and unchanged, an ancestor of HEAD; clean tree; outputs absent.
Outputs: results/phase5_sdc_calibration.csv (tau at full precision), results/phase5_sdc_val.csv.gz (keys + SDC).
Usage (Slurm job slurm/phase5_calibrate.slurm):  python src/phase5/calibrate_sdc.py
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase5 import core, guards  # noqa: E402

OUT_CAL = os.path.join("results", "phase5_sdc_calibration.csv")
OUT_SDC = os.path.join("results", "phase5_sdc_val.csv.gz")
DATASET = "QaTa-COV19"


def sdc_rows(args):
    """SDC of every GATE_CAL row of one model x seed (validation; rows and maps as features.py)."""
    m, s, cal_ids = args
    from phase3 import features as Fe
    rows = []
    for cond, var, fcsv, fnpz in Fe.passes(m, DATASET, s, "val"):
        if cond not in core.CONDS:
            continue
        pc = pd.read_csv(fcsv, dtype={"sample_id": str, "variant_id": str}, keep_default_na=False)
        pl = np.load(fnpz)
        for r in pc.itertuples():
            if r.sample_id in cal_ids:
                rows.append((DATASET, m, s, r.sample_id, cond, var, r.donor_sample_id, core.sdc(pl[r.sample_id])))
    return rows


def calibration_row(m, s, curve, ch, why):
    r = {"model": core.NAME[m], "seed": s, "threshold": core.tau_from_t(ch.threshold), "t_neg_sdc": float(ch.threshold),
         "choice": why, "n_candidates": len(curve), "objective": ch.objective}
    for c in core.CONDS:
        r[f"{c}_delta"] = ch[f"delta_{c}"]
    for c in core.CONDS:
        r[f"fallback_rate_{c}"] = ch[f"fallback_rate_{c}"]
    for c in core.CONDS:
        r[f"n_rows_{c}"] = int(ch[f"n_rows_{c}"])
    return r


def main():
    commit = guards.committed_unchanged(REPO, guards.SPEC)
    guards.clean_tree(REPO)
    guards.outputs_absent(REPO, [OUT_CAL, OUT_SDC])
    from phase3 import gates as G
    from phase4 import guard
    guard.require_dev_split("val")
    dt = {"sample_id": str, "variant_id": str, "donor_sample_id": str}
    F = pd.read_csv(os.path.join(REPO, "results", "phase3_reliability_features.csv.gz"), dtype=dt, keep_default_na=False)
    L = pd.read_csv(os.path.join(REPO, "results", "phase3_feature_labels.csv.gz"), dtype=dt, keep_default_na=False)
    D = F.merge(L, on=core.KEYS, validate="one_to_one")
    D = D[(D.dataset == DATASET) & D.condition.isin(G.CONDS)]
    cal_ids = set(D[D.partition == "GATE_CAL"].sample_id)
    guard.require_dev_ids(cal_ids)
    with Pool(9) as pool:
        parts = pool.map(sdc_rows, [(m, s, cal_ids) for m in core.METHODS for s in core.SEEDS])
    Sdc = pd.DataFrame([x for p in parts for x in p], columns=core.KEYS + ["sdc"])
    rows, keep = [], []
    for m in core.METHODS:
        for s in core.SEEDS:
            cal = D[(D.method == m) & (D.seed == s) & (D.partition == "GATE_CAL")].copy()
            cal = cal.merge(Sdc, on=core.KEYS, how="left", validate="one_to_one")
            if cal.sdc.isna().any():
                guards.stop(f"{m} {s}: GATE_CAL rows without an SDC value")
            cal["neg_sdc"] = -cal.sdc
            curve, ch, why = G.calibrate(cal, "neg_sdc")
            rows.append(calibration_row(m, s, curve, ch, why))
            keep.append(cal[core.KEYS + ["sdc"]])
    out = pd.DataFrame(rows)
    out["spec_commit"] = commit
    guards.write_atomic(out, REPO, OUT_CAL, float_format="%.17g")
    guards.write_atomic(pd.concat(keep), REPO, OUT_SDC, float_format="%.17g")
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(out.round(6).to_string(index=False))


if __name__ == "__main__":
    main()
