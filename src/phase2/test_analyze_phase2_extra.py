"""Tests of src/phase2/analyze_phase2_extra.py on synthetic data with known answers (no real results, no text).

Synthetic QaTa set, two methods (lvit, reclmis), three seeds, 4 images in 2 patients (sub-S1: 2 images, sub-S2: 2):
  C0 = 0.8, C1 = 0.7; C4 donor Dice 0.75 (changed: location), C5 0.6 (count+location), C6 donors 0.2 / 0.4
  (scope+count+location). Image sub-S2 #4 has no C5 donor, so S = the other 3 images.
  severity       S size, means on S, C0-C4 = 0.05, C0-C6 = 0.5, unit delta with 2 units
  by attribute   harm per changed-attribute group (target mean first)
  susceptibility H(lvit) = 0.5; reclmis C6 is 0.1 lower -> H_A - H_B = -0.1 (image and unit weighted)
  C2 change      synthetic packed masks: 1 of 4 images changes 3 pixels, 1 flips from empty to non-empty
  cases          quantile positions round(q (n - 1)) on the sorted harm
Usage:  python src/phase2/test_analyze_phase2_extra.py      (exit status 1 if any check fails)
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2 import analyze_phase2_extra as X  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


IDS = ["sub-S1_ses-E1_x", "sub-S1_ses-E2_x", "sub-S2_ses-E3_x", "sub-S2_ses-E4_x"]
UNIT = {i: i.split("_")[0] for i in IDS}


def synthetic():
    rows, man = [], []
    for m, shift in (("lvit", 0.0), ("reclmis", 0.1)):
        for s in X.SEEDS:
            for k, i in enumerate(IDS):
                def add(c, v, d, donor=""):
                    rows.append(dict(dataset="QaTa-COV19", method=m, seed=s, sample_id=i, condition=c, variant_id=v,
                                     donor_sample_id=donor, unit=UNIT[i], dice=d, miou=d, hd95=1.0, assd=1.0, empty=False))
                add("C0", "0", 0.8)
                add("C1", "0", 0.7)
                add("C2", "0", 0.8)
                add("C4", "d1", 0.75, "q")
                if k < 3:
                    add("C5", "d1", 0.6, "q")
                add("C6", "d1", 0.2 - shift, "q")
                add("C6", "d2", 0.4 - shift, "r")
    for k, i in enumerate(IDS):
        man.append(dict(dataset="QaTa-COV19", target_id=i, condition="C4", variant="d1", donor_id="q", changed_attributes="location"))
        if k < 3:
            man.append(dict(dataset="QaTa-COV19", target_id=i, condition="C5", variant="d1", donor_id="q",
                            changed_attributes="count+location"))
        for v, d in (("d1", "q"), ("d2", "r")):
            man.append(dict(dataset="QaTa-COV19", target_id=i, condition="C6", variant=v, donor_id=d,
                            changed_attributes="scope+count+location"))
    return pd.DataFrame(rows), pd.DataFrame(man)


def main():
    pc, man = synthetic()
    X.METHODS = ["lvit", "reclmis"]
    X.PAIRS = [("lvit", "reclmis")]
    X.DATASETS = ["QaTa-COV19"]

    sev = X.severity(pc, man).set_index("method")
    r = sev.loc["LViT-T"]
    check("S = images with C4, C5 and C6 (3 of 4)", r.n_images_S == 3 and np.isclose(r.coverage_S, 0.75))
    check("S identical to each method's eligible set", bool(r.S_same_as_method_eligible))
    check("means on S: C0 0.8, C1 0.7, C4 0.75, C5 0.6, C6 0.3 (donor mean)",
          np.allclose([r.mean_C0, r.mean_C1, r.mean_C4, r.mean_C5, r.mean_C6], [0.8, 0.7, 0.75, 0.6, 0.3]))
    check("C0-C4 = 0.05, C0-C6 = 0.5 (image and unit weighted)",
          np.isclose(r["image_delta_C0-C4"], 0.05) and np.isclose(r["unit_delta_C0-C6"], 0.5))
    check("constant unit differences: CI undefined (NaN)", np.isnan(r["unit_ci95_low_C0-C6"]))

    att = X.severity_by_attribute(pc, man)
    a = att[(att.method == "LViT-T")].set_index("condition")
    check("by attribute: C4 location harm 0.05 on 3 targets", np.isclose(a.loc["C4"].mean_harm, 0.05)
          and a.loc["C4"].n_targets == 3 and a.loc["C4"].changed_attributes == "location")
    check("by attribute: C6 harm 0.5 from 6 donor pairs", np.isclose(a.loc["C6"].mean_harm, 0.5)
          and a.loc["C6"].n_donor_pairs == 6)

    sus = X.susceptibility(pc, man)
    m = sus[sus.kind == "method"].set_index("method_a")
    check("susceptibility: H lvit 0.5, reclmis 0.6; C1-C6 lvit 0.4",
          np.isclose(m.loc["LViT-T"].H, 0.5) and np.isclose(m.loc["RecLMIS"].H, 0.6) and np.isclose(m.loc["LViT-T"]["C1-C6"], 0.4))
    p = sus[sus.kind == "pair"].iloc[0]
    check("pair LViT-T - RecLMIS: H difference -0.1 (image and unit)", np.isclose(p.image_delta_H, -0.1)
          and np.isclose(p.unit_delta_H, -0.1) and p.n_units == 2)

    # unit CI with non-constant unit differences: equals the t interval
    d, lo, hi = X.unit_ci([0.1, 0.3])
    from scipy import stats
    half = stats.t.ppf(0.975, 1) * np.std([0.1, 0.3], ddof=1) / np.sqrt(2)
    check("unit CI = mean +- t(0.975, n-1) SE", np.isclose(d, 0.2) and np.isclose(hi - d, half) and np.isclose(d - lo, half))

    # C2 mask change on synthetic packed masks
    tmp = tempfile.mkdtemp(prefix="p2x_")
    try:
        for mth in X.METHODS:
            for s in X.SEEDS:
                out = os.path.join(tmp, mth, "QaTa-COV19", f"seed{s}")
                os.makedirs(out)
                base = {i: np.zeros((224, 224), bool) for i in IDS}
                for i in IDS[:3]:
                    base[i][10:20, 10:20] = True
                c2 = {i: v.copy() for i, v in base.items()}
                c2[IDS[0]][30, 30:33] = True                        # 3 pixels added
                c2[IDS[3]][50, 50] = True                           # empty -> non-empty
                np.savez(os.path.join(out, "masks_C0_0.npz"), **{i: np.packbits(v) for i, v in base.items()})
                np.savez(os.path.join(out, "masks_C2_0.npz"), **{i: np.packbits(v) for i, v in c2.items()})
        mc = X.c2_mask_change(tmp)
        ch = X.c2_change(pc, mc).set_index("method")
        check("C2: 2 of 4 images change (0.5), 1 flip per seed",
              np.isclose(ch.loc["LViT-T"].frac_mask_changed, 0.5) and ch.loc["LViT-T"].empty_flips_total == 3)
        check("C2: |dDice| = 0 when the per-case Dice is unchanged", np.isclose(ch.loc["LViT-T"].mean_abs_dDice, 0))
        check("C2: pixels_changed per image", sorted(mc[(mc.method == "LViT-T") & (mc.seed == 1001)].pixels_changed) == [0, 0, 1, 3])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # case selection: harm per image differs by a known amount
    pc2 = pc.copy()
    for k, i in enumerate(IDS):
        pc2.loc[(pc2.sample_id == i) & (pc2.condition == "C6"), "dice"] -= 0.01 * k
    picks = X.select_cases(pc2, man, "QaTa-COV19")
    check("case selection: positions round(q (n-1)) on sorted harm",
          [p["sample_id"] for p in picks] == [IDS[0], IDS[1], IDS[2], IDS[2], IDS[3]], [p["sample_id"] for p in picks])
    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
