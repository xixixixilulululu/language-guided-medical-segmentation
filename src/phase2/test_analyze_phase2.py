"""Tests of src/phase2/analyze_phase2.py on synthetic per-case files with known answers (no real results).

Synthetic QaTa set: 6 images in 3 patients (sub-S1: 3 images, sub-S2: 2, sub-S3: 1), one method, three seeds.
  donor averaging   C6 has 3 donors with Dice 0.2 / 0.4 / 0.6 -> donor mean 0.4 per image and seed
  image vs unit     a difference that lives only in the 3 images of sub-S1 gives image-weighted 3/6 x and
                    unit-weighted 1/3 x
  common sets       C5 is missing for one image: C0-C5 uses 5 images
  t-test / Holm     unit-weighted t-test equals scipy's one-sample t-test on the unit means; Holm over 11
  Phase 3 rule      seed-wise thresholds: repeated / unstable / no signal as constructed
Usage:  python src/phase2/test_analyze_phase2.py      (exit status 1 if any check fails)
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2 import analyze_phase2 as A  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


IDS = [f"sub-S{p}_ses-E{i}_run-1_bp-chest_cr" for p, i in [(1, 1), (1, 2), (1, 3), (2, 4), (2, 5), (3, 6)]]


def write(root, seed, rows):
    out = os.path.join(root, "val", "lvit", "QaTa-COV19", f"seed{seed}")
    os.makedirs(out, exist_ok=True)
    df = pd.DataFrame(rows)
    for (c, v), g in df.groupby(["condition", "variant_id"]):
        g = g.assign(dataset="QaTa-COV19", split="val", method="lvit", config="x", checkpoint_sha256="x", seed=seed,
                     miou=g.dice, hd95=1.0, assd=1.0, empty=False)
        g[["dataset", "split", "method", "config", "checkpoint_sha256", "seed", "sample_id", "condition",
           "variant_id", "donor_sample_id", "dice", "miou", "hd95", "assd", "empty"]].to_csv(
            os.path.join(out, f"per_case_{c}_{v}.csv"), index=False)


def synthetic(root, c5_drop_by_seed):
    """C0 = 0.8; C1 = 0.8 - 0.06 on sub-S1 images only; C6 donors 0.2/0.4/0.6 (mean 0.4);
    C5 = C0 - c5_drop_by_seed[seed] on every image except the last; other conditions = C0."""
    for seed in A.SEEDS:
        rows = []
        for k, sid in enumerate(IDS):
            base = 0.8
            rows.append(dict(sample_id=sid, condition="C0", variant_id="0", donor_sample_id="", dice=base))
            rows.append(dict(sample_id=sid, condition="C1", variant_id="0", donor_sample_id="",
                             dice=base - (0.06 if sid.startswith("sub-S1_") else 0.0)))
            for c in ("C2", "C3a", "C3b", "C3c"):
                rows.append(dict(sample_id=sid, condition=c, variant_id="0", donor_sample_id="", dice=base))
            for v, x in zip(("d1", "d2", "d3"), (0.2, 0.4, 0.6)):
                rows.append(dict(sample_id=sid, condition="C6", variant_id=v, donor_sample_id="z", dice=x))
                rows.append(dict(sample_id=sid, condition="C4", variant_id=v, donor_sample_id="z", dice=base))
                if k < len(IDS) - 1:
                    rows.append(dict(sample_id=sid, condition="C5", variant_id=v, donor_sample_id="z",
                                     dice=base - c5_drop_by_seed[seed]))
        write(root, seed, rows)


def main():
    tmp = tempfile.mkdtemp(prefix="p2ana_")
    try:
        synthetic(tmp, {1001: 0.03, 1002: 0.025, 1003: 0.0})
        df = A.load(tmp, "val")
        img = A.image_seed_table(df)
        c6 = img[(img.condition == "C6")].dice
        check("donor averaging: C6 = mean of 0.2 / 0.4 / 0.6", np.allclose(c6, 0.4))
        check("units from the ids (3 patients)", df.unit.nunique() == 3)

        pw = A.pairwise_table(img, "val").set_index("comparison")
        r = pw.loc["C0-C1"]
        check("image-weighted delta C0-C1 = 3/6 x 0.06", np.isclose(r.image_weighted_delta, 0.03), r.image_weighted_delta)
        check("unit-weighted delta C0-C1 = 1/3 x 0.06", np.isclose(r.unit_weighted_delta, 0.02), r.unit_weighted_delta)
        t = stats.ttest_1samp([0.06, 0.0, 0.0], 0.0)
        check("unit t-test = scipy one-sample t-test on unit means", np.isclose(r.p_raw, t.pvalue))
        check("C0-C5 uses the common eligible set (5 images)", pw.loc["C0-C5"].n_images == 5)
        check("C0-C6 image delta = 0.8 - 0.4", np.isclose(pw.loc["C0-C6"].image_weighted_delta, 0.4))
        check("constant difference: inference undefined, Holm p NA", np.isnan(pw.loc["C0-C6"].p_raw)
              and np.isnan(pw.loc["C0-C6"].p_holm))
        check("11 comparisons", len(pw) == 11)
        check("QaTa with 3 units is flagged exploratory", bool(pw.loc["C0-C1"].exploratory))

        rule = A.phase3_rule(img, "val")
        cls = rule.drop_duplicates("comparison").set_index("comparison").classification
        # C0-C5: seeds 1001 (0.03) and 1002 (0.025) >= 0.02 with unit delta > 0 -> repeated
        check("Phase 3: C0-C5 repeated signal (2 of 3 seeds)", cls["C0-C5"] == "repeated signal", cls["C0-C5"])
        check("Phase 3: C0-C6 repeated signal", cls["C0-C6"] == "repeated signal")
        check("Phase 3: C1-C5 no signal (0.03 - 0.03 / 0.025 - 0.03 ... below 0.01)", cls["C1-C5"] == "no signal",
              cls["C1-C5"])
        s, per_md, outcome = A.phase3_summary(rule)
        check("Phase 3 outcome: basis for Phase 3", outcome == "basis for Phase 3")
        donors = A.phase3_per_donor(df, "val")
        d = donors[(donors.comparison == "C0-C6") & (donors.seed == 1001)].set_index("variant_id").delta_img
        check("per-donor C0-C6 = 0.6 / 0.4 / 0.2", np.allclose(d[["d1", "d2", "d3"]], [0.6, 0.4, 0.2]))

        # unstable: image-weighted meets in 2 seeds, but the unit-weighted delta is <= 0 in them
        tmp2 = os.path.join(tmp, "b")
        synthetic(tmp2, {1001: 0.0, 1002: 0.0, 1003: 0.0})
        for seed in (1001, 1002):            # C5: d = +0.15 on 3 sub-S1 images, -0.16 on 2 sub-S2: img 0.026, unit -0.005
            p = os.path.join(tmp2, "val", "lvit", "QaTa-COV19", f"seed{seed}")
            for v in ("d1", "d2", "d3"):
                f = os.path.join(p, f"per_case_C5_{v}.csv")
                x = pd.read_csv(f)
                x.loc[x.sample_id.str.startswith("sub-S1_"), "dice"] = 0.8 - 0.15
                x.loc[x.sample_id.str.startswith("sub-S2_"), "dice"] = 0.8 + 0.16
                x.to_csv(f, index=False)
        rule2 = A.phase3_rule(A.image_seed_table(A.load(tmp2, "val")), "val")
        c2 = rule2[rule2.comparison == "C0-C5"]
        check("Phase 3: image-only hits -> unstable signal", (c2.classification == "unstable signal").all(),
              f"img {c2.delta_img.round(3).tolist()} unit {c2.delta_unit.round(3).tolist()}")
        check("test split refused while the lock is closed", os.system(
            f"{sys.executable} {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'analyze_phase2.py')} "
            f"--split test --runs-root {tmp} --results-dir {tmp} > /dev/null") >> 8 == 5)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
