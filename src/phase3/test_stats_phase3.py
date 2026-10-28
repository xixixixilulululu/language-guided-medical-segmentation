"""Tests of src/phase3/stats_phase3.py on synthetic rows with known answers (no real results).

  aggregation   donors within image and seed -> seeds within image -> images within group -> groups equally
                (differs from a flat row mean on purpose)
  families      9 primary (LR-Gate - LANG, C4-C6 x 3 methods) and 9 secondary (LR-Gate - D-Gate), Holm within
                each family and part; descriptive / MosMed rows carry no p-value
  margin        C0 / C2 gate - LANG rows carry -0.005 and the CI flag; other rows do not
  pairing       strategies on different rows are refused
Usage:  python src/phase3/test_stats_phase3.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import stats_phase3 as P  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


rng = np.random.default_rng(0)
CONDS = ["C0", "C1", "C2", "C3a", "C4", "C5", "C6"]
rows = []
for ds, part in (("QaTa-COV19", "test"), ("MosMedData+", "MosMedData+ (all)")):
    for m in P.METHOD_ORDER:
        for c in CONDS:
            donors = ["d1", "d2", "d3"] if c in P.MISMATCH else ["0"]
            # groups g1 (images a, b), g2 (c), g3..g8 (one image each)
            for g, imgs in [("g1", ["a", "b"]), ("g2", ["c"])] + [(f"g{k}", [f"i{k}"]) for k in range(3, 9)]:
                for i in imgs:
                    for s in (1001, 1002, 1003):
                        for v in donors:
                            lang = 0.5
                            d_lr, d_d = rng.normal(0.02, 0.03), rng.normal(0.01, 0.03)
                            if (ds, m, c) == ("QaTa-COV19", "LViT-T", "C6") and g in ("g1", "g2"):
                                # crafted cell: image a gains 0.3 only for seed 1001 donor d1; b gains 0.06; c 0.02
                                d_lr = {"a": 0.3 if (s, v) == (1001, "d1") else 0.0, "b": 0.06, "c": 0.02}[i]
                            if i == "b" and c in P.MISMATCH and v != "d1":
                                continue                                   # image b has only one donor
                            for st, val in (("LANG", lang), ("LR-Gate", lang + d_lr), ("D-Gate", lang + d_d)):
                                rows.append(dict(dataset=ds, part=part, group=g, method=m, seed=s, sample_id=i,
                                                 condition=c, variant_id=v, strategy=st, dice=val))
S = pd.DataFrame(rows)

# aggregation on the crafted cell
gd = P.group_differences(S, "LR-Gate", "LANG")
u, n_img = gd[("QaTa-COV19", "test", "LViT-T", "C6")]
check("group g1 = mean(image a, image b) = (0.1/3 + 0.06) / 2", np.isclose(u["g1"], (0.1 / 3 + 0.06) / 2), f"{u['g1']:.6f}")
check("group g2 = 0.02", np.isclose(u["g2"], 0.02))
check("n_images counts images, not rows", n_img == 9, str(n_img))
cell = S[(S.dataset == "QaTa-COV19") & (S.method == "LViT-T") & (S.condition == "C6") & S.group.isin(["g1", "g2"])]
flat = (cell[cell.strategy == "LR-Gate"].dice.to_numpy() - cell[cell.strategy == "LANG"].dice.to_numpy()).mean()
check("differs from a flat row mean (test is informative)", not np.isclose(flat, u[["g1", "g2"]].mean()))

R = P.pairwise(S, "synthetic")
q = R[R.dataset == "QaTa-COV19"]
prim, sec = q[q.family == "primary"], q[q.family == "secondary"]
check("9 primary comparisons: LR-Gate - LANG on C4-C6 x 3 methods",
      len(prim) == 9 and set(prim.comparison) == {"LR-Gate - LANG"} and set(prim.condition) == set(P.MISMATCH))
check("9 secondary comparisons: LR-Gate - D-Gate on C4-C6", len(sec) == 9 and set(sec.comparison) == {"LR-Gate - D-Gate"})
# Holm by hand
for name, fam in (("primary", prim), ("secondary", sec)):
    p = fam.p_raw.to_numpy()
    o = np.argsort(p)
    adj = np.maximum.accumulate(np.minimum(1, (9 - np.arange(9)) * p[o]))
    want = np.empty(9)
    want[o] = adj
    check(f"Holm across the 9 {name} comparisons", np.allclose(fam.p_holm.to_numpy(), want))
# t-test over groups by hand (one primary row)
from scipy import stats  # noqa: E402
r = prim[(prim.method == "RecLMIS") & (prim.condition == "C5")].iloc[0]
uu = P.group_differences(S, "LR-Gate", "LANG")[("QaTa-COV19", "test", "RecLMIS", "C5")][0].values
check("p_raw = one-sample t over the 8 groups", r.n_groups == 8 and np.isclose(r.p_raw, stats.ttest_1samp(uu, 0).pvalue))
half = stats.t.ppf(0.975, 7) * uu.std(ddof=1) / np.sqrt(8)
check("95% t-CI over groups", np.isclose(r.ci95_low, uu.mean() - half) and np.isclose(r.ci95_high, uu.mean() + half))
desc = R[~R.family.isin(["primary", "secondary"])]
check("descriptive and MosMed rows have no p-value", desc.p_raw.isna().all() and desc.p_holm.isna().all())
check("D-Gate - LANG is descriptive", set(q[q.comparison == "D-Gate - LANG"].family) == {"descriptive"})
check("MosMed rows are exploratory", set(R[R.dataset == "MosMedData+"].family) == {"exploratory (MosMed)"})
mg = R[R.margin.notna()]
check("margin only on QaTa C0 / C2 gate - LANG",
      set(mg.condition) == {"C0", "C2"} and set(mg.comparison) == {"LR-Gate - LANG", "D-Gate - LANG"}
      and set(mg.dataset) == {"QaTa-COV19"} and (mg.margin == -0.005).all())
check("margin flag = CI lower bound >= -0.005", (mg.ci95_low_above_margin == (mg.ci95_low >= -0.005)).all())
# pairing
bad = S.drop(S[(S.strategy == "LANG") & (S.sample_id == "c")].index[:1])
try:
    P.group_differences(bad, "LR-Gate", "LANG")
    check("unpaired rows refused", False)
except ValueError:
    check("unpaired rows refused", True)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
