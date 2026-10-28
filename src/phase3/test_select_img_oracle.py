"""Tests of src/phase3/select_img_oracle.py on synthetic data with known answers (no real results).

  3.1C  group-weighted choice differs from image-weighted (a large group favours U-Net, groups favour nnU-Net);
        a tie within 1e-6 goes to nnU-Net
  3.1D  actual / oracle gains, neutral tolerance, S_FIT intersection, group weighting, headroom flag
Usage:  python src/phase3/test_select_img_oracle.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import select_img_oracle as S  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


# group A: 3 images, group B: 1 image
fit = pd.DataFrame({"sample_id": ["a1", "a2", "a3", "b1"], "group_id": ["A", "A", "A", "B"]})
rows = []
for s in S.SEEDS:
    for i in ["a1", "a2", "a3"]:
        rows += [dict(arch="unet", seed=s, sample_id=i, dice=0.9), dict(arch="nnunet", seed=s, sample_id=i, dice=0.8)]
    rows += [dict(arch="unet", seed=s, sample_id="b1", dice=0.1), dict(arch="nnunet", seed=s, sample_id="b1", dice=0.5)]
img = pd.DataFrame(rows)
sel, arch = S.choose_img(img, fit)
t = sel.set_index("arch")
check("image-weighted means: U-Net 0.70, nnU-Net 0.725", np.isclose(t.loc["U-Net", "image_weighted_dice"], 0.7)
      and np.isclose(t.loc["nnU-Net", "image_weighted_dice"], 0.725))
check("group-weighted: U-Net 0.5, nnU-Net 0.65 -> nnU-Net chosen", arch == "nnunet"
      and np.isclose(t.loc["U-Net", "group_weighted_dice"], 0.5) and np.isclose(t.loc["nnU-Net", "group_weighted_dice"], 0.65))
img2 = img.copy()
img2.loc[img2.arch == "unet", "dice"] = img2.loc[img2.arch == "nnunet", "dice"].values + 5e-7
check("tie within 1e-6 -> nnU-Net", S.choose_img(img2, fit)[1] == "nnunet")
img3 = img.copy()
img3.loc[img3.arch == "unet", "dice"] = img3.loc[img3.arch == "nnunet", "dice"].values + 0.01
check("U-Net higher by 0.01 -> U-Net", S.choose_img(img3, fit)[1] == "unet")

# 3.1D: LANG C4-C6; b1 has no C5 -> S_FIT = a1..a3
lang = []
for s in S.SEEDS:
    for i in ["a1", "a2", "a3", "b1"]:
        lang.append(dict(method="lvit", seed=s, sample_id=i, condition="C4", variant_id="d1", dice=0.8))   # = IMG (0.8)
        if i != "b1":
            lang.append(dict(method="lvit", seed=s, sample_id=i, condition="C5", variant_id="d1", dice=0.7))  # IMG +0.1
        lang.append(dict(method="lvit", seed=s, sample_id=i, condition="C6", variant_id="d1", dice=0.9))   # IMG -0.1
        lang.append(dict(method="lvit", seed=s, sample_id=i, condition="C6", variant_id="d2", dice=0.6))   # IMG +0.2
lang = pd.DataFrame(lang)
imgn = img[img.arch == "nnunet"][["seed", "sample_id", "dice"]]
r = S.oracle_rows(lang, imgn, fit)
a6 = r[(r.condition == "C6") & (r.variant_id == "d1") & r.sample_id.str.startswith("a")]
check("actual / oracle gain per row (group A, C6 d1: -0.1 / 0)", np.isclose(a6.actual_gain, -0.1).all()
      and np.isclose(a6.oracle_gain, 0).all())
check("neutral when |gain| <= 1e-6", (r[(r.condition == "C4") & r.sample_id.str.startswith("a")].outcome == "neutral").all())
t = S.summaries(r)
c6 = t[(t.condition == "C6") & (t.seed == 1001)].iloc[0]
check("S_FIT = 3 images (b1 lacks C5)", c6.n_images == 3 and c6.n_S_fit_images == 3)
check("C6 oracle gain: donor mean of 0 and 0.2 = 0.1", np.isclose(c6.oracle_gain_group_weighted, 0.1))
check("C6 actual gain: mean of -0.1 and +0.2 = 0.05", np.isclose(c6.actual_gain_group_weighted, 0.05))
check("headroom = mean over C4, C5, C6 = (0 + 0.1 + 0.1) / 3, not insufficient",
      np.isclose(c6.headroom_mean_over_seeds, 0.2 / 3) and not c6.insufficient_headroom)
print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
