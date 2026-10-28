"""Tests of src/phase3/features.py and src/phase3/gates.py on synthetic data with known answers (no real results).

  features        F1-F5 of small maps computed by hand; F2 empty-mask rules (both empty -> 0, one empty -> 1)
  nested weights  weighted sum = donors within image -> images within group -> groups equally (brute force)
  calibration     Delta(c, t) of every candidate equals a brute-force nested average; feasible maximum; ties to
                  the higher threshold; never-fallback when no positive feasible objective; C0 / C2 on all CAL
                  images, C4-C6 on S_CAL
  LR weights      each condition sums to the same total, groups balanced within condition, mean 1
  one class       fit_lr reports "unsupported"
Usage:  python src/phase3/test_phase3_gates.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import features as FT  # noqa: E402
from phase3 import gates as G  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


# ---- features ----
pl = np.zeros((2, 2), np.float32); pl[0, 0] = 0.9; pl[0, 1] = 0.6           # B_lang: 2 pixels
pi = np.zeros((2, 2), np.float32); pi[0, 0] = 0.8; pi[1, 1] = 0.7           # B_img: 2 pixels, overlap 1
f = FT.features(pl, pi)
H = lambda p: -(np.clip(p, 1e-6, 1 - 1e-6) * np.log(np.clip(p, 1e-6, 1 - 1e-6))  # noqa: E731
                + (1 - np.clip(p, 1e-6, 1 - 1e-6)) * np.log(1 - np.clip(p, 1e-6, 1 - 1e-6)))
check("F1 = mean |P_lang - P_img|", np.isclose(f[0], (0.1 + 0.6 + 0 + 0.7) / 4))
check("F2 = 1 - Dice(B_lang, B_img) = 1 - 2*1/4", np.isclose(f[1], 0.5))
check("F3 = |area difference| = 0", np.isclose(f[2], 0.0))
check("F4 / F5 = mean binary entropy (clipped)", np.isclose(f[3], H(pl.astype(np.float64)).mean())
      and np.isclose(f[4], H(pi.astype(np.float64)).mean()))
z = np.zeros((2, 2), np.float32)
check("F2 both empty -> 0", FT.features(z, z)[1] == 0.0)
check("F2 one empty -> 1", FT.features(pl, z)[1] == 1.0)

# ---- synthetic calibration data: 4 groups, 6 images, conditions C0, C2 (1 row), C4-C6 (2 donors) ----
rng = np.random.default_rng(5)
imgs = [("g1", "a"), ("g1", "b"), ("g2", "c"), ("g3", "d"), ("g3", "e"), ("g4", "f")]
rows = []
for g, i in imgs:
    for c in G.CONDS:
        for v in (["0"] if c in ("C0", "C2") else ["d1", "d2"]):
            if c == "C5" and i == "f":
                continue                                   # f not in S_CAL
            rows.append(dict(group=g, sample_id=i, condition=c, variant_id=v, score=float(rng.uniform()),
                             gain=float(rng.normal(0.05 if c in G.MISMATCH else -0.004, 0.05))))
cal = pd.DataFrame(rows)


def brute(cal, c, t):
    d = G.calibration_sets(cal)[c]
    d = d.assign(x=np.where(d.score > t, d.gain, 0.0))
    per_img = d.groupby(["group", "sample_id"]).x.mean()
    return per_img.groupby(level="group").mean().mean()


w = G.nested_weights(cal[cal.condition == "C4"])
check("nested weights sum to 1", np.isclose(w.sum(), 1.0))
curve, chosen, why = G.calibrate(cal, "score")
ok = all(np.isclose(curve.loc[k, f"delta_{c}"], brute(cal, c, t)) for k, t in curve.threshold.items() for c in G.CONDS)
check("Delta(c, t) equals the brute-force nested average for every candidate and condition", ok)
check("C4-C6 use S_CAL (image f excluded), C0 uses all 6 images",
      G.calibration_sets(cal)["C4"].sample_id.nunique() == 5 and G.calibration_sets(cal)["C0"].sample_id.nunique() == 6)
check("candidates = distinct CAL scores + both infinities", len(curve) == cal.score.nunique() + 2)
f_ = curve[curve.feasible]
check("chosen = feasible maximum (or never-fallback)", (why == "feasible maximum" and np.isclose(chosen.objective, f_.objective.max()))
      or (why.startswith("never") and chosen.threshold == np.inf))
check("+inf is feasible with Delta 0", curve[curve.threshold == np.inf][[f"delta_{c}" for c in G.CONDS]].abs().max().max() == 0)
# tie: two thresholds with identical switching set -> higher threshold wins
cal2 = cal.copy(); cal2["gain"] = np.where(cal2.condition.isin(G.MISMATCH), 0.1, 0.0)
cal2["score"] = np.where(cal2.condition.isin(G.MISMATCH), 0.9, 0.1)
_, ch2, _ = G.calibrate(cal2, "score")
check("ties go to the higher threshold (0.1, not -inf)", ch2.threshold == 0.1, ch2.threshold)
cal3 = cal.copy(); cal3["gain"] = -0.1
_, ch3, why3 = G.calibrate(cal3, "score")
check("no positive feasible objective -> never fall back (+inf)", ch3.threshold == np.inf and why3.startswith("never"))

# ---- LR weights and one-class case ----
fit = cal.assign(y=(cal.gain > 0.005).astype(int))
lw = G.lr_weights(fit)
tot = lw.groupby(fit.condition).sum()
check("LR weights: equal total per condition", np.allclose(tot, tot.iloc[0]))
check("LR weights: mean 1", np.isclose(lw.mean(), 1.0))
g4 = lw[(fit.condition == "C4")].groupby(fit.group).sum()
check("LR weights: groups balanced within a condition", np.allclose(g4, g4.iloc[0]))
for k in G.FEATS:
    fit[k] = rng.normal(size=len(fit))
_, lr1, info1 = G.fit_lr(fit.assign(y=0))
check("one class -> LR-Gate unsupported", lr1 is None and "unsupported" in info1["status"])
sc, lr, info = G.fit_lr(fit)
check("two classes -> fitted with mean-1 weights", lr is not None and np.isclose(info["weight_mean"], 1.0))
print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
