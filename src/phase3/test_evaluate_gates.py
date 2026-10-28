"""Tests of src/phase3/evaluate_gates.py on synthetic rows with known answers (no real results).

  strategies    LANG / IMG / gates take the metrics of the selected prediction; ORACLE keeps LANG on ties;
                NULL takes the C1 metrics
  aggregation   image-weighted: donors within image and seed, then images, then mean over seeds
  routing       beneficial / harmful / neutral switches (|dDice| <= 1e-6) and the net gain
Usage:  python src/phase3/test_evaluate_gates.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import evaluate_gates as E  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


rows = []
for s in (1001, 1002, 1003):
    for i, g in (("a", "g1"), ("b", "g2")):
        for v, ld, idc, dsw in (("d1", 0.6, 0.8, True), ("d2", 0.9, 0.7, True), ("d3", 0.5, 0.5, True)):
            r = dict(dataset="QaTa-COV19", part="GATE_CAL", group=g, method="lvit", seed=s, sample_id=i, condition="C6",
                     variant_id=v, d_switch=dsw, lr_switch=False)
            for m in E.METRICS:
                r[f"lang_{m}"], r[f"img_{m}"], r[f"null_{m}"] = (ld, idc, 0.3) if m != "empty" else (0.0, 0.0, 0.0)
            rows.append(r)
D = pd.DataFrame(rows)
S = E.strategy_metrics(D)
d = S[(S.strategy == "D-Gate") & (S.seed == 1001) & (S.sample_id == "a")].set_index("variant_id").dice
check("D-Gate takes the IMG metrics when switched", np.allclose(d[["d1", "d2", "d3"]], [0.8, 0.7, 0.5]))
o = S[(S.strategy == "ORACLE") & (S.seed == 1001) & (S.sample_id == "a")].set_index("variant_id")
check("ORACLE: better of the two; tie keeps LANG (switch 0)", np.allclose(o.dice[["d1", "d2", "d3"]], [0.8, 0.9, 0.5])
      and o.switch["d3"] == 0)
check("NULL takes the C1 metrics", np.allclose(S[S.strategy == "NULL (C1)"].dice, 0.3))
res = E.image_weighted(S).set_index("strategy")
check("image-weighted Dice: LANG (0.6+0.9+0.5)/3", np.isclose(res.loc["LANG", "dice"], 2.0 / 3))
check("fallback rate D-Gate 1, LR-Gate 0", res.loc["D-Gate", "fallback_rate"] == 1 and res.loc["LR-Gate", "fallback_rate"] == 0)
rq = E.routing(D).set_index("gate").loc["D-Gate"]
check("routing: 6 beneficial, 6 harmful, 6 neutral of 18 switched", (rq.n_switched, rq.n_beneficial, rq.n_harmful, rq.n_neutral)
      == (18, 6, 6, 6), (rq.n_switched, rq.n_beneficial, rq.n_harmful, rq.n_neutral))
check("routing: net gain (0.2 - 0.2 + 0) / 3 = 0", np.isclose(rq.net_dice_gain_image_weighted, 0.0))
print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
