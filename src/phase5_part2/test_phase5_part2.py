"""Tests of src/phase5_part2 on synthetic data only (no real data is read).

  reconciliation  exact masks, stored masks, |Delta Dice| <= 1e-6 (boundary), any pixel difference fails
  judging rule    all three C4-C6 lower bounds > 0 and a range of width > 0; boundaries at 0 and width 0
  conclusion      the plan's two sentences; any not-consistent U-Net row -> backbone-dependent
  parsing         the Phase 5 better_than_both_p_range text (np.float64(...), 0.0, none)
  counts          206,595 rows and 2081 / 472, 2113 / 474 asserted
  outputs         the run log is not a result file; long table columns
Usage:  python src/phase5_part2/test_phase5_part2.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase5 import guards as Gu  # noqa: E402
from phase5_part2 import compute as C  # noqa: E402
from phase5_part2 import reconcile_unet as U  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return e.code == Gu.STOP_EXIT
    return False


# reconciliation
p = {"a": np.full((4, 4), 0.2, np.float32), "b": np.full((4, 4), 0.9, np.float32)}
p["b"][0, 0] = 0.5                                                     # exactly 0.5 is background
packed = {k: np.packbits(v > 0.5) for k, v in p.items()}
arch = {k: ((v > 0.5) * 255).astype(np.uint8) for k, v in p.items()}
new, old = pd.Series({"a": 0.5, "b": 0.8}), pd.Series({"a": 0.5, "b": 0.8})
r = U.reconcile_rows(["a", "b"], p, packed, arch, new, old)
check("identical masks and Dice -> rule met", U.rule_met(r) and r.mask_pixels_differing.sum() == 0)
arch2 = dict(arch); arch2["b"] = arch["b"].copy(); arch2["b"][0, 0] = 255
r2 = U.reconcile_rows(["a", "b"], p, packed, arch2, new, old)
check("one differing pixel -> rule not met", not U.rule_met(r2) and r2.mask_pixels_differing.max() == 1)
r3 = U.reconcile_rows(["a", "b"], p, packed, arch, pd.Series({"a": 0.5, "b": 0.800001}), old)
check("|Delta Dice| = 1e-6 -> still met", U.rule_met(r3), f"{r3.abs_dice_diff.max():.3g}")
r4 = U.reconcile_rows(["a", "b"], p, packed, arch, pd.Series({"a": 0.5, "b": 0.800003}), old)
check("|Delta Dice| = 3e-6 -> not met", not U.rule_met(r4))
pk2 = dict(packed); pk2["a"] = np.packbits(np.ones((4, 4), bool))
check("stored masks.npz inconsistent with P > 0.5 -> not met", not U.rule_met(U.reconcile_rows(["a", "b"], p, pk2, arch, new, old)))
s = U.summary(r2, 1001)
check("summary counts", s["cases_mask_differs"] == 1 and s["max_pixels_differing"] == 1 and not s["rule_met"])

# judging rule and conclusion
check("all lower bounds > 0 and width > 0 -> consistent", C.row_consistent([0.01, 0.02, 0.03], (0.1, 0.5)))
check("a lower bound exactly 0 -> not consistent", not C.row_consistent([0.0, 0.02, 0.03], (0.1, 0.5)))
check("a negative lower bound -> not consistent", not C.row_consistent([0.01, -0.001, 0.03], (0.1, 0.5)))
check("empty range -> not consistent", not C.row_consistent([0.01, 0.02, 0.03], None))
check("range of width 0 -> not consistent", not C.row_consistent([0.01, 0.02, 0.03], (0.3, 0.3)))
check("all U-Net rows consistent -> not specific to nnU-Net", C.overall([True, True, True]) == C.CONCLUSION[True]
      and "not specific to nnU-Net" in C.overall([True, True, True]))
check("one row not consistent -> backbone-dependent", C.overall([True, False, True]) == C.CONCLUSION[False]
      and "backbone-dependent" in C.overall([True, False, True]))

# parsing of the Phase 5 text
check("np.float64 range parsed", C.parse_range("np.float64(0.07337152046896658) - np.float64(0.15997087152019895)")
      == (0.07337152046896658, 0.15997087152019895))
check("0.0 start parsed", C.parse_range("0.0 - np.float64(0.6885532641289703)") == (0.0, 0.6885532641289703))
check("'none' and empty -> None", C.parse_range("none") is None and C.parse_range("") is None and C.parse_range(float("nan")) is None)

# counts
rows = []
for c, (n_img, n_grp) in C.COUNTS.items():
    donors = 3 if c in ("C4", "C5", "C6") else 1
    for i in range(n_img):
        for v in range(donors):
            rows.append({"condition": c, "sample_id": f"i{i}", "group": f"g{min(i, n_grp - 1)}"})
base = pd.DataFrame(rows)
D = pd.concat([base] * 9, ignore_index=True)
check("206,595 rows with the expected images and groups pass", not stops(C.check_counts, D) and len(D) == C.N_ROWS)
check("a missing row stops", stops(C.check_counts, D.iloc[:-1]))
D2 = D.copy(); D2.loc[D2.condition == "C6", "group"] = "g0"
check("wrong group count stops", stops(C.check_counts, D2))

# outputs
ro = C.result_outputs()
check("result outputs exclude the run log (append-only)", C.OUTS["log"] not in ro and len(ro) == 6)
Dl = pd.DataFrame({"dataset": ["QaTa-COV19"], "group": ["g"], "method": ["lvit"], "seed": [1001], "sample_id": ["a"],
                   "condition": ["C0"], "variant_id": ["0"], **{f"lang_{k}": [0.5] for k in C.core.METRICS},
                   **{f"img_{k}": [0.7] for k in C.core.METRICS}})
L = C.long_table(Dl, {"LANG": "lang", C.IMG: "img"})
check("long table: one row per strategy with part 'test' and the metrics",
      len(L) == 2 and set(L.strategy) == {"LANG", C.IMG} and (L.part == "test").all() and L[L.strategy == C.IMG].dice.iloc[0] == 0.7)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
