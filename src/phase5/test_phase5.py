"""Tests of src/phase5 on synthetic data only (no real data is read).

  SDC            formula, the all-zero case SDC(0, 0) = 0, B = P > 0.5 (0.5 itself is background), float64
  thresholds     gates.calibrate on -SDC selects the same switches as 'SDC < tau' with tau = -t; ties -> fewer
                 fallbacks; never-fallback (+inf -> tau = -inf) when no positive objective
  Fixed Average  0.5 / 0.5 average; exactly 0.5 is not foreground; scored with the common evaluator
  crossover      normal case, equal slopes -> none, outside [0, 1] -> none; better-than-both interval
  aggregation    donors -> seeds -> images -> groups
  self-checks    spec not committed / changed; dirty tree; output present; atomic gzip write
Usage:  python src/phase5/test_phase5.py      (exit status 1 if any check fails)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import gates as G  # noqa: E402
from phase5 import core as C  # noqa: E402
from phase5 import guards as Gu  # noqa: E402

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


# SDC
p = np.array([[0.9, 0.6], [0.3, 0.0]], np.float32)
want = 2 * (0.9 + 0.6) / ((0.9 + 0.6 + 0.3) + 2)
check("SDC formula 2 sum(PB) / (sum P + sum B)", abs(C.sdc(p) - want) < 1e-6, f"{C.sdc(p):.6f} vs {want:.6f}")
check("SDC(0, 0) = 0", C.sdc(np.zeros((4, 4))) == 0.0)
check("SDC with P only below 0.5 is 0 (B empty, denominator > 0)", C.sdc(np.full((2, 2), 0.4)) == 0.0)
check("SDC: 0.5 itself is background (strict >)", C.sdc(np.array([0.5, 1.0])) == 2 * 1.0 / (1.5 + 1))
check("SDC = 1 for a binary map", C.sdc(np.array([1.0, 1.0, 0.0])) == 1.0)

# thresholds: -SDC through gates.calibrate equals 'SDC < tau'
rng = np.random.default_rng(0)
rows = []
for c in G.CONDS:
    for i in range(40):
        g = f"g{i // 2}"
        s = float(rng.uniform(0, 1))
        gain = (0.3 if s < 0.35 else -0.02) if c in G.MISMATCH else (-0.0001 if s < 0.1 else -0.01)
        for v in (["d1", "d2"] if c in G.MISMATCH else ["0"]):
            rows.append({"condition": c, "sample_id": f"i{i}", "group": g, "variant_id": v, "gain": gain, "sdc": s})
cal = pd.DataFrame(rows)
cal["neg_sdc"] = -cal.sdc
curve, ch, why = G.calibrate(cal, "neg_sdc")
tau = C.tau_from_t(ch.threshold)
check("a positive objective is found", why == "feasible maximum" and ch.objective > 0, f"tau {tau:.4f}")
check("switches 'neg_sdc > t' == 'SDC < tau'", np.array_equal(cal.neg_sdc.to_numpy() > ch.threshold, C.sdc_switch(cal.sdc, tau)))
rate = C.sdc_switch(cal[cal.condition == "C4"].sdc, tau).mean()
check("calibrate's C4 fallback rate equals the share of rows with SDC < tau", abs(rate - ch["fallback_rate_C4"]) < 1e-12)
# tie: an extra row with zero gain gives the same objective; the chosen threshold must switch fewer rows
tie = pd.DataFrame([{"condition": c, "sample_id": f"t{i}", "group": f"t{i}", "variant_id": "0", "gain": gain, "sdc": s}
                    for c in G.CONDS for i, (s, gain) in enumerate([(0.1, 0.2 if c in G.MISMATCH else 0.0), (0.2, 0.0), (0.9, 0.0)])])
tie["neg_sdc"] = -tie.sdc
_, cht, _ = G.calibrate(tie, "neg_sdc")
taut = C.tau_from_t(cht.threshold)
check("tie between switching 1 or 2 rows -> fewer fallbacks (tau between 0.1 and 0.2)",
      0.1 < taut <= 0.2 and C.sdc_switch(tie[tie.condition == "C4"].sdc, taut).sum() == 1, f"tau {taut}")
neg = tie.assign(gain=-0.1)
_, chn, whyn = G.calibrate(neg, "neg_sdc")
check("no positive objective -> never fall back (t = +inf, tau = -inf)", chn.threshold == np.inf and C.tau_from_t(chn.threshold) == -np.inf
      and not C.sdc_switch([0.0, 1.0], C.tau_from_t(chn.threshold)).any())

# Fixed Average
pa, ba = C.fixed_average(np.array([0.25, 0.75, 1.0, 0.0], np.float32), np.array([0.75, 0.25, 0.2, 0.0], np.float32))
check("Fixed Average 0.5 / 0.5: exactly 0.5 (0.25 and 0.75) is background", pa[0] == 0.5 and pa[1] == 0.5
      and ba.tolist() == [False, False, True, False])
pf, bf = C.fixed_average(np.array([0.4], np.float32), np.array([0.6], np.float32))
check("Fixed Average in float64: float32 0.4 and 0.6 average to 0.5000000149 -> foreground", pf.dtype == np.float64
      and pf[0] > 0.5 and bf.tolist() == [True], f"{pf[0]!r}")
gt = np.zeros((224, 224), np.uint8); gt[10:20, 10:20] = 1
m = np.zeros((224, 224), bool); m[10:20, 10:20] = True
sc = C.score_mask(m, gt)
check("scored with the common evaluator: identical mask -> Dice 1, HD95 0, not empty", sc["dice"] == 1.0 and sc["hd95"] == 0.0 and sc["empty"] == 0.0)
check("empty prediction -> Dice 0, HD95 = diagonal, empty 1", C.score_mask(np.zeros((224, 224), bool), gt)["empty"] == 1.0
      and C.score_mask(np.zeros((224, 224), bool), gt)["hd95"] > 316)

# crossover
check("crossover normal case: dc = -0.02, dw = +0.06 -> p* = 0.25", abs(C.crossover(-0.02, 0.06) - 0.25) < 1e-12)
check("crossover equal slopes -> none", C.crossover(0.03, 0.03) is None)
check("crossover outside [0, 1] -> none", C.crossover(0.02, 0.01) is None and C.crossover(-0.05, -0.01) is None)
check("crossover at the ends is reported", C.crossover(0.0, 0.05) == 0.0 and C.crossover(-0.05, 0.0) == 1.0)
check("better interval: A worse clean, better wrong -> (p*, 1]", C.better_interval(-0.02, 0.06) == (0.25, 1.0))
check("better interval: A better clean, worse wrong -> [0, p*)", C.better_interval(0.02, -0.06) == (0.0, 0.25))
check("better than both = intersection", C.intersect(C.better_interval(-0.01, 0.09), C.better_interval(0.03, -0.01)) == (0.1, 0.75))
check("no overlap -> none", C.intersect((0.0, 0.2), (0.5, 1.0)) is None and C.better_interval(-0.01, -0.02) is None)
check("expected Dice", abs(C.expected(0.8, 0.6, 0.25) - 0.75) < 1e-12)

# aggregation donors -> seeds -> images -> groups
S = pd.DataFrame([("m", "C4", "g1", "a", 1, 0.0), ("m", "C4", "g1", "a", 1, 1.0), ("m", "C4", "g1", "a", 2, 1.0),
                  ("m", "C4", "g1", "b", 1, 0.0), ("m", "C4", "g2", "c", 1, 1.0)],
                 columns=["method", "condition", "group", "sample_id", "seed", "dice"])
# a: seed1 mean 0.5, seed2 1.0 -> 0.75; b 0; g1 = 0.375; g2 = 1 -> 0.6875
check("group-weighted mean (donors -> seeds -> images -> groups)", abs(C.group_weighted_mean(S)[("m", "C4")] - 0.6875) < 1e-12)

# self-checks
d = tempfile.mkdtemp()
run = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, check=True)  # noqa: E731
run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
os.makedirs(os.path.join(d, "configs", "phase5")); open(os.path.join(d, Gu.SPEC), "w").write("{}\n")
check("spec not committed -> stop", stops(Gu.committed_unchanged, d, Gu.SPEC))
run("add", "-A"); run("commit", "-qm", "spec")
check("committed spec passes", not stops(Gu.committed_unchanged, d, Gu.SPEC))
open(os.path.join(d, Gu.SPEC), "w").write('{"x": 1}\n')
check("changed spec -> stop", stops(Gu.committed_unchanged, d, Gu.SPEC))
check("dirty tree -> stop", stops(Gu.clean_tree, d))
run("checkout", "-q", "--", Gu.SPEC)
os.makedirs(os.path.join(d, "results")); open(os.path.join(d, "results", "x.csv"), "w").write("x")
check("existing output -> stop", stops(Gu.outputs_absent, d, ["results/x.csv"]))
Gu.write_atomic(pd.DataFrame({"a": [1, 2]}), d, "results/y.csv.gz")
check("atomic write produces a readable gzip file and no temporary file",
      pd.read_csv(os.path.join(d, "results", "y.csv.gz")).a.tolist() == [1, 2] and not os.path.exists(os.path.join(d, "results", "y.csv.gz.tmp")))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
