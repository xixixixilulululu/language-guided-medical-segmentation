"""Tests of src/phase4/vertical_train_check.py with synthetic masks and git repositories (no real data).

  groups      upper-only / lower-only (one- or two-sided); middle, all, unspecified, mixed -> not counted
  centroid    row 111.x -> upper, row 112 -> lower; band share
  exclusions  lateral, empty mask, parse not ok, not in a group (first reason)
  rules       pass; rule 1 / 2 just failing; rule 3 -> insufficient sample
  self-checks rule not committed / changed / committed twice / not an ancestor; dirty tree; output present
Usage:  python src/phase4/test_vertical_train_check.py      (exit status 1 if any check fails)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import option1_train_check as O  # noqa: E402
from phase4 import vertical_train_check as V  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def m(*rows):
    x = np.zeros((224, 224), np.uint8)
    for r in rows:
        x[r, 5] = 1
    return x


g = V.group_of_location
check("upper-only one-sided and two-sided", g("left:upper") == "upper_only" and g("left:upper;right:upper") == "upper_only")
check("lower-only one-sided and two-sided", g("right:lower") == "lower_only" and g("left:lower;right:lower") == "lower_only")
check("middle / all / unspecified / mixed / multi-zone not counted",
      all(g(x) is None for x in ("left:middle", "left:all", "right:unspecified", "left:upper;right:lower",
                                 "left:upper+middle", "left:upper+lower", "", None)))
c, b = V.centroid_and_band(m(111, 112), "upper_only")
check("centroid 111.5 is upper (< 112)", c == 111.5 and c < V.MID_ROW)
check("centroid 112 is lower (>= 112)", V.centroid_and_band(m(112), "lower_only")[0] == 112.0)
check("band share: rows 74 in, 75 out of the upper band", V.centroid_and_band(m(74, 75), "upper_only")[1] == 0.5)
check("band share: rows 148 out, 149 in of the lower band", V.centroid_and_band(m(148, 149), "lower_only")[1] == 0.5)
check("empty mask -> None", V.centroid_and_band(m(), "upper_only") == (None, None))
R = V.records([("sub-S1_ses-E1_run-1_bp-chest_vp-ll_dx", True, "left:upper", m(10)), ("covid_1", True, "left:upper", m()),
               ("covid_2", False, "left:upper", m(10)), ("covid_3", True, "left:middle", m(10)), ("covid_4", True, "left:upper", m(10))])
check("exclusions in order, first reason only", list(R.excluded) == ["lateral_view", "mask_empty", "parse_not_ok", "not_in_a_group", ""])


def items(n_up, n_low, wrong_up=0, wrong_low=0):
    it = [(f"covid_u{i}", True, "left:upper", m(200) if i < wrong_up else m(20)) for i in range(n_up)]
    it += [(f"covid_l{i}", True, "right:lower", m(20) if i < wrong_low else m(200)) for i in range(n_low)]
    return it


s = V.evaluate(V.records(items(40, 40)))[1]
check("pass example", s["result"] == "pass" and s["rule_1"] and s["rule_2"] and s["rule_3"])
s = V.evaluate(V.records(items(40, 40, wrong_up=5)))[1]                      # upper 35/40 = 0.875; combined 75/80
check("rule 2 just fails (upper 0.875), rule 1 holds (0.9375)", s["rule_1"] and not s["rule_2"] and s["result"] == "fail")
s = V.evaluate(V.records(items(40, 40, wrong_up=5, wrong_low=4)))[1]         # combined 71/80 = 0.8875
check("rule 1 fails at combined 0.8875", not s["rule_1"] and s["result"] == "fail")
s = V.evaluate(V.records(items(29, 40)))[1]
check("rule 3: 29 upper-only reports -> insufficient sample", not s["rule_3"] and s["result"] == "insufficient sample")
rows = V.evaluate(V.records(items(40, 40)))[0]
check("descriptive quantiles and band share present", rows[0]["centroid_row_q50"] == 20.0 and rows[1]["median_band_share"] == 1.0)


def repo():
    d = tempfile.mkdtemp()
    run = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, check=True)  # noqa: E731
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    os.makedirs(os.path.join(d, "configs", "phase4")); open(os.path.join(d, V.RULE), "w").write("{}\n")
    run("add", "-A"); run("commit", "-qm", "rule")
    return d, run


def refused(d):
    try:
        O.preflight(repo=d, rule=V.RULE, out=V.OUT)
    except SystemExit as e:
        return e.code == O.PREFLIGHT_EXIT
    return False


d, run = repo()
check("clean repository with the committed rule passes", not refused(d))
open(os.path.join(d, V.RULE), "w").write('{"x":1}\n')
check("refused: rule changed after its commit", refused(d))
run("commit", "-qam", "change")
check("refused: rule committed twice", refused(d))
d, run = repo(); open(os.path.join(d, "dirty"), "w").write("x")
check("refused: dirty working tree", refused(d))
d, run = repo(); os.makedirs(os.path.join(d, "results")); open(os.path.join(d, V.OUT), "w").write("x")
run("add", "-A"); run("commit", "-qm", "o")
check("refused: output present", refused(d))
d, run = repo(); run("checkout", "-q", "--orphan", "x"); run("rm", "-rq", "--cached", "."); open(os.path.join(d, "y"), "w").write("y")
run("add", "y"); run("commit", "-qm", "u"); open(os.path.join(d, V.RULE), "w").write("{}\n")
check("refused: rule commit not an ancestor of HEAD", refused(d))
d2 = tempfile.mkdtemp(); subprocess.run(["git", "-C", d2, "init", "-q"], check=True)
os.makedirs(os.path.join(d2, "configs", "phase4")); open(os.path.join(d2, V.RULE), "w").write("{}\n")
check("refused: rule not committed", refused(d2))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
