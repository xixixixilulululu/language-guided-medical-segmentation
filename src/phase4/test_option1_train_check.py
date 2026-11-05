"""Tests of src/phase4/option1_train_check.py with synthetic masks, records and git repositories (no real data).

  sides       primary (> 1/2, < 1/2, = 1/2 excluded) and robust (>= 2/3, <= 1/3, between excluded) definitions
  agreement   conventions A / B, Wilson lower bound (known value), patient majority vote with ties excluded
  rules       a passing example, then for each of rules 1-7 an example that just fails it
  self-checks the program refuses to run: rule not committed, committed twice, changed after commit, rule
              commit not an ancestor of HEAD, dirty working tree, output already present
Usage:  python src/phase4/test_option1_train_check.py      (exit status 1 if any check fails)
"""
import os
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import option1_train_check as O  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def mask(n_left, n_right):
    m = np.zeros((224, 224), np.uint8)
    m[0, :n_left] = 1                                     # columns 0..n_left-1 (image left, n_left <= 112)
    m[1, 112:112 + n_right] = 1
    return m


# lesion side
check("fraction: empty mask -> None", O.lesion_fraction(mask(0, 0)) is None)
check("fraction: column 111 left, 112 right", O.lesion_fraction(mask(1, 0)) == 0 and O.lesion_fraction(mask(0, 1)) == 1)
check("primary: 0.5 excluded, > / < decide", O.side_primary(0.5) is None and O.side_primary(0.51) == "image_right"
      and O.side_primary(0.49) == "image_left")
check("robust: 2/3 and 1/3 included, between excluded", O.side_robust(O.lesion_fraction(mask(1, 2))) == "image_right"
      and O.side_robust(O.lesion_fraction(mask(2, 1))) == "image_left" and O.side_robust(0.5) is None
      and O.side_robust(0.6) is None)
check("one-sided: left / right only, 'unspecified' allowed, both sides or none -> None",
      O.one_sided("left:upper+lower") == "left" and O.one_sided("right:unspecified") == "right"
      and O.one_sided("left:all;right:lower") is None and O.one_sided("") is None)
check("convention A: left report agrees with image right", O.agrees("left", "image_right", "A")
      and O.agrees("right", "image_left", "A") and not O.agrees("left", "image_left", "A"))
check("convention B: left report agrees with image left", O.agrees("left", "image_left", "B")
      and O.agrees("right", "image_right", "B"))
check("Wilson lower bound 9/10 = 0.5958", abs(O.wilson_lower(9, 10) - 0.59585) < 1e-4, f"{O.wilson_lower(9, 10):.5f}")
check("Wilson lower bound 100/100 = 0.9630", abs(O.wilson_lower(100, 100) - 0.96301) < 1e-4)


def build(n_sub=40, n_cov=40, wrong_sub=0, wrong_cov=0, right_only_wrong=0, borderline=0):
    """Records where convention A holds except for the given numbers of wrong reports."""
    items = []
    for grp, n, wrong in (("sub-S{p}_ses-E{i}_run-1_bp-chest_vp-pa_dx", n_sub, wrong_sub), ("covid_{i}", n_cov, wrong_cov)):
        for i in range(n):
            side = "left" if i % 2 == 0 else "right"
            ok_img = (mask(0, 9) if side == "left" else mask(9, 0))          # A: patient left on image right
            bad_img = (mask(9, 0) if side == "left" else mask(0, 9))
            wrong_here = i < wrong or (side == "right" and i // 2 < right_only_wrong)
            m = bad_img if wrong_here else ok_img
            if i < borderline:                                              # primary side kept, robust excluded
                m = mask(4, 5) if side == "left" else mask(5, 4)
            items.append((grp.format(p=i // 2, i=i), True, f"{side}:lower", O.lesion_fraction(m)))
    items += [("sub-S999_ses-E1_run-1_bp-chest_vp-ll_dx", True, "left:lower", 0.9),
              ("covid_x", True, "left:all;right:all", 0.5), ("covid_y", False, "left:lower", 0.9), ("covid_z", True, "left:lower", None)]
    return O.records(items)


pat = lambda s: s.split("_ses")[0] if s.startswith("sub-") else ""  # noqa: E731
rows, s = O.evaluate(build(), pat)
check("passing example: all 7 rules true, winner A, result pass", s["result"] == "pass" and s["winning_convention"] == "A"
      and all(s[f"rule_{k}"] for k in range(1, 8)), str({k: s[k] for k in s if k.startswith("rule")}))
g = {(r["definition"], r["group"]): r for r in rows}
check("exclusions counted once each", [g[("primary", "all")][f"n_excl_{k}"] for k in O.EXCL] == [1, 1, 1, 1])
check("convention B = 1 - A", abs(g[("primary", "all")]["agreement_A"] + g[("primary", "all")]["agreement_B"] - 1) < 1e-12)
# each rule just failing
s1 = O.evaluate(build(wrong_cov=21), pat)[1]
check("rule 1 fails when covid_* prefers B (no winner)", not s1["rule_1"] and s1["result"] == "fail")
s2 = O.evaluate(build(wrong_sub=5), pat)[1]                                 # 35/40 = 0.875
check("rule 2 fails at agreement 0.875 (< 0.90)", s2["rule_1"] and not s2["rule_2"] and s2["result"] == "fail")
s3 = O.evaluate(build(n_sub=40, wrong_sub=3), pat)[1]                       # 37/40 = 0.925, Wilson 0.801
check("rule 3 fails: agreement 0.925 but Wilson lower bound < 0.85", s3["rule_2"] and not s3["rule_3"])
s4 = O.evaluate(build(n_sub=200, n_cov=200, right_only_wrong=16), pat)[1]   # right-only 168/200 = 0.84; groups 0.92
check("rule 4 fails when right-only reports drop below 0.85 overall", s4["rule_3"] and not s4["rule_4"] and s4["rule_7"],
      str({k: s4[k] for k in s4 if k.startswith("rule")}))
s5 = O.evaluate(build(n_cov=29), pat)[1]
check("rule 5: 29 covid_* reports -> insufficient sample", not s5["rule_5"] and s5["result"] == "insufficient sample")
s6 = O.evaluate(build(n_sub=200, n_cov=200, borderline=172), pat)[1]       # robust excludes most -> rule 5 flips
check("rule 6 fails when the robust definition changes rule 5", s6["rule_5"] and not s6["rule_6"], s6["robust_rules_1_to_5"])
items7 = [(f"sub-S{i}_ses-E{j}_run-1_bp-chest_vp-pa_dx", True, "left:lower", O.lesion_fraction(mask(0, 9) if j < 2 else mask(9, 0)))
          for i in range(100) for j in range(3)]
R7 = O.records(items7[:])
# 100 patients x 3 reports: 2 agree, 1 disagree -> patient agrees; make 11 patients disagree by majority
R7 = O.records([(sid, ok, loc, (0.0 if int(sid.split("_ses")[0][5:]) < 11 and int(sid.split("ses-E")[1][0]) < 2 else f))
                for sid, ok, loc, f in items7] + [(f"covid_{i}", True, "left:lower", 1.0) for i in range(300)])
s7 = O.evaluate(R7, pat)[1]
check("rule 7 fails at patient agreement 89/100", s7["patients_voting"] == 100 and s7["patients_agreeing"] == 89
      and not s7["rule_7"], f"{s7['patients_agreeing']}/{s7['patients_voting']}")
R7t = O.records([("sub-S1_ses-E1_x", True, "left:a", 1.0), ("sub-S1_ses-E2_x", True, "left:a", 0.0)])
check("patient tie excluded and counted", O.patient_votes(R7t, "A", pat) == (0, 0, 1))


# self-checks on temporary git repositories
def repo_with_rule():
    d = tempfile.mkdtemp()
    run = lambda *a: subprocess.run(["git", "-C", d, *a], capture_output=True, check=True)  # noqa: E731
    run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
    os.makedirs(os.path.join(d, "configs", "phase4"))
    open(os.path.join(d, O.RULE), "w").write("{}\n")
    run("add", "-A"); run("commit", "-q", "-m", "rule")
    return d, run


def refused(d):
    try:
        O.preflight(repo=d)
    except SystemExit as e:
        return e.code == O.PREFLIGHT_EXIT
    return False


d, run = repo_with_rule()
check("clean repository with a committed rule passes", not refused(d))
d2 = tempfile.mkdtemp()
subprocess.run(["git", "-C", d2, "init", "-q"], check=True)
os.makedirs(os.path.join(d2, "configs", "phase4")); open(os.path.join(d2, O.RULE), "w").write("{}\n")
check("refused: rule not committed", refused(d2))
d, run = repo_with_rule()
open(os.path.join(d, O.RULE), "w").write('{"x": 1}\n')
check("refused: rule changed after its commit (uncommitted)", refused(d))
run("commit", "-qam", "change")
check("refused: rule committed twice", refused(d))
d, run = repo_with_rule()
run("checkout", "-q", "--orphan", "other"); run("rm", "-rq", "--cached", "."); open(os.path.join(d, "x"), "w").write("x")
run("add", "x"); run("commit", "-q", "-m", "unrelated")
open(os.path.join(d, O.RULE), "w").write("{}\n")                   # same content on disk, commit not an ancestor
check("refused: rule commit not an ancestor of HEAD", refused(d))
d, run = repo_with_rule()
open(os.path.join(d, "dirty.txt"), "w").write("x")
check("refused: dirty working tree", refused(d))
d, run = repo_with_rule()
os.makedirs(os.path.join(d, "results")); open(os.path.join(d, O.OUT), "w").write("x")
run("add", "-A"); run("commit", "-q", "-m", "out")
check("refused: output already present", refused(d))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
