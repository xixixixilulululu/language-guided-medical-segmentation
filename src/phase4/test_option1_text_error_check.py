"""Tests of src/phase4/option1_text_error_check.py with synthetic data (no external or real data is read).

  classes     R1 (same text up to case / whitespace), R2 (same side), R3 (opposite side), R4 (two-sided, no side,
              not parsed)
  matching    case-insensitive ids, 'mask_' / '.png' stripped, foreign items dropped, missing or duplicated ids stop
  test key    deleted after loading
  Fisher      two-sided p of a known table
  rules       interpretation boundaries (50% with p < 0.01, 25%) and the 5% flag of the agreeing group
  self-checks rule / clean tree / output via option1_train_check.preflight; external sha256 mismatch stops
Usage:  python src/phase4/test_option1_text_error_check.py      (exit status 1 if any check fails)
"""
import json
import os
import subprocess
import sys
import tempfile

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase2 import texts as T  # noqa: E402
from phase4 import option1_text_error_check as E  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def stops(fn, *a):
    try:
        fn(*a)
    except SystemExit as e:
        return e.code == E.STOP_EXIT
    return False


L = "Unilateral pulmonary infection, one infected area, lower left lung."
Rt = "Unilateral pulmonary infection, one infected area, lower right lung."
check("R1: same text up to case and whitespace", E.classify(L, "  unilateral  pulmonary infection, one infected area, LOWER left lung. ", "left", T.parse) == "R1")
check("R2: changed wording, same side", E.classify(L, "Unilateral pulmonary infection, one infected area, upper left lung.", "left", T.parse) == "R2")
check("R3: opposite side", E.classify(L, Rt, "left", T.parse) == "R3")
check("R4: now two-sided", E.classify(L, "Bilateral pulmonary infection, two infected areas, lower left lung and lower right lung.", "left", T.parse) == "R4")
check("R4: not parsed", E.classify(L, "something else entirely", "left", T.parse) == "R4")

d = {"train": [{"image_path": "mask_covid_1.png", "caption": "a"}, {"image_path": "mask_sub-s1_ses-e1_x.png", "caption": "b"},
               {"image_path": "mask_other_9.png", "caption": "z"}],
     "valid": [{"image_path": "mask_covid_2.png", "caption": "c"}], "test": [{"image_path": "mask_covid_3.png", "caption": "t"}]}
f = os.path.join(tempfile.mkdtemp(), "a.json")
json.dump(d, open(f, "w"))
loaded = E.load_external(f)
check("test key deleted after loading", "test" not in loaded and set(loaded) == {"train", "valid"})
m = E.match_training(loaded, ["covid_1", "sub-S1_ses-E1_x", "covid_2"])
check("case-insensitive match, prefix / suffix stripped, foreign item dropped",
      m == {"covid_1": "a", "sub-S1_ses-E1_x": "b", "covid_2": "c"})
check("missing training id stops", stops(E.match_training, loaded, ["covid_1", "covid_2", "sub-S1_ses-E1_x", "covid_3"]))
dup = {"train": d["train"], "valid": d["valid"] + [{"image_path": "mask_COVID_1.png", "caption": "x"}]}
check("duplicated training id stops", stops(E.match_training, dup, ["covid_1", "sub-S1_ses-E1_x", "covid_2"]))

from scipy.stats import fisher_exact  # noqa: E402
_, p_ref = fisher_exact([[22, 22], [0, 1085]])
check("Fisher two-sided p equals scipy on the same table", abs(E.analyse(pd.DataFrame(
    [{"sample_id": f"covid_{i}", "group": "covid", "report_side": "left", "image_side": "image_right" if i < 44 else "image_left",
      "agree_B": i >= 44, "original": L, "revised": (Rt if i < 22 else L)} for i in range(44 + 1085)]), T.parse)[1]["fisher_p_two_sided"] - p_ref) < 1e-15)
check("interpretation: 50% with p < 0.01 -> most", E.interpret(22, 44, 0.009).endswith("explain most of the disagreement"))
check("interpretation: 50% but p = 0.01 -> part", E.interpret(22, 44, 0.01).endswith("explain part of the disagreement"))
check("interpretation: 11/44 = 25% -> part", E.interpret(11, 44, 0.5).endswith("part of the disagreement"))
check("interpretation: 10/44 < 25% -> little", E.interpret(10, 44, 0.5).endswith("little of the disagreement"))


def frame(n_agree_r3):
    rows = [{"sample_id": f"covid_{i}", "group": "covid", "report_side": "left", "image_side": "image_left", "agree_B": True,
             "original": L, "revised": Rt if i < n_agree_r3 else L} for i in range(100)]
    rows += [{"sample_id": f"sub-S{i}_ses-E1_x", "group": "sub", "report_side": "left", "image_side": "image_right",
              "agree_B": False, "original": L, "revised": L} for i in range(4)]
    return pd.DataFrame(rows)


s5 = E.analyse(frame(5), T.parse)[1]
s4 = E.analyse(frame(4), T.parse)[1]
check("agreeing group flag at 5/100 (>= 5%)", s5["agree_group_flag"] != "" and s5["r3_agree"] == 5)
check("no flag at 4/100", s4["agree_group_flag"] == "")
rows, s, _ = E.analyse(frame(5), T.parse)
c = {(r["subset"], r["option1_group"]): r for r in rows}
check("counts split by subset and group", c[("covid", "agree_B")]["R3"] == 5 and c[("covid", "agree_B")]["R1"] == 95
      and c[("sub", "disagree_B")]["R1"] == 4 and c[("all", "agree_B")]["n"] == 100)
check("revised agreement uses the revised side", abs(s["revised_agreement_B"] - (95 + 0) / 104) < 1e-12)

# self-checks
d0 = tempfile.mkdtemp()
run = lambda *a: subprocess.run(["git", "-C", d0, *a], capture_output=True, check=True)  # noqa: E731
run("init", "-q"); run("config", "user.email", "t@t"); run("config", "user.name", "t")
os.makedirs(os.path.join(d0, "configs", "phase4")); open(os.path.join(d0, E.RULE), "w").write("{}\n")
run("add", "-A"); run("commit", "-qm", "rule")
ext = os.path.join(tempfile.mkdtemp(), "ext.json"); open(ext, "w").write("{}")
import hashlib  # noqa: E402
good = hashlib.sha256(b"{}").hexdigest()
check("all self-checks met -> runs", not stops(E.preflight, d0, ext, good))
check("external sha256 mismatch stops", stops(E.preflight, d0, ext, "0" * 64))
check("external file missing stops", stops(E.preflight, d0, ext + ".missing", good))
open(os.path.join(d0, "dirty"), "w").write("x")
check("dirty working tree stops", stops(E.preflight, d0, ext, good))
os.remove(os.path.join(d0, "dirty")); open(os.path.join(d0, E.RULE), "w").write('{"x":1}\n')
check("rule changed after its commit stops", stops(E.preflight, d0, ext, good))
run("checkout", "-q", "--", E.RULE); os.makedirs(os.path.join(d0, "results")); open(os.path.join(d0, E.OUT), "w").write("x")
run("add", "-A"); run("commit", "-qm", "out")
check("output already present stops", stops(E.preflight, d0, ext, good))

# rule v2: capitalisation
check("capitalise changes only the first character (after stripping)",
      E.capitalise("  bilateral pulmonary infection, lower left lung.  ") == "Bilateral pulmonary infection, lower left lung."
      and E.capitalise("unilateral X Y") == "Unilateral X Y" and E.capitalise("") == "")
P2 = E.parse_v2(T.parse)
pb = P2("bilateral pulmonary infection, two infected areas, lower left lung and upper right lung.")
pu = P2("unilateral pulmonary infection, one infected area, middle right lung.")
pn = P2("no pulmonary infection, zero infected areas.")
check("lower-case 'bilateral ...' parses after capitalisation", pb["parse_ok"] and pb["scope"] == "Bilateral" and pb["location"] == "left:lower;right:upper")
check("lower-case 'unilateral ...' parses after capitalisation", pu["parse_ok"] and pu["scope"] == "Unilateral" and pu["location"] == "right:middle")
check("lower-case 'no ...' parses after capitalisation", pn["parse_ok"] and pn["scope"] == "No" and pn["count"] == 0)
check("without capitalisation the lower-case text is not parsed (the v1 failure)", not T.parse("bilateral pulmonary infection, two infected areas, lower left lung and upper right lung.")["parse_ok"])
import hashlib as _h  # noqa: E402
repo_v1 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), E.RULE_V1)
check("v1 rule file unchanged (sha256 of its commit b811353)", _h.sha256(open(repo_v1, "rb").read()).hexdigest() == E.RULE_V1_SHA)
check("script reads rule v2 and records rule_version v2", E.RULE.endswith("option1_text_error_rule_v2.json") and E.RULE_VERSION == "v2")
d1 = tempfile.mkdtemp()
run1 = lambda *a: subprocess.run(["git", "-C", d1, *a], capture_output=True, check=True)  # noqa: E731
run1("init", "-q"); run1("config", "user.email", "t@t"); run1("config", "user.name", "t")
os.makedirs(os.path.join(d1, "configs", "phase4")); open(os.path.join(d1, E.RULE), "w").write("{}\n")
run1("add", "-A"); run1("commit", "-qm", "rule v2")
check("fresh repository with rule v2 and no v1 file runs", not stops(E.preflight, d1, ext, good))
open(os.path.join(d1, E.RULE_V1), "w").write("changed v1\n"); run1("add", "-A"); run1("commit", "-qm", "v1 changed")
check("a changed v1 rule file stops (clean tree, everything else valid)", stops(E.preflight, d1, ext, good))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
