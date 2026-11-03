"""Tests of src/phase4/stage0_coverage.py on a synthetic manifest (no real data is read).

  denominator   C0 + >= 1 location-only C4 donor + >= 1 C6 donor (a count-only C4 target is not in it)
  eligibility   lateral view, empty IMG, unmappable C0, no mappable C4 / C6 donor make a target ineligible
  groups        a group counts as eligible if at least one of its images is eligible
Usage:  python src/phase4/test_stage0_coverage.py      (exit status 1 if any check fails)
"""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import stage0_coverage as X  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


LAT = "sub-S9_ses-E1_run-1_bp-chest_vp-ll_dx"
rows = []
for t, c4_donor, c4_change in (("t1", "d1", "location"), ("t2", "d3", "location"), (LAT, "d1", "location"),
                               ("t3", "d1", "count"), ("t4", "d1", "location"), ("t5", "d1", "location")):
    rows += [(t, "C0", "", ""), (t, "C4", c4_donor, c4_change), (t, "C6", "d2" if t != "t5" else "d3", "scope+count+location")]
man = pd.DataFrame(rows, columns=["sample_id", "condition", "donor_sample_id", "changed_attributes"])
loc = {"t1": "left:all", "t2": "right:lower", "t4": "", "t5": "left:upper", LAT: "left:lower", "t3": "left:lower",
       "d1": "left:upper", "d2": "right:all", "d3": "left:all;right:unspecified"}
T = X.target_table(man, loc, {"t1": True, "t2": True, LAT: True, "t3": True, "t4": True, "t5": False})
E = T.set_index("sample_id")
check("count-only C4 target is not in the denominator", "t3" not in E.index and len(E) == 5)
check("t1 eligible", bool(E.eligible["t1"]))
check("no mappable C4 donor ('unspecified')", E.first_reason["t2"] == "no_mappable_c4_location_only_donor")
check("lateral view first", E.first_reason[LAT] == "lateral_view")
check("C0 location not parsed", E.first_reason["t4"] == "c0_location_not_parsed")
check("empty IMG before the donor checks", E.first_reason["t5"] == "img_empty" and bool(E.no_mappable_c6_donor["t5"]))
T["partition"] = ["GATE_CAL", "GATE_FIT", "GATE_FIT", "GATE_FIT", "GATE_FIT"]
T["group"] = ["g3", "g1", "g1", "g2", "g2"]
c = X.coverage(T).set_index(["table", "part", "item"])
check("images: 1 of 5 eligible", c.loc[("coverage", "all", "images")].numerator == 1)
check("groups: >= 1 eligible image (only g3, which holds t1)", c.loc[("coverage", "all", "groups (>= 1 eligible image)")][["numerator", "denominator"]].tolist() == [1, 3])
check("groups with all images eligible: 0", c.loc[("coverage_descriptive", "all", "groups (all images eligible)")].numerator == 0)
check("first reasons add up to the ineligible images",
      sum(c.loc[("unavailable_reason_first", "all", k)].numerator for k in X.REASONS) == 4)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
