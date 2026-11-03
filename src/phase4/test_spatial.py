"""Tests of src/phase4/spatial.py on synthetic masks (no real data is read).

  grid        integer boundaries: rows 74 | 75 and 148 | 149, columns 111 | 112
  orientation patient left on image right / on image left; max_of_both keeps the higher agreement per row
  union       several zones of one side are united; "all" = the three cells; two sides
  score       agreement = IMG pixels in allowed cells / all IMG pixels; mismatch = 1 - agreement
  unavailable lateral view, empty IMG, empty location, "unspecified", unknown token; never 0
Usage:  python src/phase4/test_spatial.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import spatial as S  # noqa: E402

FAILED = []
RAD, NAT, BOTH = S.ORIENTATIONS
PA = "sub-S1_ses-E1_run-1_bp-chest_vp-pa_cr"


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def px(*points):
    m = np.zeros((224, 224), bool)
    for r, c in points:
        m[r, c] = True
    return m


def agree(mask, loc, o=RAD, sid=PA):
    return S.spatial_score(sid, mask, loc, o)["agreement"]


# grid boundaries (patient left on image right: patient-left cells are columns 112-223)
check("row 74 is upper, row 75 is middle", agree(px((74, 200)), "left:upper") == 1 and agree(px((75, 200)), "left:upper") == 0
      and agree(px((75, 200)), "left:middle") == 1)
check("row 148 is middle, row 149 is lower", agree(px((148, 200)), "left:middle") == 1 and agree(px((149, 200)), "left:lower") == 1
      and agree(px((149, 200)), "left:middle") == 0)
check("row 0 upper, row 223 lower", agree(px((0, 200)), "left:upper") == 1 and agree(px((223, 200)), "left:lower") == 1)
check("column 111 is image-left, column 112 image-right", agree(px((10, 111)), "right:upper") == 1
      and agree(px((10, 112)), "left:upper") == 1 and agree(px((10, 111)), "left:upper") == 0)
sizes = [S.allowed_region({("left", z)}, RAD).sum() for z in S.ZONES]
check("cell sizes 75 / 74 / 75 rows x 112 columns", sizes == [75 * 112, 74 * 112, 75 * 112], str(sizes))
full = S.allowed_region({(s, z) for s in S.SIDES for z in S.ZONES}, RAD)
check("the six cells tile the image exactly", full.all() and sum(S.allowed_region({(s, z)}, RAD).sum()
                                                                for s in S.SIDES for z in S.ZONES) == 224 * 224)
# orientation
m = px((10, 10))                                    # image-left upper
check("patient left on image right: image-left pixel is patient right", agree(m, "right:upper", RAD) == 1 and agree(m, "left:upper", RAD) == 0)
check("patient left on image left: image-left pixel is patient left", agree(m, "left:upper", NAT) == 1 and agree(m, "right:upper", NAT) == 0)
two = px((10, 10), (10, 200), (100, 200), (200, 200))   # 1 image-left, 3 image-right
check("max_of_both keeps the higher agreement of the row", agree(two, "left:all", BOTH) == 0.75
      and agree(two, "left:all", RAD) == 0.75 and agree(two, "left:all", NAT) == 0.25)
# union, all, two sides
m = px((10, 200), (100, 200), (200, 200), (200, 10))
check("union of zones of one side", agree(m, "left:upper+middle") == 0.5)
check("'all' = the three cells of that side", agree(m, "left:all") == 0.75)
check("two sides united", agree(m, "left:upper;right:lower") == 0.5)
check("mismatch = 1 - agreement", S.spatial_score(PA, m, "left:upper+middle", RAD)["mismatch"] == 0.5)
check("left:all;right:all gives agreement 1", agree(m, "left:all;right:all") == 1)
# unavailable
r = lambda sid, mask, loc: S.spatial_score(sid, mask, loc, RAD)  # noqa: E731
check("lateral view unavailable (even with a valid location and IMG)",
      r("sub-S1_ses-E1_acq-1_run-1_bp-chest_vp-ll_dx", px((10, 10)), "left:all")["reason"] == S.LATERAL)
check("lateral checked before empty IMG", r("sub-S1_ses-E1_run-1_bp-chest_vp-ll_cr", px(), "left:all")["reason"] == S.LATERAL)
check("empty IMG unavailable", r(PA, px(), "left:all")["reason"] == S.IMG_EMPTY)
check("covid_* and sub-* without view token are not excluded",
      r("covid_123", px((10, 10)), "right:upper")["agreement"] == 1 and r("sub-S1_ses-E1_run-1_bp-chest_cr", px((10, 10)), "right:upper")["agreement"] == 1)
check("empty location unavailable", r(PA, px((10, 10)), "")["reason"] == S.LOC_EMPTY and r(PA, px((10, 10)), None)["reason"] == S.LOC_EMPTY
      and r(PA, px((10, 10)), float("nan"))["reason"] == S.LOC_EMPTY)
check("'unspecified' side unavailable", r(PA, px((10, 10)), "left:all;right:unspecified")["reason"] == S.LOC_UNSPECIFIED)
check("unknown token unavailable", r(PA, px((10, 10)), "left:top")["reason"] == S.LOC_UNKNOWN
      and r(PA, px((10, 10)), "centre:upper")["reason"] == S.LOC_UNKNOWN)
u = r(PA, px(), "left:all")
check("unavailable rows carry no number (not 0)", u["agreement"] is None and u["mismatch"] is None)
try:
    S.spatial_score(PA, px((1, 1)), "left:all", "image_left")
    check("unknown orientation refused", False)
except ValueError:
    check("unknown orientation refused", True)

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
