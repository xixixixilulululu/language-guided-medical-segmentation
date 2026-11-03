"""Tests of src/phase4/stage0_mirror.py on hand-written cell sets (no data is read).

  a  D ⊇ T (equal sets, strict superset, all six cells)
  b  only mirror(D) ⊇ T (one-sided swap, bilateral asymmetric swap)
  c  neither (different zone, partial cover)
  order   a is checked before b (symmetric sets are a, never b)
Usage:  python src/phase4/test_stage0_mirror.py      (exit status 1 if any check fails)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import stage0_mirror as M  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def c(*cells):
    return {(s, z) for s, z in cells}


L, R = "left", "right"
ALL6 = c(*[(s, z) for s in (L, R) for z in ("upper", "middle", "lower")])
check("equal sets -> a", M.classify(c((L, "lower")), c((L, "lower"))) == "a")
check("donor superset -> a", M.classify(c((L, "lower")), c((L, "lower"), (L, "upper"), (R, "middle"))) == "a")
check("donor all six cells -> a for any target", M.classify(c((R, "upper"), (L, "middle")), ALL6) == "a")
check("one-sided swap -> b", M.classify(c((L, "lower")), c((R, "lower"))) == "b")
check("bilateral asymmetric swap -> b", M.classify(c((L, "upper"), (R, "lower")), c((R, "upper"), (L, "lower"))) == "b")
check("symmetric target and donor -> a, not b", M.classify(c((L, "lower"), (R, "lower")), c((L, "lower"), (R, "lower"))) == "a")
check("different zone -> c", M.classify(c((L, "lower")), c((L, "upper"))) == "c")
check("partial cover -> c", M.classify(c((L, "lower"), (L, "upper")), c((L, "lower"))) == "c")
check("mirror of a swap partial -> c", M.classify(c((L, "lower"), (R, "upper")), c((R, "lower"))) == "c")
check("mirror swaps sides only", M.mirror(c((L, "upper"), (R, "lower"))) == frozenset(c((R, "upper"), (L, "lower"))))
spec = {"lookup": {"table_patient_side_zones": {"left:all": {"left": ["upper", "middle", "lower"]},
                                                "x": {"unavailable": "location_unspecified"}}}}
check("spec lookup: unavailable entries dropped, 'all' = three cells",
      M.cells_from_spec(spec) == {"left:all": frozenset(c((L, "upper"), (L, "middle"), (L, "lower")))})

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
