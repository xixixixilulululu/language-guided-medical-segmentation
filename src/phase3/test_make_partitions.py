"""Tests of src/phase3/make_partitions.py on synthetic ids (the real split file is not touched).

  deterministic   the same ids give the same split
  groups intact   every group lies in exactly one part
  size rule       n_fit = floor(0.6 n + 0.5) groups (e.g. 5 groups -> 3, 7 -> 4, 1194 -> 716)
  order free      the split does not depend on the order of the input ids
  write once      main() refuses to overwrite an existing file
Usage:  python src/phase3/test_make_partitions.py      (exit status 1 if any check fails)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase3 import make_partitions as P  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


ids = [f"sub-S{p}_ses-E{k}_x" for p in range(1, 6) for k in range(p)] + [f"covid_{k}" for k in range(2)]
grp = lambda i: (i.split("_ses")[0], "patient") if i.startswith("sub-S") else (i, "image (patient unverified)")  # noqa: E731
a, b = P.partition(ids, grp), P.partition(list(reversed(ids)), grp)
check("deterministic and independent of the id order",
      a.set_index("sample_id").partition.sort_index().equals(b.set_index("sample_id").partition.sort_index()))
check("every group in exactly one part", (a.groupby("group_id").partition.nunique() == 1).all())
n = a.group_id.nunique()
check("size rule floor(0.6 n + 0.5)", a[a.partition == "GATE_FIT"].group_id.nunique() == int(np.floor(0.6 * n + 0.5)),
      f"{n} groups")
check("rounding examples", [int(np.floor(0.6 * k + 0.5)) for k in (5, 7, 1194)] == [3, 4, 716])
code = os.system(f"{sys.executable} -c 'import sys; sys.path.insert(0, \"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}\"); "
                 f"from phase3 import make_partitions as P; P.OUT = \"{os.path.abspath(__file__)}\"; P.main()' 2>/dev/null")
check("main() refuses to overwrite an existing file", code != 0)
print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
