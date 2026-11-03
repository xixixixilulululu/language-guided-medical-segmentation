"""Tests of src/phase4/guard.py (synthetic ids and paths; no data is read).

  split   only "train" / "val" pass; "test" exits with TEST_EXIT_CODE
  path    runs/phase*/test, Test_Folder, imagesTs, labelsTs are refused; validation paths and names that only
          contain "test" as a substring pass
  ids     an id from the test list is refused; validation ids pass
Usage:  python src/phase4/test_guard.py      (exit status 1 if any check fails)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase4 import guard as G  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def refused(fn, *a, **k):
    try:
        fn(*a, **k)
    except SystemExit as e:
        return e.code == G.TEST_EXIT_CODE
    return False


check("split 'test' refused", refused(G.require_dev_split, "test"))
check("splits 'val' and 'train' pass", not refused(G.require_dev_split, "val") and not refused(G.require_dev_split, "train"))
for p in ("runs/phase2/test/ddcmd/QaTa-COV19/seed1001/probs_C0_0.npz", "runs/phase3/test/img/nnunet",
          "/home/u/datasets/QaTa-COV19/Test_Folder/img/x.png", "/n/nnUNet_raw/Dataset501_QaTaCOV19/imagesTs/a_0000.png",
          "/n/nnUNet_raw/Dataset501_QaTaCOV19/labelsTs/a.png", "/d/common224/QaTa-COV19/test/images/a.png"):
    check(f"test path refused: {p}", refused(G.require_dev_path, p))
for p in ("runs/phase3/val/img/nnunet/QaTa-COV19/seed1001/masks.npz", "src/phase4/test_guard.py",
          "results/phase3_pairwise_results_val.csv", "/n/nnUNet_raw/Dataset501_QaTaCOV19/imagesTr/a_0000.png"):
    check(f"development path passes: {p}", not refused(G.require_dev_path, p))
known = G.test_ids(loader=lambda d: [f"{d}_test_a", f"{d}_test_b"])
check("injected test id list covers both datasets", known == {"QaTa-COV19_test_a", "QaTa-COV19_test_b",
                                                              "MosMedData+_test_a", "MosMedData+_test_b"})
check("a test id is refused", refused(G.require_dev_ids, ["val_1", "QaTa-COV19_test_b"], known))
check("validation ids pass", not refused(G.require_dev_ids, ["val_1", "val_2"], known))

print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
sys.exit(1 if FAILED else 0)
