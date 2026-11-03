"""Phase 4 test-split guard: every Phase 4 entry point refuses anything that belongs to the test split.

PHASE4_PLAN.md: "Do not use the existing test split to design, tune, choose, or evaluate the new method in this
phase." Phase 4 code calls these checks before reading data; a hit exits with TEST_EXIT_CODE (no data read).
  require_dev_split(split)   only "val" (and "train" for the definition of rules) are accepted
  require_dev_path(path)     paths of test data are refused: a path component "test" (runs/phase*/test, the common
                             224 test folder), "Test_Folder" (LViT layout), "imagesTs" / "labelsTs" (nnU-Net)
  require_dev_ids(ids)       ids that are test sample ids of QaTa-COV19 or MosMedData+ are refused; the test id
                             list is the file listing of the common 224 test folders (names only, no content)
Tests: src/phase4/test_guard.py.
"""
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
TEST_EXIT_CODE = 6
DEV_SPLITS = {"train", "val"}
TEST_COMPONENTS = {"test", "Test_Folder", "imagesTs", "labelsTs"}
DATASETS = ["QaTa-COV19", "MosMedData+"]


def refuse(reason):
    print(f"PHASE 4 TEST GUARD - refused: {reason}", file=sys.stderr)
    sys.exit(TEST_EXIT_CODE)


def require_dev_split(split):
    if split not in DEV_SPLITS:
        refuse(f"split {split!r} (Phase 4 uses development data only)")


def require_dev_path(path):
    parts = set(os.path.normpath(str(path)).split(os.sep))
    hit = parts & TEST_COMPONENTS
    if hit:
        refuse(f"test-split path {path} (component {sorted(hit)[0]!r})")


_TEST_IDS = None


def test_ids(loader=None):
    """Set of all test sample ids (names only). `loader(dataset)` is injectable for the unit tests."""
    global _TEST_IDS
    if loader is not None:
        return set().union(*[set(loader(d)) for d in DATASETS])
    if _TEST_IDS is None:
        from lgms import data
        _TEST_IDS = set().union(*[set(data.split_ids(d, "test")) for d in DATASETS])
    return _TEST_IDS


def require_dev_ids(ids, known_test_ids=None):
    bad = set(ids) & (known_test_ids if known_test_ids is not None else test_ids())
    if bad:
        refuse(f"{len(bad)} test sample id(s), e.g. {sorted(bad)[0]}")
