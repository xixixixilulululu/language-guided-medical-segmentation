"""Tests of the Phase 2 lock (src/phase2/lock.py) and of its gates. No GPU, no test data, no network.

A scratch git repository with a local bare repository as its "origin" stands in for GitHub; nothing touches
the real repository. Lock states:
  case 1  no frozen record                                   -> closed
  case 2  record written, a recorded file changed later      -> closed
  case 3  record for a commit that is not pushed             -> closed
  case 4  record without the Phase 3 decision record         -> closed
  case 5  record for a pushed commit, everything unchanged   -> open
Gates (with the scratch repository's lock CLOSED):
  - texts.read_texts(..., "test") exits with code 5, and pandas.read_excel is never called (patched to fail);
  - conditions.build(..., "test") exits with code 5 the same way;
  - the command line `conditions.py --split test` of the real repository exits with code 5 (no record yet);
  - lock.link_test refuses (code 5) and links nothing into a scratch text view;
  - with the lock OPEN, read_texts(..., "test") gets past the gate and reads the (fake) test file, and
    log_test_run writes start / end lines;
  - validation texts are read without any lock (real text view).
Usage:  python src/phase2/test_phase2_lock.py      (exit status 1 if any check fails)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pandas as pd

SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, SRC)
from phase2 import conditions as C  # noqa: E402
from phase2 import lock as L  # noqa: E402
from phase2 import texts as T  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def sh(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)


def scratch_repo(tmp):
    """Repository with the locked files (dummy content) and a bare 'origin'; returns its path."""
    origin, repo = os.path.join(tmp, "origin.git"), os.path.join(tmp, "repo")
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", origin], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    sh(repo, "config", "user.email", "t@t"); sh(repo, "config", "user.name", "t")
    sh(repo, "remote", "add", "origin", origin)
    for path in L.LOCKED_FILES:
        os.makedirs(os.path.dirname(os.path.join(repo, path)), exist_ok=True)
        with open(os.path.join(repo, path), "w") as f:
            f.write(f"dummy {path}\n")
    sh(repo, "add", "-A"); sh(repo, "commit", "-q", "-m", "files")
    sh(repo, "push", "-q", "origin", "main")
    return repo


def commit_record(repo, push=True):
    rec = L.freeze(repo)
    sh(repo, "add", L.RECORD); sh(repo, "commit", "-q", "-m", "record")
    if push:
        sh(repo, "push", "-q", "origin", "main")
    return rec


def exit_code(fn):
    try:
        fn()
    except SystemExit as e:
        return e.code
    return 0


def fail_read_excel(*a, **k):
    raise AssertionError("a text file was opened")


def main():
    tmp = tempfile.mkdtemp(prefix="p2lock_")
    real_read_excel = pd.read_excel
    try:
        repo = scratch_repo(tmp)
        ok, reasons, _ = L.check(repo)
        check("case 1: no record -> closed", not ok, reasons[0] if reasons else "")

        commit_record(repo)
        ok, reasons, _ = L.check(repo)
        check("case 5: pushed record, unchanged -> open", ok, "; ".join(reasons))

        with open(os.path.join(repo, "src/phase2/texts.py"), "a") as f:
            f.write("# changed\n")
        ok, reasons, _ = L.check(repo)
        check("case 2: recorded file changed -> closed", not ok and any("changed" in r for r in reasons))
        sh(repo, "checkout", "--", "src/phase2/texts.py")

        with open(os.path.join(repo, "src/phase2/texts.py"), "a") as f:   # new version, record not pushed
            f.write("# v2\n")
        sh(repo, "commit", "-qam", "v2")
        commit_record(repo, push=False)
        ok, reasons, _ = L.check(repo)
        check("case 3: record commit not pushed -> closed", not ok and any("not on origin" in r for r in reasons))
        sh(repo, "push", "-q", "origin", "main")
        ok, _, _ = L.check(repo)
        check("case 3b: same record after the push -> open", ok)

        rec = json.load(open(os.path.join(repo, L.RECORD)))
        del rec["files"][L.DECISION]                                          # record without the decision
        json.dump(rec, open(os.path.join(repo, L.RECORD), "w"))
        ok, reasons, _ = L.check(repo)
        check("case 4: decision record not covered -> closed", not ok and any("not covered" in r for r in reasons))
        sh(repo, "checkout", "--", L.RECORD)
        os.remove(os.path.join(repo, L.DECISION))
        ok, reasons, _ = L.check(repo)
        check("case 4b: decision record missing -> closed", not ok and any("Phase 3" in r for r in reasons))
        sh(repo, "checkout", "--", L.DECISION)

        # --- gates with a CLOSED lock (scratch repository without record) ---
        closed = scratch_repo(os.path.join(tmp, "closed"))
        view = os.path.join(tmp, "view")
        for ds in L.DATASETS:                                                  # fake view WITH a test file
            p = L.text_file(ds, "test", view)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            pd.DataFrame({"Image": ["x.png"], "Description": ["dummy"]}).to_excel(p, index=False)
        pd.read_excel = fail_read_excel
        code = exit_code(lambda: T.read_texts("QaTa-COV19", "test", view=view, repo=closed))
        check("gate: read_texts(test) with the lock closed -> exit 5, no file opened", code == L.CLOSED_EXIT_CODE,
              f"exit {code}")
        orig = L.REPO
        L.REPO = closed                                                         # build() uses the default repo
        code = exit_code(lambda: C.build("MosMedData+", "test", view=view))
        L.REPO = orig
        check("gate: conditions.build(test) with the lock closed -> exit 5, no file opened",
              code == L.CLOSED_EXIT_CODE, f"exit {code}")
        pd.read_excel = real_read_excel

        r = subprocess.run([sys.executable, os.path.join(SRC, "phase2", "conditions.py"), "--split", "test",
                            "--out", os.path.join(tmp, "x.csv")], capture_output=True, text=True)
        check("gate: conditions.py --split test in the real repository -> exit 5", r.returncode == 5
              and "LOCK CLOSED" in r.stdout and not os.path.exists(os.path.join(tmp, "x.csv")), f"exit {r.returncode}")

        view2 = os.path.join(tmp, "view2")
        code = exit_code(lambda: L.link_test(closed, view=view2, source=view))
        check("gate: link-test with the lock closed -> exit 5, nothing linked", code == L.CLOSED_EXIT_CODE
              and not os.path.exists(L.text_file("QaTa-COV19", "test", view2)), f"exit {code}")

        # --- with the lock OPEN the gate lets the test read through (fake test file) and logs ---
        opened = []
        pd.read_excel = lambda p, *a, **k: (opened.append(p), real_read_excel(p, *a, **k))[1]
        try:
            T.read_texts("QaTa-COV19", "test", view=view, repo=repo)
        except KeyError:
            pass                                                                # dummy file has no real ids
        pd.read_excel = real_read_excel
        check("gate: read_texts(test) with the lock open -> the test file is read", len(opened) == 1)
        info = L.require_open(repo)
        L.log_test_run("start", info, "test", "QaTa-COV19", "lvit", 1001, "C4", repo=repo)
        L.log_test_run("end", info, "test", "QaTa-COV19", "lvit", 1001, "C4", repo=repo)
        lines = open(os.path.join(repo, L.TEST_LOG)).read().splitlines()
        check("log_test_run writes header + start + end with the lock commit",
              len(lines) == 3 and info["commit"] in lines[1])

        # --- validation needs no lock (real text view) ---
        v = T.read_texts("MosMedData+", "val")
        check("validation texts are read without the lock", len(v) == 273)
    finally:
        pd.read_excel = real_read_excel
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL CHECKS PASSED" if not FAILED else f"{len(FAILED)} FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
