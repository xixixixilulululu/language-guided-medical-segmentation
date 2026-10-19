"""Tests of the program lock (src/phase1/analysis_lock.py): the four cases, in a temporary git repository.

A scratch repository with a local bare repository as its "origin" stands in for GitHub; nothing touches
the real repository, the data or the network.
  case 1  no frozen record                      -> lock closed
  case 2  record written, file changed later    -> lock closed
  case 3  record for a commit not pushed        -> lock closed
          (also: pushed only to another branch, e.g. a review branch -> lock closed)
  case 4  record for a pushed commit, unchanged -> lock open
Also: analyze_phase1.py --split test stops with exit code 5 before reading any data when the lock is
closed (run against the scratch repository, whose runs folder does not exist).

Usage:  python src/phase1/test_analysis_lock.py      (exit status 1 if any check fails)
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from phase1 import analysis_lock as L  # noqa: E402

FAILED = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))
    if not ok:
        FAILED.append(name)


def sh(repo, *args):
    subprocess.run(["git", "-C", repo, *args], check=True, capture_output=True)


def make_repo(tmp):
    """Scratch repo with the locked files (dummy content) and a bare 'origin'."""
    origin, repo = os.path.join(tmp, "origin.git"), os.path.join(tmp, "repo")
    subprocess.run(["git", "init", "--bare", "-q", "-b", "main", origin], check=True)
    subprocess.run(["git", "init", "-q", "-b", "main", repo], check=True)
    sh(repo, "config", "user.email", "test@example.com"); sh(repo, "config", "user.name", "test")
    for p in L.LOCKED_FILES:
        os.makedirs(os.path.dirname(os.path.join(repo, p)), exist_ok=True)
        with open(os.path.join(repo, p), "w") as f:
            f.write(f"# dummy {p}\n")
    sh(repo, "add", "."); sh(repo, "commit", "-q", "-m", "analysis")
    sh(repo, "remote", "add", "origin", origin); sh(repo, "push", "-q", "origin", "main")
    return repo


def main():
    tmp = tempfile.mkdtemp(prefix="lock_test_")
    try:
        repo = make_repo(tmp)
        ok, reasons, _ = L.check(repo)
        check("case 1: no record -> closed", not ok and "no frozen record" in reasons[0], reasons[0])

        L.freeze(repo)
        sh(repo, "add", L.RECORD); sh(repo, "commit", "-q", "-m", "freeze"); sh(repo, "push", "-q", "origin", "main")
        ok, reasons, info = L.check(repo)
        check("case 4: frozen, pushed, unchanged -> open", ok and not reasons, str(reasons))

        path = os.path.join(repo, L.LOCKED_FILES[0])
        original = open(path).read()
        with open(path, "a") as f:
            f.write("# edited after fixing\n")
        ok, reasons, _ = L.check(repo)
        check("case 2: file changed after fixing -> closed", not ok and any("changed since" in r for r in reasons),
              str(reasons))
        with open(path, "w") as f:                       # also committed but not pushed: still closed
            f.write(original + "# edited and committed\n")
        sh(repo, "commit", "-q", "-am", "edit")
        ok, reasons, _ = L.check(repo)
        check("case 2b: changed file committed (record unchanged) -> closed", not ok, str(reasons))

        # case 3: new analysis version frozen locally but never pushed
        L.freeze(repo)
        sh(repo, "add", L.RECORD); sh(repo, "commit", "-q", "-m", "freeze v2 (not pushed)")
        ok, reasons, _ = L.check(repo)
        check("case 3: frozen commit not pushed -> closed", not ok and any("not on origin/main" in r for r in reasons),
              str(reasons))
        # case 3c: the same commit pushed only to another branch (e.g. a review branch): still closed,
        # because the lock accepts only origin/main
        sh(repo, "push", "-q", "origin", "HEAD:refs/heads/phase1-analysis-draft")
        ok, reasons, _ = L.check(repo)
        check("case 3c: pushed only to a branch other than main -> closed",
              not ok and any("not on origin/main" in r for r in reasons), str(reasons))
        sh(repo, "push", "-q", "origin", "main")
        ok, reasons, _ = L.check(repo)
        check("case 3 -> 4 after pushing: open", ok, str(reasons))

        # analyze_phase1.py --split test must stop before reading anything when the lock is closed
        os.remove(os.path.join(repo, L.RECORD))
        r = subprocess.run([sys.executable, os.path.join(L.REPO, "src", "analyze_phase1.py"), "--split", "test",
                            "--lock-repo", repo, "--runs-root", os.path.join(tmp, "no_runs")],
                           capture_output=True, text=True)
        check("analyze_phase1 --split test with a closed lock: exit 5 before reading data",
              r.returncode == L.CLOSED_EXIT_CODE and "PROGRAM LOCK CLOSED" in r.stdout, f"exit {r.returncode}")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
