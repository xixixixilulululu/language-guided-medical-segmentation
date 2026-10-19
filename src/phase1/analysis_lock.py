"""Program lock for the Phase 1 analysis (the "first lock"; the data lock is the train+val-only data view).

No Phase 1 test-set result may exist before the analysis code is fixed. The lock is open only if ALL hold:
  1. the frozen record configs/phase1_frozen/analysis_lock.json exists;
  2. every recorded file (the analysis program and the code / text it uses) has the recorded sha256 now;
  3. the recorded commit is on GitHub: after `git fetch`, it is an ancestor of origin/main;
  4. in that commit, every recorded file has the recorded sha256 (what is on GitHub is what runs).
The record itself is written by `freeze` (only after the user has confirmed the analysis) and then
committed and pushed in a separate commit.

Commands:
  python src/phase1/analysis_lock.py check       report the lock state (exit 0 open, 5 closed)
  python src/phase1/analysis_lock.py freeze      write the record for HEAD (files must be committed, unchanged)
  python src/phase1/analysis_lock.py link-test   only if `check` passes: link the test split into the Phase 1
                                                 data view and append a line to runs/phase1_lock_log.jsonl
analyze_phase1.py calls require_open() before it reads anything when --split test.
Tests: src/phase1/test_analysis_lock.py (the four cases: no record, file changed, commit not pushed, open).
"""
import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RECORD = os.path.join("configs", "phase1_frozen", "analysis_lock.json")
# Files whose content defines the Phase 1 analysis (paths relative to the repository)
LOCKED_FILES = ["src/analyze_phase1.py", "src/analyze_phase0.py", "src/phase1/analysis_lock.py",
                "reports/templates/PHASE1_RESULTS_methods.md"]
LOCK_LOG = os.path.join("runs", "phase1_lock_log.jsonl")
VIEW = os.path.expanduser("~/datasets/phase1/common224_trainval")
COMMON = os.path.expanduser("~/datasets/common224")
DATASETS = ["QaTa-COV19", "MosMedData+"]
CLOSED_EXIT_CODE = 5


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path):
    with open(path, "rb") as f:
        return sha256_bytes(f.read())


def git(repo, *args, check=True):
    return subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True, check=check)


def check(repo=REPO, remote="origin", branch="main", fetch=True):
    """(is_open, reasons, info). `reasons` lists every failed condition (empty when open)."""
    reasons, info = [], {}
    rec_path = os.path.join(repo, RECORD)
    if not os.path.exists(rec_path):                                        # 1. record
        return False, [f"no frozen record ({RECORD}): the analysis has not been fixed"], info
    rec = json.load(open(rec_path))
    commit, files = rec["commit"], rec["files"]
    info.update(commit=commit, files=files)
    for path, digest in files.items():                                      # 2. current content
        full = os.path.join(repo, path)
        if not os.path.exists(full):
            reasons.append(f"{path}: missing")
        elif sha256_file(full) != digest:
            reasons.append(f"{path}: changed since it was fixed (sha256 differs)")
    if fetch:                                                               # 3. on GitHub
        r = git(repo, "fetch", "--quiet", remote, branch, check=False)
        if r.returncode != 0:
            reasons.append(f"git fetch {remote} {branch} failed: {r.stderr.strip()[:200]}")
    remote_ref = f"{remote}/{branch}"
    head = git(repo, "rev-parse", "--verify", "--quiet", remote_ref, check=False).stdout.strip()
    info["remote_head"] = head
    if git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
        reasons.append(f"recorded commit {commit} does not exist in this repository")
    elif not head or git(repo, "merge-base", "--is-ancestor", commit, remote_ref, check=False).returncode != 0:
        reasons.append(f"recorded commit {commit} is not on {remote_ref} (not pushed to GitHub)")
    else:                                                                   # 4. content of that commit
        for path, digest in files.items():
            r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, check=False)
            if r.returncode != 0:
                reasons.append(f"{path}: not in commit {commit}")
            elif sha256_bytes(r.stdout) != digest:
                reasons.append(f"{path}: content in commit {commit} differs from the record")
    return not reasons, reasons, info


def require_open(repo=REPO):
    """Exit with CLOSED_EXIT_CODE (before anything else is read) unless the lock is open."""
    ok, reasons, info = check(repo)
    if not ok:
        print("PROGRAM LOCK CLOSED - the Phase 1 analysis is not fixed on GitHub:", *reasons, sep="\n  ")
        sys.exit(CLOSED_EXIT_CODE)
    print(f"program lock open: analysis fixed in commit {info['commit']} (on {info['remote_head'][:12]})")
    return info


def freeze(repo=REPO, files=None):
    """Write the record for HEAD: every locked file must be committed and unchanged in the working tree."""
    files = files or LOCKED_FILES
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    rec = {"commit": commit, "frozen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "files": {}}
    for path in files:
        if git(repo, "status", "--porcelain", "--", path).stdout.strip():
            raise SystemExit(f"{path} has uncommitted changes: commit it first")
        rec["files"][path] = sha256_bytes(subprocess.run(["git", "-C", repo, "show", f"HEAD:{path}"],
                                                         capture_output=True, check=True).stdout)
    os.makedirs(os.path.dirname(os.path.join(repo, RECORD)), exist_ok=True)
    with open(os.path.join(repo, RECORD), "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def link_test(repo=REPO, view=VIEW, common=COMMON):
    """Open the data lock: link the test split into the view, only if the program lock is open. Logged."""
    info = require_open(repo)
    for ds in DATASETS:
        for name in ("test", "test.txt"):
            src, dst = os.path.join(common, ds, name), os.path.join(view, ds, name)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    entry = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": "link-test",
             "analysis_commit": info["commit"], "origin_main": info["remote_head"], "files": info["files"],
             "repo_head": git(repo, "rev-parse", "HEAD").stdout.strip()}
    os.makedirs(os.path.join(repo, os.path.dirname(LOCK_LOG)), exist_ok=True)
    with open(os.path.join(repo, LOCK_LOG), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("test split linked into", view)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "freeze", "link-test"])
    args = ap.parse_args()
    if args.command == "check":
        ok, reasons, info = check()
        print("program lock OPEN" if ok else "program lock CLOSED:", *reasons, sep="\n  ")
        sys.exit(0 if ok else CLOSED_EXIT_CODE)
    if args.command == "freeze":
        rec = freeze()
        print(f"record written: {RECORD} (commit {rec['commit']}); commit and push it")
    else:
        link_test()


if __name__ == "__main__":
    main()
