"""Phase 2 lock: no Phase 2 work on the TEST split before the Phase 2 rules and analysis are frozen on GitHub.

Two independent locks (as in Phase 1, decisions D14 of reports/PHASE2_SETUP_PLAN.md):
  1. Program lock (this file). Open only if ALL hold:
     a. the frozen record configs/phase2_frozen/phase2_lock.json exists;
     b. it covers the Phase 3 decision record (configs/phase2_frozen/phase3_decision.json: Phase 3 protocol
        frozen, or Phase 3 cancelled) and PHASE2_PROTOCOL.md, and the decision record exists;
     c. every recorded file has the recorded sha256 now;
     d. the recorded commit is on GitHub: after `git fetch`, an ancestor of origin/main;
     e. in that commit, every recorded file has the recorded sha256.
  2. Text view: Phase 2 reads the LViT text files through ~/datasets/phase2/text_view/, which holds only the
     Train and Val text files until `link-test` (allowed only when the program lock is open) adds Test.
Every Phase 2 test-split entry point (condition generator, inference driver) calls require_open() before it
reads any test file and exits with CLOSED_EXIT_CODE otherwise. This covers ALL conditions, C0 included (D14).
Test runs are logged in results/phase2_test_eval_log.tsv (log_test_run).

Commands:
  python src/phase2/lock.py check       report the lock state (exit 0 open, 5 closed)
  python src/phase2/lock.py make-view   create the text view with Train/Val only (no lock needed)
  python src/phase2/lock.py freeze      write the record for HEAD (files committed and unchanged)
  python src/phase2/lock.py link-test   only if `check` passes: add the Test text files to the view (logged)
Tests: src/phase2/test_phase2_lock.py.
"""
import argparse
import datetime
import json
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "src"))
from phase1.analysis_lock import git, sha256_bytes, sha256_file  # noqa: E402  (shared helpers, unchanged)

RECORD = os.path.join("configs", "phase2_frozen", "phase2_lock.json")
DECISION = os.path.join("configs", "phase2_frozen", "phase3_decision.json")
PROTOCOL = os.path.join("reports", "PHASE2_PROTOCOL.md")
# Files that define Phase 2 (the record must cover all of them; inference driver and analysis are added
# here when they are written, before the freeze)
LOCKED_FILES = ["src/phase2/lock.py", "src/phase2/texts.py", "src/phase2/conditions.py", "src/phase2/infer.py",
                PROTOCOL, DECISION]
REQUIRED_IN_RECORD = [DECISION, PROTOCOL]
CLOSED_EXIT_CODE = 5
TEXT_SOURCE = os.path.expanduser("~/datasets")                      # LViT layout: <ds>/<S>_Folder/<S>_text.xlsx
TEXT_VIEW = os.path.expanduser(os.environ.get("LGMS_P2_TEXT_VIEW", "~/datasets/phase2/text_view"))
DATASETS = ["QaTa-COV19", "MosMedData+"]
FOLDER = {"train": "Train", "val": "Val", "test": "Test"}
LOCK_LOG = os.path.join("runs", "phase2_lock_log.jsonl")
TEST_LOG = os.path.join("results", "phase2_test_eval_log.tsv")
TEST_LOG_COLUMNS = ["event", "time", "job_id", "program", "dataset", "method", "seed", "condition",
                    "lock_commit", "origin_main", "fetch"]


def text_file(dataset, split, view=None):
    """Path of a split's text file inside the view (the file exists only if the split is linked)."""
    s = FOLDER[split]
    return os.path.join(view or TEXT_VIEW, dataset, f"{s}_Folder", f"{s}_text.xlsx")


def check(repo=REPO, remote="origin", branch="main", fetch=True, fetch_required=True):
    """(is_open, reasons, info); `reasons` lists every failed condition (empty when open).

    fetch_required=False (compute nodes without network): if `git fetch` fails, the local origin/main ref
    (updated by every push from the login node) is used; the failure is recorded in info["fetch"]."""
    reasons, info = [], {}
    rec_path = os.path.join(repo, RECORD)
    if not os.path.exists(rec_path):                                              # a. record
        return False, [f"no frozen record ({RECORD}): Phase 2 has not been frozen"], info
    with open(rec_path) as f:
        rec = json.load(f)
    commit, files = rec["commit"], rec["files"]
    info.update(commit=commit, files=files)
    for path in REQUIRED_IN_RECORD:                                               # b. decision + protocol
        if path not in files:
            reasons.append(f"{path}: not covered by the frozen record")
    if not os.path.exists(os.path.join(repo, DECISION)):
        reasons.append(f"{DECISION}: missing (Phase 3 protocol or cancellation not recorded)")
    for path, digest in files.items():                                            # c. current content
        full = os.path.join(repo, path)
        if not os.path.exists(full):
            reasons.append(f"{path}: missing")
        elif sha256_file(full) != digest:
            reasons.append(f"{path}: changed since it was frozen (sha256 differs)")
    if fetch:                                                                     # d. on GitHub
        r = git(repo, "fetch", "--quiet", remote, branch, check=False)
        info["fetch"] = "ok" if r.returncode == 0 else f"failed: {r.stderr.strip()[:120]}"
        if r.returncode != 0 and fetch_required:
            reasons.append(f"git fetch {remote} {branch} failed: {r.stderr.strip()[:200]}")
    remote_ref = f"{remote}/{branch}"
    head = git(repo, "rev-parse", "--verify", "--quiet", remote_ref, check=False).stdout.strip()
    info["remote_head"] = head
    if git(repo, "cat-file", "-e", f"{commit}^{{commit}}", check=False).returncode != 0:
        reasons.append(f"recorded commit {commit} does not exist in this repository")
    elif not head or git(repo, "merge-base", "--is-ancestor", commit, remote_ref, check=False).returncode != 0:
        reasons.append(f"recorded commit {commit} is not on {remote_ref} (not pushed to GitHub)")
    else:                                                                         # e. content of that commit
        for path, digest in files.items():
            r = subprocess.run(["git", "-C", repo, "show", f"{commit}:{path}"], capture_output=True, check=False)
            if r.returncode != 0:
                reasons.append(f"{path}: not in commit {commit}")
            elif sha256_bytes(r.stdout) != digest:
                reasons.append(f"{path}: content in commit {commit} differs from the record")
    return not reasons, reasons, info


def require_open(repo=REPO, fetch_required=False, what="test split"):
    """Exit with CLOSED_EXIT_CODE unless the lock is open. Call it BEFORE any test file is read."""
    ok, reasons, info = check(repo, fetch_required=fetch_required)
    if not ok:
        print(f"PHASE 2 LOCK CLOSED - {what} refused:", *reasons, sep="\n  ")
        sys.exit(CLOSED_EXIT_CODE)
    print(f"Phase 2 lock open: frozen in commit {info['commit']} (origin/main {info['remote_head'][:12]}, "
          f"fetch {info.get('fetch')})")
    return info


def log_test_run(event, info, program, dataset, method="", seed="", condition="", repo=REPO):
    """One line per start / end of a Phase 2 test-split run (results/phase2_test_eval_log.tsv)."""
    path = os.path.join(repo, TEST_LOG)
    new = not os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    row = [event, time.strftime("%Y-%m-%d %H:%M:%S"), os.environ.get("SLURM_JOB_ID", "none"), program, dataset,
           method, str(seed), condition, info.get("commit", ""), info.get("remote_head", ""), info.get("fetch", "")]
    with open(path, "a") as f:
        if new:
            f.write("\t".join(TEST_LOG_COLUMNS) + "\n")
        f.write("\t".join(row) + "\n")


def freeze(repo=REPO, files=None):
    """Write the record for HEAD: every locked file must exist, be committed and be unchanged."""
    files = files or LOCKED_FILES
    commit = git(repo, "rev-parse", "HEAD").stdout.strip()
    rec = {"commit": commit, "frozen": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "files": {}}
    for path in files:
        if git(repo, "status", "--porcelain", "--", path).stdout.strip():
            raise SystemExit(f"{path} has uncommitted changes: commit it first")
        r = subprocess.run(["git", "-C", repo, "show", f"HEAD:{path}"], capture_output=True, check=False)
        if r.returncode != 0:
            raise SystemExit(f"{path} is not in HEAD")
        rec["files"][path] = sha256_bytes(r.stdout)
    os.makedirs(os.path.dirname(os.path.join(repo, RECORD)), exist_ok=True)
    with open(os.path.join(repo, RECORD), "w") as f:
        json.dump(rec, f, indent=2)
    return rec


def make_view(view=None, source=TEXT_SOURCE):
    """Text view with the Train and Val text files only (symbolic links; the Test file is never linked here)."""
    view = view or TEXT_VIEW
    for ds in DATASETS:
        for split in ("train", "val"):
            src, dst = text_file(ds, split, source), text_file(ds, split, view)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if not os.path.lexists(dst):
                os.symlink(src, dst)
    print("text view (train/val only):", view)


def link_test(repo=REPO, view=None, source=TEXT_SOURCE):
    """Open the text lock: link the Test text files into the view, only if the program lock is open. Logged."""
    info = require_open(repo, fetch_required=True, what="link-test")
    view = view or TEXT_VIEW
    for ds in DATASETS:
        src, dst = text_file(ds, "test", source), text_file(ds, "test", view)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.lexists(dst):
            os.symlink(src, dst)
    entry = {"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "action": "link-test",
             "lock_commit": info["commit"], "origin_main": info["remote_head"], "files": info["files"],
             "repo_head": git(repo, "rev-parse", "HEAD").stdout.strip()}
    os.makedirs(os.path.join(repo, os.path.dirname(LOCK_LOG)), exist_ok=True)
    with open(os.path.join(repo, LOCK_LOG), "a") as f:
        f.write(json.dumps(entry) + "\n")
    print("test text files linked into", view)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "make-view", "freeze", "link-test"])
    args = ap.parse_args()
    if args.command == "check":
        ok, reasons, _ = check()
        print("Phase 2 lock OPEN" if ok else "Phase 2 lock CLOSED:", *reasons, sep="\n  ")
        sys.exit(0 if ok else CLOSED_EXIT_CODE)
    if args.command == "make-view":
        make_view()
    elif args.command == "freeze":
        rec = freeze()
        print(f"record written: {RECORD} (commit {rec['commit']}); commit and push it")
    else:
        link_test()


if __name__ == "__main__":
    main()
