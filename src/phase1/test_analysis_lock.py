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

Training programs (the same four cases): the gate the three drivers call before evaluating the test split
(common.require_split) is run against a FAKE data view whose test link exists (a dummy test.txt; no real
test data is touched). With the program lock closed, the gate must stop with exit code 4 and write no
"start" line to the test-evaluation log, even though the test link exists. With the lock open but no test
link (data lock), it must also stop. Only with both open is the evaluation allowed and logged with the
fixed analysis commit. A source check confirms that each driver's final_eval calls the gate before it
builds the dataset of a split, and writes the "end" line after the common evaluation.

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


def make_view(tmp, with_test):
    """Fake data view: <view>/QaTa-COV19/{val.txt, test.txt (optional)} with dummy content."""
    view = os.path.join(tmp, "view_test" if with_test else "view_notest")
    os.makedirs(os.path.join(view, "QaTa-COV19"), exist_ok=True)
    for split in ["val"] + (["test"] if with_test else []):
        with open(os.path.join(view, "QaTa-COV19", f"{split}.txt"), "w") as f:
            f.write("dummy\n")
    return view


def gate(repo, view, eval_log):
    """Run the drivers' test gate; returns (exit code or 0, number of 'start' lines in the log)."""
    from phase1 import common
    os.environ["LGMS_COMMON_ROOT"] = view
    code = 0
    try:
        common.require_split("QaTa-COV19", "test", run={"method": "lvit", "dataset": "QaTa-COV19", "seed": 1001},
                             lock_repo=repo, eval_log=eval_log)
    except SystemExit as e:
        code = e.code
    n = sum(1 for line in open(eval_log) if line.startswith("start")) if os.path.exists(eval_log) else 0
    return code, n


def driver_source_order():
    """Each driver's final_eval: require_split before make_dataset; log_test_eval('end') after final_evaluation."""
    ok = True
    for m in ["lvit", "reclmis", "ddcmd"]:
        src = open(os.path.join(L.REPO, "src", "phase1", f"train_{m}.py")).read()
        body = src[src.index("def final_eval("):src.index("def main(")]
        good = (body.index("common.require_split(") < body.index("make_dataset(")
                and body.index("runner.final_evaluation(") < body.index('common.log_test_eval("end"'))
        check(f"driver {m}: final_eval gates every split before loading it, logs the end", good)
        ok &= good
    return ok


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
        view_t, view_n, elog = make_view(tmp, True), make_view(tmp, False), os.path.join(tmp, "eval_log.tsv")
        check("drivers, case 1: no record + test link exists -> no test evaluation (exit 4, nothing logged)",
              gate(repo, view_t, elog) == (4, 0))

        L.freeze(repo)
        sh(repo, "add", L.RECORD); sh(repo, "commit", "-q", "-m", "freeze"); sh(repo, "push", "-q", "origin", "main")
        ok, reasons, info = L.check(repo)
        check("case 4: frozen, pushed, unchanged -> open", ok and not reasons, str(reasons))
        check("drivers, case 4 without test link (data lock) -> no test evaluation (exit 4)",
              gate(repo, view_n, elog) == (4, 0))
        code, n = gate(repo, view_t, elog)
        logged = open(elog).read().splitlines()[-1].split("\t") if os.path.exists(elog) else []
        check("drivers, case 4: both locks open -> evaluation allowed, 'start' logged with the fixed commit",
              code == 0 and n == 1 and logged[6] == info["commit"], f"exit {code}, {n} start line(s)")

        path = os.path.join(repo, L.LOCKED_FILES[0])
        original = open(path).read()
        with open(path, "a") as f:
            f.write("# edited after fixing\n")
        ok, reasons, _ = L.check(repo)
        check("case 2: file changed after fixing -> closed", not ok and any("changed since" in r for r in reasons),
              str(reasons))
        check("drivers, case 2: file changed + test link exists -> no test evaluation",
              gate(repo, view_t, elog) == (4, 1))
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
        check("drivers, case 3/3c: not on origin/main + test link exists -> no test evaluation",
              gate(repo, view_t, elog) == (4, 1))
        sh(repo, "push", "-q", "origin", "main")
        ok, reasons, _ = L.check(repo)
        check("case 3 -> 4 after pushing: open", ok, str(reasons))
        # no network (compute node): the drivers fall back to the local origin/main ref and record it;
        # analyze_phase1 / link-test require a successful fetch
        good_url = subprocess.run(["git", "-C", repo, "remote", "get-url", "origin"], capture_output=True,
                                  text=True).stdout.strip()
        sh(repo, "remote", "set-url", "origin", os.path.join(tmp, "unreachable.git"))
        ok_d, _, info_d = L.check(repo, fetch_required=False)
        ok_a, reasons_a, _ = L.check(repo)
        check("no network: drivers use the local origin/main (open, fetch failure recorded)",
              ok_d and info_d.get("fetch", "").startswith("failed"), info_d.get("fetch", ""))
        check("no network: analysis / link-test stay closed (fetch required)",
              not ok_a and any("git fetch" in r for r in reasons_a), str(reasons_a)[:120])
        sh(repo, "remote", "set-url", "origin", good_url)

        # analyze_phase1.py --split test must stop before reading anything when the lock is closed
        os.remove(os.path.join(repo, L.RECORD))
        r = subprocess.run([sys.executable, os.path.join(L.REPO, "src", "analyze_phase1.py"), "--split", "test",
                            "--lock-repo", repo, "--runs-root", os.path.join(tmp, "no_runs")],
                           capture_output=True, text=True)
        check("analyze_phase1 --split test with a closed lock: exit 5 before reading data",
              r.returncode == L.CLOSED_EXIT_CODE and "PROGRAM LOCK CLOSED" in r.stdout, f"exit {r.returncode}")
        driver_source_order()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL CHECKS PASSED" if not FAILED else f"FAILED: {FAILED}")
    sys.exit(1 if FAILED else 0)


if __name__ == "__main__":
    main()
