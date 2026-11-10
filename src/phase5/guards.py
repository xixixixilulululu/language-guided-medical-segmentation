"""Phase 5 self-checks shared by the programs (exit status 10 on failure)."""
import hashlib
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SPEC = os.path.join("configs", "phase5", "spec.json")
STOP_EXIT = 10


def stop(msg):
    print(f"PHASE 5 STOPPED: {msg}", file=sys.stderr)
    sys.exit(STOP_EXIT)


def git(repo, *a):
    return subprocess.run(["git", "-C", repo, *a], capture_output=True, text=True)


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def committed_unchanged(repo, path):
    """The file is tracked, has exactly one commit, equals that commit and the commit is an ancestor of HEAD."""
    commits = git(repo, "log", "--format=%H", "--", path).stdout.split()
    if not commits:
        stop(f"{path} is not committed")
    if len(commits) != 1:
        stop(f"{path} was changed after its first commit ({len(commits)} commits)")
    blob = subprocess.run(["git", "-C", repo, "show", f"{commits[0]}:{path}"], capture_output=True).stdout
    if hashlib.sha256(blob).hexdigest() != sha256_file(os.path.join(repo, path)):
        stop(f"{path} differs from its committed content")
    if git(repo, "merge-base", "--is-ancestor", commits[0], "HEAD").returncode != 0:
        stop(f"the commit of {path} is not an ancestor of HEAD")
    return commits[0]


def clean_tree(repo):
    if git(repo, "status", "--porcelain").stdout.strip():
        stop("the working tree has uncommitted changes")


def on_origin(repo):
    if git(repo, "merge-base", "--is-ancestor", "HEAD", "origin/main").returncode != 0:
        stop("HEAD is not on origin/main (push first)")


def outputs_absent(repo, outs):
    for o in outs:
        if os.path.exists(os.path.join(repo, o)):
            stop(f"{o} exists; it is never overwritten")


def write_atomic(df, repo, out, **kw):
    p = os.path.join(repo, out)
    if os.path.exists(p):
        stop(f"{out} exists; it is never overwritten")
    df.to_csv(p + ".tmp", index=False, compression="gzip" if out.endswith(".gz") else None, **kw)
    os.replace(p + ".tmp", p)
