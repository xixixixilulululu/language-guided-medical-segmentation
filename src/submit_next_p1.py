"""Submit exactly ONE Phase 1 job (called by the user, by hand, via slurm/submit_next_p1.sh).

Same rules as the Phase 0 tool (src/submit_next.py, unchanged): cluster fair-use policy of 2026-10-13 --
at most 2 jobs per user (1 running + 1 waiting), 12 h per job, jobs submitted by hand. This tool submits at
most one job per call, never runs in a loop and is never called by a job or a background process. It never
passes --dependency (the cluster's admission filter chains a second job itself and refuses user ones).

Each call:
  1. Refuses if the user already has 2 jobs, or already has a job waiting in the queue.
  2. Picks the next job (plan rows: quick_check, quick_check_<method>, or method/dataset/seed runs):
     a. the next SEGMENT of a training run whose last segment ended at the wall-clock guard
        (last line of <run>/segments.jsonl has "ended": "segment"), if no job of that run is queued/running;
     b. otherwise the first not yet submitted row of slurm/phase1_plan.tsv.
  3. Submits it with sbatch and records it in runs/phase1_submit_log.tsv.

Usage:  slurm/submit_next_p1.sh            submit the next job
        slurm/submit_next_p1.sh --dry-run  show what would be submitted, submit nothing
"""
import csv
import datetime
import json
import os
import subprocess
import sys

REPO = os.environ.get("LGMS_SUBMIT_REPO",
                      "/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation")
PLAN = os.path.join(REPO, "slurm", "phase1_plan.tsv")
LOG = os.path.join(REPO, "runs", "phase1_submit_log.tsv")
NAMES = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD", "quick_check": "Phase 1 quick check",
         "diag_ddcmd": "DD-CMD diagnostic"}
# Rows that are not training runs: method -> Slurm script (no METHOD/DATASET/SEED export)
SPECIAL = {"diag_ddcmd": "phase1_ddcmd_diag.slurm"}
MAX_JOBS = 2


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def my_jobs():
    """[(job_id, state, name)] of the current user (any partition)."""
    out = sh(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%t|%j"])
    return [tuple(line.split("|")) for line in out.splitlines() if line.strip()]


def read_tsv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_log(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        w = csv.DictWriter(f, fieldnames=["time", "job_id", "key", "label", "note"], delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def key(r):
    """Plan-row key; the optional 'tag' column distinguishes a resubmission (e.g. after a cancelled job)."""
    tag = (r.get("tag") or "").strip()
    return f"{r['method']}|{r['dataset']}|{r['seed']}" + (f"|{tag}" if tag and tag != "-" else "")


def qc_methods(r):
    """Quick-check rows: 'quick_check' (all methods) or 'quick_check_<method>' (that method only)."""
    m = r["method"]
    return m[len("quick_check_"):] if m.startswith("quick_check_") else None


def label(r, segment=None):
    if r["method"] in SPECIAL:
        return NAMES[r["method"]]
    if r["method"].startswith("quick_check"):
        return NAMES["quick_check"] + (f" ({NAMES[qc_methods(r)]} only)" if qc_methods(r) else "")
    return f"{NAMES[r['method']]} {r['dataset']} seed{r['seed']}" + (f" (segment {segment})" if segment else "")


def job_name(r):
    if r["method"] in SPECIAL:
        return "p1_" + r["method"]
    return "p1_quick_check" if r["method"].startswith("quick_check") else f"p1-{r['method']}-{r['dataset'].split('-')[0]}-{r['seed']}"


def segments(r):
    """Lines of <run>/segments.jsonl (one 'start' and one 'end' line per finished segment)."""
    path = os.path.join(REPO, "runs", r["method"], r["dataset"], f"seed{r['seed']}", "segments.jsonl")
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def pick_next(plan, log, jobs):
    """(row, segment number or None) of the next job, or (None, None)."""
    names_in_queue = {j[2] for j in jobs}
    submitted = {row["key"] for row in log}
    for r in plan:                                   # a. continuation segments first
        if r["method"].startswith("quick_check") or r["method"] in SPECIAL or key(r) not in submitted or job_name(r) in names_in_queue:
            continue
        seg = segments(r)
        if seg and seg[-1].get("ended") == "segment":
            return r, sum(1 for s in seg if "ended" in s) + 1
    for r in plan:                                   # b. next row of the plan
        if key(r) not in submitted:
            return r, None
    return None, None


def show_status(jobs):
    print("Your jobs now:" if jobs else "Your jobs now: none")
    for j in jobs:
        print(f"  {j[0]}  {'running' if j[1] == 'R' else 'waiting (' + j[1] + ')'}  {j[2]}")


def main():
    dry = "--dry-run" in sys.argv
    plan, log, jobs = read_tsv(PLAN), read_tsv(LOG), my_jobs()
    n_wait = sum(1 for j in jobs if j[1] != "R")
    if len(jobs) >= MAX_JOBS or n_wait >= 1:
        reason = (f"you already have {len(jobs)} jobs (limit {MAX_JOBS}: 1 running + 1 waiting)"
                  if len(jobs) >= MAX_JOBS else "you already have a job waiting in the queue (limit: 1 waiting)")
        print(f"NOT SUBMITTED: {reason}. Run this again when a job has finished or started.")
        r, segment = pick_next(plan, log, jobs)
        print(f"Next when allowed: {label(r, segment) if r else 'nothing (plan complete)'}\n")
        show_status(jobs)
        sys.exit(1)
    r, segment = pick_next(plan, log, jobs)
    if r is None:
        print("Nothing to submit: every row of slurm/phase1_plan.tsv has been submitted and no run waits for a segment.\n")
        show_status(jobs)
        return
    if r["method"] in SPECIAL:
        cmd = ["sbatch", "--parsable", f"--job-name={job_name(r)}", os.path.join(REPO, "slurm", SPECIAL[r["method"]])]
    elif r["method"].startswith("quick_check"):
        cmd = ["sbatch", "--parsable", f"--job-name={job_name(r)}"]
        if qc_methods(r):
            cmd.append(f"--export=ALL,QC_METHODS={qc_methods(r)}")
        cmd.append(os.path.join(REPO, "slurm", "phase1_quick_check.slurm"))
    else:
        cmd = ["sbatch", "--parsable", f"--job-name={job_name(r)}",
               f"--export=ALL,METHOD={r['method']},DATASET={r['dataset']},SEED={r['seed']}",
               os.path.join(REPO, "slurm", "train_phase1.slurm")]
    lab = label(r, segment)
    if dry:
        print("DRY RUN - would submit:", lab, "\n  ", " ".join(cmd), "\n")
        show_status(jobs)
        return
    job_id = sh(cmd).strip().split(";")[0]
    running = [j[0] for j in jobs if j[1] == "R"]
    note = (f"segment {segment}" if segment else "plan row") + (f"; queued behind running job {running[0]}" if running else "")
    append_log({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "job_id": job_id,
                "key": key(r) + (f"|seg{segment}" if segment else ""), "label": lab, "note": note})
    print(f"SUBMITTED: job {job_id} - {lab} (12 h limit)"
          + (f"; the cluster queues it behind running job {running[0]}" if running else "") + "\n")
    show_status(my_jobs())


if __name__ == "__main__":
    main()
