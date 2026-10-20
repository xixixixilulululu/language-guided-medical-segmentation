"""Submit exactly ONE Phase 1 job (called by the user, by hand, via slurm/submit_next_p1.sh).

Same rules as the Phase 0 tool (src/submit_next.py, unchanged): cluster fair-use policy of 2026-10-13 --
at most 2 jobs per user (1 running + 1 waiting), 12 h per job, jobs submitted by hand. This tool submits at
most one job per call, never runs in a loop and is never called by a job or a background process. It never
passes --dependency (the cluster's admission filter chains a second job itself and refuses user ones).

Each call:
  1. Refuses if the user already has 2 jobs, or already has a job waiting in the queue.
  2. Picks the next job (plan rows: quick_check, quick_check_<method>, or method/dataset/seed runs):
     a. the next SEGMENT of a training run whose last segment ended at the wall-clock guard (last line of
        <run>/segments.jsonl has "ended": "segment") or has no end line (killed by the 12 h limit or
        crashed), if no job of that run is queued/running;
     b. otherwise the first not yet submitted row of slurm/phase1_plan.tsv.
  3. Submits it with sbatch and records it in runs/phase1_submit_log.tsv.

Day / night (user rule of 2026-10-19: long QaTa runs at night, short MosMed runs in the day):
  --night   take the next QaTa-COV19 row instead of the next row in file order
  --day     take the next MosMed+ row instead of the next row in file order
  A continuation segment always comes first (e.g. LViT-T segment 2), whatever the option.
A training row is skipped if a job of the same name is already queued or its run has already started
(segments.jsonl exists), so a run is never submitted twice.

Usage:  slurm/submit_next_p1.sh [--day | --night]            submit the next job
        slurm/submit_next_p1.sh [--day | --night] --dry-run  show what would be submitted, submit nothing
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
# Runs on hold (method<TAB>dataset<TAB>seed<TAB>reason): never submitted or continued until removed from the file
HOLD = os.path.join(REPO, "slurm", "phase1_hold.tsv")
NAMES = {"lvit": "LViT-T", "reclmis": "RecLMIS", "ddcmd": "DD-CMD", "quick_check": "Phase 1 quick check",
         "diag_ddcmd": "DD-CMD diagnostic"}
# Rows that are not training runs: method -> Slurm script (no METHOD/DATASET/SEED export)
SPECIAL = {"diag_ddcmd": "phase1_ddcmd_diag.slurm"}
MAX_JOBS = 2


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


ENDING = {"CG"}   # completing: the job has ended and is only being cleaned up (not running, not waiting)


def my_jobs():
    """[(job_id, state, name)] of the current user (any partition)."""
    out = sh(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%t|%j"])
    return [tuple(line.split("|")) for line in out.splitlines() if line.strip()]


def active(jobs):
    """Jobs that count for the limit: everything except ending (CG) jobs."""
    return [j for j in jobs if j[1] not in ENDING]


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
    note = ""
    if segment and "ended" not in (segments(r) or [{}])[-1]:
        note = "; the previous segment did not end normally - check its log first"
    return f"{NAMES[r['method']]} {r['dataset']} seed{r['seed']}" + (f" (segment {segment}{note})" if segment else "")


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


def on_hold():
    """{method|dataset|seed} of the runs on hold."""
    return {f"{h['method']}|{h['dataset']}|{h['seed']}" for h in read_tsv(HOLD)}


def run_started(r):
    return bool(segments(r))


def pick_next(plan, log, jobs, dataset=None):
    """(row, segment number or None) of the next job, or (None, None). dataset: only rows of this dataset
    (--day / --night); continuation segments are always considered first."""
    names_in_queue = {j[2] for j in jobs}
    held = on_hold()
    submitted = {row["key"] for row in log}
    for r in plan:                                   # a. continuation segments first
        if (r["method"].startswith("quick_check") or r["method"] in SPECIAL or key(r) not in submitted
                or f"{r['method']}|{r['dataset']}|{r['seed']}" in held) or job_name(r) in names_in_queue:
            continue
        seg = segments(r)
        # continue a run whose last segment ended at the wall-clock guard, or whose last segment has no end
        # line at all (killed by the 12 h Slurm limit, or crashed: the label asks to check the log first)
        if seg and (seg[-1].get("ended") == "segment" or "ended" not in seg[-1]):
            return r, sum(1 for s in seg if "start_epoch" in s) + 1
    for r in plan:                                   # b. next row of the plan
        if key(r) in submitted or (dataset and r["dataset"] != dataset) \
                or f"{r['method']}|{r['dataset']}|{r['seed']}" in held:
            continue
        training = r["method"] in ("lvit", "reclmis", "ddcmd")
        if training and (job_name(r) in names_in_queue or run_started(r)):
            continue                                 # never submit the same run twice
        return r, None
    return None, None


def show_status(jobs):
    print("Your jobs now:" if jobs else "Your jobs now: none")
    for j in jobs:
        state = {"R": "running", "PD": "waiting (PD)", "CG": "ending (CG), not counted"}.get(j[1], j[1])
        print(f"  {j[0]}  {state}  {j[2]}")


def time_limit(script):
    """The job's time limit as written in the Slurm script (#SBATCH --time=...), e.g. '12:00:00'."""
    with open(script) as f:
        for line in f:
            if line.startswith("#SBATCH") and "--time=" in line:
                return line.split("--time=", 1)[1].split()[0]
    return "unknown"


def refusal(jobs):
    """Reason not to submit now, or None. Ending (CG) jobs count neither as waiting nor for the limit."""
    n_active = len(active(jobs))
    n_wait = sum(1 for j in jobs if j[1] == "PD")              # waiting = pending only
    if n_active >= MAX_JOBS:
        return f"you already have {n_active} jobs (limit {MAX_JOBS}: 1 running + 1 waiting)"
    if n_wait >= 1:
        return "you already have a job waiting in the queue (limit: 1 waiting)"
    return None


def main():
    dry = "--dry-run" in sys.argv
    dataset = "QaTa-COV19" if "--night" in sys.argv else ("MosMedData+" if "--day" in sys.argv else None)
    plan, log, jobs = read_tsv(PLAN), read_tsv(LOG), my_jobs()
    reason = refusal(jobs)
    if reason:
        print(f"NOT SUBMITTED: {reason}. Run this again when a job has finished or started.")
        r, segment = pick_next(plan, log, jobs, dataset)
        print(f"Next when allowed: {label(r, segment) if r else 'nothing (plan complete)'}\n")
        show_status(jobs)
        sys.exit(1)
    r, segment = pick_next(plan, log, jobs, dataset)
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
        print("DRY RUN - would submit:", lab, f"(time limit {time_limit(cmd[-1])})", "\n  ", " ".join(cmd), "\n")
        show_status(jobs)
        return
    job_id = sh(cmd).strip().split(";")[0]
    running = [j[0] for j in jobs if j[1] == "R"]
    note = (f"segment {segment}" if segment else "plan row") + (f"; queued behind running job {running[0]}" if running else "")
    append_log({"time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), "job_id": job_id,
                "key": key(r) + (f"|seg{segment}" if segment else ""), "label": lab, "note": note})
    print(f"SUBMITTED: job {job_id} - {lab} (time limit {time_limit(cmd[-1])}, from the Slurm script)"
          + (f"; the cluster queues it behind running job {running[0]}" if running else "") + "\n")
    show_status(my_jobs())


if __name__ == "__main__":
    main()
