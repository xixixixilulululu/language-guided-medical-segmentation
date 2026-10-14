"""Submit exactly ONE job: the next one of the Phase 0 plan (called by the user via slurm/submit_next.sh).

Cluster fair-use policy (2026-10-13): at most 2 jobs per user (1 running + 1 waiting), 12 h per job,
jobs submitted by hand. This tool never submits more than one job per call, never runs in a loop, and is
never called by any job or background process.

Each call:
  1. Refuses if the user already has 2 jobs, or already has a job waiting in the queue.
  2. Picks the next job:
     a. an extra nnU-Net segment, if a run has used all its planned segments, is not complete (no DONE
        marker) and has no job in the queue (e.g. snapshot selection was cut by the time limit);
     b. otherwise the first not yet submitted row of slurm/phase0_plan.tsv. Rows whose run is already
        complete are recorded as skipped. An nnU-Net segment > 1 whose previous segment is still in the
        queue is submitted with --dependency=afterany:<previous job>, but only once that segment is
        running (if it is still waiting, the call refuses).
  3. Submits it with sbatch (each .slurm file sets --time=12:00:00) and records it in
     runs/phase0_submit_log.tsv.
  4. Shows: what was just submitted, what is running, what is waiting, and what comes next.

Usage:  slurm/submit_next.sh            submit the next job
        slurm/submit_next.sh --dry-run  show what would be submitted, submit nothing
"""
import csv
import datetime
import os
import subprocess
import sys

REPO = os.environ.get("LGMS_SUBMIT_REPO",
                      "/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation")
PLAN = os.path.join(REPO, "slurm", "phase0_plan.tsv")
LOG = os.path.join(REPO, "runs", "phase0_submit_log.tsv")
NAMES = {"unet": "U-Net", "transunet": "TransUNet", "nnunet": "nnU-Net"}
MAX_JOBS = 2


def sh(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout


def my_jobs():
    """[(job_id, state, name)] of the current user (any partition)."""
    out = sh(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i|%t|%j"])
    return [tuple(line.split("|")) for line in out.splitlines() if line.strip()]


def read_plan():
    with open(PLAN) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def read_log():
    if not os.path.exists(LOG):
        return []
    with open(LOG) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def append_log(row):
    new = not os.path.exists(LOG)
    with open(LOG, "a") as f:
        w = csv.DictWriter(f, fieldnames=["time", "job_id", "key", "label", "note"], delimiter="\t")
        if new:
            w.writeheader()
        w.writerow(row)


def key(r):
    return f"{r['method']}|{r['dataset']}|{r['seed']}|{r['segment']}"


def label(method, dataset, seed, segment):
    s = f"{NAMES[method]} {dataset} seed{seed}"
    return s + (f" (segment {segment})" if segment not in ("-", "") else "")


def run_complete(method, dataset, seed):
    out = os.path.join(REPO, "runs", method, dataset, f"seed{seed}")
    if method == "nnunet":
        return os.path.exists(os.path.join(out, "DONE"))
    return os.path.exists(os.path.join(out, "eval_test", "per_case.csv"))


def pick_next(plan, log, jobs):
    """Return (item dict, refusal message or None, skipped rows). Pure: changes nothing."""
    submitted = {r["key"]: r for r in log}
    in_queue = {j[0]: j[1] for j in jobs}
    skipped = []
    # a. extra nnU-Net segment for a run that used all planned segments but is not complete
    runs = {}
    for r in plan:
        if r["method"] == "nnunet":
            runs.setdefault((r["dataset"], r["seed"]), []).append(r)
    for (ds, seed), rows in runs.items():
        keys = [key(r) for r in rows]
        extra = [l for l in log if l["key"].startswith(f"nnunet|{ds}|{seed}|extra")]
        all_keys = keys + [l["key"] for l in extra]
        if all(k in submitted for k in keys) and not run_complete("nnunet", ds, seed):
            ids = [submitted[k]["job_id"] for k in all_keys if k in submitted and submitted[k]["job_id"].isdigit()]
            if not any(i in in_queue for i in ids):
                n = len(extra) + 1
                return ({"method": "nnunet", "dataset": ds, "seed": seed, "segment": f"extra{n}",
                         "est_hours": "1-12", "dependency": None,
                         "why": "all planned segments ended but the run is not complete"}, None, skipped)
    # b. first not yet submitted plan row
    for r in plan:
        k = key(r)
        if k in submitted:
            continue
        if run_complete(r["method"], r["dataset"], r["seed"]):
            skipped.append(r)
            continue
        item = dict(r, dependency=None, why=f"plan row {r['order']}")
        if r["method"] == "nnunet" and r["segment"] not in ("1", "-"):
            prev = submitted.get(f"nnunet|{r['dataset']}|{r['seed']}|{int(r['segment']) - 1}")
            if prev is None:
                return (item, f"segment {int(r['segment']) - 1} of this run has not been submitted", skipped)
            state = in_queue.get(prev["job_id"])
            if state == "PD":
                return (item, f"the previous segment (job {prev['job_id']}) has not started yet - submit this "
                              "one after it starts running", skipped)
            if state is not None:
                item["dependency"] = prev["job_id"]          # starts only after the previous segment ends
        return (item, None, skipped)
    return (None, None, skipped)


def show_status(jobs, plan, log):
    run = [j for j in jobs if j[1] == "R"]
    wait = [j for j in jobs if j[1] != "R"]
    print("Running :", "; ".join(f"{j[0]} {j[2]}" for j in run) or "none")
    print("Waiting :", "; ".join(f"{j[0]} {j[2]} ({j[1]})" for j in wait) or "none")
    nxt, why, _ = pick_next(plan, log, jobs)
    if nxt is None:
        print("Next    : the plan is complete - nothing left to submit")
    else:
        print(f"Next    : {label(nxt['method'], nxt['dataset'], nxt['seed'], nxt['segment'])}, ~{nxt['est_hours']} h"
              + (f" [{nxt.get('when')}]" if nxt.get("when") else "") + (f" - not yet possible: {why}" if why else ""))
    done = sum(1 for r in log if r["note"] != "skipped (run already complete)")
    print(f"Plan    : {len(plan)} rows, {done} submitted so far (log: runs/phase0_submit_log.tsv)")


def main():
    dry = "--dry-run" in sys.argv
    plan, log, jobs = read_plan(), read_log(), my_jobs()
    n_wait = sum(1 for j in jobs if j[1] != "R")
    if len(jobs) >= MAX_JOBS or n_wait >= 1:
        reason = (f"you already have {len(jobs)} jobs (limit {MAX_JOBS}: 1 running + 1 waiting)"
                  if len(jobs) >= MAX_JOBS else "you already have a job waiting in the queue (limit: 1 waiting)")
        print(f"NOT SUBMITTED: {reason}. Run this again when a job has finished or started.\n")
        show_status(jobs, plan, log)
        sys.exit(1)

    item, refusal, skipped = pick_next(plan, log, jobs)
    if item is None:
        print("Nothing to submit: every row of the plan has been submitted.\n")
        show_status(jobs, plan, log)
        return
    lab = label(item["method"], item["dataset"], item["seed"], item["segment"])
    if refusal:
        print(f"NOT SUBMITTED: next is {lab}, but {refusal}.\n")
        show_status(jobs, plan, log)
        sys.exit(1)

    script = os.path.join(REPO, "slurm", f"train_{item['method']}.slurm")
    jobname = f"{item['method']}-{item['dataset'].split('-')[0]}-{item['seed']}"
    cmd = ["sbatch", "--parsable", f"--job-name={jobname}",
           f"--export=ALL,DATASET={item['dataset']},SEED={item['seed']}"]
    if item["dependency"]:
        cmd.append(f"--dependency=afterany:{item['dependency']}")
    cmd.append(script)
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if dry:
        print("DRY RUN - would run:", " ".join(cmd))
        for r in skipped:
            print("  (would record as skipped, run already complete:", label(r["method"], r["dataset"], r["seed"], r["segment"]), ")")
        print()
        show_status(jobs, plan, log)
        return
    for r in skipped:
        append_log({"time": now, "job_id": "-", "key": key(r), "note": "skipped (run already complete)",
                    "label": label(r["method"], r["dataset"], r["seed"], r["segment"])})
    job_id = sh(cmd).strip().split(";")[0]
    k = (f"nnunet|{item['dataset']}|{item['seed']}|{item['segment']}" if str(item["segment"]).startswith("extra")
         else key(item))
    append_log({"time": now, "job_id": job_id, "key": k, "label": lab,
                "note": item["why"] + (f"; after job {item['dependency']}" if item["dependency"] else "")})
    print(f"SUBMITTED: job {job_id} - {lab} (~{item['est_hours']} h, 12 h limit)"
          + (f", starts after job {item['dependency']} ends" if item["dependency"] else "") + "\n")
    show_status(my_jobs(), plan, read_log())


if __name__ == "__main__":
    main()
