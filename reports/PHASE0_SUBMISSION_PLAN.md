# Phase 0 — Submission plan for the remaining runs

## Cluster rules (fair-use policy, 2026-10-13)

The supervisor set these rules for the `gpu` partition (2 nodes). The queued jobs that disappeared on
2026-10-12/13 were removed under them.

1. At most **2 jobs per user**: 1 running and 1 waiting in the queue.
2. At most **12 hours per job** (`#SBATCH --time=12:00:00` in every `.slurm` file).
3. Jobs are **submitted by hand**, one at a time; no batch submission.

How the project follows them:
- **One job per call.** The user submits with `slurm/submit_next.sh`. Each call submits exactly one job,
  the next row of `slurm/phase0_plan.tsv`, and refuses when 2 jobs exist or a job is already waiting.
  `--dry-run` shows what would be submitted.
- **The cluster orders the jobs itself.** Its admission filter (`/etc/slurm/job_submit.lua`) chains our
  second job behind the running one with `afterany`, so a queued job shows the reason "Dependency"
  until the running one ends. User-supplied `--dependency` is refused, so `submit_next.sh` never
  passes one.
- **No job submits another job.** nnU-Net segments no longer queue their successor. When a segment
  reaches the 12 h limit, its phone notification says that the next segment must be submitted by hand.
- **Notifications.** Every job notifies at start and at end; each message ends with "You can run
  slurm/submit_next.sh to submit the next job."

## State at the start of this plan

**6 of 18 runs are complete:**
- U-Net seed 1001 (QaTa-COV19, MosMedData+);
- nnU-Net seeds 1001 and 1002 (QaTa-COV19, MosMedData+).

**12 runs remain:**

| Method | Seeds | Runs | Time per run | Total |
|---|---|---:|---|---:|
| U-Net | 1002, 1003 | 4 | 2–4 h | ≈ 12 h |
| TransUNet | 1001, 1002, 1003 | 6 | 2.5–6.8 h | ≈ 28 h |
| nnU-Net | 1003 (2 datasets) | 2 | ≈ 23.5 h, as two 12 h segments | ≈ 47 h |

In total about **87 GPU hours**, run one job at a time.

## Order (re-ordered 2026-10-14 by the real start of job 282)

Job 282 (row 1) started on 2026-10-13 at 23:38 EDT and reaches its 12 h limit on 2026-10-14 at about
11:38. The remaining rows were re-ordered by that real time. Only the order changed; no setting changed.
The rules for the order:
- **Daytime: short jobs** (U-Net, TransUNet MosMedData+), so that the next job can be submitted
  promptly while the user is awake.
- **Bedtime (about 23:00–01:00): long jobs.** The job queued at bedtime is a long one (nnU-Net segment);
  the job running at bedtime is a medium-long one (TransUNet / U-Net QaTa-COV19). The queued job starts
  during the night and runs into the next morning or midday.

Durations are measured epoch times × planned epochs. U-Net assumes early stopping as for seed 1001.

| # | Job | Est. duration | Slot |
|---:|---|---:|---|
| 1 | nnU-Net QaTa-COV19 seed 1003, segment 1 (job 282, epochs 0–~516) | 12.0 h | night (running) |
| 2 | U-Net MosMedData+ seed 1002 | 2.1 h | day |
| 3 | TransUNet MosMedData+ seed 1001 | 2.5 h | day |
| 4 | U-Net QaTa-COV19 seed 1002 | 3.8 h | day |
| 5 | TransUNet QaTa-COV19 seed 1001 | 6.8 h | evening (running at bedtime) |
| 6 | nnU-Net QaTa-COV19 seed 1003, segment 2 (epochs 500–1000, selection, evaluation) | 11.8 h | night (queued at bedtime) |
| 7 | U-Net MosMedData+ seed 1003 | 2.1 h | day |
| 8 | TransUNet MosMedData+ seed 1002 | 2.5 h | day |
| 9 | U-Net QaTa-COV19 seed 1003 | 3.8 h | evening (running at bedtime) |
| 10 | nnU-Net MosMedData+ seed 1003, segment 1 | 12.0 h | night (queued at bedtime) |
| 11 | TransUNet MosMedData+ seed 1003 | 2.5 h | day |
| 12 | TransUNet QaTa-COV19 seed 1002 | 6.8 h | day |
| 13 | TransUNet QaTa-COV19 seed 1003 | 6.8 h | evening (running at bedtime) |
| 14 | nnU-Net MosMedData+ seed 1003, segment 2 (+ selection, evaluation) | 11.6 h | night (queued at bedtime) |

If an nnU-Net run is not complete after its two segments, `submit_next.sh` submits an extra segment for
it before the next plan row (about 0.5 h, selection and evaluation only). This is likely for QaTa-COV19
seed 1003. Segment 1 will end near epoch 516 with its last checkpoint at epoch 500, so segment 2 needs
about 11.7 h of training plus about 0.5 h of selection.

## Estimated timeline (EDT)

The estimate assumes a free node whenever our next job is due, and that the next job is always submitted
while the current one runs.

| # | Start | End | Submit the next job |
|---:|---|---|---|
| 1 | Tue 10-13 23:38 | Wed 10-14 11:38 | now (row 2) |
| 2 | Wed 10-14 11:38 | 13:44 | after row 2 starts |
| 3 | 13:44 | 16:14 | after row 3 starts |
| 4 | 16:14 | 20:02 | after row 4 starts |
| 5 | 20:02 | Thu 10-15 02:50 | after row 5 starts (evening): queue the night job, row 6 |
| 6 | Thu 10-15 02:50 | 14:38 | in the morning |
| 7 | 14:38 | 16:44 | after row 7 starts |
| 8 | 16:44 | 19:14 | after row 8 starts |
| 9 | 19:14 | 23:02 | after row 9 starts (evening): queue the night job, row 10 |
| 10 | Thu 10-15 23:02 | Fri 10-16 11:02 | in the morning |
| 11 | Fri 10-16 11:02 | 13:32 | after row 11 starts |
| 12 | 13:32 | 20:20 | after row 12 starts |
| 13 | 20:20 | Sat 10-17 03:08 | after row 13 starts (evening): queue the night job, row 14 |
| 14 | Sat 10-17 03:08 | **Sat 10-17 ≈ 14:45** | — |

**Estimated completion: Saturday 2026-10-17 about 14:45 EDT, or about 15:15 with the likely extra
segment for QaTa-COV19 seed 1003.** Other users' jobs (first come, first served) or late submissions
add to this.

## What the user does

- **After each "started" notification**, run `slurm/submit_next.sh`. It queues the following job, which
  then starts as soon as the running one ends.
- **In the evening**, after the evening job (rows 5, 9, 13) has started, submit the night job (an nnU-Net
  segment). It then starts during the night and runs into the next day.
- **On a "STOPPED at the 12 h time limit" notification for U-Net or TransUNet**, do nothing
  automatically. This needs a decision (see PHASE0_RESULTS.md, methods).
- **On a "FAILED" or "CANCELLED before its time limit" notification**, look at it before submitting
  anything else.
