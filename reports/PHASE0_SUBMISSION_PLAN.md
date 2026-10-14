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

## Order

The ~12-hour nnU-Net segments are placed overnight. Short jobs run in the daytime, when the next job can
be submitted while the current one runs. Durations are measured epoch times × planned epochs; U-Net
assumes early stopping as for seed 1001.

| # | Job | Est. duration | Slot |
|---:|---|---:|---|
| 1 | nnU-Net QaTa-COV19 seed 1003, segment 1 (epochs 0–~510) | 12.0 h | night |
| 2 | TransUNet MosMedData+ seed 1001 | 2.5 h | day |
| 3 | TransUNet QaTa-COV19 seed 1001 | 6.8 h | day |
| 4 | nnU-Net QaTa-COV19 seed 1003, segment 2 (epochs 500–1000, selection, evaluation) | 11.8 h | night |
| 5 | U-Net QaTa-COV19 seed 1002 | 3.8 h | day |
| 6 | U-Net MosMedData+ seed 1002 | 2.1 h | day |
| 7 | TransUNet MosMedData+ seed 1002 | 2.5 h | day |
| 8 | U-Net MosMedData+ seed 1003 | 2.1 h | day |
| 9 | TransUNet MosMedData+ seed 1003 | 2.5 h | day |
| 10 | nnU-Net MosMedData+ seed 1003, segment 1 | 12.0 h | night |
| 11 | TransUNet QaTa-COV19 seed 1002 | 6.8 h | day |
| 12 | U-Net QaTa-COV19 seed 1003 | 3.8 h | day |
| 13 | nnU-Net MosMedData+ seed 1003, segment 2 (+ selection, evaluation) | 11.6 h | night |
| 14 | TransUNet QaTa-COV19 seed 1003 | 6.8 h | day |

If an nnU-Net run is not complete after its two segments, `submit_next.sh` submits an extra segment for
it before the next plan row. This happens, for example, when snapshot selection is cut by the 12 h limit;
the extra segment takes about 0.5 h.

## Estimated timeline

The estimate assumes:
- the first job is submitted on 2026-10-13 at about 23:30 (EDT);
- a node is free whenever our next job is due;
- the next job is always submitted while the current one runs.

| # | Start | End |
|---:|---|---|
| 1 | Tue 10-13 23:30 | Wed 10-14 11:30 |
| 2 | Wed 10-14 11:30 | 14:00 |
| 3 | 14:00 | 20:50 |
| 4 | Wed 10-14 20:50 | Thu 10-15 08:40 |
| 5 | Thu 10-15 08:40 | 12:30 |
| 6 | 12:30 | 14:35 |
| 7 | 14:35 | 17:05 |
| 8 | 17:05 | 19:10 |
| 9 | 19:10 | 21:40 |
| 10 | Thu 10-15 21:40 | Fri 10-16 09:40 |
| 11 | Fri 10-16 09:40 | 16:30 |
| 12 | 16:30 | 20:20 |
| 13 | Fri 10-16 20:20 | Sat 10-17 07:55 |
| 14 | Sat 10-17 07:55 | **Sat 10-17 ≈ 14:45** |

**Estimated completion: Saturday 2026-10-17 afternoon.** Delays from other users' jobs (first come,
first served) or from late submissions are added on top. The previous estimate with a full queue was
2026-10-15 afternoon. The new rules add about 2 days, because only one of our jobs runs at a time.

## What the user does

- **After each "started" notification**, run `slurm/submit_next.sh`. It queues the following job, which
  then starts as soon as the running one ends.
- **In the evening**, make sure the night job is running or waiting, and the next morning job is queued
  behind it.
- **On a "STOPPED at the 12 h time limit" notification for U-Net or TransUNet**, do nothing
  automatically. This needs a decision (see PHASE0_RESULTS.md, methods).
- **On a "FAILED" or "CANCELLED before its time limit" notification**, look at it before submitting
  anything else.
