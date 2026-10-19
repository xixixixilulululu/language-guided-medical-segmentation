#!/bin/bash
# DD-CMD diagnostic (quick-check job 317 hung after Lightning's sanity check, before the end of the first
# training epoch). Small subset (64 train / 32 val cases, QaTa-COV19, 2 epochs, validation only; the test
# split is never touched). Four variants of the DATA LOADING only, each limited to 10 minutes:
#   A. 0 workers                                   (the official setting)
#   B. 8 workers, sharing strategy file_system     (the official train.py sets this strategy)
#   C. 8 workers, as in job 317                    (expected to hang: stack dumps show where)
#   D. 8 workers, spawn instead of fork
# Stack dumps of the main process and every data-loader worker every 60 s (src/phase1/diag_ddcmd.py).
# Time limit 1 h (user decision of 2026-10-19), so that the next job waits at most 1 h.
# Submitted by hand only (slurm/submit_next_p1.sh).
#SBATCH --job-name=p1_ddcmd_diag
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=01:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="DD-CMD diagnostic"
D=runs/phase1_quick_check/diag_job${SLURM_JOB_ID:-none}
mkdir -p $D
lgms_notify_start
echo "ulimit -n $(ulimit -n) | nproc $(nproc) | $(date)"
RESULT=""
variant() {   # variant <name> <args...>: run with a 10-minute limit, record the exit code (124 = timeout)
  local name=$1; shift
  echo "=== $(date +%T) variant $name: $*"
  timeout --signal=TERM --kill-after=30 600 python -X faulthandler src/phase1/diag_ddcmd.py --out $D/$name "$@" \
    > $D/$name.log 2>&1
  local rc=$?
  echo "=== $(date +%T) variant $name: exit $rc"
  RESULT="$RESULT $name=$rc"
}
variant A_workers0 --workers 0
variant B_workers8_filesystem --workers 8 --sharing file_system
variant C_workers8 --workers 8
variant D_workers8_spawn --workers 8 --spawn
echo "result:$RESULT"
NOTIFY_RESULT="exit codes:$RESULT (0 = ok, 124 = hung, stopped after 10 min)"
echo "done $(date)"
