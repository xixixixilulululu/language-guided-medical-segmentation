#!/bin/bash
# Phase 5, Tasks 5.3-5.4 (step 5): the one QaTa test computation, after the freeze (post-hoc supplementary).
# src/phase5/compute_test_split.py refuses to run unless: Phase 2 lock open; spec, freeze document, manifest and SDC
# calibration committed once and unchanged; clean tree; HEAD on origin/main; every manifest file (data and code)
# has its recorded sha256; outputs absent. Then 206,595 rows asserted and every archived mask checked before any
# output. Run once; logged in results/phase5_test_run_log.tsv. Submitted by hand only.
#SBATCH --job-name=p5_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 test computation (post-hoc, frozen)"
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase5/compute_test_split.py || exit 1
NOTIFY_RESULT="Phase 5 test outputs written (results/phase5_*.csv)"
echo "done $(date)"
