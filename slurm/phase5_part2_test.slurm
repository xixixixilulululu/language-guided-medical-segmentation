#!/bin/bash
# Phase 5 Part 2, Tasks 5.2.3-5.2.5 (step 5): the one U-Net sensitivity computation on QaTa test, after the freeze.
# src/phase5_part2/compute.py refuses to run unless: Phase 2 lock open; spec, freeze document and manifest committed
# once and unchanged; clean tree; HEAD on origin/main; every manifest file (data and code) has its recorded sha256;
# result files absent. Run once; logged in results/phase5_part2_test_run_log.tsv. Submitted by hand only.
#SBATCH --job-name=p5p2_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 Part 2 test computation (U-Net sensitivity, frozen)"
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase5_part2/compute.py || exit 1
NOTIFY_RESULT="Phase 5 Part 2 outputs written (results/phase5_part2_*.csv)"
echo "done $(date)"
