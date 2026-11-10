#!/bin/bash
# Phase 5, Task 5.1A (step 3): SDC-Fallback thresholds on QaTa GATE_CAL (validation only; CPU work, no model run).
# src/phase5/calibrate_sdc.py: SDC of every GATE_CAL row of C0, C2, C4, C5, C6 for 3 language models x 3 seeds from the
# stored float32 LANG maps (rows as src/phase3/features.py), then src/phase3/gates.py calibrate() on -SDC per model x
# seed (rules: configs/phase5/spec.json). The program refuses to run unless the spec is committed once and unchanged,
# the working tree is clean and the outputs do not exist. No test data are read. Submitted by hand only.
#SBATCH --job-name=p5_calibrate
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 SDC calibration (GATE_CAL, validation)"
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase5/calibrate_sdc.py || exit 1
NOTIFY_RESULT="SDC thresholds written (results/phase5_sdc_calibration.csv)"
echo "done $(date)"
