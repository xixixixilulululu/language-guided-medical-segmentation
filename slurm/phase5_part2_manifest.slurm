#!/bin/bash
# Phase 5 Part 2, Task 5.2.2 (step 4): sha256 manifest of every input of the U-Net test computation (data and code).
# src/phase5_part2/make_manifest.py -> configs/phase5_part2/artifact_manifest.tsv. Hashes files only. Submitted by hand.
#SBATCH --job-name=p5p2_manifest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 Part 2 artifact sha256 manifest"
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase5_part2/make_manifest.py || exit 1
NOTIFY_RESULT="manifest written (configs/phase5_part2/artifact_manifest.tsv)"
echo "done $(date)"
