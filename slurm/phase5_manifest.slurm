#!/bin/bash
# Phase 5, Task 5.2 (step 4): sha256 manifest of every input of the Phase 5 test computation (data and code).
# src/phase5/make_manifest.py -> configs/phase5/artifact_manifest.tsv (validation and test probability maps, archived
# masks and per-case files of C0, C2, C4-C6; IMG maps; QaTa test ground truth and id list; existing tables; SDC
# calibration outputs; Phase 5 code and every in-repository module it imports; slurm/phase5_test.slurm and its
# environment scripts). Reads files only to hash them; computes no result. Submitted by hand only.
#SBATCH --job-name=p5_manifest
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 artifact sha256 manifest"
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase5/make_manifest.py || exit 1
NOTIFY_RESULT="manifest written (configs/phase5/artifact_manifest.tsv)"
echo "done $(date)"
