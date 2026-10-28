#!/bin/bash
# Phase 3, Task 3.8: computational cost per image on a fixed validation subset (src/phase3/cost.py).
# 64 QaTa validation images (the first in sorted id order), C0 text, seed 1001: LANG only, IMG only (nnU-Net),
# LANG + IMG + D-Gate and LANG + IMG + LR-Gate for each language model; per-image latency and peak GPU memory.
# Validation only; no gate, threshold or model is changed. Submitted by hand only.
#SBATCH --job-name=p3_cost
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 3 computational cost (Task 3.8, validation subset)"
for F in configs/phase1_frozen/lvit configs/phase1_frozen/reclmis configs/phase1_frozen/ddcmd; do
  sed "s#\$HOME#$HOME#" $F.sha256 | sha256sum -c --quiet - || { echo "frozen files changed: $F.sha256"; exit 1; }
done
echo "frozen Phase 1 drivers verified"
export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed1001
lgms_notify_start
python src/phase3/cost.py || exit 1
NOTIFY_RESULT="cost table written (results/phase3_computational_cost.csv)"
echo "done $(date)"
