#!/bin/bash
# Phase 0 main run: U-Net, one dataset x one seed (training + common evaluation of the best checkpoint).
# Submit: sbatch --export=ALL,DATASET=QaTa-COV19,SEED=1001 slurm/train_unet.slurm
#SBATCH --job-name=unet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="U-Net $DATASET seed$SEED"
OUT="runs/unet/$DATASET/seed$SEED"
python src/train_unet.py --dataset "$DATASET" --seed "$SEED" --out "$OUT"
NOTIFY_RESULT="best val Dice $(json_value $OUT/training_summary.json best_val_dice)"
