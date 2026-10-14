#!/bin/bash
# Phase 0 main run: TransUNet, one dataset x one seed (training + common evaluation of the best epoch).
# Submit by hand: slurm/submit_next.sh (or: sbatch --export=ALL,DATASET=QaTa-COV19,SEED=1001 slurm/train_transunet.slurm)
#SBATCH --job-name=transunet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="TransUNet $DATASET seed$SEED"
OUT="runs/transunet/$DATASET/seed$SEED"
LGMS_EPOCHS_CSV="$OUT/epochs.csv"          # time-limit notification: last / best epoch
lgms_notify_start
# The long command runs in the background and the script waits for it: bash handles SIGTERM (time limit /
# scancel) only between commands, so a foreground child that is slow to exit (data-loader workers) would
# delay the notification trap past Slurm's kill timeout. 'wait' returns at once on a signal.
python src/train_transunet.py --dataset "$DATASET" --seed "$SEED" --out "$OUT" &
wait $!
NOTIFY_RESULT="best val Dice $(json_value $OUT/training_summary.json best_val_dice)"
