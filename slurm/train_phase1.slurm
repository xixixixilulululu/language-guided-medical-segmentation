#!/bin/bash
# Phase 1 main run (one segment): METHOD (lvit | reclmis | ddcmd) x DATASET x SEED, training and, when
# training has finished, the common evaluation of the best validation-Dice checkpoint (val + test).
# Submitted by hand only: slurm/submit_next_p1.sh (it passes METHOD, DATASET, SEED with --export).
# Segments: if the run directory already holds a resume checkpoint, training continues from it. The driver's
# wall-clock guard ends a segment before the 12 h limit (exit code 3); the user then submits the next
# segment by hand. A job never submits anything.
#SBATCH --job-name=phase1
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
declare -A NAME=([lvit]="LViT-T" [reclmis]="RecLMIS" [ddcmd]="DD-CMD")
JOB_LABEL="${NAME[$METHOD]} $DATASET seed$SEED"
OUT="runs/$METHOD/$DATASET/seed$SEED"
LGMS_SEGMENTED=1                        # if the 12 h limit is ever reached anyway: "submit the next segment"
lgms_notify_start
# Background + wait: bash handles SIGTERM only between commands (see slurm/train_unet.slurm)
rc=0
python "src/phase1/train_$METHOD.py" --dataset "$DATASET" --seed "$SEED" --out "$OUT" &
wait $! || rc=$?
if [ "$rc" = 3 ]; then
  lgms_segment_ended "last completed epoch $(tail -n 1 "$OUT/epochs.csv" | cut -d, -f1), best val Dice $(json_value $OUT/training_summary.json best_val_dice)"
  exit 0
fi
[ "$rc" = 0 ] || exit "$rc"
NOTIFY_RESULT="best val Dice $(json_value $OUT/training_summary.json best_val_dice) at epoch $(json_value $OUT/training_summary.json best_epoch 0)"
echo "done $(date)"
