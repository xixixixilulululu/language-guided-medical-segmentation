#!/bin/bash
# Phase 1 SENSITIVITY CHECK (post hoc, not a main run): LViT-T on MosMedData+ with batch size 24, one SEED.
# Plan: reports/PHASE1_SENSITIVITY_LVIT_PLAN.md. The driver src/phase1/train_lvit.py is the frozen main-run
# driver, unchanged; only the configuration differs (configs/lvit_sensitivity_b24.yaml: batch_size 24).
# Output: runs/lvit_b24/MosMedData+/seed$SEED (never read by src/analyze_phase1.py; no main-run file is touched).
# Submitted by hand only: slurm/submit_next_p1.sh --day (it passes DATASET and SEED with --export).
# Segments as in train_phase1.slurm: the wall-clock guard ends a segment before the 12 h limit (exit code 3);
# the user then submits the next segment by hand. A job never submits anything.
#SBATCH --job-name=p1_lvit_b24
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
[ "$DATASET" = "MosMedData+" ] || { echo "sensitivity check is for MosMedData+ only (got $DATASET)"; exit 1; }
JOB_LABEL="LViT-T batch24 (sensitivity) $DATASET seed$SEED"
OUT="runs/lvit_b24/$DATASET/seed$SEED"
CONFIG=configs/lvit_sensitivity_b24.yaml
LGMS_SEGMENTED=1                        # if the 12 h limit is ever reached anyway: "submit the next segment"
# Frozen main-run record of LViT-T (driver, shared code, text features, official commit) must still hold,
# and the sensitivity configuration must have its recorded sha256.
F=configs/phase1_frozen/lvit
sed "s#\$HOME#$HOME#" $F.sha256 | sha256sum -c --quiet - || { echo "frozen files changed: $F.sha256"; exit 1; }
sha256sum -c --quiet configs/phase1_frozen/lvit_b24.sha256 || { echo "$CONFIG changed"; exit 1; }
read -r REPO_DIR REPO_COMMIT < $F.commit
[ "$(git -C $REPO_DIR rev-parse HEAD)" = "$REPO_COMMIT" ] && [ -z "$(git -C $REPO_DIR status --porcelain --untracked-files=no)" ] \
  || { echo "$REPO_DIR is not at the frozen commit $REPO_COMMIT (or has local changes)"; exit 1; }
echo "frozen LViT-T driver verified ($F.sha256, $REPO_DIR @ $REPO_COMMIT); sensitivity config $CONFIG verified"
lgms_notify_start
# Background + wait: bash handles SIGTERM only between commands (see slurm/train_unet.slurm)
rc=0
python src/phase1/train_lvit.py --config "$CONFIG" --dataset "$DATASET" --seed "$SEED" --out "$OUT" &
wait $! || rc=$?
if [ "$rc" = 3 ]; then
  lgms_segment_ended "last completed epoch $(tail -n 1 "$OUT/epochs.csv" | cut -d, -f1), best val Dice $(json_value $OUT/training_summary.json best_val_dice)"
  exit 0
fi
[ "$rc" = 0 ] || exit "$rc"
NOTIFY_RESULT="best val Dice $(json_value $OUT/training_summary.json best_val_dice) at epoch $(json_value $OUT/training_summary.json best_epoch 0)"
echo "done $(date)"
