#!/bin/bash
# Phase 0 main run: nnU-Net v2 2d, fold 0 (= LViT train/val), one dataset x one seed, then snapshot
# selection on the full validation set and the common evaluation (src/nnunet_select.py).
#
# SEGMENTED, because the gpu partition limits a job to 12 h (since 2026-10-12) while one nnU-Net run
# needs ~23 h. Every job is one segment:
#   1. If the run is complete (DONE marker), exit silently.
#   2. Otherwise first queue the next segment (--dependency=afterany on this job), so that the run goes
#      on whatever stops this one.
#   3. If training is not finished (no checkpoint_final.pth): continue training with nnU-Net's own resume
#      (nnUNetv2_train --c, from checkpoint_latest.pth, written every 50 epochs). The time limit stops
#      the segment; at most 50 epochs are recomputed by the next one.
#   4. Once training is finished: snapshot selection + final prediction + common evaluation, then DONE
#      and the queued next segment is cancelled. If the time limit interrupts this step, the next
#      segment repeats it from the start.
# Training settings, checkpoint selection and evaluation are exactly those of the unsegmented run.
# Submit: sbatch --export=ALL,DATASET=QaTa-COV19,SEED=1001 slurm/train_nnunet.slurm
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
source $REPO/slurm/nnunet_local.sh          # preprocessed data on node-local disk
declare -A ID=( [QaTa-COV19]=501 [MosMedData+]=502 )
declare -A NAME=( [QaTa-COV19]=Dataset501_QaTaCOV19 [MosMedData+]=Dataset502_MosMedDataPlus )
JOB_LABEL="nnU-Net $DATASET seed$SEED"
export LGMS_SEED=$SEED
export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed$SEED   # one results tree per seed
MODEL_DIR=$nnUNet_results/${NAME[$DATASET]}/nnUNetTrainerLGMS__nnUNetPlans__2d
FOLD=$MODEL_DIR/fold_0
OUT=runs/nnunet/$DATASET/seed$SEED

# 1. nothing left to do (e.g. a queued segment after the run completed): exit without a notification
if [ -f "$OUT/DONE" ]; then
  trap - EXIT TERM INT; eval "$LGMS_CLEANUP"; echo "run already complete"; exit 0
fi

# Progress record (also the source for the segment table in PHASE0_RESULTS.md):
# runs/nnunet/<dataset>/seed<k>/segments.tsv  =  job_id  start_time  start_point
# start_point = epochs already completed (last 50-epoch checkpoint), or "final" if training was finished.
mkdir -p "$OUT"
if [ -f "$FOLD/checkpoint_final.pth" ]; then
  start_point=final
else
  start_point=$(ls "$FOLD"/checkpoint_ep*.pth 2>/dev/null | sed -E 's/.*ep0*([0-9]+)\.pth/\1/' | sort -n | tail -1 || true)
  start_point=${start_point:-0}
fi
prev_point=$(tail -n 1 "$OUT/segments.tsv" 2>/dev/null | cut -f3 || true)
printf "%s\t%s\t%s\n" "$SLURM_JOB_ID" "$(date '+%Y-%m-%d %H:%M')" "$start_point" >> "$OUT/segments.tsv"
# Guard against an endless chain of failing segments: stop if the previous segment made no progress.
if [ -n "$prev_point" ] && [ "$prev_point" = "$start_point" ]; then
  echo "previous segment made no progress (start point $start_point) -- not queuing another segment"
  exit 3                                     # -> "FAILED" notification; needs a manual look
fi

# 2. queue the next segment first
LGMS_SEGMENT_NEXT=$(sbatch --parsable --dependency=afterany:$SLURM_JOB_ID --job-name="$SLURM_JOB_NAME" \
    --export=ALL,DATASET=$DATASET,SEED=$SEED $REPO/slurm/train_nnunet.slurm)
echo "next segment queued: job $LGMS_SEGMENT_NEXT"

# 3. training (one segment)
if [ ! -f "$FOLD/checkpoint_final.pth" ]; then
  # resume point = last 50-epoch checkpoint (checkpoint_latest.pth and checkpoint_epXXXX.pth share the cadence)
  if [ "$start_point" != 0 ]; then
    $REPO/src/notify.sh "$JOB_LABEL: segment started (job $SLURM_JOB_ID), resuming after epoch $start_point" "Slurm"
  fi
  stage_dataset ${NAME[$DATASET]}
  nnUNetv2_train ${ID[$DATASET]} 2d 0 -tr nnUNetTrainerLGMS --c
fi

# 4. training finished: selection on the full validation set, final prediction and common evaluation
python src/nnunet_select.py --dataset "$DATASET" --model-dir "$MODEL_DIR" --out "$OUT"
touch "$OUT/DONE"
scancel "$LGMS_SEGMENT_NEXT" || true          # the queued next segment is not needed any more
LGMS_SEGMENT_NEXT=""
NOTIFY_RESULT="best val Dice $(json_value $OUT/selection.json best_val_dice) at epoch $(json_value $OUT/selection.json best_epoch 0)"
