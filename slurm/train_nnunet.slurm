#!/bin/bash
# Phase 0 main run: nnU-Net v2 2d, fold 0 (= LViT train/val), one dataset x one seed, then snapshot
# selection on the full validation set and the common evaluation (src/nnunet_select.py).
#
# SEGMENTED: a job may run at most 12 h (cluster fair-use policy) while one nnU-Net run needs ~23 h.
# Every job is ONE segment, submitted by hand (slurm/submit_next.sh); a segment never submits anything:
#   1. If the run is complete (DONE marker), exit (notification: nothing to do).
#   2. Check that checkpoint_latest.pth loads and holds the epoch of the last 50-epoch snapshot; a
#      segment that starts where the previous one started (no progress) stops as well.
#   3. If training is not finished (no checkpoint_final.pth): continue training with nnU-Net's own resume
#      (nnUNetv2_train --c, from checkpoint_latest.pth, written every 50 epochs). The 12 h limit stops
#      the segment; the notification then says that the next segment must be submitted by hand.
#      At most 50 epochs are recomputed by the next segment.
#   4. Once training is finished: snapshot selection + final prediction + common evaluation, then DONE.
#      If the time limit interrupts this step, the next segment repeats it from the start.
# Training settings, checkpoint selection and evaluation are exactly those of the unsegmented run.
# Submit (by hand): slurm/submit_next.sh   (or sbatch --export=ALL,DATASET=QaTa-COV19,SEED=1003 slurm/train_nnunet.slurm)
#SBATCH --job-name=nnunet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
source $REPO/slurm/nnunet_local.sh          # preprocessed data on node-local disk
declare -A ID=( [QaTa-COV19]=501 [MosMedData+]=502 )
declare -A NAME=( [QaTa-COV19]=Dataset501_QaTaCOV19 [MosMedData+]=Dataset502_MosMedDataPlus )
JOB_LABEL="nnU-Net $DATASET seed$SEED"
LGMS_SEGMENTED=1                            # time-limit notification: "submit the next segment by hand"
export LGMS_SEED=$SEED
export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed$SEED   # one results tree per seed
MODEL_DIR=$nnUNet_results/${NAME[$DATASET]}/nnUNetTrainerLGMS__nnUNetPlans__2d
FOLD=$MODEL_DIR/fold_0
OUT=runs/nnunet/$DATASET/seed$SEED

# 1. nothing left to do
if [ -f "$OUT/DONE" ]; then
  echo "run already complete"; NOTIFY_RESULT="run was already complete - nothing done"; exit 0
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
# 2a. Guard: stop if the previous segment made no training progress (e.g. a crash right after starting).
#     Not applied to "final": selection + evaluation (< 1 h) is simply repeated if it was interrupted.
if [ -n "$prev_point" ] && [ "$prev_point" = "$start_point" ] && [ "$start_point" != final ]; then
  echo "previous segment made no progress (start point $start_point) -- stopping, needs a manual look"
  exit 3                                     # -> "FAILED" notification
fi

# 2b. Check the resume checkpoint: it must load and hold the epoch of the last snapshot.
if [ "$start_point" != 0 ] && [ "$start_point" != final ]; then
  python - "$FOLD/checkpoint_latest.pth" "$start_point" <<'PYEOF' || { echo "checkpoint_latest.pth check FAILED"; exit 4; }
import sys, torch
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
need = {"network_weights", "optimizer_state", "grad_scaler_state", "logging", "current_epoch", "init_args"}
missing = need - set(ck)
assert not missing, f"missing keys: {missing}"
assert int(ck["current_epoch"]) == int(sys.argv[2]), f"current_epoch {ck['current_epoch']} != last snapshot {sys.argv[2]}"
print(f"checkpoint_latest.pth OK: current_epoch {ck['current_epoch']}, {len(ck['network_weights'])} weight tensors")
PYEOF
fi

case "$start_point" in
  0)     lgms_notify_start "training from epoch 0 (segment 1)" ;;
  final) lgms_notify_start "training finished earlier - snapshot selection and final evaluation only" ;;
  *)     lgms_notify_start "resuming after epoch $start_point" ;;
esac

# 3. training (one segment)
if [ ! -f "$FOLD/checkpoint_final.pth" ]; then
  stage_dataset ${NAME[$DATASET]}
  # The long command runs in the background and the script waits for it: bash handles SIGTERM (time limit /
  # scancel) only between commands, so a foreground child that is slow to exit (data-loader workers) would
  # delay the notification trap past Slurm's kill timeout. 'wait' returns at once on a signal.
  nnUNetv2_train ${ID[$DATASET]} 2d 0 -tr nnUNetTrainerLGMS --c &
  wait $!
fi

# 4. training finished: selection on the full validation set, final prediction and common evaluation
python src/nnunet_select.py --dataset "$DATASET" --model-dir "$MODEL_DIR" --out "$OUT" &
wait $!
touch "$OUT/DONE"
NOTIFY_RESULT="run complete, best val Dice $(json_value $OUT/selection.json best_val_dice) at epoch $(json_value $OUT/selection.json best_epoch 0)"
