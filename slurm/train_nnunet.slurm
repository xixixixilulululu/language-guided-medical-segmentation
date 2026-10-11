#!/bin/bash
# Phase 0 main run: nnU-Net v2 2d, fold 0 (= LViT train/val), one dataset x one seed, then snapshot
# selection on the full validation set and the common evaluation (src/nnunet_select.py).
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
stage_dataset ${NAME[$DATASET]}
export LGMS_SEED=$SEED
export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed$SEED   # one results tree per seed
# --c continues from checkpoint_latest.pth if the job is resubmitted after an interruption
nnUNetv2_train ${ID[$DATASET]} 2d 0 -tr nnUNetTrainerLGMS --c
python src/nnunet_select.py --dataset "$DATASET" \
       --model-dir "$nnUNet_results/${NAME[$DATASET]}/nnUNetTrainerLGMS__nnUNetPlans__2d" \
       --out "runs/nnunet/$DATASET/seed$SEED"
NOTIFY_RESULT="best val Dice $(json_value runs/nnunet/$DATASET/seed$SEED/selection.json best_val_dice) at epoch $(json_value runs/nnunet/$DATASET/seed$SEED/selection.json best_epoch 0)"
