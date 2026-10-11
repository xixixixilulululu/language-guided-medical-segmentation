#!/bin/bash
# Task 0.3 quick check for U-Net and TransUNet (both datasets):
#   steps 1-4 (batch, forward, shapes, loss), step 5 (small-subset training, with the full evaluation
#   pipeline at the end), and timing of full-size epochs used for the 18-run schedule.
# Run as job 206 while U-Net still used batch 2. Its U-Net steps 1-5 are superseded by
# slurm/quick_check_unet_final.slurm (batch 24, LViT paper settings); the U-Net batch-24 timing and all
# TransUNet results of this job are used.
#SBATCH --job-name=qc_unet_tu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="Quick check U-Net + TransUNet"
Q=runs/quick_check

for ds in QaTa-COV19 MosMedData+; do
  # steps 1-4
  python src/quick_check.py --method unet      --dataset $ds --out $Q/steps1-4/unet_$ds.json
  python src/quick_check.py --method transunet --dataset $ds --out $Q/steps1-4/transunet_$ds.json
  # step 5: small subset (first 32 train / 32 val cases), whole pipeline incl. final evaluation
  python src/train_unet.py      --dataset $ds --seed 1001 --subset-train 32 --subset-val 32 --max-epochs 40 \
         --out $Q/subset/unet_$ds
  python src/train_transunet.py --dataset $ds --seed 1001 --subset-train 48 --subset-val 32 --max-epochs 60 \
         --out $Q/subset/transunet_$ds
done

# timing: full-size epochs with the real configuration (no final evaluation)
for ds in QaTa-COV19 MosMedData+; do
  python src/train_unet.py      --dataset $ds --seed 1001 --max-epochs 2 --no-final-eval --out $Q/timing/unet_bs2_$ds
  python src/train_unet.py      --dataset $ds --seed 1001 --max-epochs 2 --no-final-eval --batch-size 24 \
         --out $Q/timing/unet_bs24_$ds   # LViT paper batch size (open question, see PHASE0_MODEL_SETUP.md)
  python src/train_transunet.py --dataset $ds --seed 1001 --max-epochs 2 --no-final-eval --out $Q/timing/transunet_$ds
done
# timing: 3 U-Net runs (batch 2) sharing the GPU, to see whether runs can be packed into one job
for s in 1001 1002 1003; do
  python src/train_unet.py --dataset QaTa-COV19 --seed $s --max-epochs 1 --no-final-eval \
         --out $Q/timing/unet_bs2_QaTa-COV19_packed3_s$s &
done
wait
echo "done $(date)"
