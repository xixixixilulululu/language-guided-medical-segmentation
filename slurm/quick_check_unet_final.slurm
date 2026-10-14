#!/bin/bash
# Task 0.3 quick check, steps 1-5, for U-Net with the final configuration (batch 24; lr 3e-4 QaTa /
# 1e-3 MosMed, LViT paper Sec. IV-C). Replaces the earlier batch-2 U-Net check in
# slurm/quick_check_unet_transunet.slurm; epoch timing for batch 24 was measured there already.
#SBATCH --job-name=qc_unet_final
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="Quick check U-Net (final config)"
Q=runs/quick_check
for ds in QaTa-COV19 MosMedData+; do
  python src/quick_check.py --method unet --dataset $ds --out $Q/steps1-4/unet_$ds.json
  # small subset: first 48 train / 32 val cases (2 iterations per epoch at batch 24), full pipeline
  rm -rf $Q/subset/unet_$ds
  python src/train_unet.py --dataset $ds --seed 1001 --subset-train 48 --subset-val 32 --max-epochs 60 \
         --out $Q/subset/unet_$ds
done
echo "done $(date)"
