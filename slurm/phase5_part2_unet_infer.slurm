#!/bin/bash
# Phase 5 Part 2, Task 5.2.1 (step 3): U-Net test inference (inference only, no training) and reconciliation.
# 1. src/phase3/infer_img.py --arch unet --dataset QaTa-COV19 --seeds 1001 1002 1003 --split test (unchanged): Phase 0
#    checkpoints of results/phase3_checkpoints.tsv (sha256 checked), configs/unet.yaml, eval mode, sigmoid, no TTA;
#    float32 probs.npz, packbits masks.npz, per_case.csv, reconcile_phase0.csv; start / end rows appended to
#    results/phase2_test_eval_log.tsv. Seeds already done are skipped.
# 2. src/phase5_part2/reconcile_unet.py: exact rule of configs/phase5_part2/spec.json; exit 1 if not met.
# Submitted by hand only.
#SBATCH --job-name=p5p2_unet_infer
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 5 Part 2 U-Net test inference and reconciliation"
python -c 'import sys; sys.path.insert(0, "src"); from phase2 import lock; lock.require_open(what="slurm/phase5_part2_unet_infer.slurm")' \
  || { echo "Phase 2 lock closed: nothing was run"; exit 5; }
echo "HEAD $(git -C $REPO rev-parse HEAD)"
lgms_notify_start
python src/phase3/infer_img.py --arch unet --dataset QaTa-COV19 --seeds 1001 1002 1003 --split test || exit 1
python src/phase5_part2/reconcile_unet.py || exit 1
NOTIFY_RESULT="U-Net test inference done; reconciliation rule met"
echo "done $(date)"
