#!/bin/bash
# Phase 3, Task 3.1A: inference-only reruns on the VALIDATION split (no training, no test data).
#   1. IMG: U-Net and nnU-Net (Phase 0 checkpoints, seeds 1001-1003, both datasets): float32 probabilities, scored
#      masks and per-case scores in one pass (src/phase3/infer_img.py); reconciliation with the Phase 0 archive.
#   2. LANG: LViT-T, RecLMIS, DD-CMD, conditions C2, C3a, C3b, C3c (no probabilities in Phase 2): float32
#      probabilities, masks, scores (src/phase3/infer_lang.py); pixel differences to the Phase 2 masks recorded.
# Resumable: finished seeds / conditions are skipped. Submitted by hand only.
#SBATCH --job-name=p3_infer_val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 3 validation inference reruns (IMG float32 + LANG C2/C3)"
# The Phase 1 drivers are imported unchanged: their frozen records must still hold (and the batch-24 config).
for F in configs/phase1_frozen/lvit configs/phase1_frozen/reclmis configs/phase1_frozen/ddcmd; do
  sed "s#\$HOME#$HOME#" $F.sha256 | sha256sum -c --quiet - || { echo "frozen files changed: $F.sha256"; exit 1; }
  read -r REPO_DIR REPO_COMMIT < $F.commit
  [ "$(git -C $REPO_DIR rev-parse HEAD)" = "$REPO_COMMIT" ] || { echo "$REPO_DIR is not at $REPO_COMMIT"; exit 1; }
done
sha256sum -c --quiet configs/phase1_frozen/lvit_b24.sha256 || { echo "configs/lvit_sensitivity_b24.yaml changed"; exit 1; }
echo "frozen Phase 1 drivers verified"
lgms_notify_start
FAILED=()
for DS in QaTa-COV19 MosMedData+; do
  echo "=== $(date +%T) IMG unet $DS"
  python src/phase3/infer_img.py --arch unet --dataset $DS || FAILED+=("unet/$DS")
  for S in 1001 1002 1003; do                          # nnU-Net: one results tree per seed
    export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed$S
    echo "=== $(date +%T) IMG nnunet $DS seed $S"
    python src/phase3/infer_img.py --arch nnunet --dataset $DS --seeds $S || FAILED+=("nnunet/$DS/$S")
  done
done
for DS in QaTa-COV19 MosMedData+; do
  for M in lvit reclmis ddcmd; do
    echo "=== $(date +%T) LANG $M $DS (C2, C3a, C3b, C3c)"
    python src/phase3/infer_lang.py --method $M --dataset $DS || FAILED+=("$M/$DS")
  done
done
echo "failed: ${#FAILED[@]} ${FAILED[*]:-}"
[ ${#FAILED[@]} -eq 0 ] || exit 1
NOTIFY_RESULT="IMG 12 runs + LANG C2/C3 for 18 checkpoints done (runs/phase3/val)"
echo "done $(date)"
