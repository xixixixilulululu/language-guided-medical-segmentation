#!/bin/bash
# One-off repair of job 214 (nnU-Net QaTa-COV19 seed 1001): one re-prediction of the test set with the
# already selected checkpoint to complete the predicted test masks (src/analysis/repredict_214_masks.py).
# Run exactly once (decision of 2026-10-12); no training, no re-scoring, no re-selection.
#SBATCH --job-name=repredict-214
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="Re-prediction for job 214 masks"
python src/analysis/repredict_214_masks.py
R=runs/nnunet/QaTa-COV19/seed1001/repredict_214/result.json
NOTIFY_RESULT="check1 mismatches $(json_value $R check1_mismatching_cases 0), check2 mismatches $(json_value $R check2_mismatching_cases 0), copied: $(python3 -c "import json;print(json.load(open('$R'))['copied_896_masks'])"), sha256 unchanged: $(python3 -c "import json;print(json.load(open('$R'))['sha256_after_unchanged'])")"
