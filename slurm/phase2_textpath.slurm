#!/bin/bash
# Phase 2, first job: text-path check on the VALIDATION split (PHASE2_PLAN Task 2.3 "First"), all 18 checkpoints.
# For each dataset x method (one process each, src/phase2/infer.py --stage textpath):
#   C0 on the full validation split + reconciliation with the archived Phase 1 validation evaluation,
#   a repeated C0 pass (numerical-noise reference), C1 and C6 (first donor) on a fixed subset of 64 cases,
#   and the effective-input audit of every validation condition (LViT-T: the words actually in its window).
# No test data is read (the Phase 2 lock stays closed). Submitted by hand only (slurm/submit_next_p1.sh).
#SBATCH --job-name=p2_textpath
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 2 text-path check (validation)"
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
for DS in MosMedData+ QaTa-COV19; do
  for M in lvit reclmis ddcmd; do
    echo "=== $(date +%T) $M $DS"
    python src/phase2/infer.py --stage textpath --method $M --dataset $DS || FAILED+=("$M/$DS")
  done
done
echo "failed: ${#FAILED[@]} ${FAILED[*]:-}"
[ ${#FAILED[@]} -eq 0 ] || exit 1
NOTIFY_RESULT="18 checkpoints done; see runs/phase2/val/*/*/seed*/textpath/summary.json"
echo "done $(date)"
