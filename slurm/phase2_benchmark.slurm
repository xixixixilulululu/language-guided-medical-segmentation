#!/bin/bash
# Phase 2: full VALIDATION benchmark (PHASE2_PLAN Task 2.3), all 18 checkpoints, conditions C1-C6 (C0 reused
# from the text-path job; PHASE2_PROTOCOL.md section 4.8). For each dataset x method (one process each,
# src/phase2/infer.py --stage benchmark --split val): the effective-input audit of the current manifest, then
# every condition pass on its eligible cases, three seeds; each pass must change the model input exactly where
# the text changes (else the job stops). Resumable: finished passes are skipped. Every pass is logged with the
# Slurm job id (passes.tsv). No test data is read (the Phase 2 lock stays closed).
#SBATCH --job-name=p2_benchmark_val
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 2 benchmark (validation)"
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
    python src/phase2/infer.py --stage benchmark --split val --method $M --dataset $DS || FAILED+=("$M/$DS")
  done
done
echo "failed: ${#FAILED[@]} ${FAILED[*]:-}"
[ ${#FAILED[@]} -eq 0 ] || exit 1
NOTIFY_RESULT="18 checkpoints, all validation condition passes done (runs/phase2/val)"
echo "done $(date)"
