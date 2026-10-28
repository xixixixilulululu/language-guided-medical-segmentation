#!/bin/bash
# DRAFT (2026-10-28) - not to be submitted before the freeze (reports/PHASE3_PROTOCOL_FREEZE.md, section 8).
# Phase 3: TEST evaluation with the frozen gates. Nothing is fitted, calibrated or selected here.
#   1. IMG: nnU-Net test inference (Phase 0 checkpoints of results/phase3_checkpoints.tsv, seeds 1001-1003, both
#      datasets; imagesTs), float32 probabilities, masks and per-case scores (src/phase3/infer_img.py --split test)
#   2. export of the IMG per-case scores, features F1-F5 from the frozen definitions (P_lang of every Phase 2 test
#      pass, amendment A1), the frozen gates on every row, Task 3.5 tables and Task 3.7 statistics
# Requires (checked below): Phase 2 lock open; the Phase 2 test benchmark finished; results/phase2_test_per_case.csv.gz
# from `python src/phase2/analyze_phase2.py --split test` (login node, by hand, after the benchmark job).
# Test runs are logged (results/phase2_test_eval_log.tsv). Submitted by hand.
#SBATCH --job-name=p3_test
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 3 test evaluation (frozen gates)"
# 1. lock (first: nothing else runs when it is closed)
python -c 'import sys; sys.path.insert(0, "src"); from phase2 import lock; lock.require_open(what="slurm/phase3_test.slurm")' \
  || { echo "Phase 2 lock closed: nothing was run"; exit 5; }
# 2. Phase 2 test inputs present
[ -f results/phase2_test_per_case.csv.gz ] || { echo "results/phase2_test_per_case.csv.gz missing: run analyze_phase2.py --split test"; exit 1; }
[ -f results/phase2_test_c0_reconciliation.csv ] || { echo "test C0 reconciliation missing (slurm/phase2_test_benchmark.slurm)"; exit 1; }
echo "lock open, Phase 2 test outputs present"
lgms_notify_start
FAILED=()
for DS in QaTa-COV19 MosMedData+; do
  for S in 1001 1002 1003; do                          # nnU-Net: one results tree per seed
    export nnUNet_results=$REPO/runs/nnunet/nnUNet_results_seed$S
    echo "=== $(date +%T) IMG nnunet $DS seed $S (test)"
    python src/phase3/infer_img.py --arch nnunet --dataset $DS --seeds $S --split test || FAILED+=("nnunet/$DS/$S")
  done
done
echo "failed: ${#FAILED[@]} ${FAILED[*]:-}"
[ ${#FAILED[@]} -eq 0 ] || exit 1
for STEP in export_per_case features evaluate_gates stats_phase3; do
  echo "=== $(date +%T) $STEP --split test"
  python src/phase3/$STEP.py --split test || { echo "$STEP failed"; exit 1; }
done
NOTIFY_RESULT="Phase 3 test: IMG 6 runs, features, gates, Task 3.5 tables and Task 3.7 statistics done"
echo "done $(date)"
