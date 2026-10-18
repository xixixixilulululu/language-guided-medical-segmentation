#!/bin/bash
# Phase 1 quick check (Milestone 1.2), all three methods x both datasets, in one job:
#   A. probes on the untrained model: pairing, forward pass, text gradient, text swap, RecLMIS losses
#   B. subset training (first 64 train / 32 val cases, 40 epochs, seed 1001; validation only, the test set
#      is never touched): loss decreases, validation Dice rises, valid masks; then the text-swap probe on
#      the trained model
#   C. resume test (MosMedData+, 32 train / 16 val cases): 6 epochs uninterrupted vs 3 + 3 epochs with a
#      segment end in between; epochs.csv of both are compared afterwards
#   D. timing: 2 full epochs on the full training and validation sets (no evaluation)
# A failed step is logged and the job continues; the job fails at the end if any step failed.
# Submitted by hand only (slurm/submit_next_p1.sh).
#SBATCH --job-name=p1_quick_check
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -uo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env_phase1.sh
JOB_LABEL="Phase 1 quick check"
Q=runs/phase1_quick_check
mkdir -p $Q
lgms_notify_start
FAILED=()
# step <name> <expected exit code> <command...>: run, log the exit code, continue on failure
step() {
  local name=$1 want=$2; shift 2
  echo "=== $(date +%T) $name"
  "$@"; local rc=$?
  echo "$name	$rc	$want" >> $Q/steps.tsv
  [ "$rc" = "$want" ] || { echo "!!! $name: exit $rc (expected $want)"; FAILED+=("$name"); }
}
: > $Q/steps.tsv

for M in lvit reclmis ddcmd; do
  for DS in QaTa-COV19 MosMedData+; do
    step "probe_init $M $DS" 0 python src/phase1/quick_check.py --method $M --dataset $DS --mode init \
         --out $Q/probe_init_${M}_$DS.json
  done
done

for M in lvit reclmis ddcmd; do
  for DS in QaTa-COV19 MosMedData+; do
    D=$Q/subset/${M}_$DS; rm -rf $D
    step "subset $M $DS" 0 python src/phase1/train_$M.py --dataset $DS --seed 1001 --subset-train 64 \
         --subset-val 32 --max-epochs 40 --eval-splits val --out $D
    CK=$D/best_model.pt; [ $M = ddcmd ] && CK=$D/checkpoints/best.ckpt
    step "probe_trained $M $DS" 0 python src/phase1/quick_check.py --method $M --dataset $DS --mode trained \
         --ckpt $CK --out $Q/probe_trained_${M}_$DS.json
  done
done

for M in lvit reclmis ddcmd; do
  R=$Q/resume/${M}; rm -rf $R; A="--dataset MosMedData+ --seed 1001 --subset-train 32 --subset-val 16 --max-epochs 6 --no-final-eval"
  step "resume_ref $M" 0 python src/phase1/train_$M.py $A --out $R/uninterrupted
  step "resume_seg1 $M" 3 python src/phase1/train_$M.py $A --stop-after-epochs 3 --out $R/segmented
  step "resume_seg2 $M" 0 python src/phase1/train_$M.py $A --out $R/segmented
done

for M in lvit reclmis ddcmd; do
  for DS in QaTa-COV19 MosMedData+; do
    D=$Q/timing/${M}_$DS; rm -rf $D
    step "timing $M $DS" 0 python src/phase1/train_$M.py --dataset $DS --seed 1001 --max-epochs 2 \
         --no-final-eval --out $D
  done
done

echo "failed steps: ${#FAILED[@]} ${FAILED[*]:-}"
NOTIFY_RESULT="${#FAILED[@]} failed steps; ask Claude to summarise"
[ ${#FAILED[@]} = 0 ] || exit 1
echo "done $(date)"
