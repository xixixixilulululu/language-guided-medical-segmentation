#!/bin/bash
# Task 0.2/0.3 for nnU-Net:
#   a) plan + preprocess (2d) on node-local disk, then store one tar per dataset on ~/datasets/nnunet/
#      (the raw datasets are exported beforehand on the login node: python src/prepare_nnunet.py);
#   b) quick check steps 1-4;
#   c) step 5 = 3 full-size epochs (250 iterations) per dataset with nnUNetTrainerLGMS_quick, then the
#      snapshot selection + test prediction pipeline on those 3 snapshots (timing for the schedule).
#SBATCH --job-name=qc_nnunet
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --time=12:00:00
#SBATCH --output=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/runs/slurm_logs/%x-%j.out
set -euo pipefail
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
JOB_LABEL="Quick check nnU-Net"
source $REPO/slurm/nnunet_local.sh
Q=runs/quick_check
NAMES="Dataset501_QaTaCOV19 Dataset502_MosMedDataPlus"

# a) preprocess locally (skipped when the tars exist already, e.g. on a resubmission)
if [ -f $NNUNET_PRE_TAR/Dataset502_MosMedDataPlus.tar ]; then
  for n in $NAMES; do stage_dataset $n; done
else
  # splits_final.json is copied in first so that the trainer finds it
  for n in $NAMES; do mkdir -p $nnUNet_preprocessed/$n; cp $NNUNET_PRE_NFS/$n/splits_final.json $nnUNet_preprocessed/$n/; done
  SECONDS=0
  nnUNetv2_plan_and_preprocess -d 501 502 -c 2d --verify_dataset_integrity -np 16
  echo "plan+preprocess: $SECONDS s"
  mkdir -p $NNUNET_PRE_TAR
  for n in $NAMES; do
    tar -cf $NNUNET_PRE_TAR/$n.tar -C $nnUNet_preprocessed $n
    cp $nnUNet_preprocessed/$n/*.json $NNUNET_PRE_NFS/$n/          # plans, fingerprint, dataset.json, splits
  done
fi
ls -la $NNUNET_PRE_TAR
python -c "
import json,os
for n in '$NAMES'.split():
    s=json.load(open(os.path.join(os.environ['nnUNet_preprocessed'],n,'splits_final.json')))
    print(n, 'folds', len(s), 'train', len(s[0]['train']), 'val', len(s[0]['val']))"

declare -A ID=( [QaTa-COV19]=501 [MosMedData+]=502 )
declare -A NAME=( [QaTa-COV19]=Dataset501_QaTaCOV19 [MosMedData+]=Dataset502_MosMedDataPlus )
export nnUNet_results=$REPO/$Q/nnunet_results
export LGMS_SEED=1001
for ds in QaTa-COV19 MosMedData+; do
  python src/quick_check.py --method nnunet --dataset $ds --out $Q/steps1-4/nnunet_$ds.json   # b)
  nnUNetv2_train ${ID[$ds]} 2d 0 -tr nnUNetTrainerLGMS_quick                                   # c)
  python src/nnunet_select.py --dataset $ds \
         --model-dir $nnUNet_results/${NAME[$ds]}/nnUNetTrainerLGMS_quick__nnUNetPlans__2d \
         --out $Q/subset/nnunet_$ds
done
echo "done $(date)"
