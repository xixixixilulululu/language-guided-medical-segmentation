# Common environment for every Phase 0 Slurm job (sourced by the .slurm files).
# Phone notification on success / failure / cancellation (ntfy), installed first:
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/notify_trap.sh
module load cuda/13.0
module load miniconda3/latest
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$HOME/envs/lgms"

export REPO=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation
export LGMS_COMMON_ROOT=$HOME/datasets/common224            # Task 0.2 output
# nnU-Net folders: raw / preprocessed data live with the datasets, results under runs/ (never in git)
export nnUNet_raw=$HOME/datasets/nnunet/nnUNet_raw
export nnUNet_preprocessed=$HOME/datasets/nnunet/nnUNet_preprocessed
export nnUNet_extTrainer=$REPO/src/nnunet_ext                # nnUNetTrainerLGMS
export nnUNet_n_proc_DA=12                                   # nnU-Net's default for unknown hosts (20 CPUs/node)
export PYTHONUNBUFFERED=1
export LGMS_WATCH_FILE=$REPO/runs/phase0_watch_jobs.tsv   # jobs watched by src/queue_watchdog.sh
cd "$REPO"
echo "host $(hostname) | job ${SLURM_JOB_ID:-none} | $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
