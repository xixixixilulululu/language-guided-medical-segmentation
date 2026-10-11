# Stage nnU-Net preprocessed data on the compute node's local NVMe (sourced by nnU-Net .slurm files,
# after env.sh). Small-file I/O on the NFS-mounted /mnt/cv_data is very slow from compute nodes
# (~3 files/s while preprocessing), so nnU-Net reads its preprocessed cases from local disk.
# The canonical copy is one tar per dataset: ~/datasets/nnunet/preprocessed_tar/<DatasetName>.tar
export NNUNET_PRE_NFS=$HOME/datasets/nnunet/nnUNet_preprocessed     # plans / splits (small files)
export NNUNET_PRE_TAR=$HOME/datasets/nnunet/preprocessed_tar
export nnUNet_preprocessed=/tmp/${USER}_nnunet_${SLURM_JOB_ID:-local}/nnUNet_preprocessed
mkdir -p "$nnUNet_preprocessed"
LGMS_CLEANUP='rm -rf /tmp/${USER}_nnunet_${SLURM_JOB_ID:-local}'   # run at exit by slurm/notify_trap.sh

stage_dataset() {  # $1 = DatasetName: unpack the preprocessed tar to local disk
  tar -xf "$NNUNET_PRE_TAR/$1.tar" -C "$nnUNet_preprocessed"
  cp "$NNUNET_PRE_NFS/$1/splits_final.json" "$nnUNet_preprocessed/$1/"   # LViT train/val fold 0
  echo "staged $1: $(ls "$nnUNet_preprocessed/$1/nnUNetPlans_2d" | wc -l) files"
}
