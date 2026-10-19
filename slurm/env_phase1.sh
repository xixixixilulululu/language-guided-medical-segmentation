# Common environment for every Phase 1 Slurm job (sourced by the Phase 1 .slurm files).
# 1. slurm/env.sh: ntfy notification trap, CUDA module, conda env lgms, REPO, LGMS_COMMON_ROOT, cd "$REPO".
# 2. The overlay venv ~/envs/lgms_p1 (created with --system-site-packages on top of lgms: same torch
#    2.14.0+cu130; adds transformers 4.57.6, pytorch_lightning 1.9.0, torchmetrics 0.10.3, monai 1.6.1,
#    clip, thop). lgms itself is not modified (decision D2, 2026-10-18).
source /mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/slurm/env.sh
source "$HOME/envs/lgms_p1/bin/activate"
# Data lock (2026-10-19): Phase 1 jobs read a view of common224 that links train and val only. The test
# split is linked into it only after the Phase 1 analysis code is fixed and pushed; until then a driver that
# reaches its test evaluation stops with exit code 4 (training and the selected checkpoint are saved).
export LGMS_COMMON_ROOT=$HOME/datasets/phase1/common224_trainval
export HF_HUB_OFFLINE=1            # pretrained weights come from the local HF cache (downloaded beforehand)
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -c "import torch, sys; assert torch.__version__ == '2.14.0+cu130', torch.__version__; print('python', sys.executable, 'torch', torch.__version__)"

# Called when a training driver exits with code 3: the wall-clock guard ended this segment before the
# 12 h limit and the run is not finished. Sends the "submit the next segment by hand" notification and
# suppresses the generic "finished" message of notify_trap.sh. Nothing is ever submitted from a job.
lgms_segment_ended() {
  "$_LGMS_NOTIFY" "$(_lgms_label): segment ended by the wall-clock guard (job ${SLURM_JOB_ID:-?})${1:+, $1}; the run is not finished - submit the next segment by hand: slurm/submit_next_p1.sh" "👉 RUN: slurm/submit_next_p1.sh" 4
  _lgms_cancelled=1   # notify_trap.sh: skip its own exit message (this one replaces it)
}
