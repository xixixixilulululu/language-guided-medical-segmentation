# Phone notifications for every Slurm job (sourced first by slurm/env.sh).
#
# Cluster fair-use policy (2026-10-13): at most 1 running + 1 queued job per user, 12 h per job, jobs are
# submitted by hand. Therefore every notification ends with the hint to submit the next job by hand.
#
#   start     "<label> started (job <id>). <hint>"                       -- lgms_notify_start, called by the job
#   success   "<label> finished (job <id>)[, <NOTIFY_RESULT>]. <hint>"
#   failure   "<label> FAILED (job <id>, exit <code>). <hint>"
#   time limit:
#     nnU-Net segment (LGMS_SEGMENTED=1): "segment ended at the 12 h time limit ...; the run is not
#                     complete - submit the next segment by hand. <hint>"
#     other jobs:     "STOPPED at the 12 h time limit[, last completed epoch N, best validation Dice at
#                     epoch M]; not handled automatically - needs your decision. <hint>"
#   cancelled before the time limit (scancel): "<label> CANCELLED before its time limit (job <id>). <hint>"
# A job script may set: JOB_LABEL (task name), NOTIFY_RESULT (short result), LGMS_CLEANUP (command run at
# exit), LGMS_SEGMENTED=1 (nnU-Net segment), LGMS_EPOCHS_CSV (U-Net / TransUNet epochs.csv, used by the
# time-limit message). Nothing here ever submits a job.

_LGMS_NOTIFY=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/src/notify.sh
_LGMS_HINT="You can run slurm/submit_next.sh to submit the next job."
_lgms_cancelled=0

_lgms_label() { echo "${JOB_LABEL:-${SLURM_JOB_NAME:-job}}"; }

# Called by the job script once JOB_LABEL is set; optional argument = extra information.
lgms_notify_start() {
  "$_LGMS_NOTIFY" "$(_lgms_label) started (job ${SLURM_JOB_ID:-?})${1:+, $1}. $_LGMS_HINT" "Slurm"
}

# True if SIGTERM came from the job's time limit (job end time reached, within 3 min), not from scancel.
_lgms_at_time_limit() {
  local end
  end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null) || return 1
  [ -n "$end" ] && [ "$end" != "N/A" ] || return 1
  [ "$(date +%s)" -ge $(( $(date -d "$end" +%s) - 180 )) ]
}

# "last completed epoch N, best validation Dice at epoch M (best_model.pt)" from an epochs.csv
# (columns: epoch, lr, train_loss, val_dice, best_val_dice, best_epoch, ...)
_lgms_epoch_info() {
  [ -n "${LGMS_EPOCHS_CSV:-}" ] && [ -f "$LGMS_EPOCHS_CSV" ] || return 0
  tail -n 1 "$LGMS_EPOCHS_CSV" | awk -F, '$1 ~ /^[0-9]+$/ {printf "last completed epoch %s, best validation Dice at epoch %s (best_model.pt)", $1, $6}'
}

_lgms_on_term() {
  _lgms_cancelled=1
  eval "${LGMS_CLEANUP:-true}"
  local label info; label=$(_lgms_label)
  if _lgms_at_time_limit; then
    if [ "${LGMS_SEGMENTED:-0}" = 1 ]; then
      "$_LGMS_NOTIFY" "$label: segment ended at the 12 h time limit (job ${SLURM_JOB_ID:-?}); the run is not complete - submit the next segment by hand. $_LGMS_HINT" "Slurm"
    else
      info=$(_lgms_epoch_info)
      "$_LGMS_NOTIFY" "$label STOPPED at the 12 h time limit (job ${SLURM_JOB_ID:-?})${info:+, $info}; not handled automatically - needs your decision. $_LGMS_HINT" "Slurm"
    fi
  else
    "$_LGMS_NOTIFY" "$label CANCELLED before its time limit (job ${SLURM_JOB_ID:-?}). $_LGMS_HINT" "Slurm"
  fi
  exit 143
}

_lgms_on_exit() {
  local rc=$?
  [ "$_lgms_cancelled" = 1 ] && return
  eval "${LGMS_CLEANUP:-true}"
  local label; label=$(_lgms_label)
  if [ "$rc" = 0 ]; then
    "$_LGMS_NOTIFY" "$label finished (job ${SLURM_JOB_ID:-?})${NOTIFY_RESULT:+, $NOTIFY_RESULT}. $_LGMS_HINT" "Slurm"
  else
    "$_LGMS_NOTIFY" "$label FAILED (job ${SLURM_JOB_ID:-?}, exit $rc). $_LGMS_HINT" "Slurm"
  fi
}

trap _lgms_on_term TERM INT
trap _lgms_on_exit EXIT

# json_value FILE KEY [DIGITS]: print a numeric value from a small JSON file (empty if missing)
json_value() {
  python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))[sys.argv[2]], int(sys.argv[3])))" \
    "$1" "$2" "${3:-4}" 2>/dev/null
}
