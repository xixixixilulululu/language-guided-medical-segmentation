# Phone notification at the end of every Slurm job (sourced first by slurm/env.sh).
#
# Sends one ntfy message (src/notify.sh) when the job script ends:
#   success   "<label> finished (job <id>)[, <NOTIFY_RESULT>]"
#   failure   "<label> FAILED (job <id>, exit <code>)"
#   cancelled "<label> CANCELLED (job <id>)"   (scancel / time limit: Slurm sends SIGTERM first)
# A job script may set, at any point before it ends:
#   JOB_LABEL      short task name, e.g. "U-Net QaTa-COV19 seed1001" (default: the Slurm job name)
#   NOTIFY_RESULT  short result, e.g. "best val Dice 0.7912"
#   LGMS_CLEANUP   a command run at exit before the notification (e.g. removing node-local data)
#   LGMS_SEGMENT_NEXT  id of the queued next segment of a segmented run (slurm/train_nnunet.slurm). A segment
#                  stopped by its 12 h time limit is reported as "segment ended ... continues in job <id>".
#                  A segment stopped by scancel BEFORE its time limit withdraws the queued next segment and
#                  asks the user to decide (no automatic continuation after someone cancelled a job).

_LGMS_NOTIFY=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/src/notify.sh
_lgms_cancelled=0

# True if SIGTERM came from the job's time limit (job end time reached, within 3 min), not from scancel.
_lgms_at_time_limit() {
  local end
  end=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %e 2>/dev/null) || return 1
  [ -n "$end" ] && [ "$end" != "N/A" ] || return 1
  [ "$(date +%s)" -ge $(( $(date -d "$end" +%s) - 180 )) ]
}

_lgms_on_term() {
  _lgms_cancelled=1
  eval "${LGMS_CLEANUP:-true}"
  local label="${JOB_LABEL:-${SLURM_JOB_NAME:-job}}"
  if [ -n "${LGMS_SEGMENT_NEXT:-}" ] && _lgms_at_time_limit; then
    "$_LGMS_NOTIFY" "$label: segment ended at the 12 h time limit (job ${SLURM_JOB_ID:-?}); training continues in job $LGMS_SEGMENT_NEXT" "Slurm"
  elif [ -n "${LGMS_SEGMENT_NEXT:-}" ]; then
    # stopped by someone before the time limit: do NOT continue automatically -- the user decides
    printf "%s\tSELF_CANCELLED\n" "$LGMS_SEGMENT_NEXT" >> "${LGMS_WATCH_FILE:-/dev/null}"
    scancel "$LGMS_SEGMENT_NEXT" 2>/dev/null || true
    "$_LGMS_NOTIFY" "$label CANCELLED before its time limit (job ${SLURM_JOB_ID:-?}); next segment $LGMS_SEGMENT_NEXT withdrawn, nothing resubmitted - waiting for your decision" "Slurm"
  else
    "$_LGMS_NOTIFY" "$label CANCELLED (job ${SLURM_JOB_ID:-?})" "Slurm"
  fi
  exit 143
}

_lgms_on_exit() {
  local rc=$?
  [ "$_lgms_cancelled" = 1 ] && return
  eval "${LGMS_CLEANUP:-true}"
  local label="${JOB_LABEL:-${SLURM_JOB_NAME:-job}}"
  if [ "$rc" = 0 ]; then
    "$_LGMS_NOTIFY" "$label finished (job ${SLURM_JOB_ID:-?})${NOTIFY_RESULT:+, $NOTIFY_RESULT}" "Slurm"
  else
    "$_LGMS_NOTIFY" "$label FAILED (job ${SLURM_JOB_ID:-?}, exit $rc)" "Slurm"
  fi
}

trap _lgms_on_term TERM INT
trap _lgms_on_exit EXIT

# json_value FILE KEY [DIGITS]: print a numeric value from a small JSON file (empty if missing)
json_value() {
  python3 -c "import json,sys; print(round(json.load(open(sys.argv[1]))[sys.argv[2]], int(sys.argv[3])))" \
    "$1" "$2" "${3:-4}" 2>/dev/null
}
