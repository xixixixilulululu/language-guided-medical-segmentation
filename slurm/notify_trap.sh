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

_LGMS_NOTIFY=/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation/src/notify.sh
_lgms_cancelled=0

_lgms_on_term() {
  _lgms_cancelled=1
  eval "${LGMS_CLEANUP:-true}"
  "$_LGMS_NOTIFY" "${JOB_LABEL:-${SLURM_JOB_NAME:-job}} CANCELLED (job ${SLURM_JOB_ID:-?})" "Slurm"
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
