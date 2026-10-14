#!/bin/bash
# RETIRED (2026-10-13): must not be started again - no background processes on the login node (CLAUDE.md).
# Queue watchdog for the Phase 0 jobs (runs on the login node; there is no cron on this cluster).
#
# Every 5 minutes it compares the watched jobs (runs/phase0_watch_jobs.tsv: job_id <TAB> label) with squeue:
#   * still in the queue                               -> fine;
#   * gone, and it ran (its runs/slurm_logs/*-<id>.out exists) -> ended normally; the job's own trap has
#     already sent its notification (finished / failed / time limit / cancelled);
#   * gone WITHOUT ever running (no log file) and not withdrawn by our own scripts (SELF_CANCELLED)
#     -> it was removed from the queue by someone else: send ONE phone notification listing all such jobs.
# It NEVER resubmits anything: the user decides (decision of 2026-10-13).
# It exits when no watched job is left in the queue. Heartbeat: runs/slurm_logs/watchdog.heartbeat.
#
# Start (detached, survives logout):  setsid nohup src/queue_watchdog.sh >> runs/slurm_logs/watchdog.log 2>&1 &
REPO=${LGMS_WATCHDOG_REPO:-/mnt/cv_data/users/hahaxixi/projects/language-guided-medical-segmentation}  # override: tests
INTERVAL=${LGMS_WATCHDOG_INTERVAL:-300}                                                          # seconds
WATCH=$REPO/runs/phase0_watch_jobs.tsv
LOGS=$REPO/runs/slurm_logs
REPORTED=$REPO/runs/phase0_watch_reported.txt     # removals already notified (one message per job)
touch "$REPORTED"
echo "$(date '+%F %T') watchdog started (pid $$)"

while true; do
  date '+%F %T' > "$LOGS/watchdog.heartbeat"
  queue=$(squeue -h -u "$USER" -o %i 2>/dev/null)
  if [ $? -ne 0 ]; then                    # controller not reachable: never treat that as a removal
    echo "$(date '+%F %T') squeue failed, skipping this check"; sleep "$INTERVAL"; continue
  fi
  removed=(); active=0
  while IFS=$'\t' read -r id label; do
    [ -z "$id" ] || [ "$label" = SELF_CANCELLED ] && continue
    grep -qx "$id" <<<"$queue" && { active=$((active + 1)); continue; }
    grep -q -P "^$id\tSELF_CANCELLED$" "$WATCH" && continue                   # withdrawn by our own scripts
    ls "$LOGS"/*-"$id".out >/dev/null 2>&1 && continue                       # it ran; its trap notified
    grep -qx "$id" "$REPORTED" && continue                                    # already reported
    removed+=("$id ($label)"); echo "$id" >> "$REPORTED"
  done < "$WATCH"
  if [ ${#removed[@]} -gt 0 ]; then
    msg="ALERT: ${#removed[@]} queued job(s) were removed from the queue without running: $(IFS='; '; echo "${removed[*]}"). Nothing resubmitted - waiting for your decision."
    echo "$(date '+%F %T') $msg"
    "$REPO/src/notify.sh" "$msg" "Slurm queue"
  fi
  if [ "$active" -eq 0 ]; then
    echo "$(date '+%F %T') no watched job left in the queue - watchdog exits"
    "$REPO/src/notify.sh" "Queue watchdog: no watched job left in the queue (all ended or removed)" "Slurm queue"
    exit 0
  fi
  sleep "$INTERVAL"
done
