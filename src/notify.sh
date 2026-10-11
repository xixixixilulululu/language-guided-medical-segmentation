#!/bin/bash
# Send a short phone notification through ntfy (https://ntfy.sh).
#
# Usage:  src/notify.sh "<message>" [title]
#   e.g.  src/notify.sh "U-Net QaTa seed1001 finished (job 1234), val Dice 0.79" "Slurm"
#
# The channel (topic) name is read from ~/.ntfy_topic, which is never committed to git.
# Messages contain only a task name, a status and a short result -- never data, file contents,
# paths or credentials. A failed notification never fails the caller (always exits 0).
# LGMS_NOTIFY_DRYRUN=1 prints the message instead of sending it (for testing).

topic_file="$HOME/.ntfy_topic"
[ -r "$topic_file" ] || { echo "notify.sh: $topic_file not found, no notification sent" >&2; exit 0; }
topic=$(tr -d '[:space:]' < "$topic_file")
msg=${1:-"(no message)"}
title=${2:-"lgms"}

if [ "${LGMS_NOTIFY_DRYRUN:-0}" = 1 ]; then echo "[dry-run ntfy] ${title}: ${msg}"; exit 0; fi
# -f: fail on HTTP errors; short timeouts so a missing network never blocks a job or a hook
curl -fsS --connect-timeout 5 --max-time 15 \
     -H "Title: ${title}" \
     -d "${msg}" "https://ntfy.sh/${topic}" > /dev/null \
  || echo "notify.sh: sending failed (no network?)" >&2
exit 0
