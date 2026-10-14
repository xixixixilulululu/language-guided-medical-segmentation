#!/bin/bash
# Send a short phone notification through ntfy (https://ntfy.sh).
#
# Usage:  src/notify.sh "<message>" [title] [priority]
#   e.g.  src/notify.sh "U-Net QaTa seed1001 finished (job 1234), val Dice 0.79" "Slurm"
#   priority: ntfy "Priority" header, 1 (min) .. 5 (max); default 3. The project uses 4 (high) for messages
#   that need the user to act: job started (submit the next one), FAILED, ALERT, next nnU-Net segment to be
#   submitted by hand, U-Net / TransUNet stopped at the 12 h limit, cancelled before the time limit.
#
# The channel (topic) name is read from ~/.ntfy_topic, which is never committed to git.
# Messages contain only a task name, a status and a short result -- never data, file contents,
# paths or credentials. A failed notification never fails the caller (always exits 0).
# LGMS_NOTIFY_DRYRUN=1 prints the message instead of sending it (for testing).
#
# E-mail channel (since 2026-10-14) for messages the user must act on: priority >= 4 and a title that is not
# "✅ No action needed" (i.e. "👉 RUN ..." and "🆘 ASK CLAUDE ..."). Settings: ~/.config/lgms/push.env (never
# printed, never committed).
#   1. SMTP with the user's own account (QQ Mail, smtp.qq.com:465) via src/send_email.py, if SMTP_USER and
#      SMTP_PASS are set. (The cluster's own Slurm mail cannot be used: /bin/mail and sendmail are missing.)
#   2. otherwise ntfy e-mail forwarding with an account token (NTFY_TOKEN), sent to the private topic
#      "<topic>-mail"; ntfy.sh refuses anonymous e-mail (code 40053).
#   3. otherwise the e-mail is skipped with one log line.
# The e-mail is always sent after, and separately from, the phone push: its failure never affects the push
# or the job (only one line is logged).

topic_file="$HOME/.ntfy_topic"
[ -r "$topic_file" ] || { echo "notify.sh: $topic_file not found, no notification sent" >&2; exit 0; }
topic=$(tr -d '[:space:]' < "$topic_file")
msg=${1:-"(no message)"}
title=${2:-"lgms"}
priority=${3:-3}
case "$priority" in [1-5]) ;; *) priority=3 ;; esac   # only valid ntfy priorities

if [ "${LGMS_NOTIFY_DRYRUN:-0}" = 1 ]; then
  mail_note=""; [ "$priority" -ge 4 ] && [[ "$title" != "✅"* ]] && mail_note=" (+ e-mail)"
  echo "[dry-run ntfy] (priority ${priority})${mail_note} ${title}: ${msg}"; exit 0
fi
# -f: fail on HTTP errors; short timeouts so a missing network never blocks a job or a hook
curl -fsS --connect-timeout 5 --max-time 15 \
     -H "Title: ${title}" \
     -H "Priority: ${priority}" \
     -d "${msg}" "https://ntfy.sh/${topic}" > /dev/null \
  || echo "notify.sh: sending failed (no network?)" >&2

# E-mail channel: only messages that need an action (priority >= 4, not "✅ No action needed")
if [ "$priority" -ge 4 ] && [[ "$title" != "✅"* ]] && [ -r "$HOME/.config/lgms/push.env" ]; then
  _env() { sed -n "s/^[[:space:]]*$1[[:space:]]*=[[:space:]]*//p" "$HOME/.config/lgms/push.env" | head -1 | tr -d "\"' \r\t"; }
  if [ -n "$(_env SMTP_USER)" ] && [ -n "$(_env SMTP_PASS)" ]; then
    timeout 40 python3 "$(dirname "$0")/send_email.py" "$title" "$msg" \
      || echo "notify.sh: e-mail (SMTP) failed; the phone push is not affected" >&2
  elif [ -n "$(_env NTFY_TOKEN)" ] && [ -n "$(_env NOTIFY_EMAIL)" ]; then
    curl -fsS --connect-timeout 5 --max-time 20 \
         -H "Authorization: Bearer $(_env NTFY_TOKEN)" \
         -H "Title: ${title}" \
         -H "Priority: ${priority}" \
         -H "Email: $(_env NOTIFY_EMAIL)" \
         -d "${msg}" "https://ntfy.sh/${topic}-mail" > /dev/null 2>&1 \
      || echo "notify.sh: e-mail (ntfy) failed; the phone push is not affected" >&2
  else
    echo "notify.sh: e-mail channel not configured (SMTP_USER/SMTP_PASS or NTFY_TOKEN in ~/.config/lgms/push.env)" >&2
  fi
fi
exit 0
