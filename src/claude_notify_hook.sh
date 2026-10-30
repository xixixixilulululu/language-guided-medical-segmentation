#!/bin/bash
# Claude Code hook -> phone notification (ntfy, via src/notify.sh). Configured in .claude/settings.json.
#
#   claude_notify_hook.sh stop          Claude finished its turn (task done, or waiting for confirmation).
#       Sends the one-line status Claude wrote to ~/.cache/lgms/claude_status before stopping
#       (e.g. "Waiting for your confirmation: quick checks done"), then deletes that file;
#       without a status file a generic line is sent.
#       Since 2026-10-18 (user request) the same line is also sent by e-mail with the title
#       "✅ Claude is waiting for you" (src/send_email.py, after the phone push; stop event only).
#   claude_notify_hook.sh notification  Claude Code needs attention (permission prompt / idle input);
#       sends Claude Code's own notification text.
#
# The hook JSON arrives on stdin; only the short "message" field is used. Always exits 0.

here=$(cd "$(dirname "$0")" && pwd)
status_file="$HOME/.cache/lgms/claude_status"
payload=$(cat)

case "$1" in
  stop)
    if [ -s "$status_file" ]; then
      msg="Claude: $(head -c 200 "$status_file" | head -n 1)"
      rm -f "$status_file"
    else
      msg="Claude finished and is waiting for you"
    fi
    ;;
  notification)
    text=$(printf '%s' "$payload" | python3 -c \
      "import json,sys; print(json.load(sys.stdin).get('message','')[:200])" 2>/dev/null)
    msg="Claude needs your attention${text:+: $text}"
    ;;
  *)
    exit 0
    ;;
esac
"$here/notify.sh" "$msg" "✅ No action needed"   # title = action for the user (none)
# Stop event: also e-mail the status (sent after the push; 12 s cap keeps the hook inside its 30 s timeout;
# a failure is only logged). LGMS_NOTIFY_DRYRUN=1 prints instead of sending.
if [ "$1" = stop ]; then
  if [ "${LGMS_NOTIFY_DRYRUN:-0}" = 1 ]; then
    echo "[dry-run e-mail] ✅ Claude is waiting for you: ${msg}"
  else
    # one log line per attempt (time, exit status, error type; never addresses or credentials), since 2026-10-30
    err=$(timeout 12 python3 "$here/send_email.py" "✅ Claude is waiting for you" "$msg" 2>&1 > /dev/null)
    rc=$?
    printf '%s\tstop\texit=%s\t%s\t%s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$rc" "${err//$'\n'/ }" \
      "$(printf '%s' "$msg" | head -c 60)" >> "$HOME/.cache/lgms/email.log"
    [ "$rc" -eq 0 ] || echo "claude_notify_hook.sh: e-mail failed (exit $rc); the phone push is not affected" >&2
  fi
fi
exit 0
