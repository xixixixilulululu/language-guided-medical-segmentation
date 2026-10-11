#!/bin/bash
# Claude Code hook -> phone notification (ntfy, via src/notify.sh). Configured in .claude/settings.json.
#
#   claude_notify_hook.sh stop          Claude finished its turn (task done, or waiting for confirmation).
#       Sends the one-line status Claude wrote to ~/.cache/lgms/claude_status before stopping
#       (e.g. "Waiting for your confirmation: quick checks done"), then deletes that file;
#       without a status file a generic line is sent.
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
"$here/notify.sh" "$msg" "Claude Code"
exit 0
