"""Send one notification e-mail through the user's own SMTP account (QQ Mail: smtp.qq.com, SSL port 465).

Called by src/notify.sh for messages that need an action (priority >= 4, not "✅ No action needed").
Only the Python standard library is used, so it runs with any python3 on the login or compute nodes.

Configuration: ~/.config/lgms/push.env (mode 600, never committed), KEY=value lines:
    NOTIFY_EMAIL  recipient address
    SMTP_USER     sender account, e.g. the QQ Mail address (QQ requires From == login account)
    SMTP_PASS     the account's SMTP authorization code (QQ Mail: Settings -> Account -> POP3/IMAP/SMTP ->
                  enable SMTP -> generate an authorization code); NOT the normal login password
    SMTP_HOST     optional, default smtp.qq.com
    SMTP_PORT     optional, default 465 (implicit SSL)
Credentials are read here from the file -- never passed on the command line or through the environment,
so they appear neither in `ps` nor in any log. Nothing secret is ever printed.

Usage:  python3 src/send_email.py "<title>" "<message>"
Exit status: 0 sent, 2 not configured, 1 sending failed. src/notify.sh ignores the status apart from logging.
"""
import os
import smtplib
import ssl
import sys
from email.message import EmailMessage

CONFIG = os.path.expanduser("~/.config/lgms/push.env")


def read_config(path=CONFIG):
    """KEY=value lines; quotes and surrounding blanks are stripped. Missing file -> {}."""
    cfg = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                cfg[k.strip()] = v.strip().strip("\"'")
    except OSError:
        pass
    return cfg


def main():
    if len(sys.argv) != 3:
        print("usage: send_email.py <title> <message>", file=sys.stderr)
        return 1
    title, body = sys.argv[1], sys.argv[2]
    cfg = read_config()
    to, user, password = cfg.get("NOTIFY_EMAIL"), cfg.get("SMTP_USER"), cfg.get("SMTP_PASS")
    if not (to and user and password):
        print("send_email: not configured (NOTIFY_EMAIL, SMTP_USER, SMTP_PASS in ~/.config/lgms/push.env)",
              file=sys.stderr)
        return 2
    host, port = cfg.get("SMTP_HOST", "smtp.qq.com"), int(cfg.get("SMTP_PORT", "465"))

    msg = EmailMessage()
    msg["Subject"] = f"[lgms] {title}"       # UTF-8 subject (emoji allowed)
    msg["From"] = user                        # QQ Mail requires From == the authenticated account
    msg["To"] = to
    msg.set_content(body + "\n\n(sent by src/notify.sh, language-guided-medical-segmentation)\n")
    try:
        with smtplib.SMTP_SSL(host, port, timeout=20, context=ssl.create_default_context()) as s:
            s.login(user, password)
            s.send_message(msg)
    except Exception as e:  # noqa: BLE001 - report the type only; messages may echo the account name
        print(f"send_email: sending failed ({type(e).__name__})", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
