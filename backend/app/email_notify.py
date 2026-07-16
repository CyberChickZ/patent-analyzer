"""Email notification — send .md report after pipeline completion.

Config via env vars:
  SMTP_HOST     (default: smtp.gmail.com)
  SMTP_PORT     (default: 587)
  SMTP_USER     sender email address
  SMTP_PASSWORD Gmail app password (or SMTP password)
  NOTIFY_EMAIL  default recipient(s), comma-separated

Per-job override: pass notify_email in the API request.
If neither SMTP_USER nor SMTP_PASSWORD is set, sending is silently skipped.
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
DEFAULT_NOTIFY = os.getenv("NOTIFY_EMAIL", "")


def _configured() -> bool:
    return bool(SMTP_USER and SMTP_PASSWORD)


def send_report(
    to: str,
    subject: str,
    md_body: str,
    job_id: str = "",
) -> str | None:
    """Send the markdown report as an email. Returns None on success, error string on failure."""
    if not _configured():
        return "SMTP not configured (SMTP_USER / SMTP_PASSWORD missing)"
    if not to:
        return "no recipient"

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    if not recipients:
        return "no valid recipient"

    msg = MIMEMultipart("alternative")
    msg["From"] = SMTP_USER
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject

    msg.attach(MIMEText(md_body, "plain", "utf-8"))

    try:
        import markdown
        html_body = markdown.markdown(md_body, extensions=["tables", "fenced_code"])
        msg.attach(MIMEText(html_body, "html", "utf-8"))
    except ImportError:
        pass

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, recipients, msg.as_string())
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"
