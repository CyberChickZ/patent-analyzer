"""Email notification — send the .md report when the pipeline finishes.

Config via env vars:
  SMTP_HOST     (default: smtp.gmail.com)
  SMTP_PORT     (default: 587)
  SMTP_USER     sender address
  SMTP_PASSWORD app password
  NOTIFY_EMAIL  default recipient(s), comma-separated

Per-job override: pass notify_email in the API request.

Nothing here is ever skipped silently. With no credentials `send_report`
returns the reason as an error string, the report node records `email_failed`,
and the submit page turns the checkbox off — because "Email the report when it
finishes", offered and not delivered, is worse than not offering it. The old
docstring said sending was "silently skipped"; it was not true of the caller
and must never become true again.

Every value is read per call, not at import. A module-level os.getenv cannot
see credentials added afterwards, which is exactly the shape of bug that lost
this deployment its prompts and its counters (leader_deploy.md §G).
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def smtp_host() -> str:
    return os.getenv("SMTP_HOST", "smtp.gmail.com")


def smtp_port() -> int:
    try:
        return int(os.getenv("SMTP_PORT", "587"))
    except ValueError:
        return 587


def smtp_user() -> str:
    return os.getenv("SMTP_USER", "")


def smtp_password() -> str:
    return os.getenv("SMTP_PASSWORD", "")


def default_notify() -> str:
    return os.getenv("NOTIFY_EMAIL", "")


def configured() -> bool:
    """Whether this deployment can send mail at all. Public: the submit page
    asks before it offers the option."""
    return bool(smtp_user() and smtp_password())


NOT_CONFIGURED = "SMTP not configured on this server (SMTP_USER / SMTP_PASSWORD are not set)"


def plan_for(notify_email: str) -> dict:
    """What can be decided about the email before the report is written.

    The report file is the attachment, so it exists first and cannot carry the
    send result. It can carry this: whether a copy was asked for, and whether
    this server is able to send one at all. Those are the two facts that were
    missing when "Email the report when it finishes" was offered by a
    deployment with no credentials.
    """
    ok = configured()
    return {"requested": bool(notify_email), "to": notify_email or "",
            "enabled": ok, "reason": "" if ok else NOT_CONFIGURED}


def send_report(
    to: str,
    subject: str,
    md_body: str,
    job_id: str = "",
) -> str | None:
    """Send the markdown report as an email. Returns None on success, error string on failure."""
    if not configured():
        return NOT_CONFIGURED
    if not to:
        return "no recipient"

    recipients = [addr.strip() for addr in to.split(",") if addr.strip()]
    if not recipients:
        return "no valid recipient"

    msg = MIMEMultipart("alternative")
    msg["From"] = smtp_user()
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
        with smtplib.SMTP(smtp_host(), smtp_port(), timeout=30) as server:
            server.starttls()
            server.login(smtp_user(), smtp_password())
            server.sendmail(smtp_user(), recipients, msg.as_string())
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}"
