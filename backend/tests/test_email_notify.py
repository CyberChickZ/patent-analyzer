import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).parent.parent))

from app import email_notify as en


@pytest.fixture(autouse=True)
def no_smtp(monkeypatch):
    for k in ("SMTP_USER", "SMTP_PASSWORD", "SMTP_HOST", "SMTP_PORT", "NOTIFY_EMAIL"):
        monkeypatch.delenv(k, raising=False)


def test_every_setting_is_read_per_call(monkeypatch):
    """A module-level os.getenv cannot see credentials added afterwards, which
    is the shape of bug that lost this deployment its prompts and its
    counters."""
    assert en.configured() is False
    monkeypatch.setenv("SMTP_USER", "a@b.c")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    assert en.configured() is True
    monkeypatch.setenv("SMTP_HOST", "mail.example.com")
    monkeypatch.setenv("SMTP_PORT", "2525")
    assert en.smtp_host() == "mail.example.com" and en.smtp_port() == 2525
    monkeypatch.setenv("SMTP_PORT", "not a number")
    assert en.smtp_port() == 587, "a bad port falls back rather than crashing the run"


def test_unconfigured_returns_the_reason_and_never_pretends_to_send():
    err = en.send_report("x@y.z", "subject", "# body")
    assert err == en.NOT_CONFIGURED
    assert "SMTP_USER" in err and "SMTP_PASSWORD" in err


def test_the_server_says_whether_it_can_send_mail(monkeypatch):
    import app.main as m
    c = TestClient(m.app)
    r = c.get("/config").json()
    assert r["email_enabled"] is False and r["email_from"] == ""
    assert "SMTP_USER" in r["email_reason"]

    monkeypatch.setenv("SMTP_USER", "reports@oregonstate.edu")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    r = c.get("/config").json()
    assert r["email_enabled"] is True and r["email_from"] == "reports@oregonstate.edu"
    assert r["email_reason"] == ""
    assert "pw" not in str(r), "the password must not leave the process"


def test_the_report_says_no_mail_is_coming_when_it_is_not():
    """The report file is written before the mail is sent, so it carries the
    half that is knowable first — and "requested but impossible" is knowable."""
    from patent_analyzer.report_sections import email_note_html, email_note_md
    plan = {"requested": True, "to": "h@oregonstate.edu", "enabled": False,
            "reason": en.NOT_CONFIGURED}
    h = email_note_html(plan)
    assert "No email was sent" in h and "h@oregonstate.edu" in h and "SMTP_USER" in h
    assert "The report itself is unaffected" in h
    assert "No email was sent" in "\n".join(email_note_md(plan))

    ok = {"requested": True, "to": "h@oregonstate.edu", "enabled": True, "reason": ""}
    assert "being emailed" in email_note_html(ok)
    assert email_note_html({"requested": False}) == "" and email_note_md(None) == []


def test_the_plan_is_decided_before_the_report_is_written(monkeypatch):
    """The report file is the attachment, so it exists first and cannot carry
    the send result. It can carry this: a copy was asked for, and this server
    cannot send one."""
    p = en.plan_for("h@oregonstate.edu")
    assert p == {"requested": True, "to": "h@oregonstate.edu",
                 "enabled": False, "reason": en.NOT_CONFIGURED}
    monkeypatch.setenv("SMTP_USER", "a@b.c")
    monkeypatch.setenv("SMTP_PASSWORD", "pw")
    assert en.plan_for("h@oregonstate.edu") == {
        "requested": True, "to": "h@oregonstate.edu", "enabled": True, "reason": ""}
    assert en.plan_for("")["requested"] is False


def test_the_node_records_email_failed_when_it_cannot_send():
    """The event is the contract: nothing about email is skipped quietly. The
    node raises it from `plan_for` BEFORE writing the report, so the reason is
    on the job even if the run dies later."""
    import inspect

    from nodes import report as rep
    src = inspect.getsource(rep)
    assert "email_notify.plan_for(" in src
    i = src.index("email_plan = email_notify.plan_for(")
    j = src.index("html = inject_html(")
    between = src[i:j]
    assert '_event("email_failed"' in between, "the failure must be recorded before the report is written"
    assert "email=email_plan" in src, "and the report must be told"
    assert '"email_status": email_status' in src, "and the job record must keep the outcome"
