"""What the cloud frontend does for somebody who has not signed in.

Everything here stops at the Google popup. Nothing in this script can sign in,
and it must not try: the account is Harry's, and AUTH_DISABLED is never going
to Cloud Run. So the checks are the ones that are answerable from outside the
gate:

  1. the page loads with no console error and no uncaught exception
  2. no /api/* request is fired before there is a token to send
  3. the Sign in button really opens accounts.google.com
  4. /api/* answers 401 quickly instead of hanging

Usage: backend/.venv/bin/python backend/evals/scratch_cloud_smoke.py [url]
"""

import json
import sys
import time

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "https://patent-analyzer-frontend-2mk262glgq-uw.a.run.app"


def main() -> None:
    out: dict = {"url": URL, "ts": time.strftime("%Y-%m-%d %H:%M:%S")}
    console: list[dict] = []
    pageerrors: list[str] = []
    requests: list[dict] = []
    responses: list[dict] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context()
        page = ctx.new_page()
        page.on("console", lambda m: console.append({"type": m.type, "text": m.text[:300]}))
        page.on("pageerror", lambda e: pageerrors.append(str(e)[:300]))
        page.on("request", lambda r: requests.append({"url": r.url, "method": r.method}))
        page.on("response", lambda r: responses.append({"url": r.url, "status": r.status}))

        t0 = time.time()
        resp = page.goto(URL, wait_until="networkidle", timeout=60000)
        out["load"] = {"status": resp.status if resp else None, "seconds": round(time.time() - t0, 2)}
        page.wait_for_timeout(4000)   # let the Firebase auth state settle

        out["console_errors"] = [c for c in console if c["type"] == "error"]
        out["console_warnings"] = [c for c in console if c["type"] == "warning"]
        out["page_errors"] = pageerrors
        api = [r for r in requests if "/api/" in r["url"]]
        out["api_requests_before_login"] = api
        out["api_responses_before_login"] = [r for r in responses if "/api/" in r["url"]]

        body = page.inner_text("body")
        # the button is found by its text, not by where it sits: the deployed
        # build is not always the one in the working tree
        btn = next((b for b in page.query_selector_all("button")
                    if "sign in" in (b.inner_text() or "").lower()), None)
        out["body_text"] = body[:800]
        out["body_has"] = {
            "sign_in_button": bool(btn),
            "sign_in_text": "Sign in" in body,
            "spinner_left_running": "Loading…" in body or "Loading..." in body,
            "please_sign_in_zh": "请先登录" in body,
        }

        # 3. the popup, and only as far as the popup
        popup_first = None
        popup_urls: list[str] = []
        popup_err = None
        dialogs: list[str] = []
        page.on("dialog", lambda d: (dialogs.append(d.message[:300]), d.dismiss()))
        if btn:
            try:
                with page.expect_popup(timeout=20000) as pop:
                    btn.click()
                popup = pop.value
                popup.on("framenavigated", lambda f: popup_urls.append(f.url[:200])
                         if f == popup.main_frame else None)
                popup.wait_for_load_state("domcontentloaded", timeout=20000)
                popup_first = popup.url
                # signInWithPopup opens Firebase's own /__/auth/handler first and
                # only then bounces to Google — the hop we are checking for is the
                # second one, and an unauthorized domain never gets there.
                try:
                    popup.wait_for_url("**accounts.google.com/**", timeout=20000)
                except Exception:
                    pass
                popup_urls.append(popup.url[:200])
                popup.close()
            except Exception as e:
                popup_err = f"{type(e).__name__}: {e}"[:300]
        seen = popup_urls + ([popup_first] if popup_first else [])
        out["popup"] = {"first_url": popup_first, "urls": popup_urls, "error": popup_err,
                        "reached_google": any("accounts.google.com" in u for u in seen),
                        "dialogs": dialogs}
        page.wait_for_timeout(1500)
        out["console_errors_after_click"] = [c for c in console if c["type"] == "error"]

        # 4. the proxy itself, with no token
        probes = {}
        for path in ("/api/config", "/api/jobs", "/api/quota", "/api/health", "/api/status/nope"):
            t = time.time()
            try:
                r = ctx.request.get(URL + path, timeout=30000)
                probes[path] = {"status": r.status, "seconds": round(time.time() - t, 2),
                                "body": r.text()[:120]}
            except Exception as e:
                probes[path] = {"error": f"{type(e).__name__}: {e}"[:200],
                                "seconds": round(time.time() - t, 2)}
        out["unauthenticated_api"] = probes

        browser.close()

    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
