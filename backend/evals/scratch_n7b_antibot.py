#!/usr/bin/env python3
"""N7b: does a real browser fingerprint get past the hosts N7 called a wall?

N7 concluded "the publisher sites refuse us" from httpx/urllib plus a swapped
User-Agent. That test cannot support that conclusion: Cloudflare's hard 403 is
usually a TLS/HTTP2 *fingerprint* verdict, and changing a header string does not
change the fingerprint. This script re-runs the same URLs with clients that do
change it.

Ladder, cheapest first, and it stops at the first method that returns a PDF:

  1 curl_cffi   impersonate="chrome" - Chrome's TLS/JA3 + HTTP2 settings
  2 pw_headless Playwright Chromium, headless
  3 pw_headed   Playwright Chromium, headed, persistent profile, waits out a JS
                challenge (methods 3 and 4 of the brief merged - see the note in
                leader_fulltext.md; the per-site request cap does not leave room
                for both, and a persistent profile only pays off on a revisit)

Politeness: at most 3 requests per host in a run, >= 10 s apart, and only ever
against articles Unpaywall reports `is_oa: true` with a CC licence. Nothing here
touches paywalled content and nothing here downloads in bulk.

    python3 evals/scratch_n7b_antibot.py --method curl_cffi
    .venv/bin/python evals/scratch_n7b_antibot.py --method pw_headless
    .venv/bin/python evals/scratch_n7b_antibot.py --method pw_headed
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

OUT = Path(__file__).parent.parent / "eval_data" / "runs" / "gold_nature" / "n7b_antibot.json"
PROFILE = Path.home() / ".cache" / "amie-n7b-profile"
DL_DIR = Path("/tmp/n7b_downloads")

PER_HOST_GAP_S = 10.0
PER_HOST_CAP = 3

_HEADFUL_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/143.0.0.0 Safari/537.36")

# One open-access article per host. Every DOI here was checked against Unpaywall
# on 2026-09-19: is_oa true, oa_status gold or hybrid, a CC licence, published
# version. The licence is recorded so the choice can be re-checked later.
TARGETS = [
    {"host": "academic.oup.com", "doi": "10.1093/hr/uhaf313", "oa": "gold / cc-by",
     "pdf": "https://academic.oup.com/hr/advance-article-pdf/doi/10.1093/hr/uhaf313/65291239/uhaf313.pdf",
     "landing": "https://doi.org/10.1093/hr/uhaf313"},
    {"host": "www.jbc.org", "doi": "10.1074/jbc.m112.403279", "oa": "hybrid / cc-by",
     "pdf": "http://www.jbc.org/article/S0021925820464300/pdf",
     "landing": "https://doi.org/10.1074/jbc.m112.403279"},
    {"host": "www.mdpi.com", "doi": "10.3390/biology14060634", "oa": "gold / cc-by",
     "pdf": "https://www.mdpi.com/2079-7737/14/6/634/pdf?version=1748588612",
     "landing": "https://www.mdpi.com/2079-7737/14/6/634"},
    {"host": "dl.acm.org", "doi": "10.1145/223904.223937", "oa": "gold (ACM Open)",
     "pdf": "https://dl.acm.org/doi/pdf/10.1145/223904.223937",
     "landing": "https://dl.acm.org/doi/10.1145/223904.223937"},
    {"host": "onlinelibrary.wiley.com", "doi": "10.1155/2024/3573796", "oa": "hybrid / cc-by",
     "pdf": "https://onlinelibrary.wiley.com/doi/pdfdirect/10.1155/2024/3573796",
     "landing": "https://doi.org/10.1155/2024/3573796"},
    {"host": "pmc.ncbi.nlm.nih.gov", "doi": "10.1016/j.euros.2024.01.010", "oa": "gold / cc-by-nc-nd",
     "pdf": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11992575/pdf/main.pdf",
     "landing": "https://pmc.ncbi.nlm.nih.gov/articles/PMC11992575/"},
]

# Substrings that mean "this is the bot-management page, not the article".
_CHALLENGE_MARKS = (
    "just a moment", "cf-browser-verification", "cf_chl", "challenge-platform",
    "enable javascript and cookies to continue", "checking your browser",
    "cloudpmc-viewer-pow", "attention required", "ddos-guard", "_incapsula_",
)


def challenge_kind(body: bytes | str) -> str:
    """Which anti-bot page this is, or "" when it is not one."""
    s = body.decode("utf-8", "replace") if isinstance(body, bytes) else body
    low = s[:200000].lower()
    if "cloudpmc-viewer-pow" in low:
        return "PMC proof-of-work"
    for m in _CHALLENGE_MARKS:
        if m in low:
            return f"challenge page ({m})"
    return ""


_last_hit: dict[str, float] = {}


def _pace(host: str) -> None:
    gap = PER_HOST_GAP_S - (time.time() - _last_hit.get(host, 0.0))
    if gap > 0:
        time.sleep(gap)
    _last_hit[host] = time.time()


def _verdict(status, ctype, body: bytes) -> dict:
    ch = challenge_kind(body)
    is_pdf = body[:4] == b"%PDF"
    return {"status": status, "content_type": ctype, "challenge": ch,
            "pdf": is_pdf, "bytes": len(body),
            "note": "PDF" if is_pdf else (ch or f"not a PDF ({ctype or '?'})")}


# ── method 1: curl_cffi ─────────────────────────────────────────────────────

def try_curl_cffi(t: dict) -> dict:
    from curl_cffi import requests as creq
    _pace(t["host"])
    try:
        r = creq.get(t["pdf"], impersonate="chrome", timeout=45, allow_redirects=True)
        return {**_verdict(r.status_code, (r.headers.get("content-type") or "").split(";")[0],
                           r.content or b""), "final_url": str(r.url)[:200]}
    except Exception as exc:
        return {"status": None, "content_type": "", "challenge": "", "pdf": False,
                "bytes": 0, "note": f"{type(exc).__name__}: {exc}"[:160]}


# ── methods 2/3: Playwright ─────────────────────────────────────────────────

def _pw_fetch(page, url: str, wait_challenge_s: float) -> dict:
    """Navigate to `url`, then read what actually came back.

    Chromium turns a PDF response into a download rather than a document, so the
    download event is the success path and `response` is the failure path.
    """
    DL_DIR.mkdir(parents=True, exist_ok=True)
    holder: dict = {}
    page.on("download", lambda d: holder.setdefault("dl", d))
    status, ctype, body = None, "", b""
    try:
        resp = page.goto(url, wait_until="domcontentloaded", timeout=60000)
        if resp is not None:
            status = resp.status
            ctype = (resp.header_value("content-type") or "").split(";")[0]
            # Headed Chrome renders a PDF in its viewer instead of downloading it.
            # Response.body() reads the bytes the browser already has, so this
            # costs no second request against the host.
            if ctype == "application/pdf":
                try:
                    body = resp.body()
                    if body[:4] == b"%PDF":
                        out = _verdict(status, ctype, body)
                        out["final_url"] = page.url[:200]
                        return out
                except Exception:
                    body = b""
    except Exception as exc:
        if "Download is starting" not in str(exc):
            status, ctype = None, f"{type(exc).__name__}"
    # Give a download, or a JS challenge, time to finish.
    deadline = time.time() + wait_challenge_s
    while time.time() < deadline and "dl" not in holder:
        try:
            html = page.content()
        except Exception:
            html = ""
        if html and not challenge_kind(html):
            break
        page.wait_for_timeout(1000)
    if "dl" in holder:
        p = DL_DIR / f"n7b_{abs(hash(url))}.pdf"
        try:
            holder["dl"].save_as(str(p))
            body = p.read_bytes()
            ctype = ctype or "application/pdf"
        except Exception as exc:
            return {"status": status, "content_type": ctype, "challenge": "", "pdf": False,
                    "bytes": 0, "note": f"download failed: {type(exc).__name__}: {exc}"[:160]}
    else:
        try:
            body = (page.content() or "").encode("utf-8", "replace")
        except Exception:
            body = b""
    out = _verdict(status, ctype, body)
    try:
        out["final_url"] = page.url[:200]
    except Exception:
        pass
    return out


def try_playwright(t: dict, headed: bool) -> dict:
    from playwright.sync_api import sync_playwright
    _pace(t["host"])
    with sync_playwright() as p:
        if headed:
            PROFILE.mkdir(parents=True, exist_ok=True)
            ctx = p.chromium.launch_persistent_context(
                str(PROFILE), headless=False, accept_downloads=True,
                viewport={"width": 1280, "height": 900},
                args=["--disable-blink-features=AutomationControlled"])
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            try:
                return _pw_fetch(page, t["pdf"], wait_challenge_s=30)
            finally:
                ctx.close()
        browser = p.chromium.launch(headless=True,
                                    args=["--disable-blink-features=AutomationControlled"])
        # Headless Chromium advertises "HeadlessChrome/..." in its UA. N7 already
        # proved a UA swap alone changes nothing, so stripping the word keeps this
        # a test of the *fingerprint* rather than a rerun of N7's header test.
        ctx = browser.new_context(accept_downloads=True,
                                  viewport={"width": 1280, "height": 900},
                                  user_agent=_HEADFUL_UA)
        page = ctx.new_page()
        try:
            return _pw_fetch(page, t["pdf"], wait_challenge_s=20)
        finally:
            ctx.close()
            browser.close()


METHODS = {
    "curl_cffi": lambda t: try_curl_cffi(t),
    "pw_headless": lambda t: try_playwright(t, headed=False),
    "pw_headed": lambda t: try_playwright(t, headed=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(METHODS))
    ap.add_argument("--only", default="", help="substring of the host, to retry one site")
    args = ap.parse_args()

    prev = json.loads(OUT.read_text()) if OUT.exists() else {}
    prev.setdefault("attempts", {})

    if args.method == "pw_headed":
        print("\n*** A Chrome window is about to open on your screen (method 3, headed). ***")
        print("*** It loads open-access article PDFs only, one per site, 10 s apart.   ***\n")
        sys.stdout.flush()
        time.sleep(3)

    targets = [t for t in TARGETS if args.only in t["host"]]
    print(f"method={args.method}  targets={len(targets)}\n")
    for t in targets:
        seen = prev["attempts"].get(t["host"], {})
        if len(seen) >= PER_HOST_CAP and args.method not in seen:
            print(f"{t['host']:<26} SKIPPED - {PER_HOST_CAP}-request cap already used")
            continue
        if any(v.get("pdf") for v in seen.values()):
            won = next(k for k, v in seen.items() if v.get("pdf"))
            print(f"{t['host']:<26} SKIPPED - already got a PDF via {won}")
            continue
        r = METHODS[args.method](t)
        seen[args.method] = r
        prev["attempts"][t["host"]] = seen
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(prev, indent=1))
        print(f"{t['host']:<26} status={str(r['status']):<6} pdf={str(r['pdf']):<5} "
              f"{r['note'][:70]}")

    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
