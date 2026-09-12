"""Where the *serving* IP stands with Google Patents.

`patent_analyzer/recall/google_patents.py` records, from a laptop, "~10
requests in a few minutes from one IP triggered Google's Sorry soft-block".
Whether a Cloud Run egress IP behaves the same is a question about this
deployment, not about the channel's code, so the measurement has to be taken
from inside the deployment.

The probe deliberately does *not* go through `google_patents._get`: that
function trips a breaker and returns `None` on the first 429/503, which is
the right behaviour for recall and the wrong one for measurement — it hides
every status after the first. Here each request is issued raw and its status
recorded, so the answer is "which request number did it break at", not
"blocked: yes".

Reachable only through Cloud Run IAM: the backend runs
--no-allow-unauthenticated and the public frontend proxy has no route to
/admin/*.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse

import httpx
from fastapi import APIRouter

router = APIRouter()

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
_XHR = "https://patents.google.com/xhr/query"

# Distinct enough that no two requests can be served from one cache entry —
# a probe that measures Google's cache instead of Google's throttle measures
# nothing.
_TERMS = ["video conferencing", "speaker tracking", "beamforming microphone", "camera pan tilt",
          "acoustic echo cancellation", "far field voice", "gesture recognition", "depth sensor",
          "face detection", "audio source localization"]


def _probe_url(i: int) -> str:
    q = f"{_TERMS[i % len(_TERMS)]} {i}"
    inner = f"q={'+'.join(q.split())}&num=10&page=0"
    return f"{_XHR}?url={urllib.parse.quote(inner, safe='')}&exp="


async def _one(client: httpx.AsyncClient, i: int) -> dict:
    t0 = time.monotonic()
    try:
        r = await client.get(_probe_url(i))
    except httpx.HTTPError as e:
        return {"i": i, "status": None, "error": type(e).__name__, "ms": int((time.monotonic() - t0) * 1000)}
    head = r.text[:400]
    return {"i": i,
            "status": r.status_code,
            "sorry": "<title>Sorry" in head,
            "bytes": len(r.content),
            "ms": int((time.monotonic() - t0) * 1000)}


@router.get("/admin/gp-probe")
async def gp_probe(n: int = 10, gap: float = 1.0):
    """Fire `n` serial Google Patents XHR searches `gap` seconds apart and
    report every status. No retries, no breaker, no cache."""
    n = max(1, min(int(n), 200))
    gap = max(0.0, min(float(gap), 30.0))
    out: list[dict] = []
    async with httpx.AsyncClient(headers={"User-Agent": _UA}, timeout=30, follow_redirects=True) as client:
        for i in range(n):
            if i and gap:
                await asyncio.sleep(gap)
            out.append(await _one(client, i))
    bad = [r for r in out if r.get("status") != 200]
    first_503 = next((r["i"] for r in out if r.get("status") in (429, 503)), None)
    first_sorry = next((r["i"] for r in out if r.get("sorry")), None)
    return {
        "n": n, "gap_s": gap,
        "ok": sum(1 for r in out if r.get("status") == 200),
        "blocked": len(bad),
        "block_rate": round(len(bad) / n, 4),
        "first_rate_limited_index": first_503,
        "first_sorry_index": first_sorry,
        "by_status": {str(k): sum(1 for r in out if r.get("status") == k)
                      for k in sorted({r.get("status") for r in out}, key=lambda x: (x is None, x))},
        "requests": out,
    }
