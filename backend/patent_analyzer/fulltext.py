"""Open-access full text for the paper side of the pool.

Why this exists: on the E4/h1h gold the examiner-cited *papers* were the weak
side — 2 of 17 resolvable NPL gold reached the pool (reach .118) against .607
on the patent side — and the M2 e2e runs downloaded 0-6 PDFs out of 30, so the
documents that did reach Phase 4 mostly arrived with an abstract at best. This
module turns a paper candidate into a downloadable PDF URL, in tiers:

    arxiv          the candidate is an arXiv paper -> arxiv.org/pdf/<id>
    oa             an open-access PDF URL is already known (OpenAlex
                   `open_access.oa_url` / `primary_location.pdf_url`, or
                   Semantic Scholar `openAccessPdf.url`), or Unpaywall names one
    abstract_only  no open-access copy found; the document is evaluated from
                   its abstract, and its landing page goes on the manual list

COMPLIANCE — read before extending this module.

Everything here reads *open-access* copies, from sources that publish them for
programmatic use. There is deliberately no paywalled tier. A proxied tier
through the OSU institutional subscription was designed and then **dropped**,
because OSU Libraries' "Responsible Use of Licensed Electronic Resources"
forbids exactly that technique:

    "Activities Not Allowed
     Systematically or programmatically downloading content from electronic
     resources. This includes, but is not limited to: ... Using scripts,
     spiders, crawlers, or other computer programs to automatically download
     content"

and states the blast radius:

    "The service providers of these products may suspend access for the entire
     OSU community if the terms of the license are violated by any user."

    -- https://library.oregonstate.edu/responsible-use-licensed-electronic-resources

Rate limiting does not change that: it lowers the chance of detection, not the
character of the act. The sanctioned route for the same goal, from the same
page, is to ask first:

    "Research sometimes involves bulk computer generated analysis of
     collections of materials; please consult with us about clearing such
     activities with database providers."

(entry point: https://library.oregonstate.edu/ask-librarian — the contact
address on that page is obfuscated against scraping as "[email protected]" and
the real address is unverified here.)

What replaces the dropped tier is `manifest_rows()`: the abstract_only
documents are exported as a list of DOI + title + landing page, and a human
opens those pages and saves the PDFs. A person reading articles is not a
script downloading them.

Output of this module is used only for internal research and technology-
transfer evaluation, and is not redistributed.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import ssl
import urllib.error
import urllib.parse
import urllib.request

import certifi

from . import metering
from .cache import kv
from .runtime_state import PeriodQuota, SerialLock, day_key

_NS = "fulltext"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())
_UA = "patent-analyzer/0.4 (https://github.com/CyberChickZ/patent-analyzer)"

UNPAYWALL_BASE = "https://api.unpaywall.org/v2"
# Unpaywall asks for a contact address on every call instead of an API key.
UNPAYWALL_EMAIL = os.environ.get("UNPAYWALL_EMAIL", "hczhang34@gmail.com")
# One request at a time with a 1 s cooldown. Unpaywall's own published limit
# could not be verified (docs.unpaywall.org did not answer and the product page
# is a JS shell), so this is the same politeness the Semantic Scholar terms ask
# for rather than a number read off their documentation.
UNPAYWALL_COOLDOWN_S = float(os.environ.get("UNPAYWALL_COOLDOWN_S", "1.0"))
# No invented daily ceiling: open-access content carries none of the risk that
# motivated a cap on the dropped proxied tier. The knob exists so a real,
# documented limit can be enforced the day we learn one. 0 = no cap.
UNPAYWALL_DAILY_CAP = int(os.environ.get("UNPAYWALL_DAILY_CAP", "0"))
UNPAYWALL_TIMEOUT_S = float(os.environ.get("UNPAYWALL_TIMEOUT_S", "20"))
CACHE_DAYS = float(os.environ.get("FULLTEXT_CACHE_DAYS", "30"))

GCS_BUCKET = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
GCS_PREFIX = os.environ.get("FULLTEXT_GCS_PREFIX", "fulltext/")
GCS_CACHE_ON = os.environ.get("FULLTEXT_GCS_CACHE", "1") not in ("0", "false", "")

TIERS = ("arxiv", "oa", "abstract_only")

_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;)\]]+)", re.I)
_ARXIV_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([^/?\s]+?)(?:v\d+)?(?:\.pdf)?(?:[?#]|$)", re.I)
_ARXIV_ID_RE = re.compile(r"^(?:arxiv:)?(\d{4}\.\d{4,5}|[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?$", re.I)

call_log: list[dict] = []


def _quota() -> PeriodQuota:
    return PeriodQuota("unpaywall", UNPAYWALL_DAILY_CAP, day_key)


# ── identifiers ──────────────────────────────────────────────────────────────

def normalise_doi(raw: str | None) -> str:
    """A bare lowercase DOI out of a DOI, a doi.org URL or a citation string."""
    if not raw:
        return ""
    m = _DOI_RE.search(str(raw))
    return m.group(1).rstrip(".").lower() if m else ""


def arxiv_id_of(*values: str | None) -> str:
    """An arXiv id from an id string or any arxiv.org URL among `values`."""
    for v in values:
        if not v:
            continue
        s = str(v).strip()
        m = _ARXIV_ID_RE.match(s)
        if m:
            return m.group(1)
        m = _ARXIV_RE.search(s)
        if m:
            return m.group(1)
    return ""


def arxiv_pdf_url(arxiv_id: str) -> str:
    return f"https://arxiv.org/pdf/{arxiv_id}" if arxiv_id else ""


def doi_slug(doi: str) -> str:
    """Cache-object name for a DOI. Keeps it readable and path-safe:
    10.1038/nature14539 -> 10.1038_nature14539."""
    d = normalise_doi(doi) or (doi or "").strip().lower()
    return re.sub(r"[^a-z0-9._-]+", "_", d).strip("_")[:180]


# ── Unpaywall ────────────────────────────────────────────────────────────────

def oa_url_from_unpaywall(payload: dict | None) -> tuple[str, str]:
    """(url, host_type) of the best open-access copy, or ("", "").

    Measured 2026-09-18 on 10.1038/nature14539: `is_oa` is true and
    `best_oa_location.host_type` is "repository", yet `url_for_pdf` is null.
    Reading only `url_for_pdf` therefore throws away real open-access copies —
    fall through to `url` and `url_for_landing_page`, and if the best location
    offers neither, try the other `oa_locations`.
    """
    if not payload or not payload.get("is_oa"):
        return "", ""
    locs = [payload.get("best_oa_location")] + list(payload.get("oa_locations") or [])
    for key in ("url_for_pdf", "url", "url_for_landing_page"):
        for loc in locs:
            if isinstance(loc, dict) and loc.get(key):
                return str(loc[key]), str(loc.get("host_type") or "")
    return "", ""


async def unpaywall(doi: str) -> tuple[dict | None, str | None]:
    """GET /v2/<doi>, KV-cached, serialised with a cooldown. (payload, error)."""
    d = normalise_doi(doi)
    if not d:
        return None, "not a DOI"
    hit = kv().get(_NS, f"unpaywall:{d}", max_age_days=CACHE_DAYS)
    if hit is not None:
        call_log.append({"doi": d, "cached": True})
        return hit.get("payload"), None
    q = _quota()
    if q.cap > 0 and not q.take():
        metering.incident("unpaywall", metering.EXHAUSTED, f"daily cap reached ({day_key()})")
        return None, "unpaywall daily cap reached"
    url = (f"{UNPAYWALL_BASE}/{urllib.parse.quote(d, safe='/')}"
           f"?email={urllib.parse.quote(UNPAYWALL_EMAIL)}")

    def _call() -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": _UA, "Accept": "application/json"})
        with urllib.request.urlopen(req, timeout=UNPAYWALL_TIMEOUT_S, context=_SSL_CTX) as r:
            return r.read()

    try:
        async with SerialLock("unpaywall", UNPAYWALL_COOLDOWN_S):
            metering.count("unpaywall")
            body = await asyncio.to_thread(_call)
        payload = json.loads(body)
    except urllib.error.HTTPError as exc:
        err = f"HTTPError: {exc.code} {exc.reason}"[:200]
        # 404 means "Unpaywall has never heard of this DOI" — a real answer,
        # worth caching so the next job does not ask again.
        if exc.code == 404:
            kv().put(_NS, f"unpaywall:{d}", {"payload": None})
        call_log.append({"doi": d, "error": err})
        return None, err
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"[:200]
        call_log.append({"doi": d, "error": err})
        metering.incident("unpaywall", metering.FAILED, f"{d}: {err}")
        return None, err
    kv().put(_NS, f"unpaywall:{d}", {"payload": payload})
    call_log.append({"doi": d, "is_oa": bool(payload.get("is_oa"))})
    return payload, None


# ── resolution ───────────────────────────────────────────────────────────────

def _known_pdf_url(doc: dict) -> str:
    """The PDF URL an upstream channel already put on the document: OpenAlex
    writes `open_access.oa_url` / `primary_location.pdf_url` into `pdf_link`
    (recall/openalex.py:64-71) and Semantic Scholar writes `openAccessPdf.url`
    there (recall/semantic_scholar.py:59-60). Costs no network call."""
    v = doc.get("pdf_link") or ""
    if isinstance(v, list):
        v = next((x for x in v if x), "")
    return v if isinstance(v, str) and v.startswith("http") else ""


def _landing_page(doc: dict, doi: str) -> str:
    u = doc.get("url") or ""
    if isinstance(u, list):
        u = next((x for x in u if x), "")
    if isinstance(u, str) and u.startswith("http"):
        return u
    return f"https://doi.org/{doi}" if doi else ""


async def resolve(doc: dict, use_unpaywall: bool = True) -> dict:
    """Best open-access PDF URL for one paper-shaped doc dict.

    Returns {"fulltext_tier", "fulltext_url", "landing_page", "doi",
    "fulltext_detail"} — a patch, so callers stay clear of in-place mutation of
    graph state. The chain is arXiv, then what OpenAlex/Semantic Scholar
    already told us, then Unpaywall, then abstract_only.
    """
    doi = normalise_doi(doc.get("doi") or doc.get("pub_num") or "")
    aid = arxiv_id_of(doc.get("arxiv_id"), doc.get("pub_num"),
                      _known_pdf_url(doc), doc.get("url"))
    out = {"doi": doi, "landing_page": _landing_page(doc, doi)}

    if aid:
        return {**out, "fulltext_tier": "arxiv", "fulltext_url": arxiv_pdf_url(aid),
                "fulltext_detail": f"arXiv:{aid}"}

    known = _known_pdf_url(doc)
    if known:
        return {**out, "fulltext_tier": "oa", "fulltext_url": known,
                "fulltext_detail": "open-access URL from OpenAlex/Semantic Scholar"}

    if use_unpaywall and doi:
        payload, err = await unpaywall(doi)
        url, host = oa_url_from_unpaywall(payload)
        if url:
            return {**out, "fulltext_tier": "oa", "fulltext_url": url,
                    "fulltext_detail": f"Unpaywall ({host or 'unknown host'})"}
        detail = err or ("Unpaywall: not open access" if payload else "Unpaywall: DOI unknown")
        return {**out, "fulltext_tier": "abstract_only", "fulltext_url": "",
                "fulltext_detail": detail}

    return {**out, "fulltext_tier": "abstract_only", "fulltext_url": "",
            "fulltext_detail": "no DOI and no open-access URL"}


async def resolve_many(docs: list[dict], use_unpaywall: bool = True) -> list[dict]:
    """resolve() over a list, sequentially — SerialLock serialises the Unpaywall
    calls anyway, so a gather would only queue them up behind the same lock."""
    return [await resolve(d, use_unpaywall=use_unpaywall) for d in docs]


def tier_counts(patches: list[dict]) -> dict[str, int]:
    counts = {t: 0 for t in TIERS}
    for p in patches:
        counts[p.get("fulltext_tier", "abstract_only")] = \
            counts.get(p.get("fulltext_tier", "abstract_only"), 0) + 1
    return counts


# ── manual-download manifest (the replacement for the dropped proxied tier) ──

def manifest_rows(docs: list[dict], patches: list[dict]) -> list[dict]:
    """The abstract_only documents, as a list a person can work through by hand:
    DOI, title, landing page, and why the automatic tiers gave up. Opening these
    pages and saving the PDFs is reading, not scripted downloading."""
    rows = []
    for doc, p in zip(docs, patches):
        if p.get("fulltext_tier") != "abstract_only":
            continue
        rows.append({"doi": p.get("doi", ""), "title": (doc.get("title") or "").strip(),
                     "landing_page": p.get("landing_page", ""),
                     "reason": p.get("fulltext_detail", "")})
    return rows


def manifest_markdown(rows: list[dict]) -> str:
    """The manifest as a Markdown table. Compliance note travels with the list."""
    head = ("| # | DOI | Title | Landing page | Why not automatic |\n"
            "|---|---|---|---|---|\n")
    body = "".join(
        f"| {i} | {r['doi'] or '-'} | {(r['title'] or '-')[:90]} | "
        f"{r['landing_page'] or '-'} | {r['reason']} |\n"
        for i, r in enumerate(rows, 1))
    note = ("\n> Open these pages in a browser and save the PDFs by hand. Do not point a "
            "script at them: OSU Libraries' Responsible Use policy forbids "
            "\"using scripts, spiders, crawlers, or other computer programs to "
            "automatically download content\", and a violation can \"suspend access for "
            "the entire OSU community\". "
            "<https://library.oregonstate.edu/responsible-use-licensed-electronic-resources>\n")
    return head + body + note


# ── GCS cache, keyed by DOI ─────────────────────────────────────────────────

def _blob(doi: str):
    from google.cloud import storage
    return storage.Client().bucket(GCS_BUCKET).blob(f"{GCS_PREFIX}{doi_slug(doi)}.pdf")


def cache_get(doi: str) -> bytes | None:
    """The cached PDF for a DOI, or None. Never raises: no credentials, no
    bucket or no object all mean the same thing to the caller."""
    if not GCS_CACHE_ON or not doi_slug(doi):
        return None
    try:
        b = _blob(doi)
        return b.download_as_bytes() if b.exists() else None
    except Exception:
        return None


def cache_put(doi: str, data: bytes) -> bool:
    if not GCS_CACHE_ON or not doi_slug(doi) or not data:
        return False
    try:
        _blob(doi).upload_from_string(data, content_type="application/pdf")
        return True
    except Exception:
        return False


def cache_uri(doi: str) -> str:
    return f"gs://{GCS_BUCKET}/{GCS_PREFIX}{doi_slug(doi)}.pdf" if doi_slug(doi) else ""
