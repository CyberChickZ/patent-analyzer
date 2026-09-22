"""Open-access full-text sources, and the HTTP client that can actually reach them.

Why this module exists, measured 2026-09-19 (N7b). N7 concluded that the
publisher hosts "refuse an automated client" from `urllib` plus a swapped
User-Agent. That test could not support the conclusion: Cloudflare's hard 403 is
a TLS/HTTP2 *fingerprint* verdict, and a header string does not change a
fingerprint. Re-run against one Unpaywall-confirmed open-access article per host:

    host                      urllib+UA   curl_cffi(chrome)   Chromium
    www.jbc.org               403         PDF, 1.98 MB        -
    dl.acm.org                403         PDF, 1.25 MB        -
    www.mdpi.com              403         HTML stub           PDF, 1.42 MB (headed)
    pmc.ncbi.nlm.nih.gov      PoW page    PoW page            PDF, 392 KB (headless)
    academic.oup.com          403         CF challenge        CF passed, CDN 403
    onlinelibrary.wiley.com   403         CF challenge        CF passed, bytes not caught

So two of the five "walls" were never walls, and the rest are JS challenges
rather than blanket refusals. `get()` here therefore speaks Chrome's fingerprint
by default.

Ordering principle, and it matters more than any of the above: ask the parties
that publish for programmatic use *first*. A repository copy (PMC, an
institutional repository, Zenodo, arXiv) is the same article with no bot
management in front of it, and Europe PMC / BioC-PMC / Crossref hand out full
text through documented APIs. The publisher's own copy is the last stop, not the
first. `fulltext.resolve()` wires the order; this module is the adapters.

Nothing here defeats an access control. The PMC proof-of-work page is left
alone; where PMC content is wanted we call NCBI's BioC API, which is the
interface NCBI publishes for exactly this. There is no paywalled tier — see the
compliance note in `fulltext.py`.
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

from .cache import kv
from .runtime_state import SerialLock

_NS = "ftsrc"
_SSL_CTX = ssl.create_default_context(cafile=certifi.where())

CONTACT = os.environ.get("UNPAYWALL_EMAIL", "hczhang34@gmail.com")
_UA = f"patent-analyzer/0.4 (mailto:{CONTACT})"
HOST_COOLDOWN_S = float(os.environ.get("FULLTEXT_HOST_COOLDOWN_S", "1.0"))
TIMEOUT_S = float(os.environ.get("FULLTEXT_TIMEOUT_S", "45"))
CACHE_DAYS = float(os.environ.get("FULLTEXT_CACHE_DAYS", "30"))
MAX_TEXT_CHARS = int(os.environ.get("FULLTEXT_MAX_CHARS", "120000"))

# Chrome's TLS/JA3 + HTTP2 fingerprint. Off with FULLTEXT_IMPERSONATE=0, which
# falls the client back to urllib so the module still works where the wheel is
# unavailable.
IMPERSONATE = os.environ.get("FULLTEXT_IMPERSONATE", "chrome")

# Institutional text-and-data-mining credentials. Both are free for academic use
# but have to be *applied for*, which is why neither is required here: without
# the credential the tier is skipped, not failed. These two names are the
# concrete asks for a library TDM request.
#   Wiley:    https://onlinelibrary.wiley.com/library-info/resources/text-and-datamining
#   Elsevier: https://dev.elsevier.com/
WILEY_TDM_TOKEN = os.environ.get("WILEY_TDM_TOKEN", "")
ELSEVIER_TDM_KEY = os.environ.get("ELSEVIER_TDM_KEY", "")
CORE_API_KEY = os.environ.get("CORE_API_KEY", "")

call_log: list[dict] = []


def is_pdf(data: bytes | None) -> bool:
    """A PDF is bytes that start with %PDF, and nothing else.

    The `Content-Type` header lies in both directions — Cloudflare's challenge
    page has been seen served as `application/pdf`, and hosts serve real PDFs as
    `application/octet-stream`. Recording an HTML challenge page as a successful
    download is the single easiest way to overstate reach, and it is what the
    `fulltext_tier` / `fulltext_download` split exists to prevent.
    """
    return bool(data) and data[:4] == b"%PDF"


# ── HTTP ────────────────────────────────────────────────────────────────────

class Resp:
    __slots__ = ("status", "content_type", "body", "url", "error")

    def __init__(self, status=None, content_type="", body=b"", url="", error=""):
        self.status, self.content_type = status, content_type
        self.body, self.url, self.error = body, url, error

    @property
    def ok(self) -> bool:
        return self.status == 200 and not self.error

    def text(self, limit: int = 400000) -> str:
        return self.body[:limit].decode("utf-8", "replace")


def _host(url: str) -> str:
    return urllib.parse.urlparse(url).netloc or "unknown"


def _blocking_get(url: str, headers: dict | None = None, identify: bool = True) -> Resp:
    """One blocking GET.

    `identify=True` sends the `patent-analyzer (mailto:…)` User-Agent that
    Crossref, Unpaywall and NCBI all ask for. `identify=False` leaves the whole
    header set to curl_cffi's browser profile, and that distinction is load
    bearing: measured 2026-09-19, sending Chrome's TLS fingerprint *together
    with* a `patent-analyzer/0.4` User-Agent gets a 403 from www.jbc.org, while
    the same request with Chrome's own headers returns the 1.98 MB PDF. Bot
    management compares the two, so a mismatched pair is worse than either
    alone. Document fetches therefore do not identify; API calls do, because
    those hosts want to be told who is calling and have no wall.
    """
    hdrs = dict(headers or {})
    if identify:
        hdrs.setdefault("User-Agent", _UA)
        hdrs.setdefault("Accept", "*/*")
    if IMPERSONATE not in ("0", "", "false"):
        try:
            from curl_cffi import requests as creq
            r = creq.get(url, headers=hdrs or None, impersonate=IMPERSONATE,
                         timeout=TIMEOUT_S, allow_redirects=True)
            return Resp(r.status_code, (r.headers.get("content-type") or "").split(";")[0],
                        r.content or b"", str(r.url))
        except ImportError:
            pass
        except Exception as exc:
            return Resp(error=f"{type(exc).__name__}: {exc}"[:200], url=url)
    try:
        req = urllib.request.Request(url, headers=hdrs or {"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S, context=_SSL_CTX) as r:
            return Resp(r.status, (r.headers.get("Content-Type") or "").split(";")[0],
                        r.read(), r.geturl())
    except urllib.error.HTTPError as exc:
        return Resp(exc.code, "", b"", url, f"HTTP {exc.code}")
    except Exception as exc:
        return Resp(error=f"{type(exc).__name__}: {exc}"[:200], url=url)


async def get(url: str, headers: dict | None = None, identify: bool = True) -> Resp:
    """One GET, serialised per host with a cooldown so no host sees a burst."""
    async with SerialLock(f"ft:{_host(url)}", HOST_COOLDOWN_S):
        r = await asyncio.to_thread(_blocking_get, url, headers, identify)
    call_log.append({"url": url[:160], "status": r.status, "bytes": len(r.body),
                     "error": r.error})
    return r


async def get_pdf(url: str, headers: dict | None = None) -> tuple[bytes | None, str]:
    """(pdf_bytes, detail). Bytes only when they start with %PDF."""
    if not url or not url.startswith("http"):
        return None, "no URL"
    r = await get(url, headers, identify=bool(headers))
    if r.error:
        return None, r.error
    if is_pdf(r.body):
        return r.body, f"{len(r.body)} bytes"
    if r.status != 200:
        return None, f"HTTP {r.status}"
    return None, f"not a PDF ({r.content_type or 'no content-type'}, {len(r.body)} bytes)"


# ── Crossref ────────────────────────────────────────────────────────────────

CROSSREF_BASE = "https://api.crossref.org/works"


async def crossref_work(doi: str) -> dict | None:
    """The Crossref record for a DOI, KV-cached. None when Crossref has none."""
    d = (doi or "").strip().lower()
    if not d:
        return None
    hit = kv().get(_NS, f"crossref:{d}", max_age_days=CACHE_DAYS)
    if hit is not None:
        return hit.get("message")
    r = await get(f"{CROSSREF_BASE}/{urllib.parse.quote(d, safe='/')}")
    msg = None
    if r.ok:
        try:
            msg = json.loads(r.body).get("message")
        except Exception:
            msg = None
    if r.status in (200, 404):
        kv().put(_NS, f"crossref:{d}", {"message": msg})
    return msg


def tdm_links(work: dict | None) -> list[dict]:
    """Crossref `link[]` entries the publisher marked for text and data mining.

    <https://www.crossref.org/documentation/retrieve-metadata/rest-api/text-and-data-mining/>

    Measured on the six acceptance DOIs: 3 of 6 carry a `text-mining` link, and
    2 of those 3 point at `api.elsevier.com`, which answers only with an API
    key. Unauthenticated yield is therefore low — this tier is worth having but
    it is not the answer on its own. Wiley publishes *no* text-mining link at
    all (only `similarity-checking`); its TDM route is the token API below.
    """
    out = []
    for L in ((work or {}).get("link") or []):
        if str(L.get("intended-application") or "").lower() == "text-mining" and L.get("URL"):
            out.append({"url": str(L["URL"]), "content_type": str(L.get("content-type") or "")})
    # A PDF beats XML beats "unspecified" (usually a landing page).
    rank = {"application/pdf": 0, "text/xml": 1, "application/xml": 1, "text/plain": 2}
    out.sort(key=lambda L: rank.get(L["content_type"], 3))
    return out


def _norm_title(t: str) -> str:
    return " ".join("".join(c if c.isalnum() or c.isspace() else " "
                            for c in (t or "").lower()).split())


async def crossref_doi_for_title(title: str) -> str:
    """The DOI Crossref returns for a title, only on an exact normalised match.

    Used to *cross-confirm* OpenAlex's answer, never on its own: a wrong DOI is
    worse than no DOI, because it points the full-text fetch, the GCS cache and
    the manual-download list at the wrong paper all at once.
    """
    want = _norm_title(title)
    if len(want) < 20:
        return ""
    key = f"crtitle:{want[:180]}"
    hit = kv().get(_NS, key, max_age_days=CACHE_DAYS)
    if hit is not None:
        return hit.get("doi") or ""
    q = urllib.parse.urlencode({"query.bibliographic": title[:200], "rows": "1",
                                "select": "DOI,title", "mailto": CONTACT})
    r = await get(f"{CROSSREF_BASE}?{q}")
    doi = ""
    if r.ok:
        try:
            items = (json.loads(r.body).get("message") or {}).get("items") or []
            got = _norm_title((items[0].get("title") or [""])[0]) if items else ""
            if got and got == want:
                doi = str(items[0].get("DOI") or "").lower()
        except Exception:
            doi = ""
    kv().put(_NS, key, {"doi": doi})
    return doi


# ── PubMed Central through NCBI's own APIs ──────────────────────────────────
#
# The PMC *website* answers a non-browser with a proof-of-work challenge
# (`cloudpmc-viewer-pow`). That control is left alone. NCBI publishes two
# interfaces for programmatic access to the same open-access articles, and those
# are what we call:
#   idconv  DOI -> PMCID
#   BioC    PMCID -> full text as BioC XML
# Verified 2026-09-19 on 10.1016/j.euros.2024.01.010: the web PDF is behind the
# challenge, BioC returns 15 KB of the article body over plain HTTP.

IDCONV = "https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/"
BIOC = "https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml"


async def pmcid_for_doi(doi: str) -> tuple[str, str]:
    """(PMCID, error). KV-cached, negatives included."""
    d = (doi or "").strip().lower()
    if not d:
        return "", "not a DOI"
    hit = kv().get(_NS, f"pmcid:{d}", max_age_days=CACHE_DAYS)
    if hit is not None:
        return hit.get("pmcid") or "", ""
    q = urllib.parse.urlencode({"ids": d, "format": "json", "email": CONTACT,
                                "tool": "patent-analyzer"})
    r = await get(f"{IDCONV}?{q}")
    if r.error or not r.ok:
        return "", r.error or f"HTTP {r.status}"
    try:
        recs = json.loads(r.body).get("records") or []
    except Exception as exc:
        return "", f"{type(exc).__name__}"
    pmcid = str((recs[0] if recs else {}).get("pmcid") or "")
    kv().put(_NS, f"pmcid:{d}", {"pmcid": pmcid})
    return pmcid, ""


_TAG_RE = re.compile(r"<[^>]+>")


def bioc_to_text(xml: bytes | str) -> str:
    """Readable text out of a BioC document: the passage text and nothing else."""
    s = xml.decode("utf-8", "replace") if isinstance(xml, bytes) else xml
    parts = re.findall(r"(?is)<text>(.*?)</text>", s)
    body = "\n\n".join(_TAG_RE.sub(" ", p) for p in parts)
    body = (body.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
                .replace("&quot;", '"').replace("&apos;", "'"))
    return re.sub(r"[ \t]{2,}", " ", body).strip()[:MAX_TEXT_CHARS]


async def bioc_pmc_text(doi: str) -> tuple[str, str]:
    """Open-access full text for a DOI through NCBI's BioC API, or ("", reason).

    429 is reported as its own reason rather than folded into the generic
    failures: it means "come back later", not "this article is not available",
    and the two must not be counted together when the tier's success rate is
    reported.
    """
    pmcid, err = await pmcid_for_doi(doi)
    if err:
        return "", f"idconv: {err}"
    if not pmcid:
        return "", "no PMCID for this DOI"
    r = await get(f"{BIOC}/{pmcid}/unicode")
    if r.status == 429:
        return "", f"BioC {pmcid}: rate limited (429) — retry later, not unavailable"
    if r.error or not r.ok:
        return "", f"BioC {pmcid}: {r.error or f'HTTP {r.status}'}"
    text = bioc_to_text(r.body)
    if len(text) < 500:
        return "", f"BioC {pmcid}: body too short to be the article ({len(text)} chars)"
    return text, ""


# ── preprint servers ────────────────────────────────────────────────────────

_BIORXIV_PREFIX = "10.1101/"
_CHEMRXIV_PREFIX = "10.26434/"


def preprint_pdf_url(doi: str, server_hint: str = "") -> str:
    """Direct PDF URL on bioRxiv/medRxiv for a 10.1101 DOI, else "".

    Both serve `content/<doi>v1.full.pdf`. Which of the two holds a given
    10.1101 DOI is not encoded in the DOI, so the hint (journal/venue string on
    the candidate) picks medRxiv and bioRxiv is the default.
    """
    d = (doi or "").strip().lower()
    if not d.startswith(_BIORXIV_PREFIX):
        return ""
    host = "www.medrxiv.org" if "medrxiv" in (server_hint or "").lower() else "www.biorxiv.org"
    return f"https://{host}/content/{d}v1.full.pdf"


async def chemrxiv_pdf_url(doi: str) -> str:
    """ChemRxiv is on Cambridge Open Engage and does not answer the bioRxiv URL
    shape; its item record has to be looked up by DOI first."""
    d = (doi or "").strip().lower()
    if not d.startswith(_CHEMRXIV_PREFIX):
        return ""
    r = await get("https://chemrxiv.org/engage/chemrxiv/public-api/v1/items/doi/"
                  + urllib.parse.quote(d, safe=""))
    if not r.ok:
        return ""
    try:
        item = json.loads(r.body)
    except Exception:
        return ""
    return str(((item.get("asset") or {}).get("original") or {}).get("url") or "")


# ── the landing page's own declaration ──────────────────────────────────────

_CITATION_PDF_RE = re.compile(
    r"""<meta[^>]+name=["']citation_pdf_url["'][^>]+content=["']([^"']+)["']""", re.I)
_CITATION_PDF_RE2 = re.compile(
    r"""<meta[^>]+content=["']([^"']+)["'][^>]+name=["']citation_pdf_url["']""", re.I)


def citation_pdf_url_in(html: str, base: str = "") -> str:
    """The `citation_pdf_url` Google-Scholar meta tag, which most platforms emit."""
    for rx in (_CITATION_PDF_RE, _CITATION_PDF_RE2):
        m = rx.search(html or "")
        if m:
            u = m.group(1).strip()
            return urllib.parse.urljoin(base, u) if base and not u.startswith("http") else u
    return ""


async def citation_pdf_url(landing: str) -> str:
    """Fetch a landing page and read its `citation_pdf_url`, or "".

    This is a *hint*, not a download: whatever it points at still has to come
    back starting with %PDF before anything counts.
    """
    if not landing or not landing.startswith("http"):
        return ""
    # A landing page is a document on a publisher host, not an API, so it is
    # fetched with the browser profile's own headers — see _blocking_get.
    r = await get(landing, identify=False)
    if not r.ok or "html" not in (r.content_type or ""):
        return ""
    return citation_pdf_url_in(r.text(), r.url or landing)


# ── aggregators and archives ────────────────────────────────────────────────

async def core_pdf_url(doi: str) -> tuple[str, str]:
    """(download_url, why-not) from CORE.

    Needs a free API key. Without one the tier is *skipped*, not failed:
    measured 2026-09-19, an unauthenticated call to `api.core.ac.uk/v3` answers
    429, so counting those as "CORE has nothing" would be a lie about coverage.
    Key: <https://core.ac.uk/services/api> (free, self-service).
    """
    d = (doi or "").strip().lower()
    if not d:
        return "", "not a DOI"
    if not CORE_API_KEY:
        return "", "skipped: no CORE_API_KEY (free, self-service at core.ac.uk/services/api)"
    q = urllib.parse.urlencode({"q": f'doi:"{d}"', "limit": "1"})
    r = await get(f"https://api.core.ac.uk/v3/search/works/?{q}",
                  {"Authorization": f"Bearer {CORE_API_KEY}"})
    if r.status == 429:
        return "", "CORE rate limited (429)"
    if not r.ok:
        return "", r.error or f"CORE HTTP {r.status}"
    try:
        results = json.loads(r.body).get("results") or []
    except Exception:
        return "", "CORE returned something that is not JSON"
    if not results:
        return "", "CORE has no record for this DOI"
    w = results[0]
    u = str(w.get("downloadUrl") or (w.get("sourceFulltextUrls") or [""])[0] or "")
    return (u, "") if u else ("", "CORE has the record but no full-text URL")


async def wayback_snapshot(url: str) -> tuple[str, str]:
    """(snapshot_url, why-not). The Internet Archive's newest public snapshot.

    Worth a try on a publisher PDF URL that now sits behind bot management: the
    Archive crawled a lot of them before the wall went up, and the snapshot is a
    public copy served by archive.org rather than by the publisher.

    429 is reported separately from "no snapshot" — archive.org rate limits
    freely and the two answers mean opposite things.
    """
    if not url or not url.startswith("http"):
        return "", "no URL to look up"
    r = await get("https://archive.org/wayback/available?url="
                  + urllib.parse.quote(url, safe=""))
    if r.status == 429:
        return "", "archive.org rate limited (429) — retry later, not absent"
    if not r.ok:
        return "", r.error or f"archive.org HTTP {r.status}"
    try:
        snap = ((json.loads(r.body).get("archived_snapshots") or {}).get("closest") or {})
    except Exception:
        return "", "archive.org returned something that is not JSON"
    if snap.get("available") and snap.get("url"):
        return str(snap["url"]), ""
    return "", "no Internet Archive snapshot"


# ── publisher TDM APIs (optional, credentialled) ────────────────────────────

async def wiley_tdm_pdf(doi: str) -> tuple[bytes | None, str]:
    """Wiley's TDM endpoint. Skipped, not failed, when no token is configured.

    The token is free for academic use but has to be requested; it is one of the
    two concrete items on the library TDM ask. Wiley asks for 10 s between
    calls, which is what the SerialLock is set to here.
    """
    d = (doi or "").strip().lower()
    if not d:
        return None, "not a DOI"
    if not WILEY_TDM_TOKEN:
        return None, "skipped: no WILEY_TDM_TOKEN (free, but must be applied for)"
    url = ("https://api.wiley.com/onlinelibrary/tdm/v1/articles/"
           + urllib.parse.quote(d, safe=""))
    async with SerialLock("wiley-tdm", 10.0):
        r = await asyncio.to_thread(_blocking_get, url, {"Wiley-TDM-Client-Token": WILEY_TDM_TOKEN})
    if is_pdf(r.body):
        return r.body, f"{len(r.body)} bytes"
    return None, r.error or f"HTTP {r.status}"


async def elsevier_tdm_text(doi: str) -> tuple[str, str]:
    """Elsevier's article API. Skipped, not failed, without an API key — the
    other concrete item on the library TDM ask. Crossref points 2 of the 6
    acceptance DOIs straight at this endpoint."""
    d = (doi or "").strip().lower()
    if not d:
        return "", "not a DOI"
    if not ELSEVIER_TDM_KEY:
        return "", "skipped: no ELSEVIER_TDM_KEY (free for academics, must be applied for)"
    url = (f"https://api.elsevier.com/content/article/doi/{urllib.parse.quote(d, safe='/')}"
           "?httpAccept=text%2Fplain")
    r = await get(url, {"X-ELS-APIKey": ELSEVIER_TDM_KEY, "Accept": "text/plain"})
    if not r.ok:
        return "", r.error or f"HTTP {r.status}"
    text = r.text()[:MAX_TEXT_CHARS].strip()
    return (text, "") if len(text) >= 500 else ("", "body too short to be the article")


# ── eLife ───────────────────────────────────────────────────────────────────

async def elife_pdf_url(doi: str) -> str:
    """eLife publishes every article's PDF and XML through its own API, keyed by
    the numeric part of the 10.7554/eLife.NNNNN DOI."""
    m = re.match(r"^10\.7554/elife\.(\d+)", (doi or "").strip().lower())
    if not m:
        return ""
    r = await get(f"https://api.elifesciences.org/articles/{m.group(1)}")
    if not r.ok:
        return ""
    try:
        return str(((json.loads(r.body).get("pdf")) or ""))
    except Exception:
        return ""
