"""Open-access full text for the paper side of the pool.

Why this exists: on the E4/h1h gold the examiner-cited *papers* were the weak
side — 2 of 17 resolvable NPL gold reached the pool (reach .118) against .607
on the patent side — and the M2 e2e runs downloaded 0-6 PDFs out of 30, so the
documents that did reach Phase 4 mostly arrived with an abstract at best. This
module turns a paper candidate into a readable copy, in tiers:

    arxiv          the candidate is an arXiv paper -> arxiv.org/pdf/<id>
    oa             an open-access PDF URL is already known (OpenAlex
                   `open_access.oa_url` / `primary_location.pdf_url`, or
                   Semantic Scholar `openAccessPdf.url`), or Unpaywall names
                   one, or Europe PMC serves the full text through its API
    abstract_only  no open-access copy found; the document is evaluated from
                   its abstract, and its landing page goes on the manual list

`fulltext_tier` says which tier answered; `fulltext_download` says what came
back. They are not the same number and the report prints both, because a
resolved URL is usually not a paper in hand: measured on the 17 examiner-cited
NPL gold, 6 resolved and 0 returned a PDF.

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
from . import fulltext_sources as fs
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

# A repository copy is the same article without bot management in front of it,
# so it is tried before the publisher's own copy. Measured 2026-09-19 (N7b): of
# the five publisher hosts N7 called a wall, two answered a Chrome-fingerprinted
# client with the PDF and three put a JS challenge in the way — while the
# repository copies of the same articles (PMC through NCBI's BioC API) came back
# over plain HTTP. An unknown host_type sorts last because there is nothing to
# justify preferring it.
HOST_RANK = {"repository": 0, "publisher": 2}
_UNKNOWN_HOST_RANK = 3
_KEY_RANK = {"url_for_pdf": 0, "url": 1, "url_for_landing_page": 2}

# Hosts that are repositories even when the field does not say so — used to rank
# the PDF URL a search channel put on the candidate, which carries no host_type.
_REPO_HOST_RE = re.compile(
    r"(arxiv\.org|biorxiv\.org|medrxiv\.org|chemrxiv\.org|osti\.gov|zenodo\.org|"
    r"ssrn\.com|hal\.science|escholarship\.org|repec\.org|europepmc\.org|"
    r"ncbi\.nlm\.nih\.gov|\.edu(/|$)|repository|eprints|dspace|scholarworks)", re.I)


def is_repository_url(url: str) -> bool:
    return bool(_REPO_HOST_RE.search(url or ""))


def oa_locations_ordered(payload: dict | None) -> list[dict]:
    """Every open-access copy Unpaywall knows, repositories first.

    Two measured facts shape the order. (1) 10.1038/nature14539: `is_oa` true,
    `best_oa_location.host_type` "repository", `url_for_pdf` **null** — reading
    only `url_for_pdf` throws away a real copy, so `url` and
    `url_for_landing_page` are kept as fallbacks. (2) N7b: publisher hosts are
    where the bot management is; repository hosts are not. So the sort is
    host_type first, and only then PDF-before-landing-page inside a host class.

    Returns [{"url", "host_type", "kind"}] with duplicates removed.
    """
    if not payload or not payload.get("is_oa"):
        return []
    locs = [payload.get("best_oa_location")] + list(payload.get("oa_locations") or [])
    rows = []
    for i, loc in enumerate(locs):
        if not isinstance(loc, dict):
            continue
        host = str(loc.get("host_type") or "")
        for key in ("url_for_pdf", "url", "url_for_landing_page"):
            if loc.get(key):
                rank = HOST_RANK.get(host, _UNKNOWN_HOST_RANK)
                if rank == _UNKNOWN_HOST_RANK and is_repository_url(str(loc[key])):
                    rank, host = 0, host or "repository"
                rows.append({"url": str(loc[key]), "host_type": host, "kind": key,
                             "_sort": (rank, _KEY_RANK[key], i)})
    rows.sort(key=lambda r: r["_sort"])
    out, seen = [], set()
    for r in rows:
        if r["url"] in seen:
            continue
        seen.add(r["url"])
        out.append({"url": r["url"], "host_type": r["host_type"], "kind": r["kind"]})
    return out


def oa_url_from_unpaywall(payload: dict | None) -> tuple[str, str]:
    """(url, host_type) of the copy to try first, or ("", "")."""
    rows = oa_locations_ordered(payload)
    return (rows[0]["url"], rows[0]["host_type"]) if rows else ("", "")


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


# ── Europe PMC ───────────────────────────────────────────────────────────────
#
# The measurement that put this here: of the 17 examiner-cited NPL gold, 6
# resolved to an open-access URL and 0 returned a PDF. Publisher hosts answer
# an automated client with a Cloudflare 403 and PubMed Central serves a
# proof-of-work challenge in place of the file. Europe PMC publishes the same
# open-access articles through a REST API meant to be called
# (`/{PMCID}/fullTextXML`, measured 2026-09-18: 200, 97 KB of JATS for
# PMC2150576), so where a paper is in Europe PMC's open-access set we ask that
# instead of scraping a page that does not want to be scraped.
#
# Coverage, measured on the two acceptance sets: 1/17 of the NPL gold and
# 12/104 of the papers a real job delivered. Modest, and real.

EPMC_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EPMC_COOLDOWN_S = float(os.environ.get("EPMC_COOLDOWN_S", "1.0"))
EPMC_TIMEOUT_S = float(os.environ.get("EPMC_TIMEOUT_S", "30"))
EPMC_MAX_CHARS = int(os.environ.get("EPMC_MAX_CHARS", "120000"))
EPMC_ON = os.environ.get("FULLTEXT_EPMC", "1") not in ("0", "false", "")

_XML_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"[ \t]*\n[ \t]*")


async def _epmc_get(path: str) -> tuple[bytes | None, str | None]:
    def _call() -> bytes:
        req = urllib.request.Request(f"{EPMC_BASE}/{path}", headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=EPMC_TIMEOUT_S, context=_SSL_CTX) as r:
            return r.read()
    try:
        async with SerialLock("europepmc", EPMC_COOLDOWN_S):
            metering.count("europepmc")
            return await asyncio.to_thread(_call), None
    except urllib.error.HTTPError as exc:
        return None, f"HTTPError: {exc.code}"
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"[:200]


async def europepmc_pmcid(doi: str) -> tuple[str, str | None]:
    """The PMCID of a DOI, only when Europe PMC holds the article itself
    (`inEPMC == "Y"`). KV-cached, including the negative answer."""
    d = normalise_doi(doi)
    if not d:
        return "", "not a DOI"
    hit = kv().get(_NS, f"epmc:{d}", max_age_days=CACHE_DAYS)
    if hit is not None:
        return hit.get("pmcid") or "", None
    q = urllib.parse.quote(f'DOI:"{d}"')
    body, err = await _epmc_get(f"search?query={q}&format=json&resultType=core")
    if err:
        return "", err
    try:
        res = (json.loads(body).get("resultList") or {}).get("result") or []
    except Exception as exc:
        return "", f"{type(exc).__name__}: {exc}"[:200]
    pmcid = ""
    if res and res[0].get("inEPMC") == "Y":
        pmcid = str(res[0].get("pmcid") or "")
    kv().put(_NS, f"epmc:{d}", {"pmcid": pmcid})
    return pmcid, None


def jats_to_text(xml: bytes | str) -> str:
    """Readable text out of a JATS full-text document. Not a parser: the body
    is what the evaluation reads, so tags go and the words stay."""
    s = xml.decode("utf-8", "replace") if isinstance(xml, bytes) else xml
    for tag in ("ref-list", "back", "table-wrap", "fig", "front-stub"):
        s = re.sub(rf"(?is)<{tag}\b.*?</{tag}>", " ", s)
    s = re.sub(r"(?is)<(title|p|sec|abstract|article-title)\b[^>]*>", "\n", s)
    s = _XML_TAG_RE.sub(" ", s)
    s = (s.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
          .replace("&quot;", '"').replace("&#x2019;", "'").replace("&apos;", "'"))
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = _WS_RE.sub("\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()[:EPMC_MAX_CHARS]


async def europepmc_text(doi: str) -> tuple[str, str | None]:
    """Open-access full text for a DOI through Europe PMC, or ("", reason)."""
    if not EPMC_ON:
        return "", "Europe PMC disabled"
    pmcid, err = await europepmc_pmcid(doi)
    if err:
        return "", f"Europe PMC lookup: {err}"
    if not pmcid:
        return "", "not in Europe PMC's open-access set"
    body, err = await _epmc_get(f"{pmcid}/fullTextXML")
    if err:
        return "", f"Europe PMC {pmcid}: {err}"
    text = jats_to_text(body or b"")
    if len(text) < 500:
        return "", f"Europe PMC {pmcid}: full text too short to be the article"
    return text, None


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


def build_plan(doc: dict, doi: str, payload: dict | None) -> list[dict]:
    """The ordered list of things to try for one paper, cheapest and least
    contentious first.

        arXiv -> repository copy -> BioC-PMC -> preprint server ->
        Crossref TDM -> publisher copy -> abstract-only

    The point of the order is that the parties who publish full text *for
    programmatic use* are asked before the parties who put bot management in
    front of it. Putting the publisher first — which is what the previous chain
    did, by reading `url_for_pdf` out of whichever location happened to have one
    — meant the hard cases were attempted first and the easy ones never.

    Steps with no `url` are handled by a named fetcher in `acquire()` (BioC-PMC
    returns text, not a file). Steps whose URL can only be discovered by asking
    (Crossref TDM, `citation_pdf_url`, CORE, Wayback) are marked `lazy` and cost
    nothing unless everything before them failed.
    """
    plan: list[dict] = []
    aid = arxiv_id_of(doc.get("arxiv_id"), doc.get("pub_num"),
                      _known_pdf_url(doc), doc.get("url"))
    if aid:
        plan.append({"source": "arxiv", "url": arxiv_pdf_url(aid), "tier": "arxiv",
                     "detail": f"arXiv:{aid}"})

    known = _known_pdf_url(doc)
    known_is_repo = bool(known) and is_repository_url(known)
    if known_is_repo:
        plan.append({"source": "repository", "url": known, "tier": "oa",
                     "detail": "repository PDF from OpenAlex/Semantic Scholar"})

    rows = oa_locations_ordered(payload)
    for r in rows:
        if r["host_type"] == "repository":
            plan.append({"source": "repository", "url": r["url"], "tier": "oa",
                         "detail": f"Unpaywall repository copy ({r['kind']})"})

    if doi:
        plan.append({"source": "bioc_pmc", "tier": "oa",
                     "detail": "NCBI BioC API (PubMed Central open-access subset)"})

    pre = fs.preprint_pdf_url(doi, f"{doc.get('venue', '')} {doc.get('journal', '')}")
    if pre:
        plan.append({"source": "preprint", "url": pre, "tier": "oa",
                     "detail": "bioRxiv/medRxiv direct PDF"})
    if doi.startswith("10.26434/"):
        plan.append({"source": "chemrxiv", "tier": "oa", "lazy": True,
                     "detail": "ChemRxiv (Cambridge Open Engage API)"})
    if doi.startswith("10.7554/elife."):
        plan.append({"source": "elife", "tier": "oa", "lazy": True,
                     "detail": "eLife article API"})

    if doi:
        plan.append({"source": "crossref_tdm", "tier": "oa", "lazy": True,
                     "detail": "Crossref link[] marked intended-application: text-mining"})

    if known and not known_is_repo:
        plan.append({"source": "publisher", "url": known, "tier": "oa",
                     "detail": "publisher PDF from OpenAlex/Semantic Scholar"})
    for r in rows:
        if r["host_type"] != "repository":
            plan.append({"source": "publisher", "url": r["url"], "tier": "oa",
                         "detail": f"Unpaywall {r['host_type'] or 'unknown host'} copy ({r['kind']})"})

    if doi:
        plan.append({"source": "citation_pdf_url", "tier": "oa", "lazy": True,
                     "detail": "citation_pdf_url meta tag on the landing page"})
        plan.append({"source": "core", "tier": "oa", "lazy": True,
                     "detail": "CORE aggregator"})
        plan.append({"source": "wayback", "tier": "oa", "lazy": True,
                     "detail": "Internet Archive snapshot"})
        plan.append({"source": "wiley_tdm", "tier": "oa", "lazy": True,
                     "detail": "Wiley TDM API (needs WILEY_TDM_TOKEN)"})
        plan.append({"source": "elsevier_tdm", "tier": "oa", "lazy": True,
                     "detail": "Elsevier article API (needs ELSEVIER_TDM_KEY)"})
    return plan


async def resolve(doc: dict, use_unpaywall: bool = True) -> dict:
    """Best open-access copy for one paper-shaped doc dict.

    Returns {"fulltext_tier", "fulltext_url", "landing_page", "doi",
    "fulltext_detail", "fulltext_source", "fulltext_plan"} — a patch, so callers
    stay clear of in-place mutation of graph state.

    `fulltext_tier` is how far *resolution* got and `fulltext_url` is the first
    thing worth trying; neither is a claim that anything was read. `acquire()`
    is what actually reads, and `fulltext_download` is what it reports. Keeping
    the two apart is the whole point: on the 17 examiner-cited NPL gold, 6
    resolved and 0 returned a PDF under the original chain.
    """
    doi = normalise_doi(doc.get("doi") or doc.get("pub_num") or "")
    out = {"doi": doi, "landing_page": _landing_page(doc, doi)}

    payload, err = (None, None)
    if use_unpaywall and doi and not arxiv_id_of(doc.get("arxiv_id"), doc.get("pub_num"),
                                                 _known_pdf_url(doc), doc.get("url")):
        payload, err = await unpaywall(doi)

    plan = build_plan(doc, doi, payload)
    # Only a step that already *has* a URL counts as resolution. A plan full of
    # steps that might turn something up (BioC-PMC, Crossref TDM, the Internet
    # Archive) is not a copy found, and calling it one would be the same
    # overstating that the tier/download split exists to stop. Those steps stay
    # in the plan and `acquire()` still runs them — if one of them delivers, it
    # shows up in `fulltext_download`, which is where it belongs.
    eager = [s for s in plan if s.get("url")]
    if eager:
        first = eager[0]
        return {**out, "fulltext_tier": first["tier"], "fulltext_url": first["url"],
                "fulltext_source": first["source"], "fulltext_detail": first["detail"],
                "fulltext_plan": plan}
    detail = err or ("Unpaywall: not open access" if payload else
                     ("Unpaywall: DOI unknown" if doi else "no DOI and no open-access URL"))
    if plan:
        detail += f" — {len(plan)} API route(s) still to try"
    return {**out, "fulltext_tier": "abstract_only", "fulltext_url": "",
            "fulltext_source": "", "fulltext_detail": detail, "fulltext_plan": plan}


# ── acquisition: the step that decides whether anything was actually read ────

async def _lazy_url(source: str, doc: dict, patch: dict) -> tuple[str, str]:
    """(url, why-not) for a plan step whose URL has to be asked for."""
    doi = patch.get("doi", "")
    if source == "crossref_tdm":
        links = fs.tdm_links(await fs.crossref_work(doi))
        return (links[0]["url"], "") if links else ("", "no text-mining link in Crossref")
    if source == "citation_pdf_url":
        u = await fs.citation_pdf_url(patch.get("landing_page", ""))
        return (u, "") if u else ("", "no citation_pdf_url on the landing page")
    if source == "core":
        return await fs.core_pdf_url(doi)
    if source == "wayback":
        return await fs.wayback_snapshot(patch.get("fulltext_url")
                                         or patch.get("landing_page", ""))
    if source == "chemrxiv":
        u = await fs.chemrxiv_pdf_url(doi)
        return (u, "") if u else ("", "not found on ChemRxiv")
    if source == "elife":
        u = await fs.elife_pdf_url(doi)
        return (u, "") if u else ("", "not found in the eLife API")
    return "", f"no fetcher for {source}"


async def acquire(doc: dict, patch: dict, max_steps: int = 14) -> dict:
    """Walk the plan until something comes back, and say honestly what came.

    Returns {"pdf": bytes|None, "text": str, "fulltext_download", "fulltext_source",
    "fulltext_detail", "attempts": [...]}. `fulltext_download` is `ok` for a PDF,
    `ok_text` for full text through an API, `failed` when every step was tried
    and none produced anything, `no_url` when there was nothing to try.

    A step counts as a success only when the bytes start with `%PDF` or the text
    is long enough to be an article. A Cloudflare challenge page served as
    `application/pdf` has been observed (N7b, onlinelibrary.wiley.com), so the
    content type is never the evidence.
    """
    attempts: list[dict] = []
    plan = patch.get("fulltext_plan") or []
    if not plan:
        return {"pdf": None, "text": "", "fulltext_download": "no_url",
                "fulltext_source": "", "fulltext_detail": patch.get("fulltext_detail", ""),
                "attempts": attempts}

    for step in plan[:max_steps]:
        src = step["source"]
        if src == "bioc_pmc":
            text, why = await fs.bioc_pmc_text(patch.get("doi", ""))
            attempts.append({"source": src, "ok": bool(text), "detail": why or f"{len(text)} chars"})
            if text:
                return {"pdf": None, "text": text, "fulltext_download": "ok_text",
                        "fulltext_source": src, "attempts": attempts,
                        "fulltext_detail": f"{step['detail']} ({len(text)} chars)"}
            continue
        if src == "elsevier_tdm":
            text, why = await fs.elsevier_tdm_text(patch.get("doi", ""))
            attempts.append({"source": src, "ok": bool(text), "detail": why or f"{len(text)} chars"})
            if text:
                return {"pdf": None, "text": text, "fulltext_download": "ok_text",
                        "fulltext_source": src, "attempts": attempts,
                        "fulltext_detail": f"{step['detail']} ({len(text)} chars)"}
            continue
        if src == "wiley_tdm":
            data, why = await fs.wiley_tdm_pdf(patch.get("doi", ""))
            attempts.append({"source": src, "ok": bool(data), "detail": why})
            if data:
                return {"pdf": data, "text": "", "fulltext_download": "ok",
                        "fulltext_source": src, "attempts": attempts,
                        "fulltext_detail": f"{step['detail']} ({why})"}
            continue

        url = step.get("url", "")
        if not url and step.get("lazy"):
            url, why = await _lazy_url(src, doc, patch)
            if not url:
                attempts.append({"source": src, "ok": False, "detail": why})
                continue
        if not url:
            continue
        data, why = await fs.get_pdf(url)
        attempts.append({"source": src, "ok": bool(data), "url": url[:160], "detail": why})
        if data:
            return {"pdf": data, "text": "", "fulltext_download": "ok",
                    "fulltext_source": src, "attempts": attempts,
                    "fulltext_detail": f"{step['detail']} ({why})"}

    tried = ", ".join(f"{a['source']}: {a['detail']}" for a in attempts[:4])
    return {"pdf": None, "text": "", "fulltext_download": "failed" if attempts else "no_url",
            "fulltext_source": "", "attempts": attempts,
            "fulltext_detail": f"every open-access route failed — {tried}"[:400]}


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


def read_counts(docs: list[dict], patches: list[dict]) -> dict[str, int]:
    """Per tier, how many documents ended up with the paper in hand - a PDF, or
    Europe PMC full text. Resolving a URL
    and holding the paper are different numbers and the report prints both:
    counting a resolved-but-unfetchable paper as read is exactly the overstating
    this module exists to stop."""
    counts = {t: 0 for t in TIERS}
    for doc, p in zip(docs, patches):
        if doc.get("local_pdf") or doc.get("oa_full_text"):
            counts[p.get("fulltext_tier", "abstract_only")] += 1
    return counts


# ── manual-download manifest (the replacement for the dropped proxied tier) ──

_DL_REASON = {"failed": "a URL was resolved but it returned no readable PDF",
              "no_url": "no URL to fetch",
              "skipped_budget": "the shared download budget ran out first"}


def manifest_rows(docs: list[dict], patches: list[dict]) -> list[dict]:
    """Every paper still without a readable PDF, as a list a person can work
    through by hand: DOI, title, landing page, and why the automatic tiers gave
    up. Opening these pages and saving the PDFs is reading, not scripted
    downloading.

    Membership is decided on `local_pdf`, not on the tier: a document can
    resolve to an open-access URL and still arrive with nothing, which is the
    common case rather than the exception (measured on the 17 examiner-cited
    NPL gold: 6 resolved to an OA URL, 0 returned a PDF — publisher hosts 403 a
    plain HTTP client and PMC puts a proof-of-work challenge in front of the
    file). A list built from the tier alone would quietly omit those.
    """
    rows = []
    for doc, p in zip(docs, patches):
        if doc.get("local_pdf") or doc.get("oa_full_text"):
            continue
        why = p.get("fulltext_detail", "")
        dl = doc.get("fulltext_download") or ""
        if dl in _DL_REASON:
            why = f"{why} — {_DL_REASON[dl]}".strip(" —")
        rows.append({"doi": p.get("doi", ""), "title": (doc.get("title") or "").strip(),
                     "landing_page": p.get("landing_page", ""),
                     "tier": p.get("fulltext_tier", ""), "reason": why})
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
