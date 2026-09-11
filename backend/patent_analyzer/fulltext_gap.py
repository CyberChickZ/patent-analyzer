"""Which evaluated references reached Phase 4 without their own full text.

The open-access chain in `fulltext.py` answers "can we fetch this?" before the
download. This module answers the question after the run: *of the references
the report is standing on, which ones did a model never actually read, and what
was tried for each*. The two are not the same list — a paper can resolve to an
arXiv PDF and still arrive with an abstract because the download budget ran out
— and only the second list is actionable by a person.

It is actionable because the remedy is manual. There is no automatic paywalled
tier and there will not be one (fulltext.py carries the policy text and the
reason). What replaces it is this list plus an upload slot: a person opens the
landing page, saves the PDF, uploads it, and the evidence step is re-run for
those references alone.

Everything here is derived from `results.json`, so it works on a finished job
whose graph checkpoint is long gone, and degrades on jobs that predate the
tiers: an unknown tier is reported as unknown rather than as a failure.
"""

from __future__ import annotations

import re

# `source` values that mean the model read the document's own text. Anything
# else — an abstract, or nothing at all — is a reference the report has not
# really checked. eval_subgraph.NOT_READ_SOURCES is the narrower set (nothing
# was read at all); "abstract" sits between the two and belongs on this list,
# because an abstract cannot establish that an element is or is not disclosed.
READ_SOURCES = ("pdf", "full_text")

POLICY_NOTE = (
    "Nothing here is fetched automatically. OSU Libraries' Responsible Use of Licensed "
    "Electronic Resources forbids \"using scripts, spiders, crawlers, or other computer "
    "programs to automatically download content\", and says a violation \"may suspend "
    "access for the entire OSU community\". Open the landing page yourself, save the PDF, "
    "and upload it here."
)
POLICY_URL = "https://library.oregonstate.edu/responsible-use-licensed-electronic-resources"


def ref_id(row: dict) -> str:
    """A stable, URL-safe id for one evaluated reference.

    Keyed on `pub_num` because that is what the scoring report, the claim chart
    and `adjudication.per_doc_coverage` all key on; a document with no pub_num
    (some paper channels return none) falls back to its title, which is what
    those same places fall back to.
    """
    raw = str(row.get("pub_num") or "").strip() or str(row.get("title") or "").strip()
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw.lower()).strip("-")
    return slug[:120] or "ref"


def _is_patentish(row: dict) -> bool:
    return (row.get("match_type") == "Patent"
            or bool(re.match(r"^[A-Z]{2}[-\s]?\d", str(row.get("pub_num") or "").upper())))


def _attempts(row: dict) -> list[dict]:
    """The tier trail for one reference: what was tried, and how it ended.

    `outcome` is one of ok / missed / failed / skipped / unknown. Each entry is
    written from a field the pipeline actually recorded; where a job predates
    that field the outcome is "unknown", never an invented failure.
    """
    tier = row.get("fulltext_tier") or ""
    detail = str(row.get("fulltext_detail") or "")
    source = str(row.get("source") or "")
    dl = str(row.get("fulltext_download") or "")
    out: list[dict] = []

    if _is_patentish(row):
        had = bool(row.get("had_bq_claims"))
        out.append({"tier": "bigquery_claims", "outcome": "ok" if had else "missed",
                    "detail": "claims text from our own copy of the patent corpus" if had else
                              "no claims in amie_patents (non-US publication, or not in our copy)"})
    if not _is_patentish(row) or tier:
        out.append({"tier": "arxiv",
                    "outcome": "ok" if tier == "arxiv" else ("unknown" if not tier else "missed"),
                    "detail": detail if tier == "arxiv" else
                              ("this job ran before the open-access tiers were recorded"
                               if not tier else "no arXiv id on the record")})
        if tier == "oa":
            out.append({"tier": "oa", "outcome": "ok", "detail": detail or "open-access copy found"})
        elif tier == "abstract_only":
            out.append({"tier": "oa", "outcome": "failed",
                        "detail": detail or "no open-access copy (OpenAlex / Semantic Scholar / Unpaywall)"})
        elif tier != "arxiv":
            out.append({"tier": "oa", "outcome": "unknown",
                        "detail": "this job ran before the open-access tiers were recorded"})

    url = row.get("fulltext_url") or ""
    if dl:
        words = {"ok": ("ok", "the PDF was downloaded"),
                 "cached": ("ok", "served from the GCS cache, keyed by DOI"),
                 "failed": ("failed", "the download did not return a readable PDF"),
                 "no_url": ("skipped", "no URL to fetch"),
                 "skipped_budget": ("skipped", "the shared PDF-download budget ran out first"),
                 "not_needed": ("skipped", "the claims text was already in hand")}
        oc, why = words.get(dl, ("unknown", dl))
        out.append({"tier": "pdf_download", "outcome": oc, "detail": why})
    elif source == "pdf":
        out.append({"tier": "pdf_download", "outcome": "ok", "detail": "the PDF was downloaded and read"})
    elif url:
        out.append({"tier": "pdf_download", "outcome": "failed",
                    "detail": "a URL was resolved but no readable PDF reached the deep read"})
    else:
        out.append({"tier": "pdf_download", "outcome": "skipped", "detail": "no URL to fetch"})

    return out


def _read_state(row: dict) -> tuple[str, str]:
    """(state, why) of what the deep read actually had. Not a judgement of the
    document — a statement about our copy of it."""
    source = str(row.get("source") or "")
    if source in READ_SOURCES:
        return "full_text", ""
    why = str(row.get("no_content_reason") or "")
    if source == "abstract":
        return "abstract_only", why or "only the abstract was available"
    if source in ("abstract_failed", "abstract_noparse"):
        return "nothing", why or f"the abstract read returned nothing ({source})"
    return "nothing", why or "no text reached a model"


def gap_rows(results: dict, uploads: dict | None = None) -> list[dict]:
    """One row per evaluated reference whose full text we do not have.

    `uploads` is the job record's `fulltext_uploads` map (ref_id -> upload
    record); a reference that has one is still listed, with its upload attached,
    so the page can show what is staged and what the re-read has already used.
    """
    uploads = uploads or {}
    sr = ((results or {}).get("evaluation") or {}).get("scoring_report") or []
    manifest = (((results or {}).get("search") or {}).get("summary") or {}).get("fulltext_oa") or {}
    by_title = {str(m.get("title") or "").strip().lower(): m for m in (manifest.get("manifest") or [])}

    from .fulltext import normalise_doi

    rows = []
    for r in sr:
        if r.get("is_source_duplicate"):
            continue
        state, why = _read_state(r)
        rid = ref_id(r)
        up = uploads.get(rid)
        if state == "full_text" and not up:
            continue
        # A job that predates the per-document stamp still has the manual
        # manifest in search_stats; it is keyed by title, which is the only key
        # both records share.
        m = by_title.get(str(r.get("title") or "").strip().lower()) or {}
        row = {
            "ref_id": rid,
            "pub_num": r.get("pub_num") or "",
            "title": r.get("title") or "",
            "match_type": r.get("match_type") or "",
            # The paper channels put the DOI in `pub_num`, and on a job that ran
            # before the per-document stamp that is the only copy of it. It is
            # the key the GCS full-text cache is named by, so it is worth
            # recovering rather than showing a reference with no identifier.
            "doi": r.get("doi") or m.get("doi") or normalise_doi(r.get("pub_num") or ""),
            "landing_page": r.get("landing_page") or m.get("landing_page") or "",
            "fulltext_url": r.get("fulltext_url") or "",
            "fulltext_tier": r.get("fulltext_tier") or "",
            "read_state": state,
            "read_reason": why or m.get("reason") or "",
            "text_chars": int(r.get("text_chars") or 0),
            "similarity_score": float(r.get("similarity_score") or 0.0),
            # Being in the manual manifest *is* the abstract_only tier —
            # fulltext.manifest_rows emits nothing else — so a job whose rows
            # predate the per-row stamp still gets a real trail, not "unknown".
            "attempts": _attempts({**r,
                                   "fulltext_tier": r.get("fulltext_tier") or ("abstract_only" if m else ""),
                                   "fulltext_detail": r.get("fulltext_detail") or m.get("reason") or ""}),
            "upload": up or None,
        }
        if not row["landing_page"] and row["doi"]:
            row["landing_page"] = f"https://doi.org/{row['doi']}"
        rows.append(row)

    # Highest-scoring first: a reference that already looks close on an abstract
    # is the one whose full text can still change the determination.
    rows.sort(key=lambda x: (-x["similarity_score"], x["title"]))
    return rows


def gap_summary(results: dict, rows: list[dict]) -> dict:
    sr = ((results or {}).get("evaluation") or {}).get("scoring_report") or []
    evaluated = sum(1 for r in sr if not r.get("is_source_duplicate"))
    return {
        "evaluated": evaluated,
        "missing": sum(1 for r in rows if r["read_state"] != "full_text"),
        "abstract_only": sum(1 for r in rows if r["read_state"] == "abstract_only"),
        "nothing": sum(1 for r in rows if r["read_state"] == "nothing"),
        "uploaded": sum(1 for r in rows if r.get("upload")),
        "pending_reread": sum(1 for r in rows if (r.get("upload") or {}).get("reread") is False),
        "full_text": evaluated - sum(1 for r in rows if r["read_state"] != "full_text"),
    }


# ── report section: which references were re-read from a supplied PDF ────────
#
# A row re-read after an upload carries different evidence from the one the
# original run produced, and the determination above it may have been recomputed
# because of it. The report has to say so: otherwise a §102 that only appeared
# after a reviewer supplied one paper reads exactly like one the pipeline found
# on its own.

UPLOAD_TIER = "user_upload"


def uploaded_rows(scoring_report: list[dict] | None) -> list[dict]:
    return [r for r in (scoring_report or []) if r.get("fulltext_tier") == UPLOAD_TIER]


def _covered_count(row: dict) -> tuple[int, int]:
    cr = row.get("checklist_results") or {}
    n = sum(1 for v in cr.values()
            if isinstance(v, dict) and (v.get("score") or 0) >= 1 and v.get("verified_quotes"))
    return n, len(cr)


def uploaded_fulltext_html(scoring_report: list[dict] | None) -> str:
    rows = uploaded_rows(scoring_report)
    if not rows:
        return ""
    from html import escape as _e
    out = ['<div class="sec"><div class="sec-t">Full Text Supplied by Hand</div>',
           f'<div class="sec-note">{len(rows)} reference(s) below were not reachable during the run and '
           'were re-read from a PDF a reviewer uploaded afterwards. The determination above was '
           'recomputed over this evidence.</div>',
           '<table class="tbl"><thead><tr><th>#</th><th>Reference</th><th>Identifier</th>'
           '<th>Elements covered</th><th>Where the text came from</th></tr></thead><tbody>']
    for i, r in enumerate(rows, 1):
        n, total = _covered_count(r)
        out.append(f'<tr><td>{i}</td><td>{_e((r.get("title") or "")[:90])}</td>'
                   f'<td>{_e(r.get("pub_num") or r.get("doi") or "—")}</td>'
                   f'<td>{n} of {total}</td><td>{_e(r.get("fulltext_detail") or "uploaded")}</td></tr>')
    out.append("</tbody></table></div>")
    return "\n".join(out)


def uploaded_fulltext_md(scoring_report: list[dict] | None) -> list[str]:
    rows = uploaded_rows(scoring_report)
    if not rows:
        return []
    out = ["## Full Text Supplied by Hand", "",
           f"{len(rows)} reference(s) below were not reachable during the run and were re-read from a "
           "PDF a reviewer uploaded afterwards. The determination above was recomputed over this evidence.",
           "", "| # | Reference | Identifier | Elements covered | Where the text came from |",
           "|---|---|---|---|---|"]
    for i, r in enumerate(rows, 1):
        n, total = _covered_count(r)
        out.append(f"| {i} | {(r.get('title') or '')[:90]} | {r.get('pub_num') or r.get('doi') or '-'} | "
                   f"{n} of {total} | {r.get('fulltext_detail') or 'uploaded'} |")
    out.append("")
    return out
