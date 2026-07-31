"""H2 data: claim-level §102 / §103 / ALLOW instances from PANORAMA's raw
parsed_CTNF (LG-AI-Research/PANORAMA, panorama.parquet, 8,143 apps).

Labels come from the first non-final office action as parsed by the dataset
authors. We borrow the labels, not the NOC4PC framing: each instance carries
its own examiner-cited references (chain-resolved for dependent claims), so
"single reference covers every element" (§102) and "union of references
covers, no single one does" (§103) are testable per claim.

Filters (leader_codebases.md §A2-C): drop claims rejected only under
§101/§112, cited references with an empty paragraph list, and reasons that
say "same reasons as claim N" / inherent / implicit / "does not explicitly".
§102 instances cite exactly one reference over the whole claim chain; §103
instances cite at least two. ALLOW = no rejection at all, in an application
that has §102/§103 rejections elsewhere.

Reference text: abstract + claims come from the parquet's own
patentsCitedByExaminer (same fields BigQuery would give, zero cost);
the description is scraped from patents.google.com one page at a time
(>= 4 s apart, at most --max-pages, cached under PANORAMA_ROOT/pages).
Documents that miss the page budget fall back to abstract + claims and
are marked text_mode="abstract_claims".

Usage:
    python3 evals/panorama_data.py --n102 40 --n103 40 --nallow 20 --max-pages 60
"""


import argparse

import html

import json

import os

import random

import re

import sys

import time

from collections import Counter

from pathlib import Path


sys.path.insert(0, str(Path(__file__).parent.parent))


PANORAMA_ROOT = Path(os.environ.get("PANORAMA_ROOT", "/tmp/panorama"))

PARQUET = PANORAMA_ROOT / "panorama.parquet"

PARQUET_URL = "https://huggingface.co/datasets/LG-AI-Research/PANORAMA/resolve/main/panorama.parquet"

PAGES = PANORAMA_ROOT / "pages"

RUN_DIR = Path(__file__).parent.parent / "eval_data" / "runs" / "h2"


_BAD_REASON = re.compile(r"same reasons? as (?:set forth in )?(?:claim|the)|inherent|implicit|does not explicitly", re.I)

_DEP_PREFIX = re.compile(r"^\s*\d+\s*\.\s*(?:The|A|An)\b[^,]{0,200}?\b(?:of|according to|as (?:recited |claimed |defined |set forth )?in)\s+claims?\s+\d+\s*,?\s*", re.I)

_CLAIM_NUM = re.compile(r"^\s*(\d+)\s*\.\s*")

_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

_PAGE_GAP_S = 4.0

_last_page_call = 0.0



def norm_pub(p: str) -> str:
    """'US 2005/0025220 A1', '20050025220', 'US7123456B2' -> digits only."""
    return re.sub(r"[^0-9]", "", p or "")



def ensure_parquet() -> Path:
    if not PARQUET.exists():
        import urllib.request
        PARQUET.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(PARQUET_URL, PARQUET)
    return PARQUET



def load_rows(limit: int | None = None) -> list[dict]:
    import pyarrow.parquet as pq
    t = pq.read_table(ensure_parquet(), columns=[
        "id", "applicationNumber", "earliestPublicationNumber", "initialClaims",
        "patentsCitedByExaminer", "parsed_CTNF"])
    if limit:
        t = t.slice(0, limit)
    return t.to_pylist()



def _claims_by_number(initial_claims: list[str]) -> dict[int, str]:
    out = {}
    for c in initial_claims or []:
        m = _CLAIM_NUM.match(c or "")
        if m:
            out[int(m.group(1))] = c.strip()
    return out



def _dependent_body(text: str) -> str:
    """'2. The method of claim 1, wherein X' -> 'wherein X'."""
    body = _DEP_PREFIX.sub("", text or "", count=1)
    if body == (text or ""):
        body = _CLAIM_NUM.sub("", text or "", count=1)
    return body.strip()



def resolve_chain(parsed: dict[int, dict], texts: dict[int, str], num: int) -> tuple[list[int], str]:
    """Walk parentClaim links to the root; return (chain numbers root..num,
    root text + each dependent's added body joined by '; ')."""
    chain, seen = [], set()
    cur = num
    while cur in parsed and cur not in seen:
        seen.add(cur)
        chain.append(cur)
        parent = parsed[cur].get("parentClaim", -1)
        if parent in (None, -1, 0) or parent == cur:
            break
        cur = parent
    chain.reverse()
    if not chain:
        return [num], texts.get(num, "")
    root = _CLAIM_NUM.sub("", texts.get(chain[0], ""), count=1).strip().rstrip(".")
    parts = [root] + [_dependent_body(texts.get(n, "")).rstrip(".") for n in chain[1:]]
    return chain, "; ".join(p for p in parts if p)



def _prior_art_reasons(claim: dict) -> list[dict]:
    return [r for r in claim.get("reasons") or [] if r.get("sectionCode") in (102, 103)]



def claim_instances(row: dict) -> list[dict]:
    """All usable (claim, label, cited) instances of one application."""
    try:
        parsed_list = json.loads(row.get("parsed_CTNF") or "null")
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed_list, list):
        return []
    parsed = {c["claimNumber"]: c for c in parsed_list if isinstance(c, dict) and "claimNumber" in c}
    texts = _claims_by_number(row.get("initialClaims") or [])
    if not any(_prior_art_reasons(c) for c in parsed.values()):
        return []
    app_pub_count: Counter = Counter()
    for c in parsed.values():
        for r in _prior_art_reasons(c):
            for cp in r.get("citedPatents") or []:
                app_pub_count[norm_pub(cp.get("patentNum"))] += 1
    out = []
    for num, c in parsed.items():
        if num not in texts:
            continue
        chain, chain_text = resolve_chain(parsed, texts, num)
        if any(n not in texts for n in chain):
            continue
        own = _prior_art_reasons(c)
        chain_reasons = [r for n in chain for r in _prior_art_reasons(parsed[n])]
        if not own:
            if c.get("isReject"):
                continue  # rejected only under §101/§112: neither ALLOW nor prior-art
            if any(parsed[n].get("isReject") and not _prior_art_reasons(parsed[n]) for n in chain[:-1]):
                continue
            label = "ALLOW"
        else:
            if any(_BAD_REASON.search(r.get("reason") or "") for r in chain_reasons):
                continue
            cps = [cp for r in chain_reasons for cp in r.get("citedPatents") or []]
            if not cps or any(not cp.get("text") for cp in cps):
                continue
            secs = {r.get("sectionCode") for r in own}
            chain_secs = {r.get("sectionCode") for r in chain_reasons}
            n_pubs = len({norm_pub(cp.get("patentNum")) for cp in cps})
            if secs == {102} and chain_secs == {102} and n_pubs == 1:
                label = "102"
            elif secs == {103} and n_pubs >= 2:
                label = "103"
            else:
                continue
        cited: dict[str, dict] = {}
        for r in chain_reasons:
            for cp in r.get("citedPatents") or []:
                key = norm_pub(cp.get("patentNum"))
                d = cited.setdefault(key, {"patentNum": cp.get("patentNum"), "pub": key, "paragraphs": []})
                d["paragraphs"] = sorted(set(d["paragraphs"]) | {int(x) for x in cp.get("text") or [] if str(x).isdigit()})
        if label == "ALLOW" and not cited:
            for key, _ in app_pub_count.most_common(3):
                cited[key] = {"patentNum": key, "pub": key, "paragraphs": []}
        if not cited:
            continue
        out.append({
            "app": row.get("applicationNumber"),
            "app_pub": row.get("earliestPublicationNumber"),
            "row_id": row.get("id"),
            "claimNumber": num,
            "chain": chain,
            "is_dependent": len(chain) > 1,
            "claim_text": texts[num],
            "chain_text": chain_text,
            "label": label,
            "cited": list(cited.values()),
            "reasons": [{"claim": n, "section": r.get("sectionCode"), "reason": r.get("reason", "")}
                        for n in chain for r in _prior_art_reasons(parsed[n])],
        })
    return out


