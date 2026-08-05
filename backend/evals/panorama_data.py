#!/usr/bin/env python3
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
the description comes from the dataset's spec_cited.zip, read member by
member over HTTP Range (no 10.9 GB download; cached under
PANORAMA_ROOT/spec), falling back to a patents.google.com page (>= 4 s
apart, at most --max-pages, cached under PANORAMA_ROOT/pages). Documents
with neither fall back to abstract + claims (text_mode="abstract_claims").

Usage:
    python3 evals/panorama_data.py --n102 40 --n103 40 --nallow 20 --max-pages 60
"""

import argparse
import html
import io
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
SPEC_DIR = PANORAMA_ROOT / "spec"
SPEC_ZIP_URL = "https://huggingface.co/datasets/LG-AI-Research/PANORAMA/resolve/main/spec_cited.zip"
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


def sample_instances(rows: list[dict], n102: int = 40, n103: int = 40, nallow: int = 20,
                     seed: int = 42, per_app_per_label: int = 2) -> list[dict]:
    """Seeded sample, at most `per_app_per_label` claims per label per
    application, independent claims first so the §102/§103 sets are not
    dominated by 'wherein' one-liners."""
    rng = random.Random(seed)
    order = list(range(len(rows)))
    rng.shuffle(order)
    quota = {"102": n102, "103": n103, "ALLOW": nallow}
    picked: list[dict] = []
    for i in order:
        if all(v <= 0 for v in quota.values()):
            break
        inst = claim_instances(rows[i])
        if not inst:
            continue
        for label in ("102", "103", "ALLOW"):
            if quota[label] <= 0:
                continue
            cands = [x for x in inst if x["label"] == label]
            if not cands:
                continue
            rng.shuffle(cands)
            cands.sort(key=lambda x: (x["is_dependent"], len(x["cited"])))
            for x in cands[:min(per_app_per_label, quota[label])]:
                picked.append(x)
                quota[label] -= 1
    return picked


def cited_texts(rows: list[dict]) -> dict[str, dict]:
    """pub digits -> {title, abstract, claims[]} from patentsCitedByExaminer."""
    out = {}
    for r in rows:
        for c in r.get("patentsCitedByExaminer") or []:
            key = norm_pub(c.get("referenceIdentifier"))
            if key and key not in out:
                out[key] = {"title": c.get("title") or "", "abstract": c.get("abstract") or "",
                            "claims": [x for x in (c.get("claims") or []) if x]}
    return out


def _gp_pub(pub_digits: str) -> str:
    return f"US{pub_digits}"


def _clean(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ")


def parse_patent_page(h: str) -> dict:
    """Title, abstract, claims, numbered description paragraphs from a
    patents.google.com page. Pre-grant pages use class="description-line",
    granted ones class="description-paragraph"."""
    m_t = re.search(r'<meta name="DC.title" content="([^"]*)"', h)
    m_abs = re.search(r'<section itemprop="abstract".*?<div[^>]*class="abstract"[^>]*>(.*?)</div>', h, re.S)
    claims = [re.sub(r"\s+", " ", _clean(b)).strip()
              for b in re.findall(r'<div id="CLM-\d+"[^>]*>.*?(?=<div id="CLM-\d+"|</section>)', h, re.S)]
    desc = []
    sec = re.search(r'<section itemprop="description".*?</section>', h, re.S)
    body = sec.group(0) if sec else h
    for num, chunk in re.findall(r'<div (?:id="[pP]-\d+" )?num="(?:p-)?(\d+)" class="description-(?:line|paragraph)">(.*?)(?=<div (?:id="[pP]-\d+" )?num=|</section>|<para-num)', body, re.S):
        txt = re.sub(r"\s+", " ", _clean(chunk)).strip()
        if txt:
            desc.append({"num": int(num), "text": txt})
    if not desc and body:
        txt = re.sub(r"\s+", " ", _clean(body)).strip()
        desc = [{"num": i + 1, "text": p} for i, p in enumerate(re.split(r"(?<=\.)\s{2,}", txt)) if len(p) > 40]
    return {"title": html.unescape(m_t.group(1)).strip() if m_t else "",
            "abstract": re.sub(r"\s+", " ", _clean(m_abs.group(1))).strip() if m_abs else "",
            "claims": [c for c in claims if c], "description": desc}


def fetch_page(pub_digits: str) -> dict | None:
    """One Google Patents page, >= 4 s after the previous one, cached on disk.
    Returns None when blocked or not found (caller falls back)."""
    global _last_page_call
    import httpx
    PAGES.mkdir(parents=True, exist_ok=True)
    cache = PAGES / f"{pub_digits}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    blocked = PAGES / "BLOCKED"
    if blocked.exists() and time.time() - blocked.stat().st_mtime < 15 * 60:
        return None
    wait = _PAGE_GAP_S - (time.time() - _last_page_call)
    if wait > 0:
        time.sleep(wait)
    _last_page_call = time.time()
    url = f"https://patents.google.com/patent/{_gp_pub(pub_digits)}/en"
    try:
        r = httpx.get(url, headers={"User-Agent": _UA}, timeout=30, follow_redirects=True)
    except httpx.HTTPError as e:
        print(f"  page {pub_digits}: {e}", file=sys.stderr)
        return None
    if r.status_code != 200 or "<title>Sorry" in r.text[:400]:
        if r.status_code == 429 or "<title>Sorry" in r.text[:400]:
            blocked.write_text(url)
            print(f"  page {pub_digits}: blocked ({r.status_code})", file=sys.stderr)
        else:
            print(f"  page {pub_digits}: HTTP {r.status_code}", file=sys.stderr)
        return None
    parsed = parse_patent_page(r.text)
    parsed["pub"] = pub_digits
    parsed["url"] = str(r.url)
    cache.write_text(json.dumps(parsed))
    return parsed


class _HTTPRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file via HTTP Range requests, so
    zipfile can read the central directory and single members of the 10.9 GB
    spec_cited.zip without downloading it (~12 MB for the directory, then
    10-30 KB per member)."""

    def __init__(self, url: str):
        import urllib.request
        r = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}))
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.url = r.geturl()
        self.pos = 0

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def tell(self):
        return self.pos

    def readable(self):
        return True

    def seekable(self):
        return True

    def read(self, n=-1):
        import urllib.request
        if n == -1:
            n = self.size - self.pos
        if n <= 0:
            return b""
        end = min(self.size, self.pos + n) - 1
        r = urllib.request.urlopen(urllib.request.Request(self.url, headers={"Range": f"bytes={self.pos}-{end}"}))
        data = r.read()
        self.pos += len(data)
        return data


_spec_zip = None


def fetch_spec_text(pub_digits: str) -> str | None:
    """Raw description text of a cited patent from PANORAMA's own
    spec_cited.zip (member spec_cited/text/spec_txt_<pub>.txt), cached on disk."""
    global _spec_zip
    import zipfile
    SPEC_DIR.mkdir(parents=True, exist_ok=True)
    cache = SPEC_DIR / f"{pub_digits}.txt"
    if cache.exists():
        return cache.read_text() or None
    if _spec_zip is None:
        _spec_zip = zipfile.ZipFile(_HTTPRangeFile(SPEC_ZIP_URL))
    name = f"spec_cited/text/spec_txt_{pub_digits}.txt"
    try:
        raw = _spec_zip.read(name).decode("utf-8", "replace")
    except KeyError:
        cache.write_text("")
        return None
    cache.write_text(raw)
    return raw or None


_PARA_MARK = re.compile(r"(?m)^\s*(?:\[(\d{4})\]|\((\d{1,4})\))\s*")


def parse_spec_text(raw: str) -> list[dict]:
    """[{num, text}] from a spec_cited text: pre-grant files number paragraphs
    '[0012]', granted ones '(12)'; a heading line before a marker is kept at
    the end of the preceding paragraph (verbatim text is what quotes need)."""
    t = (raw or "").replace("\r\n", "\n")
    marks = list(_PARA_MARK.finditer(t))
    if not marks:
        chunks = [c.strip() for c in re.split(r"\n\s*\n", t) if c.strip()]
        return [{"num": i + 1, "text": re.sub(r"\s+", " ", c)} for i, c in enumerate(chunks)]
    out = []
    lead = t[:marks[0].start()].strip()
    if lead:
        out.append({"num": 0, "text": re.sub(r"\s+", " ", lead)})
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(t)
        body = re.sub(r"\s+", " ", t[m.end():end]).strip()
        if body:
            out.append({"num": int(m.group(1) or m.group(2)), "text": body})
    return out


def render_doc(doc: dict) -> str:
    """Text handed to the evaluator: title, abstract, numbered description, claims."""
    parts = [f"Title: {doc.get('title', '')}", ""]
    if doc.get("abstract"):
        parts += ["Abstract", doc["abstract"], ""]
    if doc.get("description"):
        parts += ["Description"] + [f"[{p['num']:04d}] {p['text']}" for p in doc["description"]] + [""]
    if doc.get("claims"):
        parts += ["Claims:"] + list(doc["claims"])
    return "\n".join(parts).strip() + "\n"


def build_docs(samples: list[dict], rows: list[dict], max_pages: int = 60) -> dict[str, dict]:
    """Document text per cited pub. Pages are fetched most-used first; the
    rest use abstract + claims from the parquet."""
    base = cited_texts(rows)
    use = Counter(c["pub"] for s in samples for c in s["cited"])
    docs: dict[str, dict] = {}
    pages_used = 0
    for pub, _ in use.most_common():
        doc = {"pub": pub, "title": "", "abstract": "", "claims": [], "description": [], "text_mode": "abstract_claims"}
        b = base.get(pub)
        if b:
            doc.update({"title": b["title"], "abstract": b["abstract"], "claims": b["claims"]})
        spec = None
        try:
            spec = fetch_spec_text(pub) if pub else None
        except Exception as e:  # network: fall through to the page scraper
            print(f"  spec {pub}: {e}", file=sys.stderr)
        if spec:
            doc["description"] = parse_spec_text(spec)
            doc["text_mode"] = "full_text"
            doc["desc_source"] = "spec_cited.zip"
        cached = (PAGES / f"{pub}.json").exists()
        if not doc["description"] and (cached or pages_used < max_pages):
            page = fetch_page(pub)
            if not cached:
                pages_used += 1
            if page and page.get("description"):
                doc["description"] = page["description"]
                doc["title"] = doc["title"] or page.get("title", "")
                doc["abstract"] = doc["abstract"] or page.get("abstract", "")
                doc["claims"] = doc["claims"] or page.get("claims", [])
                doc["text_mode"] = "full_text"
                doc["desc_source"] = "patents.google.com"
        if not (doc["abstract"] or doc["claims"] or doc["description"]):
            doc["text_mode"] = "missing"
        doc["text"] = render_doc(doc)
        docs[pub] = doc
    return docs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n102", type=int, default=40)
    ap.add_argument("--n103", type=int, default=40)
    ap.add_argument("--nallow", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pages", type=int, default=60)
    ap.add_argument("--no-pages", action="store_true", help="abstract + claims only")
    args = ap.parse_args()

    rows = load_rows()
    samples = sample_instances(rows, args.n102, args.n103, args.nallow, args.seed)
    docs = build_docs(samples, rows, max_pages=0 if args.no_pages else args.max_pages)
    for s in samples:
        for c in s["cited"]:
            c["text_mode"] = docs[c["pub"]]["text_mode"]
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    (RUN_DIR / "samples.json").write_text(json.dumps(samples, indent=1, ensure_ascii=False))
    (RUN_DIR / "docs.json").write_text(json.dumps(docs, ensure_ascii=False))
    labels = Counter(s["label"] for s in samples)
    dep = Counter((s["label"], s["is_dependent"]) for s in samples)
    modes = Counter(d["text_mode"] for d in docs.values())
    print(f"samples: {dict(labels)}  dependent: {dict(dep)}")
    print(f"docs: {len(docs)} unique, text modes {dict(modes)}, "
          f"avg chars {sum(len(d['text']) for d in docs.values()) // max(len(docs), 1)}")
    print(f"wrote {RUN_DIR / 'samples.json'}")


if __name__ == "__main__":
    main()
