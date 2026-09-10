"""Is this document one of the right ones? — the claims-level judge (M1).

An examiner cites a reference for what it claims or discloses, and the
abstract cannot carry that. Measured on h1o's US20150126796A1: the abstract
screen called 5,434 documents worth reading and the two examiner-cited
families sat at rank 598 and 1,841 of every abstract-level ordering we tried
(by elements touched, by cosine, by graph source). Reading the claims of a
2,000-document window judged both of them True — one at rank 51 — so the
signal is in the claims, not in the abstract (evals/claims_screen_probe.py).

The same probe kept 743 of 1,696 (44%), which is too generous to rank with.
So a document only counts as touching an element when the model can name the
CLAIM NUMBER that touches it (leader, 2026-09-18): pointing at a location is
harder to fake than a yes.

That alone was still too generous — m1a's first paper judged 17% of round 0
and 22% of round 1 GOOD, so every one of 15 elements was "covered by >=3
GOOD" after a single round and the loop stopped with reach 0/5. Hence STRONG:
a document counts towards coverage only when it touches at least two elements,
each with a claim number. A one-element GOOD is still a seed candidate for the
next round — it is just not evidence that an element is covered.
"""

from __future__ import annotations

import asyncio
import json
import os
import re

BATCH = int(os.environ.get("GOOD_BATCH", "8"))
CLAIMS_CHARS = int(os.environ.get("GOOD_CLAIMS_CHARS", "4000"))
CONCURRENCY = int(os.environ.get("GOOD_CONCURRENCY", "4"))
INDEPENDENT_ONLY = os.environ.get("GOOD_INDEPENDENT_ONLY", "0") == "1"
STRONG_TOUCHES = int(os.environ.get("M1_STRONG_TOUCHES", "2"))

SCHEMA = {
    "type": "OBJECT",
    "properties": {"verdicts": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "i": {"type": "INTEGER"},
            "touches": {"type": "ARRAY", "items": {
                "type": "OBJECT",
                # a patent points at a claim number; a paper has no claims, so it points with a
                # verbatim quote that is located in its own text — the same demand, in the form
                # the document can answer (leader, 2026-09-18)
                "properties": {"element_id": {"type": "STRING"}, "claim_number": {"type": "INTEGER"},
                               "quote": {"type": "STRING"}, "reason": {"type": "STRING"}},
                "required": ["element_id"]}},
        },
        "required": ["i", "touches"]}}},
    "required": ["verdicts"],
}

_CLAIM_SPLIT = re.compile(r"(?m)^\s*(\d{1,3})\s*[.)]\s+")


def _locates(quote: str, text: str) -> bool:
    """A paper's pointer has to be checkable the way a claim number is."""
    try:
        from ..quote_verify import locate_quote
        return bool(locate_quote(quote, text, 0.9)[0])
    except Exception:
        return quote.lower()[:60] in (text or "").lower()


def independent_claims(claims_text: str, cap: int = 6) -> str:
    """The independent claims only (no "of claim N" back-reference), which is
    where the scope lives — halves the tokens when GOOD_INDEPENDENT_ONLY=1."""
    parts = _CLAIM_SPLIT.split(claims_text or "")
    out = []
    for n, body in zip(parts[1::2], parts[2::2]):
        if not re.search(r"\b(of|in)\s+claim\s+\d", body[:200], re.I):
            out.append(f"{n}. {body.strip()}")
    return "\n".join(out[:cap]) if out else (claims_text or "")


def _prompt(elements: list[dict], batch: list[tuple[int, dict, str]]) -> str:
    els = "\n".join(f'  {e["id"]}: {e["text"][:220]}' for e in elements)
    rows = []
    for i, d, body in batch:
        if d.get("_no_claims"):
            rows.append(f'[{i}] {d.get("pub_num") or d.get("title") or ""} · {d.get("title") or ""}\n'
                        f'  NO CLAIMS (not a patent) — TEXT: {body[:CLAIMS_CHARS]}')
        else:
            text = independent_claims(body) if INDEPENDENT_ONLY else body
            rows.append(f'[{i}] {d.get("pub_num") or ""} · {d.get("title") or ""}\n  CLAIMS: {text[:CLAIMS_CHARS]}')
    return f"""You are a US patent examiner. For each document below, decide which elements of the
invention its CLAIMS touch — the same subject matter in any wording, in any field, at any level
of generality (a broader claim reads on a narrower element; a different art reciting the same
mechanism still counts).

You may only report an element when you can POINT AT WHERE the document says it.
 - a document shown with CLAIMS: give "claim_number", the number of the claim that touches it;
 - a document shown as NO CLAIMS: give "quote", copied VERBATIM from the text above — it is
   checked against that text, and a quote that cannot be found does not count.
If you cannot point, do not report that element. Report nothing for a document that touches no
element — an empty list is the right answer for most documents.

ELEMENTS:
{els}

DOCUMENTS:
{chr(10).join(rows)}

Answer with JSON {{"verdicts": [{{"i": <document index>, "touches": [{{"element_id": "<element id>",
"claim_number": <claim number, for a document shown with CLAIMS>, "quote": "<verbatim, for a
document shown as NO CLAIMS>", "reason": "<=12 words"}}, ...]}}, ...]}} — one entry per document,
with an empty "touches" list where nothing is touched."""


async def judge(elements: list[dict], docs: list[dict], claims: dict[str, str], call=None) -> dict:
    """Mark each doc with `good_touches` (element id -> claim number or quote)
    and `good` (bool).

    `claims` maps a canonical publication number to its claims text. A document
    with no claims is judged on its abstract instead rather than skipped: papers
    have no claims at all, and skipping them meant a paper could never be GOOD,
    never seed a later round and never be delivered — structurally, whatever the
    recall did (found by the N5 paper-channel probe, 2026-09-18). A paper must
    still point at where it says the thing, with a verbatim quote that is
    located in its own text; pointing is the part that matters, and a claim
    number is only how a patent points."""
    from ..recall.bigquery_patents import _canon_pub
    if call is None:
        from app import llm as _llm
        _model = _llm.stage_model("screen")

        async def call(system, user, response_schema=None):
            return await _llm.call_llm(system, user, response_schema=response_schema, model=_model)
    have, n_claims, n_abstract = [], 0, 0
    for i, d in enumerate(docs):
        c = claims.get(_canon_pub(d.get("pub_num") or ""), "")
        if c:
            have.append((i, d, c))
            n_claims += 1
            continue
        text = (d.get("full_text") or d.get("abstract") or d.get("snippet") or "").strip()
        if len(text) >= 120:
            d["_no_claims"] = True
            have.append((i, d, text))
            n_abstract += 1
    if not have:
        return {"judged": 0, "calls": 0, "good": 0, "with_claims": 0, "with_abstract": 0}
    system = "You are a US patent examiner reading claims. Output JSON only."
    batches = [have[x:x + BATCH] for x in range(0, len(have), BATCH)]
    sem = asyncio.Semaphore(CONCURRENCY)

    async def _one(batch):
        async with sem:
            try:
                return json.loads(await call(system, _prompt(elements, batch), response_schema=SCHEMA))
            except Exception:
                return {}
    results = await asyncio.gather(*(_one(b) for b in batches))
    valid = {e["id"] for e in elements}
    bodies = {i: b for i, _d, b in have}
    n_good = 0
    for data in results:
        for v in (data.get("verdicts") or []):
            try:
                i = int(v["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= i < len(docs):
                continue
            body = bodies.get(i, "")
            no_claims = bool(docs[i].get("_no_claims"))
            touches = {}
            for t in (v.get("touches") or []):
                eid, num = str(t.get("element_id") or ""), t.get("claim_number")
                if eid not in valid:
                    continue
                if not no_claims:
                    if isinstance(num, int) and num > 0:
                        touches.setdefault(eid, num)
                    continue
                quote = str(t.get("quote") or "").strip()
                if quote and _locates(quote, body):
                    touches.setdefault(eid, quote[:120])
            docs[i]["good_touches"] = touches
            docs[i]["good_reasons"] = {str(t.get("element_id")): str(t.get("reason") or "")[:80]
                                       for t in (v.get("touches") or []) if t.get("element_id")}
            docs[i]["good"] = bool(touches)
            n_good += 1 if touches else 0
    for _i, d, _b in have:
        d.pop("_no_claims", None)
    return {"judged": len(have), "calls": len(batches), "good": n_good,
            "with_claims": n_claims, "with_abstract": n_abstract}


def is_strong(d: dict) -> bool:
    """Touches >= STRONG_TOUCHES elements, each with a claim number."""
    return len(d.get("good_touches") or {}) >= STRONG_TOUCHES


def rank_good(docs: list[dict]) -> list[dict]:
    """GOOD documents, most elements touched first (the seed order for the
    next round and the delivery order). Strong ones lead, so a seed cap takes
    them before the one-element hits."""
    good = [d for d in docs if d.get("good")]
    return sorted(good, key=lambda d: (not is_strong(d), -len(d.get("good_touches") or {}),
                                       -float(d.get("prune_cos") or 0.0)))


def rank_for(docs: list[dict], element_ids, cap: int = 0) -> list[dict]:
    """GOOD documents that touch one of `element_ids` first — how an uncovered
    element pulls the next round's seeds towards itself."""
    want = set(element_ids or ())
    ranked = rank_good(docs)
    out = ([d for d in ranked if want & set(d.get("good_touches") or {})]
           + [d for d in ranked if not (want & set(d.get("good_touches") or {}))])
    return out[:cap] if cap else out


def coverage(docs: list[dict], elements: list[dict], strong_only: bool = True) -> dict[str, int]:
    """How many GOOD documents cover each element. Only STRONG ones count by
    default — see the module docstring for why a one-element hit does not."""
    out = {e["id"]: 0 for e in elements}
    for d in docs:
        if strong_only and not is_strong(d):
            continue
        for eid in (d.get("good_touches") or {}):
            if eid in out:
                out[eid] += 1
    return out


def uncovered(cover: dict[str, int], target: int = 1) -> list[str]:
    """Element ids with fewer than `target` strong GOOD behind them."""
    return [eid for eid, n in cover.items() if n < target]
