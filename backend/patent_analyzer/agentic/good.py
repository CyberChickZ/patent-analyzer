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

SCHEMA = {
    "type": "OBJECT",
    "properties": {"verdicts": {"type": "ARRAY", "items": {
        "type": "OBJECT",
        "properties": {
            "i": {"type": "INTEGER"},
            "touches": {"type": "ARRAY", "items": {
                "type": "OBJECT",
                "properties": {"element_id": {"type": "STRING"}, "claim_number": {"type": "INTEGER"},
                               "reason": {"type": "STRING"}},
                "required": ["element_id", "claim_number"]}},
        },
        "required": ["i", "touches"]}}},
    "required": ["verdicts"],
}

_CLAIM_SPLIT = re.compile(r"(?m)^\s*(\d{1,3})\s*[.)]\s+")


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
    for i, d, claims in batch:
        text = independent_claims(claims) if INDEPENDENT_ONLY else claims
        rows.append(f'[{i}] {d.get("pub_num") or ""} · {d.get("title") or ""}\n  CLAIMS: {text[:CLAIMS_CHARS]}')
    return f"""You are a US patent examiner. For each document below, decide which elements of the
invention its CLAIMS touch — the same subject matter in any wording, in any field, at any level
of generality (a broader claim reads on a narrower element; a different art reciting the same
mechanism still counts).

You may only report an element when you can name the NUMBER of the claim that touches it. If you
cannot point at a specific claim, do not report that element. Report nothing for a document whose
claims touch no element — an empty list is the right answer for most documents.

ELEMENTS:
{els}

DOCUMENTS:
{chr(10).join(rows)}

Answer with JSON {{"verdicts": [{{"i": <document index>, "touches": [{{"element_id": "<element id>",
"claim_number": <the claim number in THAT document>, "reason": "<=12 words"}}, ...]}}, ...]}} —
one entry per document, with an empty "touches" list where nothing is touched."""


async def judge(elements: list[dict], docs: list[dict], claims: dict[str, str], call=None) -> dict:
    """Mark each doc with `good_touches` (element id -> claim number) and
    `good` (bool). `claims` maps a canonical publication number to its claims
    text; documents without claims are left unjudged. Returns accounting."""
    from ..recall.bigquery_patents import _canon_pub
    if call is None:
        from app import llm as _llm
        _model = _llm.stage_model("screen")

        async def call(system, user, response_schema=None):
            return await _llm.call_llm(system, user, response_schema=response_schema, model=_model)
    have = [(i, d, claims.get(_canon_pub(d.get("pub_num") or ""), "")) for i, d in enumerate(docs)]
    have = [(i, d, c) for i, d, c in have if c]
    if not have:
        return {"judged": 0, "calls": 0, "good": 0, "with_claims": 0}
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
    n_good = 0
    for data in results:
        for v in (data.get("verdicts") or []):
            try:
                i = int(v["i"])
            except (KeyError, TypeError, ValueError):
                continue
            if not 0 <= i < len(docs):
                continue
            touches = {}
            for t in (v.get("touches") or []):
                eid, num = str(t.get("element_id") or ""), t.get("claim_number")
                if eid in valid and isinstance(num, int) and num > 0:
                    touches.setdefault(eid, num)
            docs[i]["good_touches"] = touches
            docs[i]["good_reasons"] = {str(t.get("element_id")): str(t.get("reason") or "")[:80]
                                       for t in (v.get("touches") or []) if t.get("element_id")}
            docs[i]["good"] = bool(touches)
            n_good += 1 if touches else 0
    return {"judged": len(have), "calls": len(batches), "good": n_good, "with_claims": len(have)}


def rank_good(docs: list[dict]) -> list[dict]:
    """GOOD documents, most elements touched first (the seed order for the
    next round and the delivery order)."""
    good = [d for d in docs if d.get("good")]
    return sorted(good, key=lambda d: (-len(d.get("good_touches") or {}), -float(d.get("prune_cos") or 0.0)))


def coverage(docs: list[dict], elements: list[dict]) -> dict[str, int]:
    """How many GOOD documents cover each element — the round's stop signal."""
    out = {e["id"]: 0 for e in elements}
    for d in docs:
        for eid in (d.get("good_touches") or {}):
            if eid in out:
                out[eid] += 1
    return out
