"""Error taxonomy for element extraction (omission / fabrication /
misclassification) plus the quote snapper shared by the extraction subgraph
and the FiNE / Pap2Pat extraction evals.

snap_quote(quote, doc_text) -> (found, loc)
    1. exact: first 60 normalized chars of the quote found in the normalized
       document (coverage_eval.quote_location's rule) -> char span
    2. fuzzy: quote_verify.locate_quote at 0.9 -> best difflib window
    3. dual:  quote_dual span/bigram thresholds -> densest token window
    none pass -> (False, None): the quote was written, not copied.

classify_errors(pred_elements, gold_elements, doc_text)
    greedy 1:1 embedding match at tau=0.7 (te005, extraction_eval.greedy_match);
    omission        gold element with no matched prediction
    fabrication     prediction whose quote does not snap, or an unmatched
                    prediction whose best cosine against the document < 0.5
    misclassification matched prediction whose kind / level differs from gold
"""

import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from patent_analyzer.quote_verify import locate_quote, normalize
from quote_dual import DocIndex, content_tokens, verify_quote_dual

HEAD_CHARS = 60
FUZZY_TAU = 0.9
MATCH_TAU = 0.7
DOC_COS_TAU = 0.5
_TOK = re.compile(r"[a-z0-9]+")


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    """quote_verify.normalize, plus original char index of every output char."""
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text or ""):
        for c in ch.lower():
            if ("a" <= c <= "z") or ("0" <= c <= "9"):
                out.append(c)
                idx.append(i)
            elif out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def _span(idx: list[int], start: int, end: int) -> list[int]:
    end = max(start + 1, min(end, len(idx)))
    return [idx[start], idx[end - 1] + 1]


def _fuzzy_span(q: str, d: str, idx: list[int]) -> tuple[float, list[int]] | None:
    n = len(q)
    step = max(1, n // 4)
    sm = SequenceMatcher(None, autojunk=False)
    sm.set_seq2(q)
    best, best_start = 0.0, None
    for start in range(0, max(1, len(d) - n + step), step):
        sm.set_seq1(d[start:start + n])
        if sm.real_quick_ratio() < FUZZY_TAU or sm.quick_ratio() < FUZZY_TAU:
            continue
        r = sm.ratio()
        if r > best:
            best, best_start = r, start
            if best >= 0.995:
                break
    if best_start is None:
        return None
    return best, _span(idx, best_start, best_start + n)


def _dual_span(quote: str, doc_text: str) -> list[int] | None:
    need = set(content_tokens(quote))
    if not need:
        return None
    toks = [(m.group(), m.start(), m.end()) for m in _TOK.finditer((doc_text or "").lower())]
    if not toks:
        return None
    n = len(need)
    w = max(2 * n, n + 10)
    counts: dict[str, int] = {}
    matched = best = 0
    best_i = None
    for i, (tok, _, _) in enumerate(toks):
        if tok in need:
            counts[tok] = counts.get(tok, 0) + 1
            if counts[tok] == 1:
                matched += 1
        if i >= w:
            out = toks[i - w][0]
            if out in need:
                counts[out] -= 1
                if counts[out] == 0:
                    matched -= 1
        if matched > best:
            best, best_i = matched, i
    if best_i is None:
        return None
    lo = max(0, best_i - w + 1)
    return [toks[lo][1], toks[best_i][2]]


def snap_quote(quote: str, doc_text: str) -> tuple[bool, dict | None]:
    """(found, {"char": [start, end], "method": exact|fuzzy|dual, "sim": float})."""
    q = normalize(quote)
    if len(q) < 10 or not doc_text:
        return False, None
    d, idx = _normalize_with_map(doc_text)
    pos = d.find(q[:HEAD_CHARS])
    if pos >= 0:
        end = pos + len(q) if d.startswith(q, pos) else pos + len(q[:HEAD_CHARS])
        return True, {"char": _span(idx, pos, end), "method": "exact", "sim": 1.0}
    found, sim = locate_quote(quote, doc_text, FUZZY_TAU)
    if found:
        hit = _fuzzy_span(q, d, idx)
        if hit:
            return True, {"char": hit[1], "method": "fuzzy", "sim": round(hit[0], 4)}
        return True, {"char": None, "method": "fuzzy", "sim": sim}
    ok, sr, br = verify_quote_dual(quote, DocIndex(doc_text))
    if ok:
        return True, {"char": _dual_span(quote, doc_text), "method": "dual", "sim": round(min(sr, br), 4)}
    return False, None


def _default_embed(texts: list[str]) -> np.ndarray:
    from extraction_eval import embed
    return embed(texts)


def _doc_chunks(doc_text: str, max_chars: int = 600) -> list[str]:
    out = []
    for para in re.split(r"\n\s*\n|\n(?=\[\d{4}\])|\n(?=\[S[\d.]+\.P\d+\])", doc_text or ""):
        para = " ".join(para.split())
        while len(para) > max_chars:
            cut = para.rfind(". ", 0, max_chars)
            cut = cut + 1 if cut > max_chars // 3 else max_chars
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if len(para) > 20:
            out.append(para)
    return out


def _text(e) -> str:
    return (e.get("text") or e.get("criterion") or e.get("feature") or "") if isinstance(e, dict) else str(e)


def classify_errors(pred_elements: list, gold_elements: list, doc_text: str,
                    tau: float = MATCH_TAU, embed_fn=None, require_quote: bool = True) -> dict:
    """pred_elements: [{id, text, evidence_quote?, kind?, level?}] (or str);
    gold_elements: [{text, kind?, level?}] (or str). require_quote=False skips
    the snap test for predictors that carry no quotes (legacy SSR checklist)."""
    from extraction_eval import greedy_match

    embed_fn = embed_fn or _default_embed
    preds = [p if isinstance(p, dict) else {"text": str(p)} for p in pred_elements]
    golds = [g if isinstance(g, dict) else {"text": str(g)} for g in gold_elements]
    for i, p in enumerate(preds):
        p.setdefault("id", f"p{i}")
    unsupported = set()
    if require_quote:
        for p in preds:
            found, _ = snap_quote(p.get("evidence_quote") or "", doc_text)
            if not found:
                unsupported.add(p["id"])

    matched, matched_g, matched_p = [], set(), set()
    if preds and golds:
        sim = embed_fn([_text(g) for g in golds]) @ embed_fn([_text(p) for p in preds]).T
        for gi, pj, s in greedy_match(sim):
            if s >= tau:
                matched.append((gi, pj, float(round(s, 4))))
                matched_g.add(gi)
                matched_p.add(pj)

    omission = [gi for gi in range(len(golds)) if gi not in matched_g]
    fabrication = [p["id"] for p in preds if p["id"] in unsupported]
    orphans = [j for j, p in enumerate(preds) if j not in matched_p and p["id"] not in unsupported]
    if orphans:
        chunks = _doc_chunks(doc_text)
        if chunks:
            cs = embed_fn([_text(preds[j]) for j in orphans]) @ embed_fn(chunks).T
            for row, j in zip(cs, orphans):
                if float(row.max()) < DOC_COS_TAU:
                    fabrication.append(preds[j]["id"])
        else:
            fabrication.extend(preds[j]["id"] for j in orphans)

    misclassification = []
    for gi, pj, _ in matched:
        g, p = golds[gi], preds[pj]
        for attr in ("kind", "level"):
            if g.get(attr) and p.get(attr) and g[attr] != p[attr]:
                misclassification.append({"pred": p["id"], "gold": gi, "attr": attr,
                                          "pred_value": p[attr], "gold_value": g[attr]})
                break
    return {
        "omission": omission, "fabrication": fabrication,
        "misclassification": misclassification, "matched": matched,
        "unsupported": sorted(unsupported),
        "n_gold": len(golds), "n_pred": len(preds),
        "quote_survival": (1 - len(unsupported) / len(preds)) if preds and require_quote else None,
    }


def error_rates(err: dict) -> dict:
    ng, np_ = max(err["n_gold"], 1), max(err["n_pred"], 1)
    return {"omission": len(err["omission"]) / ng,
            "fabrication": len(err["fabrication"]) / np_,
            "misclassification": len(err["misclassification"]) / max(len(err["matched"]), 1)}
