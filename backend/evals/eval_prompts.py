"""Eval-only prompt variants (never imported by nodes/ or app/).

evaluate_multi_quote: the full_text SSR prompt of
app.llm.evaluate_single_document_text with one change — every criterion
returns `evidence_quotes: [up to MAX_QUOTES verbatim excerpts]` instead of a
single `evidence_quote`, so recall is no longer capped at one passage per
feature (E2 §4.2). Score 0/1/2 semantics are unchanged. Goes through
app.llm.call_llm so evals/llm_cache.install() applies.
"""

import json
import re

MAX_QUOTES = 5


def build_multi_quote_prompt(invention_summary: str, checklist: list, prior_art_text: str,
                             prior_art_title: str, prior_art_type: str,
                             max_quotes: int = MAX_QUOTES) -> tuple[str, str]:
    from app.llm import _format_criteria_for_eval
    cl_text = _format_criteria_for_eval(checklist)
    system = "You are a US patent examiner. Output JSON only."
    scoring_instruction = (
        "For EACH criterion assign score: 2=Present, 1=Partial, 0=Absent.\n"
        f"For score 1 or 2 you MUST list evidence_quotes: 1 to {max_quotes} verbatim excerpts "
        "copied from the document (exact wording, 10-40 words each). List EVERY passage "
        "of the document that discloses the criterion — different paragraphs, claims or the "
        "abstract — one excerpt per passage, most direct first. Never merge text from two "
        "places into one excerpt. If you cannot quote anything, score 0. "
        "Do NOT infer beyond what the text states.")
    output_schema = (
        '"checklist_results": {\n'
        '    "<criterion>": {"score": 0|1|2, "analysis": "...", '
        '"evidence_quotes": ["verbatim excerpt", ...], '
        '"match": true|false},\n    ...all items...\n  }')
    prompt = f"""INVENTION: {invention_summary}

CRITERIA ({len(checklist)} items):
{cl_text}

PRIOR ART "{prior_art_title}" ({prior_art_type}) — full text with numbered paragraphs:
<document>
{prior_art_text}
</document>

{scoring_instruction}

JSON output:
{{
  "anticipation_assessment": "1-2 sentences",
  "key_teachings": "1-2 sentences",
  "rs_synopsis": "One sentence: what this prior art does",
  {output_schema}
}}"""
    return system, prompt


def _clean_quotes(item: dict, max_quotes: int) -> None:
    raw = item.get("evidence_quotes")
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        raw = [item["evidence_quote"]] if item.get("evidence_quote") else []
    seen, out = set(), []
    for q in raw:
        q = str(q or "").strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            out.append(q)
    item["evidence_quotes"] = out[:max_quotes]
    item["evidence_quote"] = out[0] if out else ""


async def evaluate_multi_quote(invention_summary: str, checklist: list, prior_art_text: str,
                               prior_art_title: str, prior_art_type: str,
                               max_quotes: int = MAX_QUOTES) -> dict:
    import app.llm as llm

    system, prompt = build_multi_quote_prompt(
        invention_summary, checklist, prior_art_text, prior_art_title, prior_art_type, max_quotes)
    try:
        resp = await llm.call_llm(system, prompt, thinking_budget=4096)
        m = re.search(r"\{.*\}", resp, re.DOTALL)
        if not m:
            return {"title": prior_art_title, "checklist_results": {}, "source": "multi_noparse"}
        result = json.loads(m.group(), strict=False)
    except Exception as e:
        return {"title": prior_art_title, "match_type": prior_art_type, "error": str(e),
                "checklist_results": {}, "source": "multi_failed"}
    result["title"] = prior_art_title
    result["match_type"] = prior_art_type
    result["source"] = "full_text_multi"
    result["keys_unaligned"] = llm._align_checklist_keys(result, checklist)
    for item in result["checklist_results"].values():
        if isinstance(item, dict):
            _clean_quotes(item, max_quotes)
    llm._backfill_match_from_score(result)
    return result
