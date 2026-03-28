"""
LLM calls via OpenAI-compatible API — MINIMAL token usage.

Design principle: tokens go to PDF analysis, not routing.
Only 6 LLM calls in entire pipeline (was 10):
  1. summarize_invention     — must be LLM
  2. decompose_invention     — must be LLM
  3. generate_checklist       — must be LLM
  4. plan_delegation          — must be LLM
  5. evaluate_single_document — must be LLM × N (bulk of tokens, as intended)
  6. generate_overall_summary — must be LLM

Removed (replaced with deterministic code):
  - detect_invention → keyword rules
  - classify_document → PyMuPDF metadata
  - classify_category → keyword rules on summary
"""

import json
import os
import re
import asyncio
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

MODEL = os.getenv("LLM_MODEL", "gpt-4o")
MAX_TOKENS = 8192

_client: AsyncOpenAI | None = None


def get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            api_key=os.environ["LLM_API_KEY"],
            base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        )
    return _client


async def call_llm(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    client = get_client()
    resp = await client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content or ""


async def call_llm_with_pdf(system: str, user: str, pdf_path: str, max_tokens: int = MAX_TOKENS) -> str:
    """Extract PDF text via PyMuPDF, send as text. Works with ALL providers."""
    import fitz
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc[:10]]
    doc.close()
    pdf_text = "\n\n---PAGE---\n\n".join(pages)
    if len(pdf_text) > 60000:
        pdf_text = pdf_text[:60000] + "\n[truncated]"
    return await call_llm(system, f"{user}\n\n--- DOCUMENT ---\n\n{pdf_text}", max_tokens)


# ═══════════════════════════════════════════════════════════════
# DETERMINISTIC: Zero LLM tokens
# ═══════════════════════════════════════════════════════════════

def detect_invention(paper_text: str) -> dict:
    """DETERMINISTIC. Checks for method/system/apparatus/composition keywords."""
    lower = paper_text[:10000].lower()
    invention_kw = [
        "we propose", "we present", "we introduce", "we develop", "we design",
        "novel method", "novel system", "novel approach", "novel framework",
        "our method", "our system", "our approach", "this paper presents",
        "we demonstrate", "we show that", "we achieve",
        "apparatus", "device", "composition", "manufacture",
        "claims", "embodiment", "wherein",
    ]
    hits = sum(1 for kw in invention_kw if kw in lower)
    if hits >= 3:
        return {"status": "present"}
    if hits >= 1:
        return {"status": "implied"}
    return {"status": "absent"}


def classify_document(paper_text: str, filename: str = "") -> str:
    """DETERMINISTIC. Uses filename + content patterns."""
    lower = paper_text[:5000].lower()
    fn = filename.lower()

    # Patent indicators
    if any(x in lower for x in ["claims", "embodiment", "wherein", "applicant", "assignee"]):
        return "patent"
    if any(x in fn for x in ["us", "ep", "cn", "wo", "jp"]) and any(c.isdigit() for c in fn):
        return "patent"

    # Paper indicators
    if any(x in lower for x in ["abstract", "introduction", "related work", "methodology", "references", "arxiv"]):
        return "paper"

    return "other"


def classify_category(summary: str) -> dict:
    """DETERMINISTIC. Keyword-based classification under 35 USC §101."""
    lower = summary.lower()

    scores = {
        "Process": 0,
        "Machine": 0,
        "Manufacture": 0,
        "Composition": 0,
        "Design": 0,
    }

    process_kw = ["method", "step", "process", "procedure", "algorithm", "pipeline", "training", "learning", "computing"]
    machine_kw = ["system", "device", "apparatus", "sensor", "processor", "robot", "hardware", "module", "circuit"]
    manufacture_kw = ["article", "product", "component", "fabricat", "manufactur", "assem"]
    composition_kw = ["compound", "mixture", "composition", "formulation", "material", "substance"]
    design_kw = ["ornamental", "design", "appearance", "shape", "visual design"]

    for kw in process_kw:
        if kw in lower:
            scores["Process"] += 1
    for kw in machine_kw:
        if kw in lower:
            scores["Machine"] += 1
    for kw in manufacture_kw:
        if kw in lower:
            scores["Manufacture"] += 1
    for kw in composition_kw:
        if kw in lower:
            scores["Composition"] += 1
    for kw in design_kw:
        if kw in lower:
            scores["Design"] += 1

    best = max(scores, key=scores.get)
    if scores[best] == 0:
        best = "None"

    return {"invention_type": best, "reasoning": f"Keyword classification: {best} ({scores[best]} keyword hits)"}


# ═══════════════════════════════════════════════════════════════
# LLM CALLS: Where tokens SHOULD be spent
# ═══════════════════════════════════════════════════════════════

async def summarize_invention(paper_text: str) -> str:
    """LLM CALL #1: Invention summary."""
    return await call_llm(
        "You are a patent analyst. Be concise.",
        f"""Summarize the invention(s). Focus on WHAT is built/done.
Use the paper's terminology. List as (1)...(2)... if multiple.
200-400 words. No preamble.

{paper_text[:15000]}""",
    )


async def decompose_invention(paper_text: str) -> str:
    """LLM CALL #2: Technical breakdown."""
    return await call_llm(
        "You are a patent analyst. Be concise.",
        f"""Bullet-list the invention's key technical elements:
components, steps, interfaces, materials, algorithms.
Use the paper's terminology. No novelty assessment.

{paper_text[:15000]}""",
    )


async def generate_checklist(summary: str, ucd: str) -> list[str]:
    """LLM CALL #3: Checklist generation."""
    resp = await call_llm(
        "You are a patent claim analyst. Output JSON only.",
        f"""Derive 20-30 testable checklist items from this invention.
Each: atomic, specific, testable. "The system includes X that performs Y."
Output as JSON array of strings. No explanation.

<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>""",
    )
    match = re.search(r'\[.*\]', resp, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return [line.strip("- •").strip() for line in resp.split("\n") if line.strip().startswith(("-", "•", "*"))]


async def plan_delegation(summary: str, ucd: str, invention_type: str) -> dict:
    """LLM CALL #4: Search delegation planning."""
    resp = await call_llm(
        "You are a USPTO examiner. Output JSON only.",
        f"""Create search plan. Output JSON with: a_core, atoms[], groups[].
atoms: id, name, keywords[], core_score, distinctiveness_score.
groups: group_id, atoms[], label, intent, anchor_terms[][], expansion_terms[][].
8-20 atoms, 3-6 groups. Follow 102/103 methodology.

<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>
<type>{invention_type}</type>""",
        max_tokens=16384,
    )
    match = re.search(r'\{.*\}', resp, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return {"atoms": [], "groups": []}


async def evaluate_single_document(
    invention_summary: str,
    checklist: list[str],
    prior_art_pdf_path: str,
    prior_art_title: str,
    prior_art_type: str,
) -> dict:
    """LLM CALL #5 (× N): Per-document evaluation. This is where tokens SHOULD go."""
    cl_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(checklist))

    system = "You are a US patent examiner. Output JSON only."
    prompt = f"""INVENTION: {invention_summary[:3000]}

CHECKLIST ({len(checklist)} items):
{cl_text}

Evaluate the attached document "{prior_art_title}" ({prior_art_type}).
For EACH item: match=true only with explicit evidence. Cite section/quote.

JSON output:
{{
  "anticipation_assessment": "102 analysis in 1-2 sentences",
  "key_teachings": "103 relevant elements in 1-2 sentences",
  "checklist_results": {{
    "<item>": {{"analysis": "evidence", "match": true/false}},
    ...all {len(checklist)} items...
  }}
}}"""

    try:
        resp = await call_llm_with_pdf(system, prompt, prior_art_pdf_path, max_tokens=16384)
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            return result
    except Exception as e:
        return {"title": prior_art_title, "match_type": prior_art_type, "error": str(e), "checklist_results": {}}
    return {"title": prior_art_title, "checklist_results": {}}


async def evaluate_batch(
    invention_summary: str,
    checklist: list[str],
    documents: list[dict],
    max_concurrent: int = 5,
) -> list[dict]:
    """Run N evaluations in parallel with semaphore."""
    sem = asyncio.Semaphore(max_concurrent)

    async def one(doc):
        async with sem:
            pdf = doc.get("local_pdf", "")
            if not pdf or not Path(pdf).exists():
                return {"title": doc.get("title", ""), "match_type": doc.get("match_type", ""), "checklist_results": {}}
            return await evaluate_single_document(
                invention_summary, checklist, pdf,
                doc.get("title", ""), doc.get("match_type", "Paper"),
            )

    return list(await asyncio.gather(*[one(d) for d in documents]))


async def generate_overall_summary(invention_summary: str, top_matches: list[dict]) -> str:
    """LLM CALL #6: Executive summary."""
    matches = "\n".join(f"- {m.get('title','')}: {m.get('similarity_score',0):.0%}" for m in top_matches[:10])
    return await call_llm(
        "You are a patent novelty analyst. Be structured and concise.",
        f"""Executive summary of novelty assessment. 200-300 words.
Assess: which aspects have prior art, which are novel.
Structure as numbered points.

INVENTION: {invention_summary[:2000]}
TOP MATCHES:
{matches}""",
    )
