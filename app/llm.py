"""
LLM calls via OpenAI-compatible API.

Supports ANY OpenAI-compatible provider:
  - OpenAI (GPT-4o, o1, etc.)
  - Google Gemini (via OpenAI compat endpoint)
  - DeepSeek
  - Groq
  - Together AI
  - Local (Ollama, vLLM, etc.)
  - Anthropic (via proxy like LiteLLM)

Environment variables:
  LLM_API_KEY      — API key (required)
  LLM_BASE_URL     — Base URL (default: https://api.openai.com/v1)
  LLM_MODEL        — Model name (default: gpt-4o)
"""

import json
import os
import re
import base64
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
    """Single LLM call. Returns response text."""
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
    """
    LLM call with PDF content.

    Strategy: extract text from PDF via PyMuPDF, send as text.
    This works with ALL providers (not just those supporting document uploads).
    """
    import fitz
    doc = fitz.open(pdf_path)
    pages_text = []
    for page in doc[:10]:  # First 10 pages
        pages_text.append(page.get_text())
    doc.close()
    pdf_text = "\n\n---PAGE BREAK---\n\n".join(pages_text)

    # Truncate if too long (most models have 128K context)
    if len(pdf_text) > 60000:
        pdf_text = pdf_text[:60000] + "\n\n[... truncated ...]"

    full_prompt = f"{user}\n\n--- PRIOR ART DOCUMENT (extracted text) ---\n\n{pdf_text}"

    return await call_llm(system, full_prompt, max_tokens)


# ─── Phase 1: IDCA ─────────────────────────────────────────────────

async def detect_invention(paper_text: str) -> dict:
    """Detect if paper contains a patentable invention."""
    system = "You are an expert patent analyst under US patent law (35 USC)."
    prompt = f"""Analyze this manuscript and determine if it discloses a concrete, useful invention.

Your response MUST begin with exactly one word (lowercase):
  present  — concrete invention clearly disclosed
  implied  — invention suggested but key details missing
  absent   — no concrete invention disclosed

Follow with 2-3 sentences explaining why.

<manuscript>
{paper_text[:15000]}
</manuscript>"""

    resp = await call_llm(system, prompt)
    first_word = resp.strip().split()[0].lower().rstrip(".:,;")
    status = first_word if first_word in ("present", "implied", "absent") else "implied"
    return {"status": status, "raw": resp}


async def classify_document(paper_text: str) -> str:
    """Classify as patent/paper/other."""
    resp = await call_llm(
        "You are a document classifier.",
        f"Classify this document. Respond with one word: patent, paper, or other.\n\n{paper_text[:5000]}",
        max_tokens=50,
    )
    first = resp.strip().split()[0].lower().rstrip(".:,;")
    return first if first in ("patent", "paper", "other") else "other"


async def summarize_invention(paper_text: str) -> str:
    """Produce invention summary."""
    return await call_llm(
        "You are an expert patent analyst.",
        f"""Summarize the invention(s) in this manuscript. Focus on WHAT is built/done, not WHY.
Use the paper's own terminology. If multiple inventions, list each as (1)...(2)...
200-400 words.

<manuscript>
{paper_text[:15000]}
</manuscript>""",
    )


async def classify_category(summary: str) -> dict:
    """Classify under 35 USC §101."""
    resp = await call_llm(
        "You are a patent classification expert.",
        f"""Classify the primary invention under 35 USC §101.

Your response MUST begin with one word:
  Process | Machine | Manufacture | Composition | Design | None

Follow with 2-3 sentences of reasoning.

<invention_summary>
{summary}
</invention_summary>""",
    )
    first = resp.strip().split()[0].rstrip(".:,;")
    valid = {"Process", "Machine", "Manufacture", "Composition", "Design", "None"}
    category = first if first in valid else "None"
    return {"invention_type": category, "reasoning": resp}


# ─── Phase 2: Decomposition ────────────────────────────────────────

async def generate_checklist(summary: str, ucd: str) -> list[str]:
    """Generate 20-30 testable checklist items."""
    resp = await call_llm(
        "You are a patent claim analyst.",
        f"""Derive 20-30 concrete, testable checklist requirements from this invention.
Each item: specific, atomic, testable. "The system includes X that performs Y."
Output as JSON array of strings.

<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>""",
    )
    match = re.search(r'\[.*\]', resp, re.DOTALL)
    if match:
        return json.loads(match.group())
    return [line.strip("- ").strip() for line in resp.split("\n") if line.strip().startswith(("-", "•"))]


async def plan_delegation(summary: str, ucd: str, invention_type: str) -> dict:
    """Create atoms + search groups."""
    resp = await call_llm(
        "You are a USPTO patent examiner planning a prior art search.",
        f"""Decompose into 8-20 atoms and 3-6 search groups following USPTO methodology.
Output as JSON with keys: a_core, atoms[], groups[].
Each atom: id, name, keywords[], core_score, distinctiveness_score.
Each group: group_id, atoms[], label, intent, anchor_terms[][], expansion_terms[][].

<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>
<type>{invention_type}</type>""",
        max_tokens=16384,
    )
    match = re.search(r'\{.*\}', resp, re.DOTALL)
    if match:
        return json.loads(match.group())
    return {"atoms": [], "groups": []}


async def decompose_invention(paper_text: str) -> str:
    """Unstructured decomposition."""
    return await call_llm(
        "You are a patent analyst.",
        f"""Extract a bullet-list technical breakdown of the invention.
Components, steps, interfaces, materials, algorithms.
Use the paper's own terminology. Do NOT assess novelty.

<manuscript>
{paper_text[:15000]}
</manuscript>""",
    )


# ─── Phase 4: Deep PDF Evaluation ──────────────────────────────────

async def evaluate_single_document(
    invention_summary: str,
    checklist: list[str],
    prior_art_pdf_path: str,
    prior_art_title: str,
    prior_art_type: str,
) -> dict:
    """
    Evaluate ONE prior art document against checklist.
    1:1 comparison — target description + ONE prior art.
    """
    checklist_text = "\n".join(f"{i+1}. {item}" for i, item in enumerate(checklist))

    system = "You are a US patent examiner evaluating prior art under 35 USC 102/103."
    prompt = f"""INVENTION UNDER REVIEW:
{invention_summary}

CHECKLIST ({len(checklist)} items):
{checklist_text}

The document below is the prior art: "{prior_art_title}" ({prior_art_type}).
Read it carefully (especially abstract, introduction, methodology, claims if patent).

For EACH checklist item, determine if this document CLEARLY discloses that requirement.
- match=true ONLY with explicit evidence (cite section/page/quote)
- match=false if missing or unclear

Also assess:
- 102 anticipation: does this SINGLE document disclose ALL elements?
- 103 key teachings: what elements could be combined with other references?

Output as JSON:
{{
  "anticipation_assessment": "...",
  "key_teachings": "...",
  "checklist_results": {{
    "<item text>": {{"analysis": "evidence...", "match": true/false}},
    ...
  }}
}}"""

    try:
        resp = await call_llm_with_pdf(system, prompt, prior_art_pdf_path, max_tokens=16384)
        match = re.search(r'\{.*\}', resp, re.DOTALL)
        if match:
            result = json.loads(match.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            return result
    except Exception as e:
        return {
            "title": prior_art_title,
            "match_type": prior_art_type,
            "error": str(e),
            "checklist_results": {},
        }

    return {"title": prior_art_title, "checklist_results": {}}


async def evaluate_batch(
    invention_summary: str,
    checklist: list[str],
    documents: list[dict],
    max_concurrent: int = 5,
) -> list[dict]:
    """Evaluate multiple documents in parallel, each 1:1."""
    sem = asyncio.Semaphore(max_concurrent)

    async def eval_one(doc: dict) -> dict:
        async with sem:
            pdf_path = doc.get("local_pdf", "")
            if not pdf_path or not Path(pdf_path).exists():
                return {
                    "title": doc.get("title", ""),
                    "match_type": doc.get("match_type", ""),
                    "error": "PDF not available",
                    "checklist_results": {},
                }
            return await evaluate_single_document(
                invention_summary,
                checklist,
                pdf_path,
                doc.get("title", ""),
                doc.get("match_type", "Paper"),
            )

    results = await asyncio.gather(*[eval_one(d) for d in documents])
    return list(results)


async def generate_overall_summary(
    invention_summary: str,
    top_matches: list[dict],
) -> str:
    """Generate executive summary of novelty assessment."""
    matches_text = "\n".join(
        f"- {m.get('title','')}: {m.get('similarity_score',0):.0%} overlap"
        for m in top_matches[:10]
    )
    return await call_llm(
        "You are a patent novelty analyst.",
        f"""Write an executive summary of the novelty assessment.

<invention>{invention_summary}</invention>
<top_matches>
{matches_text}
</top_matches>

Assess: which aspects have prior art coverage, which appear novel.
200-300 words.""",
    )
