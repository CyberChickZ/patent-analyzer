"""
LLM calls via Google GenAI (Vertex AI) — MINIMAL token usage.

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
import contextvars
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

GC_PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")
MAX_TOKENS = 8192

_client: genai.Client | None = None

# Hook so the pipeline can observe every LLM call (system, user, response, thoughts)
_llm_hook: contextvars.ContextVar[Callable[[str, str, str, str], None] | None] = contextvars.ContextVar(
    "_llm_hook", default=None
)


def set_llm_hook(hook: Callable[[str, str, str, str], None] | None):
    _llm_hook.set(hook)


def _emit(system: str, user: str, response: str, thoughts: str = ""):
    hook = _llm_hook.get()
    if hook is not None:
        try:
            hook(system, user, response, thoughts)
        except Exception:
            pass


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(
            vertexai=True,
            project=GC_PROJECT,
            location="us-west1",
        )
    return _client


def _build_config(system: str, max_tokens: int, thinking_budget: int) -> types.GenerateContentConfig:
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
    )
    if thinking_budget > 0:
        try:
            config.thinking_config = types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            )
        except Exception:
            # SDK version may not support ThinkingConfig — fall back silently
            pass
    return config


def _extract_text_and_thoughts(resp) -> tuple[str, str]:
    """Split response candidate parts into (visible_text, thought_summary)."""
    text_parts: list[str] = []
    thought_parts: list[str] = []
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        if content is None:
            continue
        parts = getattr(content, "parts", None) or []
        for part in parts:
            ptext = getattr(part, "text", None) or ""
            if not ptext:
                continue
            if getattr(part, "thought", False):
                thought_parts.append(ptext)
            else:
                text_parts.append(ptext)
    text = "\n".join(text_parts) if text_parts else (getattr(resp, "text", None) or "")
    thoughts = "\n".join(thought_parts)
    return text, thoughts


async def call_llm(
    system: str,
    user: str,
    max_tokens: int = MAX_TOKENS,
    thinking_budget: int = 0,
) -> str:
    client = get_client()
    config = _build_config(system, max_tokens, thinking_budget)
    resp = await client.aio.models.generate_content(
        model=MODEL,
        contents=[types.Part.from_text(text=user)],
        config=config,
    )
    text, thoughts = _extract_text_and_thoughts(resp)
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    _emit(system, user, text, thoughts)
    return text


async def call_llm_with_pdf(
    system: str,
    user: str,
    pdf_path: str,
    max_tokens: int = MAX_TOKENS,
    thinking_budget: int = 0,
) -> str:
    """Extract PDF text via PyMuPDF, send as text."""
    import fitz
    doc = fitz.open(pdf_path)
    pages = [page.get_text() for page in doc[:10]]
    doc.close()
    pdf_text = "\n\n---PAGE---\n\n".join(pages)
    if len(pdf_text) > 60000:
        pdf_text = pdf_text[:60000] + "\n[truncated]"
    return await call_llm(
        system,
        f"{user}\n\n--- DOCUMENT ---\n\n{pdf_text}",
        max_tokens,
        thinking_budget=thinking_budget,
    )


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

    if any(x in lower for x in ["claims", "embodiment", "wherein", "applicant", "assignee"]):
        return "patent"
    if any(x in fn for x in ["us", "ep", "cn", "wo", "jp"]) and any(c.isdigit() for c in fn):
        return "patent"

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
    hits = scores[best]
    if hits == 0:
        return {"invention_type": "None", "reasoning": "No category keywords matched"}
    return {"invention_type": best, "reasoning": f"Keyword classification: {best} ({hits} keyword hits)"}


# ═══════════════════════════════════════════════════════════════
# LLM CALLS: Where tokens SHOULD be spent
# ═══════════════════════════════════════════════════════════════

def _feedback_block(feedback: dict | None) -> str:
    if not feedback:
        return ""
    issues = feedback.get("issues") or []
    issue_lines = "\n".join(f"- {i}" for i in issues) if issues else "- (none specified)"
    suggestion = feedback.get("suggestion") or ""
    prev = feedback.get("previous_response") or ""
    return f"""

════ FEEDBACK FROM YOUR PREVIOUS ATTEMPT ════
A self-review of your previous output flagged these issues:
{issue_lines}

Reviewer's suggestion: {suggestion}

Your previous output (for reference, do NOT just repeat it):
{prev[:1500]}

CRITICAL: Regenerate from scratch addressing the issues. Use ONLY facts present in the input below. Do not invent details.
"""


async def detect_and_summarize_invention(first_page_text: str) -> dict:
    """ONE LLM CALL that does both: judge whether a patentable invention is
    disclosed, AND if so produce the canonical summary used by the rest of the
    pipeline.

    Input is intentionally just the first page (~2-5k chars) — that's where the
    abstract + introduction live, which is exactly where authors state what they
    built in their most concise form. Subsequent phases (decompose, evaluate)
    still see the full paper_text where they need it.

    Returns:
        {
          "has_innovation": bool,    # if false, pipeline stops cleanly
          "reasoning":      str,     # 1-2 sentences explaining the decision
          "doc_type":       "paper" | "patent" | "other",
          "category":       "Process" | "Machine" | "Manufacture" | "Composition" | "Design" | "None",
          "summary":        str,     # 200-400 word canonical invention summary
        }
    """
    resp = await call_llm(
        "You are a patent analyst. You read the first page of a document and "
        "judge whether it discloses a concrete, patentable invention. Output JSON only.",
        f"""════ TASK ════
Read the FIRST PAGE OF DOCUMENT below (typically abstract + introduction).
Decide:
  1. Does this page describe a concrete invention — something built, made,
     synthesized, designed, or a novel method/process?
     (vs. a survey, opinion, commentary, course material, dataset description,
      review article, etc.)
  2. If yes: classify the document type, the §101 category, and write a
     200-400 word summary of WHAT was built/done in the author's own terminology.
     Focus on the technical contribution, not motivation.

Output strictly this JSON, no preamble:
{{
  "has_innovation": true | false,
  "reasoning": "1-2 sentences explaining the has_innovation decision",
  "doc_type":  "paper" | "patent" | "other",
  "category":  "Process" | "Machine" | "Manufacture" | "Composition" | "Design" | "None",
  "summary":   "200-400 word invention summary, using the paper's own terminology, focused on WHAT is built/done. List as (1)...(2)... if multiple. Empty string if has_innovation is false."
}}

════ FIRST PAGE OF DOCUMENT ════
```
{first_page_text}
```
""",
        thinking_budget=4096,
    )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            return {
                "has_innovation": bool(d.get("has_innovation", True)),
                "reasoning":      str(d.get("reasoning", "") or ""),
                "doc_type":       str(d.get("doc_type", "other") or "other"),
                "category":       str(d.get("category", "None") or "None"),
                "summary":        str(d.get("summary", "") or ""),
            }
        except json.JSONDecodeError:
            pass
    # Parse failure → safest default is to proceed with raw response as summary
    return {
        "has_innovation": True,
        "reasoning":      "(JSON parse failed — defaulting to proceed)",
        "doc_type":       "other",
        "category":       "None",
        "summary":        resp[:2000],
    }


async def summarize_invention(paper_text: str, feedback: dict | None = None) -> str:
    return await call_llm(
        "You are a patent analyst. Be concise. Only state facts present in the input — never invent details.",
        f"""════ TASK (template) ════
Summarize the invention(s). Focus on WHAT is built/done.
Use the paper's terminology. List as (1)...(2)... if multiple.
200-400 words. No preamble.
Critical: only use information present in the input below. Do not hallucinate specifics (numbers, names, components) that are not explicitly stated.{_feedback_block(feedback)}

════ INPUT (paper text, first 15000 chars) ════
{paper_text[:15000]}""",
        thinking_budget=4096,
    )


async def decompose_invention(paper_text: str, feedback: dict | None = None) -> str:
    return await call_llm(
        "You are a patent analyst. Be concise. Only list elements explicitly described in the input.",
        f"""════ TASK (template) ════
Bullet-list the invention's key technical elements:
components, steps, interfaces, materials, algorithms.
Use the paper's terminology. No novelty assessment.
Only include elements explicitly described in the input.{_feedback_block(feedback)}

════ INPUT (paper text, first 15000 chars) ════
{paper_text[:15000]}""",
        thinking_budget=4096,
    )


async def generate_checklist(summary: str, ucd: str) -> list[str]:
    resp = await call_llm(
        "You are a patent claim analyst. Output JSON only.",
        f"""════ TASK (template) ════
Derive 20-30 testable checklist items from this invention.
Each: atomic, specific, testable. "The system includes X that performs Y."
Output as JSON array of strings. No explanation.

════ INPUT (from prior LLM steps) ════
<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>""",
        thinking_budget=4096,
    )
    match = re.search(r'\[.*\]', resp, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return [line.strip("- •").strip() for line in resp.split("\n") if line.strip().startswith(("-", "•", "*"))]


async def plan_delegation(summary: str, ucd: str, invention_type: str, feedback: dict | None = None) -> dict:
    resp = await call_llm(
        "You are a USPTO examiner. Output JSON only.",
        f"""════ TASK (template) ════
Create search plan. Output JSON with: a_core, atoms[], groups[].
atoms: id, name, keywords[], core_score, distinctiveness_score.
groups: group_id, atoms[], label, intent, anchor_terms[][], expansion_terms[][].
8-20 atoms, 3-6 groups. Follow 102/103 methodology.{_feedback_block(feedback)}

════ INPUT (from prior LLM steps) ════
<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>
<type>{invention_type}</type>""",
        thinking_budget=4096,
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
        resp = await call_llm_with_pdf(system, prompt, prior_art_pdf_path, thinking_budget=8192)
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


async def refine_search_query(
    invention_summary: str,
    group_label: str,
    group_intent: str,
    weak_results: list[dict],
    original_queries: list[str],
) -> dict:
    """
    Adaptive search harness: LLM looks at weak results and proposes refined queries.
    Returns {"queries": [str, str, ...], "reasoning": "..."}
    """
    weak_titles = "\n".join(f"- ({d.get('semantic_score', 0):.2f}) {d.get('title', '')[:150]}" for d in weak_results[:5])
    orig_q = "\n".join(f"- {q}" for q in original_queries)
    resp = await call_llm(
        "You are a USPTO patent search expert. Output JSON only. You refine failed search queries based on what was found.",
        f"""════ TASK (template) ════
A search group's queries returned only weakly-relevant results. Look at what we found and propose 1-2 REFINED queries that would find more relevant prior art.

Strategy hints:
- If results are too generic → add specific technical terms from the invention
- If results are off-topic → use stricter quoted phrases
- If results are in wrong domain → add domain-restricting terms
- Try a different phrasing, synonyms, or more specific technical jargon

Output strict JSON:
{{
  "reasoning": "1-2 sentences why the original queries failed and what your refinement targets",
  "queries": ["refined query 1", "refined query 2"]
}}

════ INPUT ════
INVENTION SUMMARY:
{invention_summary[:1500]}

SEARCH GROUP:
- label: {group_label}
- intent: {group_intent}

ORIGINAL QUERIES (these failed):
{orig_q}

TOP RESULTS WE FOUND (weak — semantic similarity in parens):
{weak_titles or "(no results)"}
""",
    )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"reasoning": "parse failed", "queries": []}


async def review_phase_output(
    phase_name: str,
    task_description: str,
    original_input: str,
    output_to_review: str,
    extra_context: str = "",
) -> dict:
    """Evolve mode: full-context review of a phase's output.

    The reviewer gets the original input AND the produced output and decides
    whether the output is good enough for the phase's task. Returns:
        {
          "good_enough": bool,
          "what_works": str,
          "what_doesnt": str,
          "next_action": "proceed" | "do_more" | "skip",
          "do_more_hint": str,
        }

    For phases 1/2 the next_action is informational (no backtracking allowed).
    For phases 3/4 the next_action drives the elastic loop.
    """
    extra_block = f"\n\n════ EXTRA CONTEXT ════\n{extra_context}" if extra_context else ""
    resp = await call_llm(
        "You are a senior reviewer of a patent novelty pipeline. You read the inputs "
        "the LLM saw and the output it produced, then judge whether the output is good "
        "enough to drive the next pipeline step. Be specific. Output JSON only.",
        f"""════ TASK ════
You are reviewing the output of phase "{phase_name}".
Phase task: {task_description}

Decide whether the output is good enough. Consider: factual accuracy vs the input,
completeness for the task, whether the next pipeline step has what it needs.

Output strictly this JSON:
{{
  "good_enough": true | false,
  "what_works": "1-2 sentences on what the output got right",
  "what_doesnt": "1-2 sentences on what is missing or wrong (or empty if good_enough)",
  "next_action": "proceed" | "do_more" | "skip",
  "do_more_hint": "if next_action=do_more, 1 sentence on what specifically to do more of"
}}

════ ORIGINAL INPUT THE PHASE SAW ════
{original_input[:20000]}

════ OUTPUT TO REVIEW ════
{output_to_review[:8000]}{extra_block}""",
        max_tokens=1024,
    )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {
        "good_enough": True,
        "what_works": "(reviewer parse failed)",
        "what_doesnt": "",
        "next_action": "proceed",
        "do_more_hint": "",
    }


async def summarize_failure(step_name: str, raw_error: str, context: str = "") -> str:
    """When a pipeline step fails (LLM exception, parse error, channel error, 0 results),
    produce ONE 1-2 sentence plain-language explanation that a developer can read in the
    timeline to understand why this step did not produce useful output.

    Cheap utility — no thinking budget.
    """
    try:
        return await call_llm(
            "You explain pipeline failures to a developer in 1-2 sentences. Be specific. No filler.",
            f"""════ TASK ════
A step in a patent novelty analysis pipeline did not produce useful output.
Explain in 1-2 sentences what went wrong, in plain language. If the cause is
ambiguous, say so and list the most likely 2 reasons. Do NOT suggest fixes — just
state what happened. Output a single short paragraph, no markdown, no preamble.

════ STEP ════
{step_name}

════ RAW ERROR / SIGNAL ════
{raw_error[:1500]}

════ CONTEXT ════
{context[:1500] if context else "(none)"}""",
            max_tokens=400,
        )
    except Exception as e:
        return f"(failure_reason summarizer itself failed: {type(e).__name__}: {e})"


async def self_check(label: str, source_text: str, generated_text: str) -> dict:
    """Verify whether generated_text is faithful to source_text.

    The caller is responsible for passing source_text that contains at least
    as much content as the original step's LLM saw — otherwise the reviewer
    will judge against an incomplete source. Internal cap is just a safety net.

    Returns {ok: bool, issues: [...], suggestion: str}.
    """
    resp = await call_llm(
        "You verify whether a generated text is faithful to a source document. "
        "Output JSON only.",
        f"""════ SOURCE DOCUMENT ════
```
{source_text[:20000]}
```

════ GENERATED TEXT (the "{label}" step produced this from the source above) ════
```
{generated_text[:6000]}
```

For each substantive claim in the GENERATED TEXT, check if it appears in or
follows from the SOURCE DOCUMENT. Output strict JSON:
{{
  "ok": true | false,
  "issues": ["short issue 1", "short issue 2"],
  "suggestion": "one-line fix or 'looks good'"
}}""",
    )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"ok": True, "issues": [], "suggestion": "self-check parse failed"}


async def generate_overall_summary(invention_summary: str, top_matches: list[dict]) -> str:
    """Generate plain-language novelty assessment for faculty inventors (not patent lawyers)."""
    matches_lines = []
    for m in top_matches[:10]:
        title = m.get('title', '')
        score = m.get('similarity_score', 0)
        teachings = m.get('key_teachings', '') or m.get('snippet', '')
        matches_lines.append(f"- **{title}** ({score:.0%} overlap)\n  Key relevant content: {teachings[:300]}")
    matches = "\n".join(matches_lines)

    return await call_llm(
        "You are explaining a patent novelty assessment to a university faculty inventor who is NOT a patent lawyer. Use plain English. Be honest, specific, and actionable. Avoid legal jargon (no '102', '103', 'anticipation', 'prior art teaches'). Don't be vague.",
        f"""════ TASK (template) ════
Write a novelty assessment of this invention for the inventor. The inventor is a faculty member who knows their research area but is not familiar with patent law.

Structure your response in EXACTLY these sections (use markdown):

## What you invented (in plain words)
2-3 sentences restating what the inventor built, in their own field's language.

## What's already out there
For the most relevant existing work (top 3-5 from the matches below), explain in 1-2 sentences EACH:
- What that prior work did
- Which specific aspects of YOUR invention it covers (or comes close to)

## What appears genuinely new
List the specific technical elements of your invention that none of the matches seem to cover. Be concrete — point at actual components, methods, or claims, not abstract concepts.

## Honest assessment
1-2 sentences. Pick one: "Looks novel and worth pursuing", "Has overlap but a clear novel angle", "Significant overlap — narrow your claims", or "Likely already known". Then explain why in plain terms.

## Suggested next steps
2-3 concrete actions the inventor can take (e.g., "Read paper X carefully — it's the closest match", "Talk to your tech transfer office about claim Y", "Focus your patent application on aspect Z").

════ INPUT (from prior LLM steps) ════
INVENTION SUMMARY:
{invention_summary[:2500]}

TOP MATCHES (sorted by overlap):
{matches}
""",
    )
