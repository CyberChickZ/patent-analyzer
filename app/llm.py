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

import asyncio
import contextvars
import json
import os
import re
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
    """Send a single PDF as a native multi-modal part to Gemini (no truncation).

    Only used for bulk initial screening where text extraction is acceptable.
    For deep eval use call_llm_with_pdfs which supports multiple files.
    """
    return await call_llm_with_pdfs(system, user, [pdf_path], max_tokens, thinking_budget)


# Vertex AI inline-bytes cap is ~20MB per request. Pad for safety.
_INLINE_PDF_CAP_BYTES = 18 * 1024 * 1024


async def call_llm_with_pdfs(
    system: str,
    user: str,
    pdf_paths: list[str],
    max_tokens: int = MAX_TOKENS,
    thinking_budget: int = 0,
) -> str:
    """Send one or more PDFs as native multi-modal parts to Gemini.

    Uploads PDF bytes directly — the model sees layout, figures, tables. No text
    extraction, no truncation. If a file exceeds Vertex's inline cap (~20MB),
    it falls back to text extraction for THAT file only (others still go as PDF)
    and prefixes a [fallback_text] marker so the LLM knows the input was lossy.
    """
    parts: list[Any] = []
    for p in pdf_paths:
        if not p or not Path(p).exists():
            continue
        size = Path(p).stat().st_size
        data = Path(p).read_bytes()
        if size <= _INLINE_PDF_CAP_BYTES:
            parts.append(types.Part.from_bytes(data=data, mime_type="application/pdf"))
        else:
            from google.cloud import storage as gcs_storage
            bucket_name = os.environ.get("GCS_BUCKET", "aime-hello-world-amie-uswest1")
            blob_path = f"patent-analyzer/tmp-llm/{Path(p).name}"
            bucket = gcs_storage.Client().bucket(bucket_name)
            blob = bucket.blob(blob_path)
            blob.upload_from_string(data, content_type="application/pdf")
            file_uri = f"gs://{bucket_name}/{blob_path}"
            parts.append(types.Part.from_uri(
                file_uri=file_uri, mime_type="application/pdf"))
    parts.append(types.Part.from_text(text=user))

    client = get_client()
    config = _build_config(system, max_tokens, thinking_budget)
    resp = await client.aio.models.generate_content(
        model=MODEL,
        contents=parts,
        config=config,
    )
    text, thoughts = _extract_text_and_thoughts(resp)
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    # Log the user prompt (without the PDF blob) + file list
    file_list = ", ".join(f"{Path(p).name} ({Path(p).stat().st_size//1024}KB)"
                          for p in pdf_paths if p and Path(p).exists())
    _emit(system, f"[PDFs: {file_list}]\n\n{user}", text, thoughts)
    return text


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

    process_kw = [
        "method", "step", "process", "procedure", "algorithm",
        "pipeline", "training", "learning", "computing",
    ]
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

INITIAL_PERSONAS = {
    "landscape": (
        "You are a senior research scientist who thinks about technology as a landscape "
        "of design choices, not a flat list of features. Given a paper, you identify the "
        "AXES of innovation — the technical dimensions where the authors made deliberate "
        "choices among known alternatives. You understand CPC taxonomy deeply and can map "
        "innovation axes to specific CPC groups. You think in terms of 'what space of "
        "solutions exists for this sub-problem, and which point in that space did the "
        "authors pick?' You do NOT extract text — you reason about the field."
    ),
    "technology": (
        "You are a deep domain specialist who knows the full landscape of technical "
        "approaches within a specific research dimension. For any technique mentioned in "
        "a paper, you can enumerate 4-8 known alternatives from the literature — including "
        "ones the paper did NOT cite. You explain HOW approaches differ mechanistically, "
        "not just that they differ. You distinguish between approaches that share a name "
        "but differ in implementation, and approaches with different names that are "
        "mathematically equivalent."
    ),
    "reviewer": (
        "You are a senior patent examiner who reviews evaluation checklists for quality. "
        "You check that each item is specific enough to be testable (not 'uses ML' but "
        "'uses a two-stage detector with separate RPN'), that no important dimension is "
        "missing, that items don't overlap, and that weights reflect actual novelty "
        "contribution. You are ruthless about vagueness — any item where a reader might "
        "say 'I'm not sure if this counts' needs to be rewritten with clearer scope."
    ),
    "decompose": (
        "You are a senior patent prosecution specialist with deep experience "
        "in claim drafting across multiple technology domains. Your expertise is in "
        "reading a technical document and identifying every distinct technical element "
        "that could form the basis of an independent or dependent claim. You think in "
        "terms of structural components, method steps, functional relationships between "
        "elements, and the specific technical choices the inventors made (as opposed to "
        "obvious alternatives). You distinguish between the invention's novel contributions "
        "and its use of known building blocks. You never invent elements that aren't "
        "described in the source — if something is ambiguous, you flag it rather than assume."
    ),
    "checklist": (
        "You are a patent claim analyst who converts invention decompositions into "
        "testable prior art search criteria. Each item you produce must be atomic (tests "
        "exactly one technical element), specific (uses the invention's actual terminology, "
        "not generic descriptions), and evidence-checkable (a reader examining a prior art "
        "document can determine match/no-match without subjective judgment). You understand "
        "that overly broad items match everything and are useless, while overly narrow items "
        "match nothing. You calibrate specificity to the level where a genuine prior art hit "
        "would be meaningful."
    ),
    "plan": (
        "You are a USPTO patent examiner with 15 years of experience conducting prior art "
        "searches under 35 USC §102 and §103. You know how to decompose an invention into "
        "searchable atoms, group those atoms into strategies that target different aspects of "
        "the invention, and construct queries that balance precision and recall. You understand "
        "that the best prior art often comes from adjacent fields — a computer vision technique "
        "might have prior art in medical imaging, robotics, or satellite imagery. You design "
        "search groups that cover both the core domain and plausible adjacent domains."
    ),
    "evaluate": (
        "You are a US patent examiner conducting a detailed prior art comparison. You read "
        "both the source invention and a candidate prior art document side-by-side, comparing "
        "specific technical elements. You are rigorous: a checklist item matches only when the "
        "prior art explicitly describes that element with clear evidence you can cite. You "
        "never infer matches from vague similarity — 'both use neural networks' is not a match "
        "for 'uses a specific dual-encoder architecture with cross-attention.' When a document "
        "is the source paper itself, you identify it as a self-match."
    ),
    "summary": (
        "You are explaining a patent novelty assessment to a university faculty inventor who "
        "is an expert in their research field but is not a patent attorney. You use the "
        "inventor's own technical vocabulary, not legal jargon. You are honest and specific: "
        "you point at concrete technical elements, not abstract concepts. When overlap exists, "
        "you say exactly which elements overlap and which don't. When something appears novel, "
        "you explain what makes it distinct from the closest prior art. You give actionable "
        "advice — not 'consult an attorney' but 'your novel angle appears to be Y, focus "
        "claims on Y.'"
    ),
}


async def craft_personas(
    doc_type: str,
    fields_map: list[str],
    cpc_subclass: str,
    cpc_context: str,
    summary_excerpt: str,
) -> dict[str, str]:
    """ONE LLM call: read initial personas + document classification → output
    5 domain-specific personas. Falls back to INITIAL_PERSONAS on failure."""
    roles_block = "\n\n".join(
        f"### {role}\n{text}" for role, text in INITIAL_PERSONAS.items()
    )
    resp = await call_llm(
        "You rewrite expert personas to be domain-specific. Output JSON only.",
        f"""Take these 5 generic expert personas and rewrite each to be specific
to the document being analyzed. Keep the same expertise level and behavioral
instructions. Add domain-specific knowledge, terminology, and awareness of
the field's typical prior art landscape. Each rewrite should be roughly the
same length as the original (±20%).

INITIAL PERSONAS:
{roles_block}

DOCUMENT CONTEXT:
- Document type: {doc_type}
- Technical fields: {', '.join(fields_map)}
- CPC subclass: {cpc_subclass}
- CPC taxonomy: {cpc_context[:2000]}
- Invention excerpt: {summary_excerpt[:500]}

Output JSON with keys: decompose, checklist, plan, evaluate, summary.
Each value is the rewritten persona string.""",
        max_tokens=3000,
    )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            if all(k in result for k in INITIAL_PERSONAS):
                return result
        except json.JSONDecodeError:
            pass
    return dict(INITIAL_PERSONAS)


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

CRITICAL: Regenerate from scratch addressing the issues. Use ONLY facts from the input below.
"""


async def detect_and_summarize_invention(
    first_page_text: str,
    source_pdf_path: str | None = None,
) -> dict:
    """ONE LLM CALL: classify document → status_determination + fields_map +
    doc_type + CPC subclass + summary. This is the IDCA step — everything
    downstream (persona, decompose, eval) depends on this output.

    Returns:
        {
          "status_determination": "Present" | "Implied" | "Absent",
          "has_innovation": bool,       # backward compat: True unless Absent
          "reasoning":      str,
          "doc_type":       "invention" | "literature_review" | "design_engineering"
                          | "talks_about_invention_but_no_invention",
          "category":       "Process" | "Machine" | "Manufacture" | "Composition" | "Design" | "None",
          "fields_map":     list[str],  # 3-7 technical field labels
          "source_citation": str,       # APA format if determinable
          "cpc_subclass":   str,        # 4-char CPC code, e.g. "G06N"
          "summary":        str,        # 200-400 word canonical invention summary
        }
    """
    system = ("You are a patent analyst. Read the document and classify it. "
              "Output JSON only.")
    task_prompt = """════ TASK ════
Read the FIRST PAGE of the attached document and perform IDCA (Invention
Detection, Classification, and Assignment).

STEP 1 — Status Determination:
  - "Present": the document describes a CONCRETE, IMPLEMENTED invention —
    something built, made, synthesized, designed, or a novel method/process
    with enough technical detail to extract patent claims.
  - "Implied": the document discusses an inventive concept but lacks concrete
    implementation (theoretical proposals, future work, "with funding we
    could build X"). There IS a recognizable claim, but no implementation.
  - "Absent": no invention at all — surveys, opinions, commentaries, course
    material, dataset descriptions, review articles, news.

STEP 2 — Document Type:
  - "invention": paper or patent presenting a novel technical contribution.
  - "design_engineering": engineering implementation report — describes HOW
    something was built/integrated, not claiming novelty.
  - "literature_review": survey, review, or meta-analysis.
  - "talks_about_invention_but_no_invention": discusses/references others'
    inventions but does not present one itself.

STEP 3 — Fields & Classification:
  - fields_map: 3-7 technical field labels from broad to specific.
    Example: ["Computer Vision", "Object Detection", "Anchor-Free Detection"]
  - cpc_subclass: best-guess 4-character CPC subclass code (e.g. G06N, H04L,
    A61B). Pick the single most relevant one.
  - category: §101 type (Process/Machine/Manufacture/Composition/Design/None).

STEP 4 — Summary (only if status is Present or Implied):
  200-400 words describing WHAT is built/done, using the paper's own
  terminology. Focus on technical contribution, not motivation. If Implied,
  describe what is proposed rather than what is implemented.

Output strictly this JSON, no preamble:
{{
  "status_determination": "Present" | "Implied" | "Absent",
  "reasoning": "1-2 sentences explaining the status decision",
  "doc_type": "invention" | "literature_review" | "design_engineering"
            | "talks_about_invention_but_no_invention",
  "category": "Process" | "Machine" | "Manufacture" | "Composition"
            | "Design" | "None",
  "fields_map": ["Field1", "Field2", "..."],
  "source_citation": "APA citation from first-page info, or empty string",
  "cpc_subclass": "G06N",
  "summary": "200-400 word summary, or empty string if Absent"
}}"""
    if source_pdf_path and Path(source_pdf_path).exists():
        resp = await call_llm_with_pdfs(
            system, task_prompt, [source_pdf_path], thinking_budget=4096)
    else:
        resp = await call_llm(
            system,
            f"{task_prompt}\n\n════ FIRST PAGE (text) ════\n"
            f"```\n{first_page_text}\n```",
            thinking_budget=4096,
        )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group())
            # Backward compat: old format had has_innovation instead of status_determination
            if "has_innovation" in d and "status_determination" not in d:
                d["status_determination"] = "Present" if d["has_innovation"] else "Absent"
            status = str(d.get("status_determination", "Present"))
            return {
                "status_determination": status,
                "has_innovation":  status != "Absent",
                "reasoning":       str(d.get("reasoning", "") or ""),
                "doc_type":        str(d.get("doc_type", "invention") or "invention"),
                "category":        str(d.get("category", "None") or "None"),
                "fields_map":      list(d.get("fields_map", []) or []),
                "source_citation": str(d.get("source_citation", "") or ""),
                "cpc_subclass":    str(d.get("cpc_subclass", "") or ""),
                "summary":         str(d.get("summary", "") or ""),
            }
        except json.JSONDecodeError:
            pass
    return {
        "status_determination": "Present",
        "has_innovation": True,
        "reasoning":      "(JSON parse failed — defaulting to proceed)",
        "doc_type":       "invention",
        "category":       "None",
        "fields_map":     [],
        "source_citation": "",
        "cpc_subclass":   "",
        "summary":        resp[:2000],
    }


async def summarize_invention(paper_text: str, feedback: dict | None = None) -> str:
    return await call_llm(
        "You are a patent analyst. Be concise. Only state facts present in the input — never invent details.",
        f"""════ TASK (template) ════
Summarize the invention(s). Focus on WHAT is built/done.
Use the paper's terminology. List as (1)...(2)... if multiple.
200-400 words. No preamble.
Critical: only use information present in the input below. Do not hallucinate specifics \
(numbers, names, components) that are not explicitly stated.{_feedback_block(feedback)}

════ INPUT (paper text, first 15000 chars) ════
{paper_text[:15000]}""",
        thinking_budget=4096,
    )


# ════════════════════════════════════════════════════════════
# Phase 2 v2: Expert-driven innovation analysis (Steps B–G)
# ════════════════════════════════════════════════════════════


async def scan_innovation_landscape(
    summary: str,
    fields_map: list[str],
    cpc_subclass: str,
    cpc_context: str,
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> list[dict]:
    """Step B: Identify innovation axes — technical dimensions where the paper
    may have novel contributions. Uses LLM's field knowledge, not text extraction."""
    system = (persona or INITIAL_PERSONAS["landscape"]) + " Output JSON only."
    fields_str = ", ".join(fields_map) if fields_map else "general technology"
    prompt = f"""Read this paper carefully. You are an expert in {fields_str} (CPC: {cpc_subclass}).

CPC taxonomy context:
{cpc_context[:2000]}

Identify 3-7 INNOVATION AXES — technical dimensions where this paper makes
deliberate design choices among known alternatives.

An innovation axis is NOT a text extract. It is a technical dimension like:
"gradient flow control strategy for domain adaptation" or
"feature alignment granularity (image-level vs instance-level)".

For each axis, specify the most relevant CPC group.

INVENTION SUMMARY:
{summary[:3000]}

Output JSON:
{{"innovation_axes": [
  {{"axis_name": "...", "axis_description": "1-sentence what this dimension is about",
    "cpc_group": "e.g. G06N3/08", "relevance": "why this axis matters for THIS paper"}}
]}}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        raw = await call_llm_with_pdfs(system, prompt, [source_pdf_path], thinking_budget=4096)
    else:
        raw = await call_llm(system, prompt, thinking_budget=4096)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group()).get("innovation_axes", [])
        except json.JSONDecodeError:
            pass
    return []


async def expand_technology_choices(
    axis: dict,
    summary: str,
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> dict:
    """Step C: For one innovation axis, enumerate known approaches from the
    literature and identify which specific one this paper chose."""
    system = (persona or INITIAL_PERSONAS["technology"]) + " Output JSON only."
    axis_name = axis.get("axis_name", "unknown")
    axis_desc = axis.get("axis_description", "")
    prompt = f"""You are analyzing the innovation axis: "{axis_name}"
({axis_desc})

Read the attached paper and answer:

1. KNOWN APPROACHES: List ALL approaches you know of in the literature for
   this technical dimension (4-8 approaches). Include ones the paper did NOT
   cite. For each, give a 1-sentence mechanistic description.

2. PAPER'S CHOICE: Which specific approach does this paper use? Quote or
   paraphrase the paper's description.

3. DIFFERENTIATOR: What specifically makes the paper's choice distinct from
   the most common alternative?

INVENTION SUMMARY:
{summary[:2000]}

Output JSON:
{{"axis": "{axis_name}",
  "known_approaches": [
    {{"name": "approach name", "description": "1-sentence mechanism"}},
    ...
  ],
  "paper_choice": "the specific approach this paper uses",
  "differentiator": "what makes it distinct from the most common alternative"
}}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        raw = await call_llm_with_pdfs(system, prompt, [source_pdf_path], thinking_budget=4096)
    else:
        raw = await call_llm(system, prompt, thinking_budget=4096)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            result["axis"] = axis_name
            return result
        except json.JSONDecodeError:
            pass
    return {"axis": axis_name, "known_approaches": [], "paper_choice": "", "differentiator": ""}


async def determine_patent_types(
    summary: str,
    technology_choices: list[dict],
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> list[str]:
    """Step D: Determine which patent types (Process/Machine/Manufacture/
    Composition/Design) apply to this invention."""
    system = (persona or "You are a patent classification expert.") + " Output JSON only."
    tc_summary = json.dumps(
        [{"axis": tc.get("axis", ""), "paper_choice": tc.get("paper_choice", "")}
         for tc in technology_choices], indent=2, ensure_ascii=False)
    prompt = f"""Based on this invention, which US patent types (35 USC §101) apply?

Types:
- Process: a method, algorithm, or sequence of steps
- Machine: a system, device, or apparatus
- Manufacture: a manufactured article
- Composition: a chemical or material composition
- Design: ornamental design of a functional item

INVENTION SUMMARY:
{summary[:2000]}

TECHNOLOGY CHOICES:
{tc_summary}

Output JSON:
{{"applicable_types": ["Process", ...],
  "reasoning": {{"Process": "why applicable or not", ...}}}}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        raw = await call_llm_with_pdfs(system, prompt, [source_pdf_path], thinking_budget=2048)
    else:
        raw = await call_llm(system, prompt, thinking_budget=2048)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            types = result.get("applicable_types", [])
            valid = {"Process", "Machine", "Manufacture", "Composition", "Design"}
            return [t for t in types if t in valid] or ["Process"]
        except json.JSONDecodeError:
            pass
    return ["Process"]


async def generate_checklist_for_type(
    patent_type: str,
    summary: str,
    technology_choices: list[dict],
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> list[dict]:
    """Step E: Generate specific, testable checklist items for one patent type.
    Each item embeds known_approaches so the evaluator can distinguish
    'same method' (Present) from 'same category, different method' (Partial)."""
    system = (persona or INITIAL_PERSONAS["checklist"]) + " Output JSON only."
    tc_text = json.dumps(technology_choices, indent=2, ensure_ascii=False)
    prompt = f"""Generate prior art evaluation checklist items for a **{patent_type}** patent.

CRITICAL RULES:
- Each item must test ONE specific technical choice, not a category.
  BAD:  "Uses gradient manipulation for domain adaptation"
  GOOD: "Uses stop-gradient (detach) on the context branch to block gradient
         flow, rather than Gradient Reversal Layer (GRL)"
- Each item must include the known alternative approaches so the evaluator
  can score Partial (different method, same category) vs Present (same method).
- Generate 10-15 items covering ALL innovation axes below.
- Weight each item by its contribution to overall novelty (weights sum to ~1.0
  across all items for this type).

INVENTION SUMMARY:
{summary[:2000]}

TECHNOLOGY CHOICES (from innovation landscape analysis):
{tc_text}

Output JSON:
{{"criteria": [
  {{"id": "c1",
    "criterion": "specific testable statement",
    "weight": 0.08,
    "patent_type": "{patent_type}",
    "axis": "which innovation axis this tests",
    "known_approaches": ["approach A", "approach B", "approach C"],
    "scale": {{
      "0": "what Absent means for this item",
      "1": "what Partial means (which alternative approaches count)",
      "2": "what Present means (the exact match)"
    }}
  }},
  ...
]}}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        raw = await call_llm_with_pdfs(system, prompt, [source_pdf_path], thinking_budget=8192)
    else:
        raw = await call_llm(system, prompt, thinking_budget=8192)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            criteria = result.get("criteria", [])
            for c in criteria:
                c["patent_type"] = patent_type
            return criteria
        except json.JSONDecodeError:
            pass
    return []


async def review_checklist(
    combined_checklist: list[dict],
    summary: str,
    technology_choices: list[dict],
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> list[dict]:
    """Step F: Expert review — merge duplicates, tighten vague items,
    fill gaps, normalize weights."""
    system = (persona or INITIAL_PERSONAS["reviewer"]) + " Output JSON only."
    cl_text = json.dumps(combined_checklist, indent=2, ensure_ascii=False)
    tc_text = json.dumps(
        [{"axis": tc.get("axis", ""), "paper_choice": tc.get("paper_choice", ""),
          "differentiator": tc.get("differentiator", "")}
         for tc in technology_choices], indent=2, ensure_ascii=False)
    prompt = f"""Review this combined checklist ({len(combined_checklist)} items).

REVIEW CRITERIA:
1. DUPLICATES: Merge near-duplicate items across patent types.
2. VAGUENESS: Rewrite any item where a reader might say "I'm not sure if
   this counts." Make the scope crystal clear.
3. GAPS: Are any innovation axes missing coverage? Add items if needed.
4. GRANULARITY: Split items that test two things at once. Merge items that
   are too narrow to be individually meaningful.
5. WEIGHTS: Normalize so total weight ≈ 1.0. No single item > 0.15.
6. KNOWN APPROACHES: Ensure every item with score=1 (Partial) has clear
   guidance on which alternative approaches count as partial matches.

TECHNOLOGY CHOICES:
{tc_text}

CURRENT CHECKLIST:
{cl_text}

Output the FINAL checklist (same JSON format, renumber IDs c1, c2, ...):
{{"criteria": [...]}}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        raw = await call_llm_with_pdfs(system, prompt, [source_pdf_path], thinking_budget=8192)
    else:
        raw = await call_llm(system, prompt, thinking_budget=8192)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            result = json.loads(m.group())
            criteria = result.get("criteria", [])
            if criteria:
                return criteria
        except json.JSONDecodeError:
            pass
    return combined_checklist


async def generate_search_queries(
    checklist: list[dict],
    summary: str,
    cpc_subclass: str,
    persona: str | None = None,
) -> dict:
    """Step G: Generate search queries from checklist items. Each item gets
    2-3 query formulations, grouped by innovation axis."""
    system = (persona or INITIAL_PERSONAS["plan"]) + " Output JSON only."
    cl_text = "\n".join(
        f'{c.get("id","")}: {c.get("criterion","")}'
        for c in checklist
    )
    prompt = f"""Turn this checklist into prior art search queries.

RULES:
- For EACH checklist item, generate 2-3 query formulations:
  (a) using the paper's exact terminology
  (b) using synonyms or alternative terminology
  (c) targeting adjacent fields where similar techniques exist
- GROUP items that share an innovation axis.
- Each group needs anchor_terms (must-have keywords) and expansion_terms
  (broaden the search).

CPC subclass: {cpc_subclass}

INVENTION SUMMARY:
{summary[:1500]}

CHECKLIST:
{cl_text}

Output JSON:
{{"groups": [
  {{"group_id": "g1", "label": "group description",
    "atoms": ["c1", "c2"],
    "intent": "what this group targets",
    "patent_query": "full Google Patents query string",
    "paper_query": "full Google Scholar query string",
    "anchor_terms": [["term1", "term2"], ["term3", "term4"]],
    "expansion_terms": [["broader1"], ["adjacent_field_term"]]
  }}
]}}"""

    raw = await call_llm(system, prompt, thinking_budget=4096)
    m = re.search(r'\{.*\}', raw, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"groups": []}


async def decompose_invention(
    paper_text: str,
    feedback: dict | None = None,
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> str:
    """3-step ask-loop decompose:
    Step 0: LLM identifies the paper's own sections/aspects.
    Step 1: For each section, extract elements with verbatim quotes (1 call per section).
    Step 2: Classify all elements and build SS structure.
    """

    system = (persona or "You are a patent analyst.") + " Output JSON only."

    # ── Step 0: Let LLM discover the paper's sections ──
    step0_prompt = f"""Read this paper and list its major technical sections or aspects.
Do NOT use generic labels — use the paper's own section titles or describe
the specific technical aspect covered. Aim for 3-7 sections that together
cover the entire invention.

Output JSON:
{{"sections": ["section name or aspect", ...]}}
{_feedback_block(feedback)}"""

    if source_pdf_path and Path(source_pdf_path).exists():
        step0_raw = await call_llm_with_pdfs(
            system, step0_prompt, [source_pdf_path], thinking_budget=2048)
    else:
        step0_raw = await call_llm(
            system,
            f"{step0_prompt}\n\n════ PAPER TEXT ════\n{paper_text[:15000]}",
            thinking_budget=2048,
        )

    sections = []
    m0 = re.search(r'\{.*\}', step0_raw, re.DOTALL)
    if m0:
        try:
            sections = json.loads(m0.group()).get("sections", [])
        except json.JSONDecodeError:
            pass
    if not sections:
        sections = ["Full document"]

    # ── Step 1: Extract elements per section ──
    all_sections_list = ", ".join(f'"{s}"' for s in sections)
    all_elements = []
    for sec in sections:
        step1_prompt = f"""════ EXTRACT ELEMENTS: "{sec}" ════
This paper has these sections/aspects: [{all_sections_list}].
You are now extracting elements for: "{sec}".

Read the FULL paper for context, but extract elements that primarily
belong to "{sec}". If an element spans multiple sections, include it
in whichever section it is most central to.

Extract every distinct technical component, method, algorithm, or design
decision relevant to this aspect.

Rules:
- Use the paper's own terminology
- Each quote: a SHORT verbatim excerpt (under 40 words)
- Do NOT add elements not described in the paper

Output JSON:
{{"section": "{sec}", "elements": [
  {{"name": "short name", "quote": "verbatim quote", "section": "{sec}"}},
  ...
]}}"""

        if source_pdf_path and Path(source_pdf_path).exists():
            raw = await call_llm_with_pdfs(
                system, step1_prompt, [source_pdf_path], thinking_budget=2048)
        else:
            raw = await call_llm(
                system,
                f"{step1_prompt}\n\n════ PAPER TEXT ════\n{paper_text[:15000]}",
                thinking_budget=2048,
            )

        m1 = re.search(r'\{.*\}', raw, re.DOTALL)
        if m1:
            try:
                parsed = json.loads(m1.group())
                for el in parsed.get("elements", []):
                    el["section"] = sec
                    all_elements.append(el)
            except json.JSONDecodeError:
                pass

    if not all_elements:
        return json.dumps(_salvage_ss_from_text(step0_raw), ensure_ascii=False)

    # Deduplicate by name (case-insensitive)
    seen_names: set[str] = set()
    deduped: list[dict] = []
    for el in all_elements:
        key = (el.get("name") or "").strip().lower()
        if key and key not in seen_names:
            seen_names.add(key)
            deduped.append(el)
    all_elements = deduped

    step1 = {"elements": all_elements}
    elements_text = json.dumps(all_elements, indent=2, ensure_ascii=False)

    system_structure = (persona or "You are a patent analyst.") + " Output JSON only."
    step2_raw = await call_llm(
        system_structure,
        f"""════ STEP 2: CLASSIFY AND STRUCTURE ════
Below are technical elements extracted from a paper WITH verbatim quotes.
Now classify each and build a Source Structure (SS).

For each element, assign a role:
- "novel" — the paper explicitly claims this as their contribution
- "building_block" — known technique used as part of the system
- "cited_prior" — explicitly cited as someone else's prior work

Then identify connections between elements (how they feed into each other).

Finally, write an SS Synopsis: ONE sentence summarizing the invention
following actor → operation → object/outcome.

EXTRACTED ELEMENTS:
{elements_text}

Output JSON:
{{
  "elements": [
    {{"id": "e1", "name": "...", "role": "novel|building_block|cited_prior",
      "source_quote": "copy the verbatim quote from the input elements",
      "description": "1-sentence what it does in this invention"}},
    ...
  ],
  "connections": [
    {{"from": "e1", "to": "e2", "relation": "short description"}},
    ...
  ],
  "synopsis": "One sentence: actor → operation → object/outcome"
}}""",
        thinking_budget=4096,
    )

    m2 = re.search(r'\{.*\}', step2_raw, re.DOTALL)
    ss = None
    if m2:
        try:
            ss = json.loads(m2.group())
        except json.JSONDecodeError:
            pass

    if not ss or not ss.get("elements"):
        ss = _promote_step1_to_ss(step1)

    ss["_raw_elements"] = step1.get("elements", [])
    _backfill_quotes(ss, step1.get("elements", []))
    return json.dumps(ss, indent=2, ensure_ascii=False)


def _promote_step1_to_ss(step1: dict) -> dict:
    """Step 2 failed — build a valid SS from Step 1's raw elements."""
    elements = []
    for i, raw in enumerate(step1.get("elements", []), 1):
        elements.append({
            "id": f"e{i}",
            "name": raw.get("name", f"element_{i}"),
            "role": "unknown",
            "source_quote": raw.get("quote", ""),
            "description": raw.get("section", ""),
        })
    return {
        "elements": elements,
        "connections": [],
        "synopsis": "(Step 2 classification failed — elements unclassified)",
    }


def _salvage_ss_from_text(raw_text: str) -> dict:
    """Both steps failed to produce JSON — extract what we can from text."""
    lines = [ln.strip() for ln in raw_text.split("\n") if ln.strip()]
    elements = []
    for i, line in enumerate(lines[:30], 1):
        cleaned = re.sub(r'^[\d\-\.\)\*]+\s*', '', line).strip()
        if len(cleaned) > 5:
            elements.append({
                "id": f"e{i}",
                "name": cleaned[:120],
                "role": "unknown",
                "source_quote": "",
                "description": "",
            })
    return {
        "elements": elements or [{"id": "e1", "name": "(extraction failed)",
                                   "role": "unknown", "source_quote": "",
                                   "description": ""}],
        "connections": [],
        "synopsis": "(Structured extraction failed — salvaged from text)",
        "_salvaged": True,
    }


def _backfill_quotes(ss: dict, raw_elements: list[dict]):
    """Ensure SS elements carry source_quote from Step 1 raw extraction."""
    raw_by_name = {}
    for r in raw_elements:
        name = (r.get("name") or "").lower().strip()
        if name:
            raw_by_name[name] = r.get("quote", "")
    for elem in ss.get("elements", []):
        if elem.get("source_quote"):
            continue
        name = (elem.get("name") or "").lower().strip()
        if name in raw_by_name:
            elem["source_quote"] = raw_by_name[name]
            continue
        for rn, rq in raw_by_name.items():
            if rn in name or name in rn:
                elem["source_quote"] = rq
                break


def compute_decompose_grounding(ucd_str: str) -> dict:
    """Compute grounding metrics for decompose output. Deterministic."""
    try:
        ss = json.loads(ucd_str) if isinstance(ucd_str, str) else ucd_str
    except (json.JSONDecodeError, TypeError):
        return {"grounding_ratio": 0.0, "novel_ratio": 0.0, "ambiguous_count": 0,
                "total_elements": 0}
    elements = ss.get("elements", [])
    if not elements:
        return {"grounding_ratio": 0.0, "novel_ratio": 0.0, "ambiguous_count": 0,
                "total_elements": 0}
    quoted = sum(1 for e in elements if e.get("source_quote"))
    novel = sum(1 for e in elements if e.get("role") == "novel")
    ambiguous = sum(1 for e in elements
                    if e.get("role") in (None, "", "unknown"))
    return {
        "grounding_ratio": round(quoted / len(elements), 4),
        "novel_ratio": round(novel / len(elements), 4),
        "ambiguous_count": ambiguous,
        "total_elements": len(elements),
    }


def compute_ssr_grounding(checklist: list) -> dict:
    """Compute grounding metrics for SSR checklist. Deterministic."""
    if not checklist:
        return {"evidence_coverage": 0.0, "weight_concentration": 0.0,
                "total_criteria": 0}
    dicts = [c for c in checklist if isinstance(c, dict)]
    if not dicts:
        return {"evidence_coverage": 1.0, "weight_concentration": 0.0,
                "total_criteria": len(checklist)}
    weights = [d.get("weight", 1.0) for d in dicts]
    total_w = sum(weights)
    max_w = max(weights) if weights else 0
    return {
        "evidence_coverage": 1.0,
        "weight_concentration": round(max_w / total_w, 4) if total_w > 0 else 0.0,
        "total_criteria": len(dicts),
    }


def compute_eval_grounding(scoring_report: list[dict]) -> dict:
    """Compute evaluation grounding metrics across all docs. Deterministic."""
    if not scoring_report:
        return {"avg_denom_coverage": 0.0, "avg_evidence_density": 0.0,
                "low_confidence_docs": 0, "total_docs": 0}
    denom_coverages = []
    evidence_densities = []
    low_conf = 0
    for doc in scoring_report:
        cr = doc.get("checklist_results", doc.get("similarity_categories", {}))
        if not cr:
            low_conf += 1
            continue
        total = len(cr)
        scored = sum(1 for v in cr.values()
                     if isinstance(v, dict) and v.get("score") is not None)
        with_evidence = sum(1 for v in cr.values()
                           if isinstance(v, dict) and v.get("evidence_quote"))
        dc = scored / total if total > 0 else 0
        ed = with_evidence / total if total > 0 else 0
        denom_coverages.append(dc)
        evidence_densities.append(ed)
        if dc < 0.5 or ed < 0.3:
            low_conf += 1

    avg_dc = sum(denom_coverages) / len(denom_coverages) if denom_coverages else 0
    avg_ed = sum(evidence_densities) / len(evidence_densities) if evidence_densities else 0
    return {
        "avg_denom_coverage": round(avg_dc, 4),
        "avg_evidence_density": round(avg_ed, 4),
        "low_confidence_docs": low_conf,
        "total_docs": len(scoring_report),
    }


def compute_entropy_profile(
    ssr_grounding: dict,
    eval_grounding: dict,
) -> dict:
    """Aggregate all grounding metrics into an entropy profile. Deterministic."""
    ec = ssr_grounding.get("evidence_coverage", 0)
    dc = eval_grounding.get("avg_denom_coverage", 0)
    ed = eval_grounding.get("avg_evidence_density", 0)

    if ec >= 0.8 and dc >= 0.7 and ed >= 0.8:
        confidence = "high"
    elif ec < 0.5 or dc < 0.4:
        confidence = "low"
    else:
        confidence = "medium"

    degradation = []
    wc = ssr_grounding.get("weight_concentration", 0)
    tc = ssr_grounding.get("total_criteria", 0)
    if wc > 0.5 and tc > 0:
        degradation.append(
            f"SSR: top criterion holds {wc:.0%} of total weight")
    lcd = eval_grounding.get("low_confidence_docs", 0)
    if lcd > 0:
        degradation.append(
            f"evaluate: {lcd} docs have low confidence scores")

    return {
        "phase2_ssr_evidence_coverage": ec,
        "phase2_ssr_weight_concentration": wc,
        "phase4_avg_denom_coverage": dc,
        "phase4_avg_evidence_density": ed,
        "phase4_low_confidence_docs": lcd,
        "overall_confidence": confidence,
        "degradation_points": degradation,
    }


async def generate_checklist(
    summary: str, ucd: str, persona: str | None = None
) -> list[str]:
    """Generate SSR (Structural Scoring Rubric) with weighted criteria and 3-level scale.
    Returns a list of dicts, but falls back to list of strings for backward compat."""
    system = (persona + " Output JSON only.") if persona else (
        "You are a patent claim analyst. Output JSON only.")

    # Try to parse ucd as structured SS
    ss = None
    try:
        ss = json.loads(ucd)
    except (json.JSONDecodeError, TypeError):
        pass

    if ss and "elements" in ss:
        novel_elements = [e for e in ss["elements"]
                          if e.get("role") == "novel"]
        elements_block = json.dumps(
            novel_elements or ss["elements"], indent=2, ensure_ascii=False)
        resp = await call_llm(
            system,
            f"""════ TASK: BUILD SSR (Structural Scoring Rubric) ════
From the Source Structure below, create 10-20 weighted evaluation criteria.

Each criterion must:
- Test ONE specific technical element
- Use the invention's actual terminology
- Have a weight (0.0-1.0) reflecting importance to the overall invention
  (weights should sum to ~1.0)
- Define a 3-level match scale specific to that criterion

Focus criteria on NOVEL elements (higher weights) but include key
building blocks too (lower weights) since their specific combination
or configuration may be novel.

SOURCE STRUCTURE ELEMENTS:
{elements_block}

INVENTION SUMMARY:
{summary[:2000]}

Output JSON:
{{"criteria": [
  {{"id": "c1", "element_id": "e1",
    "criterion": "what to test — specific, testable statement",
    "weight": 0.15,
    "scale": {{
      "0": "what Absent looks like for this criterion",
      "1": "what Partial looks like",
      "2": "what Present looks like"
    }}
  }},
  ...
]}}""",
            thinking_budget=4096,
        )
    else:
        resp = await call_llm(
            system,
            f"""════ TASK ════
Derive 15-20 testable criteria from this invention.
Each: atomic, specific, testable, with a weight (0.0-1.0, sum to ~1.0).
"The system includes X that performs Y."
Output JSON: {{"criteria": [{{"id":"c1","criterion":"...","weight":0.1,
"scale":{{"0":"absent","1":"partial","2":"present"}}}},...]}}.

════ INPUT ════
<summary>{summary}</summary>
<breakdown>{ucd}</breakdown>""",
            thinking_budget=4096,
        )

    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            parsed = json.loads(m.group())
            criteria = parsed.get("criteria", [])
            if criteria and isinstance(criteria[0], dict):
                return criteria
        except json.JSONDecodeError:
            pass

    # Fallback: try as flat array
    m2 = re.search(r'\[.*\]', resp, re.DOTALL)
    if m2:
        try:
            return json.loads(m2.group())
        except json.JSONDecodeError:
            pass
    return [line.strip("- •").strip()
            for line in resp.split("\n")
            if line.strip().startswith(("-", "•", "*"))]


async def plan_delegation(
    summary: str, ucd: str, invention_type: str,
    feedback: dict | None = None, persona: str | None = None,
) -> dict:
    resp = await call_llm(
        (persona + " Output JSON only.") if persona else "You are a USPTO examiner. Output JSON only.",
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


def _extract_pdf_text(path: str, max_pages: int = 10, max_chars: int = 60000) -> str:
    import fitz
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc[:max_pages]]
    doc.close()
    t = "\n\n---PAGE---\n\n".join(pages)
    return t[:max_chars] + ("\n[truncated]" if len(t) > max_chars else "")


def _format_criteria_for_eval(checklist: list) -> str:
    """Format SSR criteria or legacy checklist for evaluation prompts.
    When known_approaches are present, include them so the evaluator can
    distinguish 'same method' (Present) from 'different method in same
    category' (Partial)."""
    lines = []
    for i, item in enumerate(checklist):
        if isinstance(item, dict) and "criterion" in item:
            c = item
            weight = c.get("weight", 0)
            scale = c.get("scale", {})
            known = c.get("known_approaches", [])
            line = f"{i+1}. [w={weight:.2f}] {c['criterion']}"
            if known:
                line += f"\n   Known alternatives: {', '.join(known)}"
            line += (
                f"\n   0={scale.get('0','absent')} | "
                f"1={scale.get('1','partial')} | "
                f"2={scale.get('2','present')}")
            lines.append(line)
        else:
            lines.append(f"{i+1}. {item}")
    return "\n".join(lines)


def _is_ssr(checklist: list) -> bool:
    return bool(checklist and isinstance(checklist[0], dict) and "weight" in checklist[0])


async def evaluate_single_document(
    invention_summary: str,
    checklist: list,
    prior_art_pdf_path: str,
    prior_art_title: str,
    prior_art_type: str,
    source_pdf_path: str | None = None,
    source_title: str | None = None,
    persona: str | None = None,
) -> dict:
    """Deep-eval one prior art doc against the invention's SSR criteria (or legacy checklist).
    Uses 3-level scoring (0=Absent, 1=Partial, 2=Present) when SSR is available."""

    use_ssr = _is_ssr(checklist)
    cl_text = _format_criteria_for_eval(checklist)

    system = (persona + " Output JSON only.") if persona else (
        "You are a US patent examiner comparing a SOURCE invention "
        "against one PRIOR ART document. Output JSON only.")

    prior_art_text = _extract_pdf_text(prior_art_pdf_path)

    source_section = ""
    if source_pdf_path and Path(source_pdf_path).exists():
        source_text = _extract_pdf_text(source_pdf_path, max_chars=40000)
        source_section = (
            f"════ SOURCE INVENTION (full text) ════\n"
            f"TITLE: {source_title or '(unknown)'}\n"
            f"<source>\n{source_text}\n</source>\n\n")

    if use_ssr:
        scoring_instruction = (
            "For EACH criterion, assign a match_score:\n"
            "  2 = Present — the prior art explicitly describes this element "
            "(cite section/quote)\n"
            "  1 = Partial — related concept exists but differs in specifics\n"
            "  0 = Absent — not found in the prior art\n"
            "Use the scale descriptions provided with each criterion as guidance.\n"
            "For score 1 or 2, you MUST include evidence_quote: a verbatim excerpt "
            "from the prior art that supports the score.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<criterion>": {"score": 0|1|2, "analysis": "why this score", '
            '"evidence_quote": "verbatim excerpt from prior art or empty string", '
            '"match": true|false},\n'
            '    ...all items...\n'
            '  }')
    else:
        scoring_instruction = (
            "For EACH checklist item set match=true only with explicit evidence "
            "from the prior art text. Include a verbatim quote as evidence.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<item>": {"analysis": "evidence", "evidence_quote": "verbatim excerpt", '
            '"match": true|false},\n'
            '    ...all items...\n'
            '  }')

    prompt = f"""{source_section}INVENTION SUMMARY:
{invention_summary[:3000]}

EVALUATION CRITERIA ({len(checklist)} items):
{cl_text}

════ PRIOR ART CANDIDATE ════
TITLE: {prior_art_title}
TYPE: {prior_art_type}
<prior_art>
{prior_art_text}
</prior_art>

TASK:
1. Is this the SAME document as the source? (identical title/authors/DOI)
   If yes, set is_source_duplicate=true.
2. {scoring_instruction}

JSON output:
{{
  "is_source_duplicate": true | false,
  "duplicate_reason": "if true",
  "anticipation_assessment": "102 analysis in 1-2 sentences",
  "key_teachings": "103 relevant elements in 1-2 sentences",
  "rs_synopsis": "One sentence: what this prior art does (actor→operation→outcome)",
  {output_schema}
}}"""

    try:
        resp = await call_llm(system, prompt, thinking_budget=8192)
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            if use_ssr:
                _backfill_match_from_score(result)
            return result
    except Exception as e:
        return {"title": prior_art_title, "match_type": prior_art_type,
                "error": str(e), "checklist_results": {}}
    return {"title": prior_art_title, "checklist_results": {}}


def _backfill_match_from_score(result: dict):
    """Ensure backward compat: set match=true when score >= 2."""
    for v in result.get("checklist_results", {}).values():
        if isinstance(v, dict) and "score" in v and "match" not in v:
            v["match"] = v["score"] >= 2


async def evaluate_single_document_text(
    invention_summary: str,
    checklist: list,
    prior_art_text: str,
    prior_art_title: str,
    prior_art_type: str,
    persona: str | None = None,
) -> dict:
    """Abstract-only evaluation. Output shape matches evaluate_single_document."""

    use_ssr = _is_ssr(checklist)
    cl_text = _format_criteria_for_eval(checklist)
    system = (persona + " Output JSON only.") if persona else (
        "You are a US patent examiner. Output JSON only.")

    if use_ssr:
        scoring_instruction = (
            "For EACH criterion assign score: 2=Present, 1=Partial, 0=Absent.\n"
            "Abstract is limited — if silent on a criterion, score 0. "
            "Do NOT infer beyond what the text states.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<criterion>": {"score": 0|1|2, "analysis": "...", '
            '"match": true|false},\n    ...all items...\n  }')
    else:
        scoring_instruction = (
            "For EACH item, match=true only when the abstract explicitly "
            "discusses that element. When silent, match=false.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<item>": {"analysis": "...", "match": true|false},\n'
            '    ...all items...\n  }')

    prompt = f"""INVENTION: {invention_summary[:3000]}

CRITERIA ({len(checklist)} items):
{cl_text}

PRIOR ART "{prior_art_title}" ({prior_art_type}) — abstract/snippet only:
<document>
{prior_art_text[:4000]}
</document>

{scoring_instruction}

JSON output:
{{
  "anticipation_assessment": "1-2 sentences",
  "key_teachings": "1-2 sentences",
  "rs_synopsis": "One sentence: what this prior art does",
  {output_schema}
}}"""
    try:
        resp = await call_llm(system, prompt, thinking_budget=4096)
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            result["source"] = "abstract"
            if use_ssr:
                _backfill_match_from_score(result)
            return result
    except Exception as e:
        return {"title": prior_art_title, "match_type": prior_art_type,
                "error": str(e), "checklist_results": {},
                "source": "abstract_failed"}
    return {"title": prior_art_title, "checklist_results": {},
            "source": "abstract_noparse"}


async def evaluate_batch(
    invention_summary: str,
    checklist: list[str],
    documents: list[dict],
    max_concurrent: int = 2,
    source_pdf_path: str | None = None,
    source_title: str | None = None,
    on_doc_done: Callable[[], None] | None = None,
    persona: str | None = None,
) -> list[dict]:
    """Evaluate a batch of candidates.

    on_doc_done: optional callback fired after each doc's eval completes
    (success or failure). Used by the pipeline to heartbeat so the zombie
    detector doesn't falsely flag this long-running phase as stuck.
    """
    sem = asyncio.Semaphore(max_concurrent)

    async def one(doc):
        async with sem:
            try:
                pdf = doc.get("local_pdf", "")
                if pdf and Path(pdf).exists():
                    res = await evaluate_single_document(
                        invention_summary, checklist, pdf,
                        doc.get("title", ""), doc.get("match_type", "Paper"),
                        source_pdf_path=source_pdf_path,
                        source_title=source_title,
                        persona=persona,
                    )
                    res["source"] = "pdf"
                    return res
                text = (doc.get("abstract") or "").strip() or (doc.get("snippet") or "").strip()
                if len(text) >= 120:
                    return await evaluate_single_document_text(
                        invention_summary, checklist, text,
                        doc.get("title", ""), doc.get("match_type", "Paper"),
                        persona=persona,
                    )
                return {"title": doc.get("title", ""), "match_type": doc.get("match_type", ""),
                        "checklist_results": {}, "source": "no_content"}
            finally:
                if on_doc_done is not None:
                    try:
                        on_doc_done()
                    except Exception:
                        pass

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
    weak_titles = "\n".join(
        f"- ({d.get('semantic_score', 0):.2f}) {d.get('title', '')[:150]}"
        for d in weak_results[:5]
    )
    orig_q = "\n".join(f"- {q}" for q in original_queries)
    resp = await call_llm(
        "You are a USPTO patent search expert. Output JSON only. "
        "You refine failed search queries based on what was found.",
        f"""════ TASK (template) ════
A search group's queries returned only weakly-relevant results. \
Look at what we found and propose 1-2 REFINED queries that would find more relevant prior art.

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


async def self_check(
    label: str,
    source_text: str,
    generated_text: str,
    source_pdf_path: str | None = None,
) -> dict:
    """Verify whether generated_text is faithful to source_text (or source PDF).

    When source_pdf_path is provided, the PDF is sent natively to the LLM
    so the check works even for scanned / image-only documents.

    Returns {ok: bool, issues: [...], suggestion: str}.
    """
    system = ("You verify whether a generated text is faithful to a source "
              "document. Output JSON only.")
    user_prompt = f"""════ GENERATED TEXT (the "{label}" step produced this) ════
```
{generated_text[:6000]}
```

For each substantive claim in the GENERATED TEXT, check if it appears in or
follows from the SOURCE DOCUMENT. Output strict JSON:
{{
  "ok": true | false,
  "issues": ["short issue 1", "short issue 2"],
  "suggestion": "one-line fix or 'looks good'"
}}"""
    if source_pdf_path and Path(source_pdf_path).exists():
        resp = await call_llm_with_pdfs(
            system, user_prompt, [source_pdf_path])
    else:
        resp = await call_llm(
            system,
            f"════ SOURCE DOCUMENT ════\n```\n{source_text[:20000]}\n```"
            f"\n\n{user_prompt}",
        )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"ok": True, "issues": [], "suggestion": "self-check parse failed"}


async def generate_overall_summary(invention_summary: str, top_matches: list[dict], persona: str | None = None) -> str:
    """Generate plain-language novelty assessment for faculty inventors (not patent lawyers)."""
    if not top_matches:
        raise ValueError(
            "generate_overall_summary called with zero matches — refusing to let the LLM "
            "hallucinate prior art. Caller must guard against empty input."
        )
    matches_lines = []
    for m in top_matches[:10]:
        title = m.get('title', '')
        score = m.get('similarity_score', 0)
        css = m.get('css', 0)
        ewss = m.get('ewss', 0)
        rs_syn = m.get('rs_synopsis', '')
        teachings = m.get('key_teachings', '') or m.get('snippet', '')
        score_str = f"CSS={css:.0%}, EWSS={ewss:.0%}" if css or ewss else f"{score:.0%} overlap"
        line = f"- **{title}** ({score_str})"
        if rs_syn:
            line += f"\n  What it does: {rs_syn}"
        line += f"\n  Key relevant content: {teachings[:300]}"
        matches_lines.append(line)
    matches = "\n".join(matches_lines)

    _default_system = (
        "You are explaining a patent novelty assessment to a university faculty "
        "inventor who is NOT a patent lawyer. Use plain English. Be honest, specific, "
        "and actionable. Avoid legal jargon (no '102', '103', 'anticipation', "
        "'prior art teaches'). Don't be vague."
    )
    return await call_llm(
        persona or _default_system,
        f"""════ TASK (template) ════
Write a novelty assessment of this invention for the inventor. \
The inventor is a faculty member who knows their research area but is not familiar with patent law.

Structure your response in EXACTLY these sections (use markdown):

## What you invented (in plain words)
2-3 sentences restating what the inventor built, in their own field's language.

## What's already out there
For the most relevant existing work (top 3-5 from the matches below), explain in 1-2 sentences EACH:
- What that prior work did
- Which specific aspects of YOUR invention it covers (or comes close to)

## What appears genuinely new
List the specific technical elements of your invention that none of the matches \
seem to cover. Be concrete — point at actual components, methods, or claims, \
not abstract concepts.

## Honest assessment
1-2 sentences. Pick one: "Looks novel and worth pursuing", \
"Has overlap but a clear novel angle", \
"Significant overlap — narrow your claims", or "Likely already known". \
Then explain why in plain terms.

## Suggested next steps
2-3 concrete actions the inventor can take \
(e.g., "Read paper X carefully — it's the closest match", \
"Talk to your tech transfer office about claim Y", \
"Focus your patent application on aspect Z").

════ INPUT (from prior LLM steps) ════
INVENTION SUMMARY:
{invention_summary[:2500]}

TOP MATCHES (sorted by overlap):
{matches}
""",
    )
