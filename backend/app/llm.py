"""
LLM calls via Google GenAI (Vertex AI).

Pipeline LLM calls:
  Phase 1: detect_and_summarize_invention, craft_personas
  Phase 2: scan_innovation_landscape, expand_technology_choices,
           determine_patent_types, generate_checklist_for_type,
           review_checklist, generate_search_queries
  Phase 4: evaluate_single_document (×N), generate_overall_summary
  Harness: self_check, refine_search_query

Deterministic (zero LLM tokens):
  detect_invention, classify_document, classify_category
"""

import asyncio
import contextvars
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types

from app import prompts
from google.genai.errors import APIError
from tenacity import retry, retry_if_exception, stop_after_attempt, wait_random_exponential

GC_PROJECT = os.getenv("GC_PROJECT", "aime-hello-world")
MODEL = os.getenv("LLM_MODEL", "gemini-2.5-pro")
MAX_TOKENS = 8192

# Per-stage override: LLM_MODEL_<STAGE> (extract / screen / eval / idca); unset → MODEL.
STAGES = ("extract", "screen", "eval", "idca")


def stage_model(stage: str) -> str:
    """Model id for a pipeline stage: LLM_MODEL_<STAGE> if set, else the global MODEL."""
    return os.getenv(f"LLM_MODEL_{stage.upper()}") or MODEL

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
        # Vertex DSQ guidance: "we recommend using the global endpoint. Unlike a
        # regional endpoint ... the global endpoint dynamically routes your
        # requests to the region with the most available capacity" (VERTEX_LOCATION)
        _client = genai.Client(
            vertexai=True,
            project=GC_PROJECT,
            location=os.getenv("VERTEX_LOCATION", "global"),
        )
    return _client


# Gemini 3 dropped the numeric budget: "The raw numeric thinking_budget parameter is
# no longer supported across all Gemini 3 models. Use the thinking_level string enum
# instead." (models/guides/gemini-3-5-flash). MINIMAL is rejected by 3.7/3.8 Flash and
# 3.x Pro ("thinking_level=\"MINIMAL\" is not available for 3.8 Flash"), so budget 0
# maps to LOW there.
_NO_MINIMAL = ("gemini-3.7-", "gemini-3.8-", "gemini-3-pro", "gemini-3.1-pro")


def _is_gemini3(model: str) -> bool:
    m = re.match(r"gemini-(\d+)", model or "")
    return bool(m) and int(m.group(1)) >= 3


def _thinking_level(model: str, thinking_budget: int) -> str:
    """Budget → level for Gemini 3 (LLM_THINKING_LEVEL overrides): 0 → MINIMAL/LOW,
    ≤2048 → LOW, ≤4096 → MEDIUM, else HIGH."""
    forced = os.getenv("LLM_THINKING_LEVEL")
    if forced:
        return forced.upper()
    if thinking_budget <= 0:
        return "LOW" if model.startswith(_NO_MINIMAL) else "MINIMAL"
    if thinking_budget <= 2048:
        return "LOW"
    if thinking_budget <= 4096:
        return "MEDIUM"
    return "HIGH"


def _build_config(system: str, max_tokens: int, thinking_budget: int,
                  response_schema: dict | None = None,
                  model: str | None = None) -> types.GenerateContentConfig:
    model = model or MODEL
    config = types.GenerateContentConfig(
        system_instruction=system,
        max_output_tokens=max_tokens,
    )
    if response_schema:
        # JSON mode: the model can only emit an instance of the schema
        config.response_mime_type = "application/json"
        config.response_schema = response_schema
    try:
        if _is_gemini3(model):
            config.thinking_config = types.ThinkingConfig(
                thinking_level=_thinking_level(model, thinking_budget),
                include_thoughts=True,
            )
        elif thinking_budget > 0:
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


_LLM_RPM = int(os.getenv("LLM_RPM", "120"))   # smoothing only; DSQ has no fixed RPM (12-way burst test: 0 × 429)
_llm_gates: dict = {}

# Per-model meter (prompt / output / thought tokens, calls, 429s) — read by evals for cost.
usage: dict[str, dict[str, int]] = {}


def _meter(model: str) -> dict[str, int]:
    return usage.setdefault(model, {"calls": 0, "prompt_tokens": 0, "output_tokens": 0,
                                    "thought_tokens": 0, "errors_429": 0, "seconds": 0.0})


def _record_usage(model: str, resp, seconds: float = 0.0) -> None:
    m = _meter(model)
    m["calls"] += 1
    m["seconds"] += seconds
    um = getattr(resp, "usage_metadata", None)
    if um is None:
        return
    m["prompt_tokens"] += int(getattr(um, "prompt_token_count", 0) or 0)
    m["output_tokens"] += int(getattr(um, "candidates_token_count", 0) or 0)
    m["thought_tokens"] += int(getattr(um, "thoughts_token_count", 0) or 0)


async def _smooth(model: str | None = None) -> None:
    """Shared per-minute gate across every process on this machine/instance, one per model."""
    model = model or MODEL
    gate = _llm_gates.get(model)
    if gate is None:
        from patent_analyzer.runtime_state import MinuteGate
        gate = _llm_gates[model] = MinuteGate(f"vertex:{model}", _LLM_RPM)
    try:
        waited = await gate.wait()
        if waited:
            print(f"[LLM] smoothed: waited {waited:.1f}s for a slot ({_LLM_RPM}/min, {model})")
    except Exception:
        pass


def _is_retryable(exc: BaseException) -> bool:
    if isinstance(exc, APIError) and exc.code in (429, 503, 500):
        # DSQ 429 = "temporary high contention for a specific shared resource", not a fixed quota
        print(f"[LLM] {exc.code} from Vertex: {str(getattr(exc, 'message', exc))[:300]}")
        if exc.code == 429:
            _meter(_current_model.get() or MODEL)["errors_429"] += 1
        return True
    name = type(exc).__name__
    return any(k in name for k in ("Timeout", "ServiceUnavailable", "ResourceExhausted"))


_retry_decorator = retry(
    retry=retry_if_exception(_is_retryable),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(4),
    reraise=True,
)


# Model of the in-flight call, so the retry predicate can attribute a 429 to it.
_current_model: contextvars.ContextVar[str | None] = contextvars.ContextVar("_current_model", default=None)


@_retry_decorator
async def call_llm(
    system: str,
    user: str,
    max_tokens: int = MAX_TOKENS,
    thinking_budget: int = 0,
    response_schema: dict | None = None,
    model: str | None = None,
) -> str:
    """model=None → the global MODEL; stages pass stage_model(<stage>)."""
    model = model or MODEL
    _current_model.set(model)
    client = get_client()
    config = _build_config(system, max_tokens, thinking_budget, response_schema, model=model)
    await _smooth(model)
    t0 = time.monotonic()
    resp = await client.aio.models.generate_content(
        model=model,
        contents=[types.Part.from_text(text=user)],
        config=config,
    )
    _record_usage(model, resp, time.monotonic() - t0)
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
    model: str | None = None,
) -> str:
    """Send a single PDF as a native multi-modal part to Gemini (no truncation).

    Only used for bulk initial screening where text extraction is acceptable.
    For deep eval use call_llm_with_pdfs which supports multiple files.
    """
    return await call_llm_with_pdfs(system, user, [pdf_path], max_tokens, thinking_budget, model=model)


# Vertex AI inline-bytes cap is ~20MB per request. Pad for safety.
_INLINE_PDF_CAP_BYTES = 18 * 1024 * 1024


@_retry_decorator
async def call_llm_with_pdfs(
    system: str,
    user: str,
    pdf_paths: list[str],
    max_tokens: int = MAX_TOKENS,
    thinking_budget: int = 0,
    image_parts: list[bytes] | None = None,
    response_schema: dict | None = None,
    model: str | None = None,
) -> str:
    """Send one or more PDFs as native multi-modal parts to Gemini.

    Uploads PDF bytes directly — the model sees layout, figures, tables. No text
    extraction, no truncation. If a file exceeds Vertex's inline cap (~20MB),
    it falls back to text extraction for THAT file only (others still go as PDF)
    and prefixes a [fallback_text] marker so the LLM knows the input was lossy.
    response_schema switches on JSON mode exactly as in call_llm.
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
    for img_data in (image_parts or []):
        parts.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))
    parts.append(types.Part.from_text(text=user))

    model = model or MODEL
    _current_model.set(model)
    client = get_client()
    config = _build_config(system, max_tokens, thinking_budget, response_schema, model=model)
    await _smooth(model)
    t0 = time.monotonic()
    resp = await client.aio.models.generate_content(
        model=model,
        contents=parts,
        config=config,
    )
    _record_usage(model, resp, time.monotonic() - t0)
    text, thoughts = _extract_text_and_thoughts(resp)
    if text.startswith("```"):
        text = text.split("```", 2)[1].strip()
        if text.startswith("json"):
            text = text[4:].strip()
    file_list = ", ".join(f"{Path(p).name} ({Path(p).stat().st_size//1024}KB)"
                          for p in pdf_paths if p and Path(p).exists())
    img_note = f" + {len(image_parts)} figure screenshots" if image_parts else ""
    _emit(system, f"[PDFs: {file_list}{img_note}]\n\n{user}", text, thoughts)
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
    domain-specific personas. Falls back to INITIAL_PERSONAS on failure."""
    roles_block = "\n\n".join(
        f"### {role}\n{text}" for role, text in INITIAL_PERSONAS.items()
    )
    keys_list = ", ".join(INITIAL_PERSONAS.keys())
    n = len(INITIAL_PERSONAS)
    resp = await call_llm(
        "You rewrite expert personas to be domain-specific. Output JSON only.",
        f"""Take these {n} generic expert personas and rewrite each to be specific
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

Output JSON with keys: {keys_list}.
Each value is the rewritten persona string.""",
        max_tokens=5000,
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
    document_text: str,
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
Read the ENTIRE attached document and perform IDCA (Invention
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

STEP 3 — Input Mode (how is this document structured?):
  - "academic_paper": formal paper with abstract, methods, results structure.
  - "informal_description": handwritten notes, project description, proposal,
    or informal write-up describing what someone is building/planning.
  - "patent_draft": document with patent claim language or structured as a
    patent application.
  - "technical_report": formal but not claiming novelty — engineering report,
    technical documentation.

STEP 4 — Fields & Classification:
  - fields_map: 3-7 technical field labels from broad to specific.
    Example: ["Computer Vision", "Object Detection", "Anchor-Free Detection"]
  - cpc_subclass: best-guess 4-character CPC subclass code (e.g. G06N, H04L,
    A61B). Pick the single most relevant one.
  - category: §101 type (Process/Machine/Manufacture/Composition/Design/None).

STEP 5 — Summary (only if status is Present or Implied):
  400-800 words describing WHAT is built/done, using the paper's own
  terminology. Cover the full technical contribution including methods,
  architecture, key results, and novel components. If Implied,
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
  "input_mode": "academic_paper" | "informal_description" | "patent_draft" | "technical_report",
  "source_citation": "APA citation from document info, or empty string",
  "cpc_subclass": "G06N",
  "publication_date": "YYYY-MM-DD if determinable from the document (arXiv date, copyright year, conference date), or empty string",
  "summary": "400-800 word summary, or empty string if Absent"
}}"""
    if source_pdf_path and Path(source_pdf_path).exists():
        resp = await call_llm_with_pdfs(
            system, task_prompt, [source_pdf_path], thinking_budget=4096,
            model=stage_model("idca"))
    else:
        resp = await call_llm(
            system,
            f"{task_prompt}\n\n════ DOCUMENT TEXT ════\n"
            f"```\n{document_text}\n```",
            thinking_budget=4096,
            model=stage_model("idca"),
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
                "input_mode":      str(d.get("input_mode", "academic_paper") or "academic_paper"),
                "category":        str(d.get("category", "None") or "None"),
                "fields_map":      list(d.get("fields_map", []) or []),
                "source_citation":  str(d.get("source_citation", "") or ""),
                "cpc_subclass":     str(d.get("cpc_subclass", "") or ""),
                "publication_date": str(d.get("publication_date", "") or ""),
                "summary":          str(d.get("summary", "") or ""),
            }
        except json.JSONDecodeError:
            pass
    return {
        "status_determination": "Present",
        "has_innovation": True,
        "reasoning":      "(JSON parse failed — defaulting to proceed)",
        "doc_type":       "invention",
        "input_mode":     "academic_paper",
        "category":       "None",
        "fields_map":     [],
        "source_citation":  "",
        "cpc_subclass":     "",
        "publication_date": "",
        "summary":          resp[:2000],
    }



# ── IDCA: structured Doc JSON (the single text layer downstream) ──
#
# How others represent a parsed paper (sources checked 2026-09-18):
# - GROBID, https://grobid.readthedocs.io/en/latest/Introduction/ : "Full text extraction and
#   structuring from PDF articles, including a model for the overall document segmentation and
#   models for the structuring of the text body (paragraph, section titles, reference and footnote
#   callouts, figures, tables, ...)"; TEI output = header (title/abstract) + body of <div><head>/<p>
#   + <figure> + <formula> + bibliography. Its JSON export has "separate sections for bibliographic
#   metadata, body text, figures and tables, and references" (Grobid-service.md).
# - MinerU content_list.json, https://opendatalab.github.io/MinerU/reference/output_files/ :
#   "stores all readable content blocks in reading order as a flat structure"; headings are text
#   blocks with `text_level: 1/2/...`, figures carry `img_caption: [..]`, formulas are
#   `{"type": "equation", "text": "$$...$$", "text_format": "latex"}`.
# - Gemini structured output, https://ai.google.dev/gemini-api/docs/structured-output :
#   "generate responses that adhere to a provided JSON Schema"; "Very large or deeply nested
#   schemas may be rejected"; "always validate values in your application".
# Hence: a FLAT section list with `level` (MinerU style, no recursive schema), figures and
# equations as separate lists (GROBID style), enforced via response_schema and validated here.

DOC_JSON_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "abstract": {"type": "STRING"},
        "sections": {"type": "ARRAY", "items": {
            "type": "OBJECT",
            "properties": {"heading": {"type": "STRING"}, "level": {"type": "INTEGER"},
                           "paragraphs": {"type": "ARRAY", "items": {"type": "STRING"}}},
            "required": ["heading", "level", "paragraphs"]}},
        "figures": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": {"label": {"type": "STRING"}, "caption": {"type": "STRING"}},
            "required": ["label", "caption"]}},
        "equations": {"type": "ARRAY", "items": {
            "type": "OBJECT", "properties": {"label": {"type": "STRING"}, "latex": {"type": "STRING"}},
            "required": ["label", "latex"]}},
        "references_count": {"type": "INTEGER"},
    },
    "required": ["title", "abstract", "sections", "figures", "equations", "references_count"],
}

_DOC_JSON_MAX_TOKENS = 65535

DOC_JSON_PROMPT = prompts.register_default("idca.docjson", """════ TASK ════
Transcribe the attached document into structured JSON. This is a TRANSCRIPTION, not a summary:
every body paragraph must be copied VERBATIM, in reading order, nothing dropped or shortened.

Rules:
- title: the document title. abstract: the abstract text (empty string if none).
- sections: one entry per heading, in reading order, FLAT (no nesting). level = 1 for a top-level
  heading ("3 Method"), 2 for a subsection ("3.2 Loss"), 3 for a sub-subsection. Keep the heading
  text as printed (with its number). Text before the first heading goes into a section with
  heading "" and level 1. Do NOT emit the reference list as a section.
- paragraphs: the section's body paragraphs, verbatim, one string each. Join lines broken by the
  page layout; remove hyphenation at line ends; drop running headers/footers and page numbers.
  Keep inline math as LaTeX ($...$). Do NOT put figure captions or table contents in paragraphs.
- figures: every figure/table caption: label ("Figure 1", "Table 2"), caption text verbatim.
- equations: every numbered display equation: label ("1", "2", ...), latex.
- references_count: number of entries in the reference list (0 if none).
{extra}""")


def _clean_doc_json(d: dict) -> dict:
    """Coerce a model response to the Doc JSON contract (drop empties, fix types)."""
    sections = []
    for sec in d.get("sections") or []:
        if not isinstance(sec, dict):
            continue
        paras = [" ".join(str(x).split()) for x in (sec.get("paragraphs") or []) if str(x or "").strip()]
        heading = " ".join(str(sec.get("heading") or "").split())
        if not paras and not heading:
            continue
        try:
            level = max(1, min(4, int(sec.get("level") or 1)))
        except (TypeError, ValueError):
            level = 1
        sections.append({"heading": heading, "level": level, "paragraphs": paras})
    figures = [{"label": " ".join(str(f.get("label") or "").split()), "caption": " ".join(str(f.get("caption") or "").split())}
               for f in (d.get("figures") or []) if isinstance(f, dict) and str(f.get("caption") or "").strip()]
    equations = [{"label": " ".join(str(e.get("label") or "").split()), "latex": str(e.get("latex") or "").strip()}
                 for e in (d.get("equations") or []) if isinstance(e, dict) and str(e.get("latex") or "").strip()]
    try:
        refs = max(0, int(d.get("references_count") or 0))
    except (TypeError, ValueError):
        refs = 0
    return {"title": " ".join(str(d.get("title") or "").split()),
            "abstract": " ".join(str(d.get("abstract") or "").split()),
            "sections": sections, "figures": figures, "equations": equations, "references_count": refs}


async def build_doc_json(document_text: str, source_pdf_path: str | None = None) -> dict | None:
    """ONE LLM CALL (JSON mode): the document as Doc JSON
    {title, abstract, sections:[{heading, level, paragraphs}], figures:[{label, caption}],
     equations:[{label, latex}], references_count}. The PDF goes to Gemini natively
    (layout, captions, math); a text input is sent as-is. None when the call or the
    parse fails — the caller keeps the fitz/plain text as the fallback text layer.
    Independent of detect_and_summarize_invention so that prompt (and its cache key)
    is untouched."""
    system = ("You are a document transcription engine. Reproduce the document's text faithfully "
              "into the requested JSON structure. Output JSON only.")
    if source_pdf_path and Path(source_pdf_path).exists():
        prompt = prompts.render("idca.docjson", extra="")
        resp = await call_llm_with_pdfs(system, prompt, [source_pdf_path], max_tokens=_DOC_JSON_MAX_TOKENS,
                                        response_schema=DOC_JSON_SCHEMA)
    else:
        prompt = prompts.render("idca.docjson", extra=f"\n════ DOCUMENT TEXT ════\n```\n{(document_text or '')[:_EXTRACTION_DOC_CAP]}\n```")
        resp = await call_llm(system, prompt, max_tokens=_DOC_JSON_MAX_TOKENS, response_schema=DOC_JSON_SCHEMA)
    data = _extraction_json(resp)
    if not data:
        return None
    doc = _clean_doc_json(data)
    if not doc["sections"] and not doc["abstract"]:
        return None
    return doc


# ════════════════════════════════════════════════════════════
# Phase 2: Expert-driven innovation analysis
# ════════════════════════════════════════════════════════════


async def scan_innovation_landscape(
    summary: str,
    fields_map: list[str],
    cpc_subclass: str,
    cpc_context: str,
    source_pdf_path: str | None = None,
    persona: str | None = None,
) -> list[dict]:
    """Identify innovation axes — technical dimensions where the paper
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
{summary}

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
    """For one innovation axis, enumerate known approaches from the
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
{summary}

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
    """Determine which patent types (Process/Machine/Manufacture/
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
{summary}

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
    """Generate specific, testable checklist items for one patent type.
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
{summary}

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
    """Expert review — merge duplicates, tighten vague items,
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
7. SEARCHABILITY — Each item must be specific enough to produce meaningful
   search results in patent/paper databases, but NOT so narrow that no prior
   art could ever match it. If an item describes a minor implementation detail
   (e.g., a specific normalization constant, a specific loss scaling trick) that
   would never appear as the primary contribution of any paper or patent,
   MERGE it into the parent axis’s broader item. A good test: "Could someone
   write a paper primarily about this specific technique?" If no, merge it.

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
    """Generate search queries from checklist items. Each item gets
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
{summary}

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


def _extract_pdf_text(path: str, max_pages: int = 10, max_chars: int = 60000) -> str:
    import fitz
    doc = fitz.open(path)
    pages = [p.get_text() for p in doc[:max_pages]]
    doc.close()
    t = "\n\n---PAGE---\n\n".join(pages)
    return t[:max_chars] + ("\n[truncated]" if len(t) > max_chars else "")


def _render_figure_pages(path: str, dpi: int = 150, max_screenshots: int = 10) -> list[tuple[int, bytes]]:
    """Detect PDF pages with figures/tables/diagrams and render them as PNG."""
    import fitz
    try:
        doc = fitz.open(path)
    except Exception:
        return []
    try:
        figure_indices = []
        for i, page in enumerate(doc):
            try:
                images = page.get_images(full=True)
                if any(img[2] > 100 and img[3] > 100 for img in images):
                    figure_indices.append(i)
                    continue
                if len(page.get_drawings()) > 50:
                    figure_indices.append(i)
            except Exception:
                pass
        results = []
        for i in figure_indices[:max_screenshots]:
            try:
                pix = doc[i].get_pixmap(dpi=dpi)
                results.append((i + 1, pix.tobytes("png")))
            except Exception:
                pass
        return results
    finally:
        doc.close()


def _format_criteria_for_eval(checklist: list) -> str:
    """Format SSR criteria or legacy checklist for evaluation prompts.
    When known_approaches are present, include them so the evaluator can
    distinguish 'same method' (Present) from 'different method in same
    category' (Partial)."""
    lines = []
    for i, item in enumerate(checklist):
        if isinstance(item, dict) and "criterion" in item:
            c = item
            weight = float(c.get("weight", 0))
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
    """Deep-eval one prior art doc against the invention's SSR criteria.
    Sends full native PDFs (source + prior art) plus figure screenshots to Gemini."""

    use_ssr = _is_ssr(checklist)
    cl_text = _format_criteria_for_eval(checklist)

    system = (persona + " Output JSON only.") if persona else (
        "You are a US patent examiner comparing a SOURCE invention "
        "against one PRIOR ART document. Output JSON only.")

    pdfs = []
    source_label = ""
    if source_pdf_path and Path(source_pdf_path).exists():
        pdfs.append(source_pdf_path)
        source_label = (
            f"FIRST attached PDF = SOURCE INVENTION "
            f"(title: {source_title or '(unknown)'}).\n")

    pdfs.append(prior_art_pdf_path)
    pa_ordinal = "SECOND" if source_label else "FIRST"

    fig_pages = _render_figure_pages(prior_art_pdf_path)
    fig_note = ""
    if fig_pages:
        pages_str = ", ".join(str(pn) for pn, _ in fig_pages)
        fig_note = (
            f"\n\nAdditionally, {len(fig_pages)} high-res screenshot(s) of "
            f"prior art pages with figures/tables are attached (pages {pages_str}). "
            f"Examine these for visual evidence.")

    if use_ssr:
        scoring_instruction = (
            "For EACH criterion, assign a match_score:\n"
            "  2 = Present — the prior art explicitly describes this element "
            "(cite section/quote)\n"
            "  1 = Partial — related concept exists but differs in specifics\n"
            "  0 = Absent — not found in the prior art\n"
            "Use the scale descriptions provided with each criterion as guidance.\n"
            "For score 1 or 2, you MUST include evidence_quotes: 1 to 5 verbatim excerpts "
            "(10-40 words each, one per passage) from the prior art that support the score.\n"
            "Pay attention to figures, tables, and diagrams — visual evidence counts.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<criterion>": {"score": 0|1|2, "analysis": "why this score", '
            '"evidence_quotes": ["verbatim excerpt from prior art", "..."], '
            '"match": true|false},\n'
            '    ...all items...\n'
            '  }')
    else:
        scoring_instruction = (
            "For EACH checklist item set match=true only with explicit evidence "
            "from the prior art text or figures. Include a verbatim quote or "
            "figure description as evidence.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<item>": {"analysis": "evidence", "evidence_quote": "verbatim excerpt", '
            '"match": true|false},\n'
            '    ...all items...\n'
            '  }')

    prompt = f"""{source_label}{pa_ordinal} attached PDF = PRIOR ART candidate.{fig_note}

INVENTION SUMMARY:
{invention_summary}

EVALUATION CRITERIA ({len(checklist)} items):
{cl_text}

PRIOR ART CANDIDATE:
TITLE: {prior_art_title}
TYPE: {prior_art_type}

Read the ENTIRE prior art document — every page, every figure, every table.

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
        resp = await call_llm_with_pdfs(
            system, prompt, pdfs, thinking_budget=8192,
            image_parts=[img for _, img in fig_pages] or None,
            model=stage_model("eval"))
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            result["keys_unaligned"] = _align_checklist_keys(result, checklist)
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


def _align_checklist_keys(result: dict, checklist: list) -> int:
    """Re-key checklist_results onto the exact criterion strings the reducer
    looks up. Models return the numbered label ("3", "3.") or a trimmed /
    paraphrased criterion; unmatched keys silently score 0 downstream.
    Returns the number of keys that could not be aligned."""
    cr = result.get("checklist_results")
    if not isinstance(cr, dict) or not checklist:
        return 0
    crits = [c.get("criterion", "") if isinstance(c, dict) else str(c) for c in checklist]
    norm = {re.sub(r"\W+", " ", c.lower()).strip(): c for c in crits}
    aligned, unmatched = {}, 0
    for k, v in cr.items():
        key = str(k).strip()
        m = re.match(r"^(\d+)\.?$", key)
        if m and 1 <= int(m.group(1)) <= len(crits):
            aligned[crits[int(m.group(1)) - 1]] = v
            continue
        if key in crits:
            aligned[key] = v
            continue
        nk = re.sub(r"\W+", " ", key.lower()).strip()
        hit = norm.get(nk) or next((c for n, c in norm.items() if nk[:40] and (nk[:40] in n or n[:40] in nk)), None)
        if hit:
            aligned[hit] = v
        else:
            aligned[key] = v
            unmatched += 1
    result["checklist_results"] = aligned
    return unmatched


async def evaluate_single_document_text(
    invention_summary: str,
    checklist: list,
    prior_art_text: str,
    prior_art_title: str,
    prior_art_type: str,
    persona: str | None = None,
    doc_mode: str = "abstract",
) -> dict:
    """Text-only evaluation. Output shape matches evaluate_single_document.

    doc_mode="abstract" (default): short snippet, silence scores 0.
    doc_mode="full_text": numbered full document; every non-zero score must
    carry a verbatim evidence_quote so it can be verified against the text.
    """

    use_ssr = _is_ssr(checklist)
    cl_text = _format_criteria_for_eval(checklist)
    system = (persona + " Output JSON only.") if persona else (
        "You are a US patent examiner. Output JSON only.")
    full = doc_mode == "full_text"
    doc_label = "full text with numbered paragraphs" if full else "abstract/snippet only"
    evidence_field = ('"evidence_quotes": ["verbatim excerpt", "..."], '
                      if full else "")

    if use_ssr:
        scoring_instruction = (
            "For EACH criterion assign score: 2=Present, 1=Partial, 0=Absent.\n"
            + ("For score 1 or 2 you MUST list evidence_quotes: 1 to 5 verbatim excerpts copied "
               "from the document (exact wording, 10-40 words each), one per passage that "
               "discloses the criterion, most direct first; never merge text from two places. "
               "If you cannot quote anything, score 0. "
               if full else
               "Abstract is limited — if silent on a criterion, score 0. ")
            + "Do NOT infer beyond what the text states.")
        output_schema = (
            '"checklist_results": {\n'
            '    "<criterion>": {"score": 0|1|2, "analysis": "...", '
            + evidence_field +
            '"match": true|false},\n    ...all items...\n  }')
    else:
        scoring_instruction = (
            f"For EACH item, match=true only when the {doc_label} explicitly "
            "discusses that element. When silent, match=false."
            + (" For match=true list 1-5 verbatim evidence_quotes." if full else ""))
        output_schema = (
            '"checklist_results": {\n'
            '    "<item>": {"analysis": "...", ' + evidence_field + '"match": true|false},\n'
            '    ...all items...\n  }')

    prompt = f"""INVENTION: {invention_summary}

CRITERIA ({len(checklist)} items):
{cl_text}

PRIOR ART "{prior_art_title}" ({prior_art_type}) — {doc_label}:
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
    try:
        resp = await call_llm(system, prompt, thinking_budget=4096, model=stage_model("eval"))
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        if m:
            result = json.loads(m.group())
            result["title"] = prior_art_title
            result["match_type"] = prior_art_type
            result["source"] = "full_text" if full else "abstract"
            result["keys_unaligned"] = _align_checklist_keys(result, checklist)
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
{invention_summary}

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
{original_input[:_EXTRACTION_DOC_CAP]}

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
            system, user_prompt, [source_pdf_path], model=stage_model("extract"))
    else:
        resp = await call_llm(
            system,
            f"════ SOURCE DOCUMENT ════\n```\n{source_text[:_EXTRACTION_DOC_CAP]}\n```"
            f"\n\n{user_prompt}",
            model=stage_model("extract"),
        )
    m = re.search(r'\{.*\}', resp, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return {"ok": True, "issues": [], "suggestion": "self-check parse failed"}


async def generate_combination_analysis(
    invention_summary: str,
    top_matches: list[dict],
    persona: str | None = None,
) -> str | None:
    """Ask whether combining top references makes the invention obvious (§103 analysis)."""
    if len(top_matches) < 2:
        return None
    refs = []
    for i, m in enumerate(top_matches[:5], 1):
        title = m.get("title", "")
        teachings = m.get("key_teachings", "") or m.get("snippet", "")
        if teachings:
            refs.append(f"Reference {i}: {title}\n  Key teachings: {teachings[:300]}")
    if len(refs) < 2:
        return None
    refs_block = "\n\n".join(refs)
    system = (
        persona or
        "You are a patent analyst assessing whether combining multiple prior art "
        "references would make an invention obvious to a person of ordinary skill. "
        "Be specific and concise."
    )
    return await call_llm(
        system,
        f"""Given this invention and the references below, assess whether a skilled
practitioner would naturally combine elements from these references to arrive
at the invention.

INVENTION:
{invention_summary}

REFERENCES:
{refs_block}

Answer in 2-4 sentences:
1. Which specific elements from which references could be combined?
2. Would this combination be natural/obvious to someone in this field, or would it require an inventive leap?
3. What specific aspect of the invention (if any) would NOT be obvious even after combining all references?

Be concrete — name the specific technical elements, not abstract concepts.""",
    )


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
{invention_summary}

TOP MATCHES (sorted by overlap):
{matches}
""",
    )


OBVIOUSNESS_EXPLAIN_PROMPT = prompts.register_default("report.obviousness_explain", """════ TASK ════
A deterministic rule has already made the determination below from verified evidence. Your job is
ONLY to write the examiner-style reasoning for it (MPEP 2143: "there must be some articulated
reasoning with some rational underpinning"). You may NOT change the determination, add or remove
references, or claim an element is disclosed when the chart says it is not.

Write 2 short paragraphs of plain prose (no headings, no bullets, no preamble):

1. Why these references, read together, could render the claim obvious: for each reference name the
   elements it supplies (use the chart), then say whether combining them is "combining prior art
   elements according to known methods to yield predictable results" (KSR rationale A) — same field,
   same problem, no change in the elements' respective functions — or whether the gap elements would
   need a routine modification. Be concrete: name the elements.

2. What cuts against it: whether an articulated reason to combine is missing, whether any reference
   teaches away or solves a different problem, and which uncovered element (if any) is more than a
   routine modification. If nothing cuts against it, say so in one sentence.

Never say the invention is or is not patentable / allowable / grantable — this is blocking-risk
reasoning over the references at hand only.

════ INPUT (rule output) ════
DETERMINATION: {label_text}
RULE REASON: {reason}

INVENTION (summary):
{summary}

REFERENCES AND THE ELEMENTS EACH DISCLOSES (located verbatim quotes in brackets):
{references}

ELEMENTS NO REFERENCE DISCLOSES:
{uncovered}""")


async def explain_obviousness(adjudication: dict, chart: dict, invention_summary: str,
                              docs_results: list[dict] | None = None) -> str:
    """One call: prose reasoning for a §103 determination the rule already
    made. Input is the rule output (references, per-element coverage,
    gaps); the model cannot change the label. Returns "" when the label is
    not "103" or the call fails."""
    if not adjudication or adjudication.get("label") != "103" or not chart or not chart.get("rows"):
        return ""
    teach = {(d.get("pub_num") or d.get("title") or ""): (d.get("key_teachings") or d.get("rs_synopsis") or "")
             for d in docs_results or [] if isinstance(d, dict)}
    refs = []
    for i, d in enumerate(chart["docs"]):
        lines = [f"Reference {i + 1}: {d.get('pub_num') or ''} {d.get('title') or ''} "
                 f"(discloses {d.get('n_covered', 0)}/{chart.get('n_elements', 0)} elements)"]
        if teach.get(d.get("key")):
            lines.append(f"  What it teaches: {teach[d['key']][:400]}")
        for r in chart["rows"]:
            c = r["cells"][i]
            if c.get("covered"):
                lines.append(f"  - {r['element']} [{c.get('quote', '')[:200]}]")
        refs.append("\n".join(lines))
    label_text = ("combination of the references below" if len(chart["docs"]) >= 2 and (adjudication.get("combo") or {}).get("docs")
                  else "primary reference below plus a secondary reference or routine modification for the gaps")
    prompt = prompts.render("report.obviousness_explain",
                            label_text=f"blocking risk under §103 — {label_text}",
                            reason=adjudication.get("reason", ""), summary=(invention_summary or "")[:3000],
                            references="\n\n".join(refs) or "(none)",
                            uncovered="\n".join(f"- {u}" for u in chart.get("uncovered") or []) or "(none — every element is disclosed by the set)")
    system = ("You are a US patent examiner writing the reasoning section of an obviousness rejection. "
              "Plain, specific prose; no legal conclusions about patentability.")
    try:
        return (await call_llm(system, prompt, max_tokens=1500)).strip()
    except Exception:
        return ""


SEARCH_FACETS_PROMPT = prompts.register_default("search.facets", """For EACH element below, give four facets of search terms:
  named     — 0-4 DISTINCTIVE NAMES that identify this element in this document, COPIED as written
              from the element text or the invention context: chemical / biological / material /
              organism / product / algorithm / protocol names. For an acronym give BOTH the acronym
              and its expansion as separate entries ("icg", "indocyanine green"). NOT generic product
              categories (tablet pc, camera, server) and NOT the document's own coinage for the
              invention. Leave empty when the element has no such name; never invent one.
  thing     — what the element IS (the core noun/mechanism): 6-10 surface forms
  place     — where/in what context it operates (domain, host system, signal): 4-8 forms
  apparatus — the concrete structural/implementation term: 3-6 forms
RULES: each form is ONE or TWO full English words (the engine stems them: write "damping" not "damp").
Use DIFFERENT vocabulary across forms: the older term, the generic term, the industrial term,
the research term, the term a competitor in another field would use.
NEVER use words so general they appear in every patent: device, member, element, portion, means, unit, system, method, apparatus, module, component, assembly.
DROP THE HEAD NOUN: write "sound damping" not "sound damping device". Lowercase. No quotes inside terms.

INVENTION CONTEXT: {summary}

ELEMENTS:
{listing}

JSON output: {{"facets": {{"<element id>": {{"named": [...], "thing": [...], "place": [...], "apparatus": [...]}}, ...}}}}""")


async def facet_elements(elements: list[dict], summary: str) -> dict[str, dict]:
    """One call: for each element give search facets in Google Patents
    vocabulary. Returns {element_id: {thing[], place[], apparatus[]}}.
    Shape follows patent-search-pilot's measured query craft: 6-14 surface
    forms per facet, two words max, no generic head nouns, vocabularies
    from different communities. Full words, not truncated stems: Google
    Patents stems unquoted keywords itself."""
    if not elements:
        return {}
    listing = "\n".join(f'{e["id"]}: {e["text"]}' for e in elements)
    system = "You write patent search facets. Output JSON only."
    prompt = prompts.render("search.facets", summary=summary[:4000], listing=listing)
    try:
        resp = await call_llm(system, prompt, thinking_budget=2048)
        m = re.search(r'\{.*\}', resp, re.DOTALL)
        data = json.loads(m.group()) if m else {}
        out = {}
        for e in elements:
            f = (data.get("facets") or {}).get(e["id"]) or {}
            out[e["id"]] = {k: [str(t).strip().lower() for t in (f.get(k) or []) if str(t).strip()][:FACET_FORMS_CAP]
                            for k in ("named", "thing", "place", "apparatus")}
        return out
    except Exception:
        return {e["id"]: {"named": [], "thing": [], "place": [], "apparatus": []} for e in elements}


# ════════════════════════════════════════════════════════════
# Extraction (line A): candidate inventions -> claim-language elements
# ════════════════════════════════════════════════════════════

EXTRACTION_LEVELS = ("core", "component", "application")
EXTRACTION_KINDS = ("structure", "step", "condition", "parameter")
_EXTRACTION_DOC_CAP = 150_000
FACET_FORMS_CAP = 10
EXTRACT_ELEMENTS_PROMPT = prompts.register_default("extract.elements", """════ TASK ════
For EACH candidate invention below, draft the independent claims and break the method
claim into elements.

CANDIDATES
{cand_lines}
{prefill_block}
For each candidate output:
  independent_claim_draft:
    method — "A method of ..., comprising: ...; ...; and ..."  (one limitation per clause)
    system — "A <apparatus/system/device> comprising: ...; ...; and ..."
    Both claims must recite EVERY component and EVERY step the document presents as part
    of the invention (typically 4-10 limitations), not a two-clause sketch.
  primary_form — "system" when the contribution is an apparatus / device / composition /
    structure (the document describes parts and how they are arranged), "method" when it is a
    process. This is the claim the elements are cut from.
  elements — the limitations of the PRIMARY claim, in order, one limitation per element:
    id             — "<candidate id>.e0" for the preamble, then .e1, .e2, ...
    text           — the limitation in claim language: one structure (with its configured-to
                     qualifier) or one action. Concatenating the element texts must reproduce
                     the primary claim.
    evidence_quote — 10-40 words COPIED verbatim from the DOCUMENT that support this limitation
    facets         — search vocabulary {{"thing": [...], "place": [...], "apparatus": [...]}},
                     1-3 short lowercase stems each (thing = what it is; place = where / in
                     what host it operates; apparatus = the concrete implementation term)
    kind           — "structure" | "step" | "condition" | "parameter"
  dependent_hints — 0-4 short refinements that could become dependent claims

RULES
- COPY the quote verbatim from the document. DO NOT paraphrase. DO NOT stitch words from
  different sentences. If nothing in the document supports a limitation, leave
  evidence_quote empty rather than inventing one.
- One limitation per element. Do not merge two actions into one element; do not split one
  action into two.
- Use the document's own terms in element text; no "novel", "improved", "efficient".
- In facets never use device/member/element/portion/means/unit/system/method/apparatus/
  module/component; drop the head noun ("sound damp" not "sound damping device").
- The preamble element (e0) names the subject ("A method of X" / "An apparatus for X") and
  carries no limitation.
{feedback_block}
Output strictly this JSON, no preamble:
{{"candidate_inventions": [
  {{"id": "inv1",
    "independent_claim_draft": {{"method": "...", "system": "..."}},
    "primary_form": "system",
    "elements": [
      {{"id": "inv1.e0", "text": "A method of ...", "evidence_quote": "...",
        "facets": {{"thing": ["..."], "place": ["..."], "apparatus": ["..."]}}, "kind": "structure"}}
    ],
    "dependent_hints": ["..."]}}
]}}

════ DOCUMENT ════
```
{document}
```""")

_FACET_BANNED = frozenset("device member element portion means unit system method apparatus module component assembly".split())

_DOC_KIND_GUIDANCE = {
    "paper": ("Look for the invention in the Method / Approach / System / Implementation sections, "
              "not in the Introduction or Related Work: what the authors built, not what they cite."),
    "manuscript": ("Related-work sections have been removed. Look for the invention in the Method / "
                   "Results / Discussion sections: what the authors built themselves."),
    "disclosure": ("The Core Idea and Novelty fields state what the inventor considers new; "
                   "the How It Works field gives the mechanism. Use them in that order."),
    "patent_draft": ("Each independent claim (or claim-like paragraph) is one candidate; keep its scope. "
                     "Dependent claims are not candidates."),
}


def _extraction_json(resp: str) -> dict | None:
    for attempt in (resp, resp[resp.find("{"):resp.rfind("}") + 1] if "{" in resp else ""):
        if not attempt:
            continue
        try:
            d = json.loads(attempt)
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            continue
    return None


def _clean_facets(f) -> dict:
    out = {}
    for k in ("thing", "place", "apparatus"):
        terms = []
        for t in (f or {}).get(k) or []:
            words = [w for w in str(t).strip().lower().split() if w]
            while words and words[-1] in _FACET_BANNED:
                words.pop()
            if words and not all(w in _FACET_BANNED for w in words):
                terms.append(" ".join(words))
        out[k] = terms[:4]
    return out


EXTRACT_CANDIDATES_PROMPT = prompts.register_default("extract.candidates", """════ TASK ════
From the DOCUMENT below (use the SUMMARY only as orientation), list 1-4 candidate
inventions — things that could each be the subject of an independent patent claim.
Order them core first, then component, then application.

level:
  core        — the main contribution as a whole; the subject of the broadest independent claim
  component   — a sub-mechanism / module / step that could stand on its own as an independent claim
  application — a use / deployment of the core in a specific setting

For each candidate:
  concept   — ONE sentence, at most 35 words: what it is, what drives it, and the feature
              that distinguishes it from ordinary practice
  cpc_pred  — up to 3 CPC group codes (e.g. "G06T7/00")

Document kind: {doc_kind}. {guidance}

If the document contains no claimable invention (survey, review, pure theory, opinion,
dataset description, commentary), output an empty list and state no_invention_reason.

Output strictly this JSON, no preamble:
{{"candidate_inventions": [{{"id": "inv1", "concept": "...", "level": "core", "cpc_pred": ["G06T7/00"]}}],
 "no_invention_reason": null}}

════ SUMMARY ════
{summary}

════ DOCUMENT ════
```
{document}
```""")


async def extract_candidates(doc_text: str, summary: str, doc_kind: str = "paper") -> dict:
    """A1 — ONE LLM CALL: list 1-4 candidate inventions from the full document.

    Returns {"candidate_inventions": [{id, concept, level, cpc_pred}],
             "no_invention_reason": None | str}
    """
    guidance = _DOC_KIND_GUIDANCE.get(doc_kind, _DOC_KIND_GUIDANCE["paper"])
    system = ("You are a patent attorney identifying what in a technical document could be "
              "claimed. Output JSON only.")
    prompt = prompts.render("extract.candidates", doc_kind=doc_kind, guidance=guidance,
                            summary=(summary or "")[:4000], document=(doc_text or "")[:_EXTRACTION_DOC_CAP])
    resp = await call_llm(system, prompt, thinking_budget=4096, model=stage_model("extract"))
    data = _extraction_json(resp) or {}
    out = []
    for i, c in enumerate((data.get("candidate_inventions") or [])[:4]):
        if not isinstance(c, dict):
            continue
        concept = " ".join(str(c.get("concept") or "").split())
        if not concept:
            continue
        level = str(c.get("level") or "").strip().lower()
        if level not in EXTRACTION_LEVELS:
            level = "core" if not out else "component"
        cpc = [str(x).strip() for x in (c.get("cpc_pred") or []) if str(x).strip()][:3]
        out.append({"id": f"inv{len(out) + 1}", "concept": concept, "level": level, "cpc_pred": cpc})
    out.sort(key=lambda c: EXTRACTION_LEVELS.index(c["level"]))
    for i, c in enumerate(out, 1):
        c["id"] = f"inv{i}"
    reason = data.get("no_invention_reason")
    reason = str(reason).strip() if reason else None
    if not out and not reason:
        reason = "model returned no candidate inventions" if data else "A1 response could not be parsed"
    return {"candidate_inventions": out, "no_invention_reason": reason if not out else None}


async def extract_elements(doc_text: str, candidates: list[dict], prefill: dict[str, list[str]] | None = None,
                           feedback: dict | None = None) -> dict:
    """A2 — ONE LLM CALL (all candidates together): independent_claim_draft
    {method, system} + primary_form + elements[{id, text, evidence_quote, facets, kind}]
    per candidate; elements are the limitations of the primary-form claim.

    prefill: {candidate_id: [limitation texts]} — element texts fixed (claim mode);
    the model only adds evidence_quote / facets / kind and may not rewrite them.
    Returns {"candidate_inventions": [<candidate + claim draft + elements + dependent_hints>]}
    """
    if not candidates:
        return {"candidate_inventions": []}
    prefill = prefill or {}
    system = ("You are a patent attorney drafting independent claims from a technical document. "
              "Output JSON only.")
    cand_lines = "\n".join(f'- {c["id"]} [{c.get("level", "core")}]: {c.get("concept", "")}' for c in candidates)
    prefill_block = ""
    if prefill:
        listing = "\n".join(f'{cid}:\n' + "\n".join(f"  {cid}.e{i}: {t}" for i, t in enumerate(texts))
                            for cid, texts in prefill.items())
        prefill_block = f"""

════ PREFILLED ELEMENTS (FIXED) ════
The element texts below are the applicant's own claim limitations. Output them EXACTLY
as given — same order, same count, same wording — adding only evidence_quote, facets
and kind for each. Do not rewrite, merge, or split them.
{listing}
"""
    prompt = prompts.render("extract.elements", cand_lines=cand_lines, prefill_block=prefill_block,
                            feedback_block=_feedback_block(feedback), document=(doc_text or "")[:_EXTRACTION_DOC_CAP])
    resp = await call_llm(system, prompt, max_tokens=16384, thinking_budget=4096, model=stage_model("extract"))
    data = _extraction_json(resp)
    if data is None:
        return {"candidate_inventions": [], "error": "A2 response could not be parsed"}
    by_id = {}
    for c in data.get("candidate_inventions") or []:
        if isinstance(c, dict) and c.get("id"):
            by_id[str(c["id"]).strip()] = c

    out = []
    for cand in candidates:
        cid = cand["id"]
        raw = by_id.get(cid) or {}
        draft = raw.get("independent_claim_draft") or {}
        elements = []
        raw_elements = [e for e in (raw.get("elements") or []) if isinstance(e, dict)]
        fixed = prefill.get(cid)
        if fixed:
            raw_elements = raw_elements[:len(fixed)] + [{}] * max(0, len(fixed) - len(raw_elements))
        for i, e in enumerate(raw_elements):
            text = fixed[i] if fixed else " ".join(str(e.get("text") or "").split())
            if not text:
                continue
            kind = str(e.get("kind") or "").strip().lower()
            elements.append({
                "id": f"{cid}.e{len(elements)}",
                "text": text,
                "evidence_quote": " ".join(str(e.get("evidence_quote") or "").split()),
                "facets": _clean_facets(e.get("facets")),
                "kind": kind if kind in EXTRACTION_KINDS else "step",
            })
        form = str(raw.get("primary_form") or "").strip().lower()
        out.append({
            **cand,
            "independent_claim_draft": {
                "method": " ".join(str(draft.get("method") or "").split()),
                "system": " ".join(str(draft.get("system") or "").split()),
            },
            "primary_form": form if form in ("method", "system") else "method",
            "elements": elements,
            "dependent_hints": [str(h).strip() for h in (raw.get("dependent_hints") or []) if str(h).strip()][:4],
        })
    return {"candidate_inventions": out}
