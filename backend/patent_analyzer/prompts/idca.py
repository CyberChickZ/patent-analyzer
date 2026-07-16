"""
Phase 1: IDCA — Invention Detection & Classification Agent prompts.

These are LLM-dependent steps. Each prompt returns structured output.
The first token of the response encodes the decision for deterministic routing:

  INVENTION_DETECTION → first word: "present" | "implied" | "absent"
  MANUSCRIPT_TYPE     → first word: "patent" | "paper" | "other"
  INVENTION_CATEGORY  → first word: one of the 6 categories
"""

INVENTION_DETECTION_PROMPT = """You are an expert Invention Detection Agent assisting with patent analysis under 35 U.S.C.

Analyze the following manuscript and determine whether it discloses a concrete technology, system, method, article, or composition that a skilled person could reasonably understand how to make or perform.

IMPORTANT: Your response MUST begin with exactly one of these words (lowercase):
  present  — a concrete, useful invention is clearly disclosed
  implied  — an invention is suggested but key details are missing
  absent   — no concrete, useful invention is disclosed

Follow with a brief explanation (2-3 sentences).

Do NOT assess novelty, patentability, or statutory category at this stage.

<manuscript>
{manuscript_text}
</manuscript>
"""

MANUSCRIPT_TYPE_PROMPT = """Classify this document. Your response MUST begin with exactly one word (lowercase):
  patent  — a patent application or granted patent
  paper   — a research/academic paper
  other   — any other document type

Follow with a one-sentence explanation.

<manuscript>
{manuscript_text}
</manuscript>
"""

INVENTION_SUMMARY_PROMPT = """You are an expert patent analyst. Provide a neutral, concrete summary of the invention(s) disclosed in this manuscript.

Rules:
- Focus on WHAT is built or done, not WHY
- Use the manuscript's own terminology
- If multiple inventions exist, list each separately
- Do NOT add features not disclosed in the manuscript
- Do NOT assess novelty or patentability
- 200-400 words

<manuscript>
{manuscript_text}
</manuscript>
"""

INVENTION_CATEGORY_PROMPT = """Classify the primary invention under 35 U.S.C. §101 statutory categories.

Your response MUST begin with exactly one of these words:
  Process       — a series of steps or actions
  Machine       — a concrete apparatus with parts
  Manufacture   — an article produced from raw materials
  Composition   — a combination of substances
  Design        — ornamental design for a manufactured article
  None          — does not fit any statutory category

Follow with reasoning (2-3 sentences) explaining why this category applies.

<invention_summary>
{summary}
</invention_summary>
"""


def parse_detection(response: str) -> str:
    """Extract detection status from response first word. Deterministic."""
    first = response.strip().split()[0].lower().rstrip(".:,;")
    if first in ("present", "implied", "absent"):
        return first
    # Fallback: scan for keywords
    lower = response.lower()
    for status in ("present", "implied", "absent"):
        if status in lower[:50]:
            return status
    return "implied"  # safe default


def parse_doc_type(response: str) -> str:
    """Extract document type from response first word. Deterministic."""
    first = response.strip().split()[0].lower().rstrip(".:,;")
    if first in ("patent", "paper", "other"):
        return first
    return "other"


def parse_category(response: str) -> str:
    """Extract invention category from response first word. Deterministic."""
    first = response.strip().split()[0].lower().rstrip(".:,;")
    valid = {"process", "machine", "manufacture", "composition", "design", "none"}
    if first in valid:
        return first.capitalize()
    return "None"
