"""
Phase 2 & 4: NAA — Novelty Assessment Agent prompts.

Phase 2 (Decomposition): LLM-dependent
Phase 4 (Evaluation): LLM-dependent per-document scoring
"""

UNSTRUCTURED_DECOMPOSITION_PROMPT = """Extract a faithful technical breakdown of the invention described below.

Rules:
- Bullet-list of key elements: components, steps, interfaces, materials, algorithms
- Use the manuscript's own terminology
- Do NOT add features not in the manuscript
- Do NOT assess novelty
- Focus on WHAT is built or done

<invention_summary>
{summary}
</invention_summary>
"""

CHECKLIST_GENERATION_PROMPT = """Derive 20-30 concrete, testable checklist requirements from the invention below.

Rules:
- Each item is a specific, atomic, testable requirement
- Expressed as: "The system/method includes/performs X that Y"
- Grounded in disclosed technical details only
- One claim per item (no compound requirements)
- Focus on what is built or done, not motivation
- Output as a JSON array of strings

<invention_summary>
{summary}
</invention_summary>

<technical_breakdown>
{ucd}
</technical_breakdown>
"""

DELEGATION_PLANNING_PROMPT = """Decompose the invention into 8-20 atomic components and create 3-6 overlapping search groups.

For each atom:
- id: short identifier (A, B, C, ...)
- name: descriptive name
- keywords: list of 5-7 search terms (both specific and broader synonyms)
- core_score: 0.0-1.0 (how central to the invention's novelty)
- distinctiveness_score: 0.0-1.0 (how unique vs. generic)

For each search group:
- group_id: G1, G2, ...
- atoms: list of atom ids included
- label: descriptive label
- intent: "core" | "combo" | "secondary"
- anchor_terms: 1-2 synonym groups (max 3 phrases each) for core concepts
- expansion_terms: 1-2 synonym groups for broadening

IMPORTANT: Maintain system context. Each atom connects to the overall system.
Search groups must find the "bottle" (complete system), not just isolated "caps".
Follow US patent examination methodology (35 USC 102/103).

Output as JSON.

<invention_summary>
{summary}
</invention_summary>

<technical_breakdown>
{ucd}
</technical_breakdown>

<invention_type>
{invention_type}
</invention_type>
"""

CHECKLIST_EVALUATION_PROMPT = """Evaluate this prior art document against each checklist item.

For each item, determine if the document clearly discloses that requirement.
- match=true ONLY if clearly and explicitly disclosed
- match=false if missing, unclear, or only partially addressed
- Provide brief analysis (1 sentence) for each

Output as JSON object where keys are checklist items and values are:
{{"analysis": "reason", "match": true/false}}

<prior_art>
Title: {title}
Type: {match_type}
ID: {pub_num}
Snippet: {snippet}
</prior_art>

<checklist>
{checklist_json}
</checklist>
"""

OVERALL_SUMMARY_PROMPT = """Write an executive summary of the novelty assessment.

Given:
- The invention summary
- The checklist
- The top scoring prior art matches with their scores

Assess:
1. Which aspects of the invention are well-covered by prior art
2. Which aspects appear novel
3. Overall risk level: HIGH (>70%), MEDIUM (40-70%), LOW (20-40%), CLEAR (<20%)

Be specific about which checklist items have the most/least coverage.
200-300 words.

<invention_summary>
{summary}
</invention_summary>

<top_matches>
{top_matches_json}
</top_matches>
"""
