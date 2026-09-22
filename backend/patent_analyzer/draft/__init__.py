"""Draft claims (Step 6): assemble independent claims from the grounded
elements, plan the avoidance from the claim chart, check definiteness,
re-check the new limitations against a small search. Pure functions here;
LLM calls live in app.llm (draft_claims / reword_limitations /
definiteness_advisory) and the node in nodes/draft.py."""
