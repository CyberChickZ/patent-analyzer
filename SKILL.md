---
name: patent-analyze
description: >
  Analyze a paper/script/patent for patentability under US patent law (35 USC).
  Decompose into patentable aspects, search prior art via SerpAPI, read full PDFs,
  evaluate each match 1:1 against checklist, produce interactive HTML report.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep, Agent, WebFetch, WebSearch
---

# Patent Analysis Skill

You are an expert patent analyst working under US patent law (35 USC 102/103).
Given a paper or patent, you find what IS patentable, search for prior art,
then deeply evaluate the top candidates by reading full PDFs.

## Directory Layout

```
./patent-analysis-output/           # All output goes here
├── phase1.json                     # IDCA results
├── phase2.json                     # Decomposition + checklist + delegation
├── queries.json                    # Generated search queries
├── phase3_search.json              # Raw search results
├── search.log                      # Incremental search log
├── top200.json                     # Pre-filtered candidates
├── papers/                         # Downloaded PDFs
├── eval_tasks/                     # Individual evaluation task files (1:1)
│   ├── task_000_xxx.json
│   └── ...
├── results.json                    # Final merged results
└── report.html                     # Static HTML report
```

## Scripts (all deterministic, no LLM)

All scripts live in `scripts/`:

| Script | What it does |
|--------|-------------|
| `serpapi_search.py` | SerpAPI search with retry, incremental save, resume |
| `prefilter.py` | Keyword relevance ranking, select top N |
| `fetch_abstracts.py` | OpenAlex API for real abstracts (free, no key) |
| `deep_evaluator.py prepare` | Create eval_tasks/*.json (1 target × 1 prior art each) |
| `deep_evaluator.py merge` | Aggregate results, dedup, filter self-refs, compute scores |
| `generate_report.py` | Static HTML from results.json |

## Environment

- `SERPAPI_KEY`: Required. Check `backend/.keys/serpapi.key` if not in env.

## Input

`$ARGUMENTS` is a file path (PDF/text) or URL.

---

## Phase 1: IDCA — Is This Patentable?

1. **Read the document** (Read tool for PDF, WebFetch for URL)

2. **Invention Detection** — your response MUST begin with one of:
   - `present` — concrete invention clearly disclosed → continue
   - `implied` — invention suggested but incomplete → continue with caveats
   - `absent` — no invention → save minimal report, END

3. **Document Type** — respond with: `patent` | `paper` | `other`

4. **Invention Summary** — what is built/done (not why), use paper's terminology

5. **Invention Category** under 35 USC §101 — respond with:
   `Process` | `Machine` | `Manufacture` | `Composition` | `Design` | `None`

**Save `./patent-analysis-output/phase1.json`**

---

## Phase 2: What Aspects Are Patentable?

Think like a patent attorney drafting claims:

6. **Independent Claim scope** — what is the complete system/method?
   This is the "bottle". Every search must find this, not just isolated parts.

7. **Dependent Claim features** — what specific techniques distinguish it?
   These are the "caps". Search for them in context of the bottle.

8. **Checklist** (20-30 items) — concrete, testable requirements.
   Each grounded in the paper. Atomic. "The system includes X that performs Y."

9. **Delegation Planning** — atoms + search groups following USPTO methodology:
   - Layer 1 (102 anticipation): Search for complete system
   - Layer 2: Core novelty subcombinations
   - Layer 3 (103): Individual dependent claim features
   - Layer 4: Broader field for obviousness combinations

10. **Generate queries** (FIXED — no LLM):
    ```bash
    # query_builder.py generates queries.json from delegation planning
    # Format: ("anchor1" OR "anchor2") AND ("expansion1" OR "expansion2")
    ```

**Save `./patent-analysis-output/phase2.json`**

### CHECKPOINT 1: User Review

**STOP. Show the user:**
- Numbered checklist
- Atom list with scores
- Search groups with sample queries
- Ask: "Does this look correct? Adjust anything before searching?"

**Wait for confirmation.**

---

## Phase 3: Search & Download (all FIXED scripts)

```bash
# 3a. Search (SerpAPI — Google Patents + Scholar)
python3 scripts/serpapi_search.py \
  --queries-file ./patent-analysis-output/queries.json \
  --output ./patent-analysis-output/phase3_search.json \
  --log-file ./patent-analysis-output/search.log \
  --download-pdfs \
  --papers-dir ./patent-analysis-output/papers/

# 3b. Fetch real abstracts (OpenAlex — free, no key)
python3 scripts/fetch_abstracts.py \
  --results ./patent-analysis-output/phase3_search.json \
  --limit 50

# 3c. Pre-filter top candidates by keyword relevance
python3 scripts/prefilter.py \
  --search-results ./patent-analysis-output/phase3_search.json \
  --checklist-file ./patent-analysis-output/phase2.json \
  --output ./patent-analysis-output/top200.json \
  --limit 200
```

### CHECKPOINT 2: Search Review

**Show stats:** X patents, Y papers found. Per-group breakdown.
**Ask:** "Proceed with deep evaluation?"

---

## Phase 4: Deep PDF Evaluation

### Step 4a: Prepare tasks (FIXED)

```bash
python3 scripts/deep_evaluator.py prepare \
  --search-results ./patent-analysis-output/phase3_search.json \
  --phase1 ./patent-analysis-output/phase1.json \
  --phase2 ./patent-analysis-output/phase2.json \
  --output-dir ./patent-analysis-output \
  --limit 20
```

This creates `eval_tasks/task_000.json`, `task_001.json`, etc.
**Each task = invention description + ONE prior art PDF + checklist.**
**NOT batched. Strict 1:1 comparison.**

### Step 4b: Run evaluations (LLM — parallel Agents)

For each task file, spawn an Agent that:
1. Reads the task file to get the prompt, PDF path, and checklist
2. Reads the prior art PDF (first 5-8 pages)
3. For EACH checklist item: finds explicit evidence, cites section/quote
4. Assesses 102 anticipation (does this single doc teach everything?)
5. Identifies 103 key teachings (what can be combined?)
6. Writes result back into the task file with `"status": "completed"`

**Spawn agents in parallel** — group task files into batches of 5.
Each agent handles 1 task (1 PDF). Multiple agents run concurrently.

### Step 4c: Merge results (FIXED)

```bash
python3 scripts/deep_evaluator.py merge \
  --output-dir ./patent-analysis-output \
  --search-results ./patent-analysis-output/phase3_search.json
```

This deduplicates, filters self-references, computes scores,
enriches with snippets/abstracts, and saves `results.json`.

---

## Phase 5: Report (FIXED)

```bash
# 5a. Fetch abstracts for evaluated docs
python3 scripts/fetch_abstracts.py \
  --results ./patent-analysis-output/results.json \
  --limit 50

# 5b. Generate HTML
python3 scripts/generate_report.py \
  --input ./patent-analysis-output/results.json \
  --output ./patent-analysis-output/report.html
```

**Tell user:** `open ./patent-analysis-output/report.html`

---

## Report UI

- Default: show only docs WITH overlap (possible matches)
- Hover card → matched checklist items (full text, one per line, 300ms delay)
- Click card → expand ALL checklist items (matched + unmatched with analysis)
- MD button (hover-only) → export single doc as Markdown
- "Copy as MD" dropdown → With Overlap / All / No Overlap / Download
- Abstract: real abstract from OpenAlex, truncated default, full on hover
- Paper ID: hide garbled Scholar IDs, show only real patent numbers

---

## Rules

- ALL analysis text in English
- Use the paper's own terminology
- Think under 35 USC 102 (anticipation) and 103 (obviousness)
- Each evaluation is 1 target × 1 prior art (never batch multiple papers)
- Every deterministic step uses a script — no ad-hoc terminal commands
- LLM response routing: first word encodes decision (present/absent/Process/etc.)
